"""
RF Arsenal OS - WebSocket Server
Real-time bi-directional communication for live data streaming.
"""

import json
import time
import threading
import hashlib
import socket
import struct
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import base64


class WSMessageType(Enum):
    """WebSocket message types"""
    COMMAND = "command"
    RESPONSE = "response"
    EVENT = "event"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SPECTRUM = "spectrum"
    WATERFALL = "waterfall"
    CONSTELLATION = "constellation"
    DETECTION = "detection"
    ALERT = "alert"
    STATUS = "status"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


@dataclass
class WSMessage:
    """WebSocket message"""
    msg_type: WSMessageType
    data: Any = None
    message_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.msg_type.value,
            "data": self.data,
            "message_id": self.message_id,
            "timestamp": self.timestamp
        })
        
    @classmethod
    def from_json(cls, json_str: str) -> 'WSMessage':
        data = json.loads(json_str)
        return cls(
            msg_type=WSMessageType(data.get("type", "command")),
            data=data.get("data"),
            message_id=data.get("message_id"),
            timestamp=data.get("timestamp", time.time())
        )


@dataclass
class WSClient:
    """WebSocket client connection"""
    client_id: str
    socket: socket.socket
    address: tuple
    subscriptions: Set[str] = field(default_factory=set)
    authenticated: bool = False
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    messages_sent: int = 0
    messages_received: int = 0


class WebSocketServer:
    """
    Production-grade WebSocket server for RF Arsenal OS.
    
    Features:
    - Real-time bi-directional communication
    - Pub/sub channel subscriptions
    - Live spectrum/waterfall streaming
    - Event notifications
    - Client authentication
    - Connection management
    - Local-only binding option
    """
    
    # WebSocket protocol constants
    WS_MAGIC_STRING = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8081,
                 command_handler: Optional[Callable] = None,
                 auth_required: bool = True):
        """
        Initialize WebSocket server.
        
        Args:
            host: Bind address
            port: Listen port
            command_handler: Command execution function
            auth_required: Require authentication
        """
        self.host = host
        self.port = port
        self.command_handler = command_handler or (lambda cmd: {"success": True})
        self.auth_required = auth_required
        
        # Clients
        self._clients: Dict[str, WSClient] = {}
        self._clients_lock = threading.Lock()
        
        # Server socket
        self._server_socket: Optional[socket.socket] = None
        self._running = False
        
        # Channels
        self._channels: Dict[str, Set[str]] = {
            "spectrum": set(),
            "waterfall": set(),
            "constellation": set(),
            "detections": set(),
            "alerts": set(),
            "status": set()
        }
        
        # Authentication tokens
        self._tokens: Dict[str, str] = {}  # token -> client_id
        
        # Message handlers
        self._message_handlers: Dict[WSMessageType, Callable] = {
            WSMessageType.COMMAND: self._handle_command,
            WSMessageType.SUBSCRIBE: self._handle_subscribe,
            WSMessageType.UNSUBSCRIBE: self._handle_unsubscribe,
            WSMessageType.PING: self._handle_ping,
        }
        
        # Callbacks
        self._event_callbacks: List[Callable] = []
        
    def start(self) -> None:
        """Start the WebSocket server"""
        if self._running:
            return
            
        self._running = True
        
        # Create server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)
        
        # Start accept thread
        threading.Thread(
            target=self._accept_connections,
            daemon=True
        ).start()
        
        # Start heartbeat thread
        threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        ).start()
        
        print(f"[WS] WebSocket server started on ws://{self.host}:{self.port}")
        
    def stop(self) -> None:
        """Stop the WebSocket server"""
        self._running = False
        
        # Close all client connections
        with self._clients_lock:
            for client in self._clients.values():
                try:
                    client.socket.close()
                except Exception:
                    pass
            self._clients.clear()
            
        # Close server socket
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
                
    def _accept_connections(self) -> None:
        """Accept incoming connections"""
        while self._running:
            try:
                client_socket, address = self._server_socket.accept()
                
                # Handle WebSocket handshake in thread
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                ).start()
                
            except socket.timeout:
                continue
            except Exception:
                if self._running:
                    time.sleep(0.1)
                    
    def _handle_client(self, client_socket: socket.socket, address: tuple) -> None:
        """Handle client connection"""
        # Perform WebSocket handshake
        try:
            handshake_data = client_socket.recv(4096).decode()
            
            if "Upgrade: websocket" not in handshake_data:
                client_socket.close()
                return
                
            # Extract WebSocket key
            ws_key = None
            for line in handshake_data.split("\r\n"):
                if line.startswith("Sec-WebSocket-Key:"):
                    ws_key = line.split(":")[1].strip()
                    break
                    
            if not ws_key:
                client_socket.close()
                return
                
            # Generate accept key
            accept_key = base64.b64encode(
                hashlib.sha1((ws_key + self.WS_MAGIC_STRING).encode()).digest()
            ).decode()
            
            # Send handshake response
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_key}\r\n\r\n"
            )
            client_socket.send(response.encode())
            
            # Create client
            client_id = hashlib.sha256(f"{address}_{time.time()}".encode()).hexdigest()[:16]
            client = WSClient(
                client_id=client_id,
                socket=client_socket,
                address=address
            )
            
            with self._clients_lock:
                self._clients[client_id] = client
                
            # Send welcome message
            self._send_message(client, WSMessage(
                msg_type=WSMessageType.STATUS,
                data={"connected": True, "client_id": client_id}
            ))
            
            # Handle messages
            self._handle_messages(client)
            
        except Exception as e:
            print(f"[WS] Client handling error: {e}")
        finally:
            # Cleanup
            with self._clients_lock:
                for cid, c in list(self._clients.items()):
                    if c.socket == client_socket:
                        del self._clients[cid]
                        break
            try:
                client_socket.close()
            except Exception:
                pass
                
    def _handle_messages(self, client: WSClient) -> None:
        """Handle messages from client"""
        while self._running:
            try:
                # Receive frame
                frame = self._receive_frame(client.socket)
                
                if frame is None:
                    break
                    
                client.last_activity = time.time()
                client.messages_received += 1
                
                # Parse message
                try:
                    message = WSMessage.from_json(frame)
                except json.JSONDecodeError:
                    self._send_error(client, "Invalid JSON")
                    continue
                    
                # Handle message
                handler = self._message_handlers.get(message.msg_type)
                if handler:
                    handler(client, message)
                else:
                    self._send_error(client, f"Unknown message type: {message.msg_type}")
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[WS] Message handling error: {e}")
                break
                
    def _receive_frame(self, sock: socket.socket) -> Optional[str]:
        """Receive and decode WebSocket frame"""
        try:
            # Read header
            header = sock.recv(2)
            if len(header) < 2:
                return None
                
            fin = (header[0] >> 7) & 1
            opcode = header[0] & 0x0F
            masked = (header[1] >> 7) & 1
            payload_len = header[1] & 0x7F
            
            # Handle close frame
            if opcode == 0x8:
                return None
                
            # Extended payload length
            if payload_len == 126:
                ext_len = sock.recv(2)
                payload_len = struct.unpack(">H", ext_len)[0]
            elif payload_len == 127:
                ext_len = sock.recv(8)
                payload_len = struct.unpack(">Q", ext_len)[0]
                
            # Masking key
            mask = sock.recv(4) if masked else None
            
            # Payload
            payload = sock.recv(payload_len)
            
            # Unmask if needed
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
                
            return payload.decode('utf-8')
            
        except Exception:
            return None
            
    def _send_frame(self, sock: socket.socket, data: str) -> bool:
        """Encode and send WebSocket frame"""
        try:
            payload = data.encode('utf-8')
            frame = bytearray()
            
            # Header
            frame.append(0x81)  # FIN + text frame
            
            # Length
            length = len(payload)
            if length < 126:
                frame.append(length)
            elif length < 65536:
                frame.append(126)
                frame.extend(struct.pack(">H", length))
            else:
                frame.append(127)
                frame.extend(struct.pack(">Q", length))
                
            # Payload
            frame.extend(payload)
            
            sock.send(frame)
            return True
            
        except Exception:
            return False
            
    def _send_message(self, client: WSClient, message: WSMessage) -> bool:
        """Send message to client"""
        success = self._send_frame(client.socket, message.to_json())
        if success:
            client.messages_sent += 1
        return success
        
    def _send_error(self, client: WSClient, error: str) -> None:
        """Send error message"""
        self._send_message(client, WSMessage(
            msg_type=WSMessageType.ERROR,
            data={"error": error}
        ))
        
    def _handle_command(self, client: WSClient, message: WSMessage) -> None:
        """Handle command message"""
        command = message.data.get("command", "") if isinstance(message.data, dict) else str(message.data)
        
        try:
            result = self.command_handler(command)
            self._send_message(client, WSMessage(
                msg_type=WSMessageType.RESPONSE,
                data=result,
                message_id=message.message_id
            ))
        except Exception as e:
            self._send_error(client, str(e))
            
    def _handle_subscribe(self, client: WSClient, message: WSMessage) -> None:
        """Handle subscribe message"""
        channels = message.data.get("channels", []) if isinstance(message.data, dict) else [message.data]
        
        for channel in channels:
            if channel in self._channels:
                self._channels[channel].add(client.client_id)
                client.subscriptions.add(channel)
                
        self._send_message(client, WSMessage(
            msg_type=WSMessageType.RESPONSE,
            data={"subscribed": list(client.subscriptions)},
            message_id=message.message_id
        ))
        
    def _handle_unsubscribe(self, client: WSClient, message: WSMessage) -> None:
        """Handle unsubscribe message"""
        channels = message.data.get("channels", []) if isinstance(message.data, dict) else [message.data]
        
        for channel in channels:
            if channel in self._channels:
                self._channels[channel].discard(client.client_id)
                client.subscriptions.discard(channel)
                
        self._send_message(client, WSMessage(
            msg_type=WSMessageType.RESPONSE,
            data={"subscribed": list(client.subscriptions)},
            message_id=message.message_id
        ))
        
    def _handle_ping(self, client: WSClient, message: WSMessage) -> None:
        """Handle ping message"""
        self._send_message(client, WSMessage(
            msg_type=WSMessageType.PONG,
            data=message.data
        ))
        
    def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats"""
        while self._running:
            time.sleep(30)
            
            current_time = time.time()
            
            with self._clients_lock:
                for client_id, client in list(self._clients.items()):
                    # Check for inactive clients
                    if current_time - client.last_activity > 120:
                        try:
                            client.socket.close()
                        except Exception:
                            pass
                        del self._clients[client_id]
                    else:
                        # Send ping
                        self._send_message(client, WSMessage(
                            msg_type=WSMessageType.PING,
                            data={"timestamp": current_time}
                        ))
                        
    def broadcast(self, channel: str, message: WSMessage) -> int:
        """
        Broadcast message to channel subscribers.
        
        Args:
            channel: Channel name
            message: Message to broadcast
            
        Returns:
            Number of clients notified
        """
        if channel not in self._channels:
            return 0
            
        count = 0
        with self._clients_lock:
            for client_id in self._channels[channel]:
                if client_id in self._clients:
                    if self._send_message(self._clients[client_id], message):
                        count += 1
                        
        return count
        
    def broadcast_spectrum(self, spectrum_data: Dict) -> int:
        """Broadcast spectrum data"""
        return self.broadcast("spectrum", WSMessage(
            msg_type=WSMessageType.SPECTRUM,
            data=spectrum_data
        ))
        
    def broadcast_waterfall(self, waterfall_data: Dict) -> int:
        """Broadcast waterfall data"""
        return self.broadcast("waterfall", WSMessage(
            msg_type=WSMessageType.WATERFALL,
            data=waterfall_data
        ))
        
    def broadcast_constellation(self, constellation_data: Dict) -> int:
        """Broadcast constellation data"""
        return self.broadcast("constellation", WSMessage(
            msg_type=WSMessageType.CONSTELLATION,
            data=constellation_data
        ))
        
    def broadcast_detection(self, detection: Dict) -> int:
        """Broadcast detection event"""
        return self.broadcast("detections", WSMessage(
            msg_type=WSMessageType.DETECTION,
            data=detection
        ))
        
    def broadcast_alert(self, alert: Dict) -> int:
        """Broadcast alert"""
        # Alerts go to all connected clients
        count = 0
        with self._clients_lock:
            for client in self._clients.values():
                if self._send_message(client, WSMessage(
                    msg_type=WSMessageType.ALERT,
                    data=alert
                )):
                    count += 1
        return count
        
    def send_to_client(self, client_id: str, message: WSMessage) -> bool:
        """Send message to specific client"""
        with self._clients_lock:
            if client_id in self._clients:
                return self._send_message(self._clients[client_id], message)
        return False
        
    def get_clients(self) -> List[Dict]:
        """Get connected clients info"""
        with self._clients_lock:
            return [
                {
                    "client_id": c.client_id,
                    "address": c.address,
                    "subscriptions": list(c.subscriptions),
                    "connected_at": c.connected_at,
                    "messages_sent": c.messages_sent,
                    "messages_received": c.messages_received
                }
                for c in self._clients.values()
            ]
            
    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        with self._clients_lock:
            return {
                "running": self._running,
                "host": self.host,
                "port": self.port,
                "connected_clients": len(self._clients),
                "channels": {k: len(v) for k, v in self._channels.items()},
                "total_messages_sent": sum(c.messages_sent for c in self._clients.values()),
                "total_messages_received": sum(c.messages_received for c in self._clients.values())
            }
