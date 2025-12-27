"""
RF Arsenal OS - REST API Server
Lightweight REST API for system control and monitoring.
Supports local-only binding for offline operation.
"""

import json
import hashlib
import time
import threading
import socket
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse


class HTTPMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class APIEndpoint:
    """API endpoint definition"""
    path: str
    method: HTTPMethod
    handler: Callable
    description: str = ""
    auth_required: bool = True
    rate_limit: int = 100  # Requests per minute
    
    
@dataclass
class APIResponse:
    """API response"""
    status_code: int = 200
    data: Any = None
    error: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    def to_json(self) -> str:
        response = {
            "success": self.status_code < 400,
            "data": self.data,
            "error": self.error,
            "timestamp": time.time()
        }
        return json.dumps(response)


class RestAPI:
    """
    Production-grade REST API for RF Arsenal OS.
    
    Features:
    - RESTful endpoint routing
    - Token-based authentication
    - Rate limiting
    - CORS support
    - Local-only binding option
    - Request logging
    - Error handling
    - Swagger-compatible documentation
    """
    
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8080,
                 command_handler: Optional[Callable] = None,
                 auth_enabled: bool = True):
        """
        Initialize REST API server.
        
        Args:
            host: Bind address (127.0.0.1 for local only)
            port: Listen port
            command_handler: Function to execute commands
            auth_enabled: Enable authentication
        """
        self.host = host
        self.port = port
        self.command_handler = command_handler or (lambda cmd: {"success": True})
        self.auth_enabled = auth_enabled
        
        # Endpoint registry
        self._endpoints: Dict[str, Dict[str, APIEndpoint]] = {}
        
        # Authentication
        self._tokens: Dict[str, Dict] = {}
        self._api_key: Optional[str] = None
        
        # Rate limiting
        self._rate_limits: Dict[str, List[float]] = {}
        
        # Server
        self._server: Optional[HTTPServer] = None
        self._running = False
        
        # Request logging
        self._request_log: List[Dict] = []
        self._max_log_entries = 1000
        
        # Register default endpoints
        self._register_default_endpoints()
        
    def _register_default_endpoints(self) -> None:
        """Register default API endpoints"""
        # System endpoints
        self.register_endpoint(APIEndpoint(
            path="/api/v1/status",
            method=HTTPMethod.GET,
            handler=self._handle_status,
            description="Get system status",
            auth_required=False
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/api/v1/health",
            method=HTTPMethod.GET,
            handler=self._handle_health,
            description="Health check",
            auth_required=False
        ))
        
        # Command endpoint
        self.register_endpoint(APIEndpoint(
            path="/api/v1/command",
            method=HTTPMethod.POST,
            handler=self._handle_command,
            description="Execute command"
        ))
        
        # Hardware endpoints
        self.register_endpoint(APIEndpoint(
            path="/api/v1/hardware",
            method=HTTPMethod.GET,
            handler=self._handle_hardware_list,
            description="List hardware"
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/api/v1/hardware/{device_id}",
            method=HTTPMethod.GET,
            handler=self._handle_hardware_info,
            description="Get hardware info"
        ))
        
        # Spectrum endpoints
        self.register_endpoint(APIEndpoint(
            path="/api/v1/spectrum",
            method=HTTPMethod.GET,
            handler=self._handle_spectrum,
            description="Get spectrum data"
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/api/v1/spectrum/scan",
            method=HTTPMethod.POST,
            handler=self._handle_spectrum_scan,
            description="Start spectrum scan"
        ))
        
        # Capture endpoints
        self.register_endpoint(APIEndpoint(
            path="/api/v1/capture",
            method=HTTPMethod.POST,
            handler=self._handle_capture_start,
            description="Start capture"
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/api/v1/capture/{capture_id}",
            method=HTTPMethod.GET,
            handler=self._handle_capture_status,
            description="Get capture status"
        ))
        
        # Mission endpoints
        self.register_endpoint(APIEndpoint(
            path="/api/v1/missions",
            method=HTTPMethod.GET,
            handler=self._handle_missions_list,
            description="List missions"
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/api/v1/missions/{mission_id}/start",
            method=HTTPMethod.POST,
            handler=self._handle_mission_start,
            description="Start mission"
        ))
        
        # Trigger endpoints
        self.register_endpoint(APIEndpoint(
            path="/api/v1/triggers",
            method=HTTPMethod.GET,
            handler=self._handle_triggers_list,
            description="List triggers"
        ))
        
        self.register_endpoint(APIEndpoint(
            path="/api/v1/triggers",
            method=HTTPMethod.POST,
            handler=self._handle_trigger_create,
            description="Create trigger"
        ))
        
        # Detection endpoints
        self.register_endpoint(APIEndpoint(
            path="/api/v1/detections",
            method=HTTPMethod.GET,
            handler=self._handle_detections,
            description="Get detections"
        ))
        
        # Export endpoint
        self.register_endpoint(APIEndpoint(
            path="/api/v1/export",
            method=HTTPMethod.POST,
            handler=self._handle_export,
            description="Export data"
        ))
        
        # Documentation
        self.register_endpoint(APIEndpoint(
            path="/api/v1/docs",
            method=HTTPMethod.GET,
            handler=self._handle_docs,
            description="API documentation",
            auth_required=False
        ))
        
    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """Register an API endpoint"""
        if endpoint.path not in self._endpoints:
            self._endpoints[endpoint.path] = {}
        self._endpoints[endpoint.path][endpoint.method.value] = endpoint
        
    def generate_api_key(self, name: str = "default", expires_hours: int = 24) -> str:
        """Generate API key"""
        key = hashlib.sha256(f"{name}_{time.time()}_{socket.gethostname()}".encode()).hexdigest()
        
        self._tokens[key] = {
            "name": name,
            "created": time.time(),
            "expires": time.time() + (expires_hours * 3600),
            "requests": 0
        }
        
        self._api_key = key
        return key
        
    def validate_token(self, token: str) -> bool:
        """Validate API token"""
        if token not in self._tokens:
            return False
            
        token_data = self._tokens[token]
        
        if token_data["expires"] < time.time():
            del self._tokens[token]
            return False
            
        token_data["requests"] += 1
        return True
        
    def _check_rate_limit(self, client_id: str, limit: int) -> bool:
        """Check if request is within rate limit"""
        current_time = time.time()
        
        if client_id not in self._rate_limits:
            self._rate_limits[client_id] = []
            
        # Remove old entries (older than 1 minute)
        self._rate_limits[client_id] = [
            t for t in self._rate_limits[client_id]
            if current_time - t < 60
        ]
        
        if len(self._rate_limits[client_id]) >= limit:
            return False
            
        self._rate_limits[client_id].append(current_time)
        return True
        
    def _handle_status(self, request: Dict) -> APIResponse:
        """Handle status request"""
        result = self.command_handler("status")
        return APIResponse(data=result)
        
    def _handle_health(self, request: Dict) -> APIResponse:
        """Handle health check"""
        return APIResponse(data={
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        })
        
    def _handle_command(self, request: Dict) -> APIResponse:
        """Handle command execution"""
        body = request.get("body", {})
        command = body.get("command", "")
        
        if not command:
            return APIResponse(status_code=400, error="Command required")
            
        try:
            result = self.command_handler(command)
            return APIResponse(data=result)
        except Exception as e:
            return APIResponse(status_code=500, error=str(e))
            
    def _handle_hardware_list(self, request: Dict) -> APIResponse:
        """Handle hardware list request"""
        result = self.command_handler("list hardware")
        return APIResponse(data=result)
        
    def _handle_hardware_info(self, request: Dict) -> APIResponse:
        """Handle hardware info request"""
        device_id = request.get("params", {}).get("device_id", "")
        result = self.command_handler(f"hardware info {device_id}")
        return APIResponse(data=result)
        
    def _handle_spectrum(self, request: Dict) -> APIResponse:
        """Handle spectrum data request"""
        result = self.command_handler("get spectrum")
        return APIResponse(data=result)
        
    def _handle_spectrum_scan(self, request: Dict) -> APIResponse:
        """Handle spectrum scan request"""
        body = request.get("body", {})
        freq_start = body.get("freq_start", 100e6)
        freq_end = body.get("freq_end", 1000e6)
        step = body.get("step", 1e6)
        
        command = f"scan spectrum {freq_start/1e6:.3f}MHz to {freq_end/1e6:.3f}MHz step {step/1e6:.3f}MHz"
        result = self.command_handler(command)
        return APIResponse(data=result)
        
    def _handle_capture_start(self, request: Dict) -> APIResponse:
        """Handle capture start request"""
        body = request.get("body", {})
        frequency = body.get("frequency", 100e6)
        duration = body.get("duration", 5.0)
        sample_rate = body.get("sample_rate", 1e6)
        
        command = f"capture iq at {frequency/1e6:.3f}MHz for {duration}s at {sample_rate/1e6:.1f}MSPS"
        result = self.command_handler(command)
        
        return APIResponse(data={
            "capture_id": result.get("capture_id", f"cap_{int(time.time())}"),
            "status": "started",
            **result
        })
        
    def _handle_capture_status(self, request: Dict) -> APIResponse:
        """Handle capture status request"""
        capture_id = request.get("params", {}).get("capture_id", "")
        result = self.command_handler(f"capture status {capture_id}")
        return APIResponse(data=result)
        
    def _handle_missions_list(self, request: Dict) -> APIResponse:
        """Handle missions list request"""
        result = self.command_handler("list missions")
        return APIResponse(data=result)
        
    def _handle_mission_start(self, request: Dict) -> APIResponse:
        """Handle mission start request"""
        mission_id = request.get("params", {}).get("mission_id", "")
        result = self.command_handler(f"start mission {mission_id}")
        return APIResponse(data=result)
        
    def _handle_triggers_list(self, request: Dict) -> APIResponse:
        """Handle triggers list request"""
        result = self.command_handler("list triggers")
        return APIResponse(data=result)
        
    def _handle_trigger_create(self, request: Dict) -> APIResponse:
        """Handle trigger create request"""
        body = request.get("body", {})
        result = self.command_handler(f"create trigger {json.dumps(body)}")
        return APIResponse(data=result)
        
    def _handle_detections(self, request: Dict) -> APIResponse:
        """Handle detections request"""
        result = self.command_handler("list detections")
        return APIResponse(data=result)
        
    def _handle_export(self, request: Dict) -> APIResponse:
        """Handle export request"""
        body = request.get("body", {})
        format_type = body.get("format", "json")
        data_type = body.get("data_type", "session")
        
        result = self.command_handler(f"export {data_type} as {format_type}")
        return APIResponse(data=result)
        
    def _handle_docs(self, request: Dict) -> APIResponse:
        """Generate API documentation"""
        docs = {
            "api_version": "1.0",
            "base_url": f"http://{self.host}:{self.port}/api/v1",
            "endpoints": []
        }
        
        for path, methods in self._endpoints.items():
            for method, endpoint in methods.items():
                docs["endpoints"].append({
                    "path": path,
                    "method": method,
                    "description": endpoint.description,
                    "auth_required": endpoint.auth_required,
                    "rate_limit": endpoint.rate_limit
                })
                
        return APIResponse(data=docs)
        
    def _match_path(self, request_path: str) -> Tuple[Optional[str], Dict[str, str]]:
        """Match request path to registered endpoint, extracting params"""
        for endpoint_path in self._endpoints:
            params = {}
            path_parts = endpoint_path.split("/")
            request_parts = request_path.split("/")
            
            if len(path_parts) != len(request_parts):
                continue
                
            match = True
            for ep_part, req_part in zip(path_parts, request_parts):
                if ep_part.startswith("{") and ep_part.endswith("}"):
                    param_name = ep_part[1:-1]
                    params[param_name] = req_part
                elif ep_part != req_part:
                    match = False
                    break
                    
            if match:
                return endpoint_path, params
                
        return None, {}
        
    def start(self) -> None:
        """Start the API server"""
        if self._running:
            return
            
        self._running = True
        self._start_time = time.time()
        
        # Create request handler
        api = self
        
        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress default logging
                
            def do_GET(self):
                self._handle_request("GET")
                
            def do_POST(self):
                self._handle_request("POST")
                
            def do_PUT(self):
                self._handle_request("PUT")
                
            def do_DELETE(self):
                self._handle_request("DELETE")
                
            def _handle_request(self, method: str):
                # Parse URL
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                query = urllib.parse.parse_qs(parsed.query)
                
                # Match endpoint
                endpoint_path, params = api._match_path(path)
                
                if not endpoint_path or method not in api._endpoints.get(endpoint_path, {}):
                    self._send_response(APIResponse(status_code=404, error="Not found"))
                    return
                    
                endpoint = api._endpoints[endpoint_path][method]
                
                # Check auth
                if api.auth_enabled and endpoint.auth_required:
                    auth_header = self.headers.get("Authorization", "")
                    token = auth_header.replace("Bearer ", "")
                    
                    if not api.validate_token(token):
                        self._send_response(APIResponse(status_code=401, error="Unauthorized"))
                        return
                        
                # Check rate limit
                client_ip = self.client_address[0]
                if not api._check_rate_limit(client_ip, endpoint.rate_limit):
                    self._send_response(APIResponse(status_code=429, error="Rate limit exceeded"))
                    return
                    
                # Parse body for POST/PUT
                body = {}
                if method in ["POST", "PUT", "PATCH"]:
                    content_length = int(self.headers.get("Content-Length", 0))
                    if content_length > 0:
                        try:
                            body = json.loads(self.rfile.read(content_length))
                        except json.JSONDecodeError:
                            self._send_response(APIResponse(status_code=400, error="Invalid JSON"))
                            return
                            
                # Build request object
                request = {
                    "method": method,
                    "path": path,
                    "params": params,
                    "query": query,
                    "body": body,
                    "headers": dict(self.headers)
                }
                
                # Log request
                api._log_request(request)
                
                # Execute handler
                try:
                    response = endpoint.handler(request)
                except Exception as e:
                    response = APIResponse(status_code=500, error=str(e))
                    
                self._send_response(response)
                
            def _send_response(self, response: APIResponse):
                self.send_response(response.status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                for key, value in response.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response.to_json().encode())
                
        # Start server in thread
        self._server = HTTPServer((self.host, self.port), RequestHandler)
        
        threading.Thread(
            target=self._server.serve_forever,
            daemon=True
        ).start()
        
        print(f"[API] REST API server started on http://{self.host}:{self.port}")
        
    def stop(self) -> None:
        """Stop the API server"""
        if self._server:
            self._server.shutdown()
        self._running = False
        
    def _log_request(self, request: Dict) -> None:
        """Log request"""
        entry = {
            "timestamp": time.time(),
            "method": request["method"],
            "path": request["path"],
            "client": request.get("headers", {}).get("Host", "unknown")
        }
        
        self._request_log.append(entry)
        
        if len(self._request_log) > self._max_log_entries:
            self._request_log = self._request_log[-self._max_log_entries:]
            
    def get_request_log(self, limit: int = 100) -> List[Dict]:
        """Get recent request log"""
        return self._request_log[-limit:]
        
    def get_status(self) -> Dict[str, Any]:
        """Get API server status"""
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "auth_enabled": self.auth_enabled,
            "endpoint_count": sum(len(m) for m in self._endpoints.values()),
            "active_tokens": len(self._tokens),
            "total_requests": len(self._request_log)
        }
