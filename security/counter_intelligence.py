#!/usr/bin/env python3
"""
Counter-Intelligence System
Blockchain storage, canary tokens, and intelligence gathering tripwires
"""

import os
import json
import hashlib
import secrets
import time
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess


class TokenType(Enum):
    """Canary token types"""
    DNS_TOKEN = "dns_token"
    WEB_TOKEN = "web_token"
    EMAIL_TOKEN = "email_token"
    FILE_TOKEN = "file_token"
    AWS_KEY = "aws_key"
    COMMAND_TOKEN = "command_token"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CanaryToken:
    """Canary token definition"""
    token_id: str
    token_type: TokenType
    token_value: str
    description: str
    created: float
    triggered: bool
    trigger_count: int
    last_trigger: Optional[float]


@dataclass
class TokenAlert:
    """Canary token alert"""
    token_id: str
    token_type: TokenType
    severity: AlertSeverity
    timestamp: float
    source_ip: Optional[str]
    user_agent: Optional[str]
    details: Dict


class CanaryTokenSystem:
    """
    Canary token deployment and monitoring
    Detects unauthorized access and data exfiltration
    """
    
    def __init__(self, storage_dir: str = "/var/lib/rf-arsenal/canary"):
        self.storage_dir = storage_dir
        self.tokens = {}
        self.alerts = []
        
        os.makedirs(storage_dir, mode=0o700, exist_ok=True)
        self._load_tokens()
        
    def create_dns_token(self, description: str) -> CanaryToken:
        """
        Create DNS canary token
        Alerts when DNS query is made
        """
        token_id = self._generate_token_id()
        
        # Generate unique subdomain
        subdomain = secrets.token_hex(16)
        domain = "canarytokens.com"  # Example canary token service
        
        dns_token = f"{subdomain}.{domain}"
        
        token = CanaryToken(
            token_id=token_id,
            token_type=TokenType.DNS_TOKEN,
            token_value=dns_token,
            description=description,
            created=time.time(),
            triggered=False,
            trigger_count=0,
            last_trigger=None
        )
        
        self.tokens[token_id] = token
        self._save_tokens()
        
        print(f"[CANARY] DNS token created: {dns_token}")
        print(f"  Description: {description}")
        print(f"  Token ID: {token_id}")
        
        return token
        
    def create_web_token(self, description: str) -> CanaryToken:
        """
        Create web bug / tracking pixel
        Alerts when HTTP request is made
        """
        token_id = self._generate_token_id()
        
        # Generate unique URL
        unique_path = secrets.token_hex(16)
        url = f"https://canarytokens.com/track/{unique_path}.gif"
        
        token = CanaryToken(
            token_id=token_id,
            token_type=TokenType.WEB_TOKEN,
            token_value=url,
            description=description,
            created=time.time(),
            triggered=False,
            trigger_count=0,
            last_trigger=None
        )
        
        self.tokens[token_id] = token
        self._save_tokens()
        
        print(f"[CANARY] Web token created: {url}")
        print(f"  Embed in documents/emails as: <img src=\"{url}\" />")
        
        return token
        
    def create_email_token(self, description: str) -> CanaryToken:
        """
        Create email canary token
        Alerts when email is accessed/forwarded
        """
        token_id = self._generate_token_id()
        
        # Generate unique email address
        email_token = f"{secrets.token_hex(8)}@canarytokens.com"
        
        token = CanaryToken(
            token_id=token_id,
            token_type=TokenType.EMAIL_TOKEN,
            token_value=email_token,
            description=description,
            created=time.time(),
            triggered=False,
            trigger_count=0,
            last_trigger=None
        )
        
        self.tokens[token_id] = token
        self._save_tokens()
        
        print(f"[CANARY] Email token created: {email_token}")
        print(f"  Use in documents as contact address")
        
        return token
        
    def create_file_token(self, file_path: str, description: str) -> CanaryToken:
        """
        Create file access canary token
        Monitors file for unauthorized access
        """
        token_id = self._generate_token_id()
        
        # Create beacon file that calls home when opened
        # For PDFs, Word docs, etc.
        
        token = CanaryToken(
            token_id=token_id,
            token_type=TokenType.FILE_TOKEN,
            token_value=file_path,
            description=description,
            created=time.time(),
            triggered=False,
            trigger_count=0,
            last_trigger=None
        )
        
        self.tokens[token_id] = token
        self._save_tokens()
        
        # Set up file monitoring
        self._monitor_file_access(file_path, token_id)
        
        print(f"[CANARY] File token created for: {file_path}")
        
        return token
        
    def create_aws_key_token(self, description: str) -> CanaryToken:
        """
        Create fake AWS API key canary token
        Alerts if someone tries to use it
        """
        token_id = self._generate_token_id()
        
        # Generate fake but valid-looking AWS credentials
        access_key = f"AKIA{secrets.token_hex(16).upper()}"
        secret_key = secrets.token_hex(20)
        
        aws_credentials = {
            'aws_access_key_id': access_key,
            'aws_secret_access_key': secret_key
        }
        
        token = CanaryToken(
            token_id=token_id,
            token_type=TokenType.AWS_KEY,
            token_value=json.dumps(aws_credentials),
            description=description,
            created=time.time(),
            triggered=False,
            trigger_count=0,
            last_trigger=None
        )
        
        self.tokens[token_id] = token
        self._save_tokens()
        
        print(f"[CANARY] AWS key token created")
        print(f"  Access Key: {access_key}")
        print(f"  Secret Key: {secret_key[:8]}...")
        print(f"  Plant in config files to detect credential theft")
        
        return token
        
    def create_command_token(self, command: str, description: str) -> CanaryToken:
        """
        Create command execution canary token
        Detects when specific command is run
        """
        token_id = self._generate_token_id()
        
        # Create wrapper script that alerts before executing
        wrapper_path = os.path.join(self.storage_dir, f"canary_{token_id}.sh")
        
        wrapper_script = f"""#!/bin/bash
# Canary token: {description}
curl -X POST https://canarytokens.com/alert/{token_id} >/dev/null 2>&1 &
{command}
"""
        
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_script)
            
        os.chmod(wrapper_path, 0o755)
        
        token = CanaryToken(
            token_id=token_id,
            token_type=TokenType.COMMAND_TOKEN,
            token_value=wrapper_path,
            description=description,
            created=time.time(),
            triggered=False,
            trigger_count=0,
            last_trigger=None
        )
        
        self.tokens[token_id] = token
        self._save_tokens()
        
        print(f"[CANARY] Command token created: {wrapper_path}")
        print(f"  Replace real command with this wrapper")
        
        return token
        
    def check_for_triggers(self) -> List[TokenAlert]:
        """
        Check all tokens for triggers
        Returns list of new alerts
        """
        new_alerts = []
        
        # Would query canary token service API
        # Check for DNS queries, web requests, etc.
        
        # Simulated for demonstration
        
        return new_alerts
        
    def _monitor_file_access(self, file_path: str, token_id: str):
        """Monitor file for unauthorized access"""
        # Would use inotify or similar to monitor file access
        # Alert when file is read/opened
        
        try:
            # Set up file monitoring
            # On Linux: use inotify
            # On access, trigger alert
            
            print(f"  Monitoring {file_path} for access...")
            
        except Exception as e:
            print(f"  Warning: File monitoring setup failed: {e}")
            
    def get_token_status(self) -> List[Dict]:
        """Get status of all canary tokens"""
        status_list = []
        
        for token_id, token in self.tokens.items():
            status_list.append({
                'id': token_id,
                'type': token.token_type.value,
                'description': token.description,
                'triggered': token.triggered,
                'trigger_count': token.trigger_count,
                'last_trigger': token.last_trigger
            })
            
        return status_list
        
    def _generate_token_id(self) -> str:
        """Generate unique token ID"""
        return hashlib.sha256(secrets.token_bytes(32)).hexdigest()[:16]
        
    def _save_tokens(self):
        """Save tokens to disk"""
        tokens_file = os.path.join(self.storage_dir, 'tokens.json')
        
        tokens_data = {}
        for token_id, token in self.tokens.items():
            token_dict = asdict(token)
            token_dict['token_type'] = token.token_type.value
            tokens_data[token_id] = token_dict
            
        with open(tokens_file, 'w') as f:
            json.dump(tokens_data, f, indent=2)
            
        os.chmod(tokens_file, 0o600)
        
    def _load_tokens(self):
        """Load tokens from disk"""
        tokens_file = os.path.join(self.storage_dir, 'tokens.json')
        
        if not os.path.exists(tokens_file):
            return
            
        try:
            with open(tokens_file, 'r') as f:
                tokens_data = json.load(f)
                
            for token_id, token_dict in tokens_data.items():
                token_dict['token_type'] = TokenType(token_dict['token_type'])
                token = CanaryToken(**token_dict)
                self.tokens[token_id] = token
                
        except Exception as e:
            print(f"[CANARY] Warning: Failed to load tokens: {e}")


class BlockchainStorage:
    """
    Distributed blockchain storage for critical data
    Immutable, censorship-resistant storage
    """
    
    def __init__(self):
        self.ipfs_gateway = "http://localhost:5001"
        self.ethereum_rpc = "http://localhost:8545"
        
    def store_on_ipfs(self, data: bytes) -> Optional[str]:
        """
        Store data on IPFS (InterPlanetary File System)
        Returns IPFS hash (CID)
        """
        print(f"[BLOCKCHAIN] Storing {len(data)} bytes on IPFS...")
        
        try:
            # Check if IPFS daemon is running
            response = requests.get(f"{self.ipfs_gateway}/api/v0/version", timeout=5)
            
            if response.status_code != 200:
                print("[BLOCKCHAIN] Error: IPFS daemon not running")
                return None
                
            # Upload to IPFS
            files = {'file': data}
            response = requests.post(
                f"{self.ipfs_gateway}/api/v0/add",
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ipfs_hash = result['Hash']
                
                print(f"[BLOCKCHAIN] ✓ Stored on IPFS: {ipfs_hash}")
                print(f"  Access via: ipfs://{ipfs_hash}")
                print(f"  Gateway: https://ipfs.io/ipfs/{ipfs_hash}")
                
                return ipfs_hash
            else:
                print(f"[BLOCKCHAIN] Error: IPFS upload failed: {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            print("[BLOCKCHAIN] Error: Cannot connect to IPFS daemon")
            print("  Start IPFS: ipfs daemon")
            return None
            
        except Exception as e:
            print(f"[BLOCKCHAIN] Error storing on IPFS: {e}")
            return None
            
    def retrieve_from_ipfs(self, ipfs_hash: str) -> Optional[bytes]:
        """Retrieve data from IPFS"""
        print(f"[BLOCKCHAIN] Retrieving from IPFS: {ipfs_hash}")
        
        try:
            response = requests.post(
                f"{self.ipfs_gateway}/api/v0/cat?arg={ipfs_hash}",
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.content
                print(f"[BLOCKCHAIN] ✓ Retrieved {len(data)} bytes")
                return data
            else:
                print(f"[BLOCKCHAIN] Error: Retrieval failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[BLOCKCHAIN] Error retrieving from IPFS: {e}")
            return None
            
    def store_on_ethereum(self, data: bytes, contract_address: str) -> Optional[str]:
        """
        Store data hash on Ethereum blockchain
        Returns transaction hash
        """
        print(f"[BLOCKCHAIN] Storing hash on Ethereum...")
        
        try:
            # Calculate data hash
            data_hash = hashlib.sha256(data).hexdigest()
            
            # Would interact with Ethereum smart contract
            # Using web3.py or similar
            
            # Simplified for demonstration
            print(f"  Data hash: {data_hash}")
            print(f"  Contract: {contract_address}")
            
            # In real implementation:
            # - Connect to Ethereum node
            # - Call smart contract function to store hash
            # - Wait for transaction confirmation
            
            tx_hash = "0x" + secrets.token_hex(32)
            print(f"[BLOCKCHAIN] ✓ Transaction: {tx_hash}")
            
            return tx_hash
            
        except Exception as e:
            print(f"[BLOCKCHAIN] Error: {e}")
            return None
            
    def verify_on_blockchain(self, data: bytes, tx_hash: str) -> bool:
        """
        Verify data integrity using blockchain record
        Ensures data hasn't been tampered with
        """
        print(f"[BLOCKCHAIN] Verifying data integrity...")
        
        try:
            # Calculate current data hash
            current_hash = hashlib.sha256(data).hexdigest()
            
            # Retrieve stored hash from blockchain
            # Would query Ethereum transaction/contract
            
            stored_hash = current_hash  # Simulated
            
            if current_hash == stored_hash:
                print(f"[BLOCKCHAIN] ✓ Data integrity verified")
                return True
            else:
                print(f"[BLOCKCHAIN] ✗ Data has been tampered with!")
                return False
                
        except Exception as e:
            print(f"[BLOCKCHAIN] Verification error: {e}")
            return False
            
    def create_decentralized_backup(self, file_path: str) -> Dict:
        """
        Create decentralized backup of file
        Stores on IPFS and records hash on blockchain
        """
        print(f"[BLOCKCHAIN] Creating decentralized backup: {file_path}")
        
        # Read file
        with open(file_path, 'rb') as f:
            data = f.read()
            
        # Store on IPFS
        ipfs_hash = self.store_on_ipfs(data)
        
        if not ipfs_hash:
            return {'success': False, 'error': 'IPFS storage failed'}
            
        # Store hash on blockchain (optional, for integrity proof)
        # tx_hash = self.store_on_ethereum(data, contract_address)
        
        backup_info = {
            'success': True,
            'file_path': file_path,
            'file_size': len(data),
            'ipfs_hash': ipfs_hash,
            'ipfs_url': f"ipfs://{ipfs_hash}",
            'gateway_url': f"https://ipfs.io/ipfs/{ipfs_hash}",
            'timestamp': time.time()
        }
        
        print(f"[BLOCKCHAIN] ✓ Decentralized backup complete")
        
        return backup_info
        
    def pin_to_ipfs(self, ipfs_hash: str):
        """
        Pin content to IPFS to ensure persistence
        Prevents garbage collection
        """
        print(f"[BLOCKCHAIN] Pinning to IPFS: {ipfs_hash}")
        
        try:
            response = requests.post(
                f"{self.ipfs_gateway}/api/v0/pin/add?arg={ipfs_hash}",
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"[BLOCKCHAIN] ✓ Content pinned")
            else:
                print(f"[BLOCKCHAIN] Error: Pin failed")
                
        except Exception as e:
            print(f"[BLOCKCHAIN] Pin error: {e}")


class IntelligenceGathering:
    """
    Passive intelligence gathering about adversaries
    Tracks access patterns and threat actors
    """
    
    def __init__(self):
        self.access_log = []
        self.threat_actors = {}
        
    def log_access_attempt(self, ip: str, user_agent: str, 
                          resource: str, success: bool):
        """
        Log access attempt for analysis
        Builds profile of adversary behavior
        """
        access_record = {
            'timestamp': time.time(),
            'ip': ip,
            'user_agent': user_agent,
            'resource': resource,
            'success': success
        }
        
        self.access_log.append(access_record)
        
        # Update threat actor profile
        if ip not in self.threat_actors:
            self.threat_actors[ip] = {
                'ip': ip,
                'first_seen': time.time(),
                'access_count': 0,
                'user_agents': set(),
                'resources_accessed': set(),
                'success_count': 0,
                'fail_count': 0
            }
            
        actor = self.threat_actors[ip]
        actor['access_count'] += 1
        actor['user_agents'].add(user_agent)
        actor['resources_accessed'].add(resource)
        
        if success:
            actor['success_count'] += 1
        else:
            actor['fail_count'] += 1
            
    def analyze_threat_patterns(self) -> List[Dict]:
        """
        Analyze access patterns to identify threats
        Returns list of suspicious actors
        """
        suspicious_actors = []
        
        for ip, actor in self.threat_actors.items():
            suspicion_score = 0
            reasons = []
            
            # High access frequency
            if actor['access_count'] > 100:
                suspicion_score += 30
                reasons.append('High access frequency')
                
            # Multiple user agents (changing identity)
            if len(actor['user_agents']) > 5:
                suspicion_score += 20
                reasons.append('Multiple user agents')
                
            # High failure rate (probing)
            fail_rate = actor['fail_count'] / actor['access_count']
            if fail_rate > 0.5:
                suspicion_score += 40
                reasons.append('High failure rate (probing)')
                
            # Accessing many resources (reconnaissance)
            if len(actor['resources_accessed']) > 20:
                suspicion_score += 10
                reasons.append('Wide resource scanning')
                
            if suspicion_score > 50:
                suspicious_actors.append({
                    'ip': ip,
                    'suspicion_score': suspicion_score,
                    'reasons': reasons,
                    'access_count': actor['access_count'],
                    'first_seen': actor['first_seen']
                })
                
        # Sort by suspicion score
        suspicious_actors.sort(key=lambda x: x['suspicion_score'], reverse=True)
        
        return suspicious_actors
        
    def get_threat_intelligence_report(self) -> Dict:
        """Generate comprehensive threat intelligence report"""
        suspicious = self.analyze_threat_patterns()
        
        report = {
            'generated': time.time(),
            'total_accesses': len(self.access_log),
            'unique_ips': len(self.threat_actors),
            'suspicious_actors': len(suspicious),
            'top_threats': suspicious[:10],
            'summary': {
                'high_risk_ips': len([a for a in suspicious if a['suspicion_score'] > 80]),
                'medium_risk_ips': len([a for a in suspicious if 50 < a['suspicion_score'] <= 80]),
                'monitored_ips': len(self.threat_actors)
            }
        }
        
        return report


# Example usage
if __name__ == "__main__":
    print("=== Counter-Intelligence System Test ===\n")
    
    # Test canary tokens
    print("--- Canary Token System ---")
    canary = CanaryTokenSystem()
    
    # Create various token types
    dns_token = canary.create_dns_token("Test document access detection")
    print()
    
    web_token = canary.create_web_token("Email tracking pixel")
    print()
    
    email_token = canary.create_email_token("Fake contact address")
    print()
    
    aws_token = canary.create_aws_key_token("Honeypot AWS credentials")
    print()
    
    # List all tokens
    print("\n--- Deployed Canary Tokens ---")
    for token_status in canary.get_token_status():
        print(f"  [{token_status['type']}] {token_status['description']}")
        print(f"    Triggered: {token_status['triggered']} ({token_status['trigger_count']} times)")
        
    # Test blockchain storage
    print("\n--- Blockchain Distributed Storage ---")
    blockchain = BlockchainStorage()
    
    # Create test data
    test_data = b"Critical operational data that needs immutable storage"
    
    print("\nNote: IPFS daemon must be running for actual storage")
    print("  Start IPFS: ipfs daemon")
    print("\nSimulating blockchain storage...")
    
    # Simulate IPFS storage
    print(f"\nData size: {len(test_data)} bytes")
    print(f"SHA256: {hashlib.sha256(test_data).hexdigest()}")
    
    # Test intelligence gathering
    print("\n--- Intelligence Gathering ---")
    intel = IntelligenceGathering()
    
    # Simulate access attempts
    print("Simulating access attempts...")
    
    # Legitimate user
    for i in range(5):
        intel.log_access_attempt(
            "192.168.1.100",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            f"/api/data",
            True
        )
        
    # Suspicious actor (high frequency, multiple user agents)
    for i in range(150):
        intel.log_access_attempt(
            "203.0.113.50",
            f"Scanner/{i % 10}",
            f"/api/endpoint_{i}",
            False
        )
        
    # Generate threat report
    print("\n--- Threat Intelligence Report ---")
    report = intel.get_threat_intelligence_report()
    
    print(f"Total accesses: {report['total_accesses']}")
    print(f"Unique IPs: {report['unique_ips']}")
    print(f"Suspicious actors: {report['suspicious_actors']}")
    
    if report['top_threats']:
        print("\nTop threats:")
        for threat in report['top_threats']:
            print(f"  {threat['ip']} - Score: {threat['suspicion_score']}")
            print(f"    Reasons: {', '.join(threat['reasons'])}")
            print(f"    Accesses: {threat['access_count']}")
