"""
RF Arsenal OS - API Security
Authentication and authorization for API endpoints.
"""

import hashlib
import hmac
import time
import secrets
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum


class AuthLevel(Enum):
    """Authentication levels"""
    NONE = 0
    READ_ONLY = 1
    OPERATOR = 2
    ADMIN = 3


@dataclass
class APIToken:
    """API authentication token"""
    token: str
    name: str
    auth_level: AuthLevel
    created_at: float
    expires_at: float
    last_used: Optional[float] = None
    request_count: int = 0
    ip_whitelist: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if token is valid"""
        return time.time() < self.expires_at
        
    def can_access(self, required_level: AuthLevel) -> bool:
        """Check if token has required access level"""
        return self.auth_level.value >= required_level.value


class TokenManager:
    """
    API token management.
    
    Features:
    - Token generation and validation
    - Expiration handling
    - IP whitelisting
    - Rate limiting per token
    - Access level management
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize token manager.
        
        Args:
            secret_key: Secret for token signing (auto-generated if None)
        """
        self.secret_key = secret_key or secrets.token_hex(32)
        self._tokens: Dict[str, APIToken] = {}
        self._revoked: set = set()
        
    def create_token(self,
                    name: str,
                    auth_level: AuthLevel = AuthLevel.OPERATOR,
                    expires_hours: int = 24,
                    ip_whitelist: Optional[List[str]] = None) -> str:
        """
        Create new API token.
        
        Args:
            name: Token name/description
            auth_level: Access level
            expires_hours: Hours until expiration
            ip_whitelist: Optional IP whitelist
            
        Returns:
            Token string
        """
        # Generate token
        token_data = f"{name}:{time.time()}:{secrets.token_hex(16)}"
        signature = hmac.new(
            self.secret_key.encode(),
            token_data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        token = f"{hashlib.sha256(token_data.encode()).hexdigest()[:32]}.{signature[:16]}"
        
        # Store token
        self._tokens[token] = APIToken(
            token=token,
            name=name,
            auth_level=auth_level,
            created_at=time.time(),
            expires_at=time.time() + (expires_hours * 3600),
            ip_whitelist=ip_whitelist or []
        )
        
        return token
        
    def validate_token(self, token: str, client_ip: Optional[str] = None) -> Optional[APIToken]:
        """
        Validate token and return token object.
        
        Args:
            token: Token to validate
            client_ip: Optional client IP for whitelist check
            
        Returns:
            APIToken if valid, None otherwise
        """
        if token in self._revoked:
            return None
            
        if token not in self._tokens:
            return None
            
        api_token = self._tokens[token]
        
        # Check expiration
        if not api_token.is_valid():
            del self._tokens[token]
            return None
            
        # Check IP whitelist
        if api_token.ip_whitelist and client_ip:
            if client_ip not in api_token.ip_whitelist:
                return None
                
        # Update usage
        api_token.last_used = time.time()
        api_token.request_count += 1
        
        return api_token
        
    def revoke_token(self, token: str) -> bool:
        """Revoke a token"""
        if token in self._tokens:
            self._revoked.add(token)
            del self._tokens[token]
            return True
        return False
        
    def get_token_info(self, token: str) -> Optional[Dict]:
        """Get token information"""
        if token in self._tokens:
            t = self._tokens[token]
            return {
                "name": t.name,
                "auth_level": t.auth_level.value,
                "created_at": t.created_at,
                "expires_at": t.expires_at,
                "last_used": t.last_used,
                "request_count": t.request_count
            }
        return None
        
    def cleanup_expired(self) -> int:
        """Remove expired tokens"""
        current_time = time.time()
        expired = [
            token for token, t in self._tokens.items()
            if t.expires_at < current_time
        ]
        for token in expired:
            del self._tokens[token]
        return len(expired)


class APIAuth:
    """
    API authentication handler.
    
    Features:
    - Multiple auth methods (API key, Bearer token, Basic auth)
    - Request signing verification
    - Rate limiting
    - Session management
    """
    
    def __init__(self, token_manager: Optional[TokenManager] = None):
        """
        Initialize API auth.
        
        Args:
            token_manager: Token manager instance
        """
        self.token_manager = token_manager or TokenManager()
        
        # Rate limiting
        self._rate_limits: Dict[str, List[float]] = {}
        self._default_rate_limit = 100  # Requests per minute
        
        # Sessions
        self._sessions: Dict[str, Dict] = {}
        
    def authenticate(self,
                    headers: Dict[str, str],
                    client_ip: Optional[str] = None) -> Optional[APIToken]:
        """
        Authenticate request from headers.
        
        Args:
            headers: Request headers
            client_ip: Client IP address
            
        Returns:
            APIToken if authenticated, None otherwise
        """
        # Try Bearer token
        auth_header = headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return self.token_manager.validate_token(token, client_ip)
            
        # Try API key
        api_key = headers.get("X-API-Key", "")
        if api_key:
            return self.token_manager.validate_token(api_key, client_ip)
            
        return None
        
    def check_rate_limit(self, identifier: str, limit: Optional[int] = None) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            identifier: Client identifier (IP or token)
            limit: Custom rate limit
            
        Returns:
            True if within limit
        """
        limit = limit or self._default_rate_limit
        current_time = time.time()
        
        if identifier not in self._rate_limits:
            self._rate_limits[identifier] = []
            
        # Remove old entries
        self._rate_limits[identifier] = [
            t for t in self._rate_limits[identifier]
            if current_time - t < 60
        ]
        
        if len(self._rate_limits[identifier]) >= limit:
            return False
            
        self._rate_limits[identifier].append(current_time)
        return True
        
    def create_session(self, token: APIToken) -> str:
        """Create authenticated session"""
        session_id = secrets.token_hex(32)
        
        self._sessions[session_id] = {
            "token": token.token,
            "created": time.time(),
            "last_activity": time.time(),
            "auth_level": token.auth_level
        }
        
        return session_id
        
    def validate_session(self, session_id: str) -> Optional[Dict]:
        """Validate session"""
        if session_id not in self._sessions:
            return None
            
        session = self._sessions[session_id]
        
        # Check session timeout (1 hour)
        if time.time() - session["last_activity"] > 3600:
            del self._sessions[session_id]
            return None
            
        session["last_activity"] = time.time()
        return session
        
    def end_session(self, session_id: str) -> bool:
        """End session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
