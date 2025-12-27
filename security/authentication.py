#!/usr/bin/env python3
"""
RF Arsenal OS - Authentication System
Provides secure authentication with password hashing and session management
"""

import os
import hashlib
import secrets
import logging
import json
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class UserCredentials:
    """User credentials and metadata"""
    username: str
    password_hash: str
    salt: str
    created_at: str
    last_login: Optional[str] = None
    failed_attempts: int = 0
    locked_until: Optional[str] = None


@dataclass
class Session:
    """User session information"""
    session_id: str
    username: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now() >= self.expires_at
    
    def is_inactive(self, timeout_minutes: int = 30) -> bool:
        """Check if session has been inactive too long"""
        timeout = timedelta(minutes=timeout_minutes)
        return datetime.now() >= (self.last_activity + timeout)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()


class AuthenticationSystem:
    """
    Secure authentication system with:
    - Password hashing (SHA-256 with salt)
    - Session management
    - Failed login attempt tracking
    - Account lockout after failed attempts
    """
    
    def __init__(self, credentials_file: str = "/var/lib/rf_arsenal/credentials.json"):
        """
        Initialize authentication system
        
        Args:
            credentials_file: Path to store encrypted credentials
        """
        self.credentials_file = Path(credentials_file)
        self.credentials_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.sessions: dict[str, Session] = {}
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 15
        self.session_duration_hours = 24
        
        # Ensure credentials file has restrictive permissions
        if self.credentials_file.exists():
            os.chmod(self.credentials_file, 0o600)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash password with salt using SHA-256
        
        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)
        
        Returns:
            Tuple of (password_hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(32)  # 256-bit salt
        
        # Combine password and salt
        salted_password = f"{password}{salt}"
        
        # Hash with SHA-256
        password_hash = hashlib.sha256(salted_password.encode()).hexdigest()
        
        return password_hash, salt
    
    def create_user(self, username: str, password: str) -> bool:
        """
        Create a new user with hashed password
        
        Args:
            username: Username
            password: Plain text password
        
        Returns:
            True if user created successfully
        """
        try:
            # Validate username
            if not username or len(username) < 3:
                logger.error("Username must be at least 3 characters")
                return False
            
            # Validate password strength
            if not self._validate_password_strength(password):
                logger.error("Password does not meet strength requirements")
                return False
            
            # Check if user already exists
            existing_creds = self._load_credentials()
            if existing_creds and existing_creds.username == username:
                logger.error(f"User already exists: {username}")
                return False
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Create credentials
            credentials = UserCredentials(
                username=username,
                password_hash=password_hash,
                salt=salt,
                created_at=datetime.now().isoformat()
            )
            
            # Save credentials
            self._save_credentials(credentials)
            
            logger.info(f"User created successfully: {username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return False
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate user and create session
        
        Args:
            username: Username
            password: Plain text password
        
        Returns:
            Session ID if authentication successful, None otherwise
        """
        try:
            # Load credentials
            credentials = self._load_credentials()
            if not credentials:
                logger.error("No user credentials found")
                return None
            
            # Check username
            if credentials.username != username:
                logger.warning(f"Authentication failed: Invalid username")
                return None
            
            # Check if account is locked
            if credentials.locked_until:
                locked_until = datetime.fromisoformat(credentials.locked_until)
                if datetime.now() < locked_until:
                    logger.warning(f"Account locked until {locked_until}")
                    return None
                else:
                    # Unlock account
                    credentials.locked_until = None
                    credentials.failed_attempts = 0
            
            # Verify password
            password_hash, _ = self.hash_password(password, credentials.salt)
            
            if password_hash != credentials.password_hash:
                # Invalid password - increment failed attempts
                credentials.failed_attempts += 1
                
                if credentials.failed_attempts >= self.max_failed_attempts:
                    # Lock account
                    lockout_until = datetime.now() + timedelta(minutes=self.lockout_duration_minutes)
                    credentials.locked_until = lockout_until.isoformat()
                    logger.warning(f"Account locked due to {self.max_failed_attempts} failed attempts")
                
                self._save_credentials(credentials)
                logger.warning(f"Authentication failed: Invalid password (attempt {credentials.failed_attempts})")
                return None
            
            # Authentication successful
            credentials.failed_attempts = 0
            credentials.locked_until = None
            credentials.last_login = datetime.now().isoformat()
            self._save_credentials(credentials)
            
            # Create session
            session_id = self._create_session(username)
            
            logger.info(f"User authenticated successfully: {username}")
            return session_id
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def validate_session(self, session_id: str) -> bool:
        """
        Validate session and check expiration
        
        Args:
            session_id: Session ID to validate
        
        Returns:
            True if session is valid and not expired
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Check expiration
        if session.is_expired():
            logger.info(f"Session expired: {session_id}")
            del self.sessions[session_id]
            return False
        
        # Check inactivity
        if session.is_inactive():
            logger.info(f"Session inactive: {session_id}")
            del self.sessions[session_id]
            return False
        
        # Update activity
        session.update_activity()
        
        return True
    
    def logout(self, session_id: str) -> bool:
        """
        Logout user and destroy session
        
        Args:
            session_id: Session ID to destroy
        
        Returns:
            True if logout successful
        """
        if session_id in self.sessions:
            username = self.sessions[session_id].username
            del self.sessions[session_id]
            logger.info(f"User logged out: {username}")
            return True
        
        return False
    
    def change_password(self, session_id: str, old_password: str, new_password: str) -> bool:
        """
        Change user password
        
        Args:
            session_id: Valid session ID
            old_password: Current password
            new_password: New password
        
        Returns:
            True if password changed successfully
        """
        try:
            # Validate session
            if not self.validate_session(session_id):
                logger.error("Invalid session")
                return False
            
            username = self.sessions[session_id].username
            
            # Load credentials
            credentials = self._load_credentials()
            if not credentials or credentials.username != username:
                logger.error("Credentials not found")
                return False
            
            # Verify old password
            old_hash, _ = self.hash_password(old_password, credentials.salt)
            if old_hash != credentials.password_hash:
                logger.error("Old password incorrect")
                return False
            
            # Validate new password
            if not self._validate_password_strength(new_password):
                logger.error("New password does not meet strength requirements")
                return False
            
            # Hash new password
            new_hash, new_salt = self.hash_password(new_password)
            
            # Update credentials
            credentials.password_hash = new_hash
            credentials.salt = new_salt
            self._save_credentials(credentials)
            
            logger.info(f"Password changed for user: {username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to change password: {e}")
            return False
    
    def _create_session(self, username: str) -> str:
        """Create a new session for user"""
        session_id = secrets.token_urlsafe(32)
        
        now = datetime.now()
        expires_at = now + timedelta(hours=self.session_duration_hours)
        
        session = Session(
            session_id=session_id,
            username=username,
            created_at=now,
            expires_at=expires_at,
            last_activity=now
        )
        
        self.sessions[session_id] = session
        
        return session_id
    
    def _validate_password_strength(self, password: str) -> bool:
        """
        Validate password meets strength requirements
        
        Requirements:
        - At least 8 characters
        - Contains uppercase and lowercase
        - Contains at least one digit
        """
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        return has_upper and has_lower and has_digit
    
    def _load_credentials(self) -> Optional[UserCredentials]:
        """Load credentials from file"""
        try:
            if not self.credentials_file.exists():
                return None
            
            with open(self.credentials_file, 'r') as f:
                data = json.load(f)
            
            return UserCredentials(**data)
            
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return None
    
    def _save_credentials(self, credentials: UserCredentials):
        """Save credentials to file"""
        try:
            with open(self.credentials_file, 'w') as f:
                json.dump(asdict(credentials), f, indent=2)
            
            # Ensure restrictive permissions
            os.chmod(self.credentials_file, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            raise


# Global authentication system instance
_auth_system: Optional[AuthenticationSystem] = None


def get_auth_system() -> AuthenticationSystem:
    """Get the global authentication system instance"""
    global _auth_system
    if _auth_system is None:
        _auth_system = AuthenticationSystem()
    return _auth_system
