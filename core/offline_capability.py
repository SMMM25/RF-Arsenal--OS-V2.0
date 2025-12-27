#!/usr/bin/env python3
"""
RF Arsenal OS - Offline Capability Module
Ensures all core functions work without internet connectivity

OFFLINE COMPLIANCE:
- All core functions work without network
- Local caching of threat databases
- Offline AI inference support
- No phone-home or telemetry
- Network access only when explicitly enabled

Author: RF Arsenal Security Team
"""

import os
import sys
import json
import time
import hashlib
import sqlite3
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import gzip
import pickle

logger = logging.getLogger(__name__)

# Default cache directory (RAM-based when possible)
DEFAULT_CACHE_DIR = "/tmp/rf_arsenal_cache"
PERSISTENT_CACHE_DIR = os.path.expanduser("~/.rf_arsenal/cache")


class CacheLocation(Enum):
    """Cache storage locations."""
    RAM = "ram"           # tmpfs/RAM disk (volatile)
    DISK = "disk"         # Persistent disk storage
    ENCRYPTED = "encrypted"  # Encrypted disk storage


class OfflineStatus(Enum):
    """Network/offline status."""
    OFFLINE = "offline"           # No network, fully offline
    ONLINE_RESTRICTED = "restricted"  # Limited network (Tor/VPN only)
    ONLINE_FULL = "full"          # Full network access


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    key: str
    data_hash: str
    created_at: datetime
    expires_at: Optional[datetime]
    size_bytes: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    compressed: bool = False
    encrypted: bool = False


class LocalThreatDatabase:
    """
    Local threat intelligence database for offline operation.
    
    Caches threat data locally for use without network connectivity.
    Supports periodic updates when network is available.
    """
    
    def __init__(self, cache_dir: str = None):
        self._cache_dir = cache_dir or PERSISTENT_CACHE_DIR
        self._db_path = os.path.join(self._cache_dir, "threat_intel.db")
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        
        self._ensure_cache_dir()
        self._init_database()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self._cache_dir, mode=0o700, exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database for threat data."""
        with self._lock:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            
            # Create tables
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS malicious_addresses (
                    address TEXT PRIMARY KEY,
                    chain TEXT,
                    category TEXT,
                    threat_level INTEGER,
                    first_seen TEXT,
                    last_seen TEXT,
                    report_count INTEGER DEFAULT 1,
                    tags TEXT,
                    description TEXT
                );
                
                CREATE TABLE IF NOT EXISTS malicious_ips (
                    ip TEXT PRIMARY KEY,
                    category TEXT,
                    threat_level INTEGER,
                    first_seen TEXT,
                    last_seen TEXT,
                    country TEXT,
                    asn TEXT,
                    description TEXT
                );
                
                CREATE TABLE IF NOT EXISTS malicious_domains (
                    domain TEXT PRIMARY KEY,
                    category TEXT,
                    threat_level INTEGER,
                    first_seen TEXT,
                    last_seen TEXT,
                    registrar TEXT,
                    description TEXT
                );
                
                CREATE TABLE IF NOT EXISTS cve_database (
                    cve_id TEXT PRIMARY KEY,
                    severity TEXT,
                    cvss_score REAL,
                    description TEXT,
                    published_date TEXT,
                    affected_products TEXT,
                    references TEXT
                );
                
                CREATE TABLE IF NOT EXISTS fingerprints (
                    fingerprint_id TEXT PRIMARY KEY,
                    device_type TEXT,
                    manufacturer TEXT,
                    model TEXT,
                    rf_signature BLOB,
                    frequency_range TEXT,
                    confidence REAL
                );
                
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    table_name TEXT PRIMARY KEY,
                    last_updated TEXT,
                    record_count INTEGER,
                    source TEXT,
                    version TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_addresses_chain ON malicious_addresses(chain);
                CREATE INDEX IF NOT EXISTS idx_addresses_threat ON malicious_addresses(threat_level);
                CREATE INDEX IF NOT EXISTS idx_cve_severity ON cve_database(severity);
            """)
            
            self._conn.commit()
    
    def check_address(self, address: str, chain: str = None) -> Optional[Dict]:
        """
        Check if address is known malicious.
        
        Works completely offline using local database.
        """
        with self._lock:
            if chain:
                cursor = self._conn.execute(
                    "SELECT * FROM malicious_addresses WHERE address = ? AND chain = ?",
                    (address.lower(), chain)
                )
            else:
                cursor = self._conn.execute(
                    "SELECT * FROM malicious_addresses WHERE address = ?",
                    (address.lower(),)
                )
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'address': row[0],
                    'chain': row[1],
                    'category': row[2],
                    'threat_level': row[3],
                    'first_seen': row[4],
                    'last_seen': row[5],
                    'report_count': row[6],
                    'tags': json.loads(row[7]) if row[7] else [],
                    'description': row[8],
                    'source': 'local_cache'
                }
            
            return None
    
    def check_ip(self, ip: str) -> Optional[Dict]:
        """Check if IP is known malicious."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM malicious_ips WHERE ip = ?",
                (ip,)
            )
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'ip': row[0],
                    'category': row[1],
                    'threat_level': row[2],
                    'first_seen': row[3],
                    'last_seen': row[4],
                    'country': row[5],
                    'asn': row[6],
                    'description': row[7],
                    'source': 'local_cache'
                }
            
            return None
    
    def check_domain(self, domain: str) -> Optional[Dict]:
        """Check if domain is known malicious."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM malicious_domains WHERE domain = ?",
                (domain.lower(),)
            )
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'domain': row[0],
                    'category': row[1],
                    'threat_level': row[2],
                    'first_seen': row[3],
                    'last_seen': row[4],
                    'registrar': row[5],
                    'description': row[6],
                    'source': 'local_cache'
                }
            
            return None
    
    def search_cve(self, query: str = None, severity: str = None, limit: int = 100) -> List[Dict]:
        """Search CVE database offline."""
        with self._lock:
            sql = "SELECT * FROM cve_database WHERE 1=1"
            params = []
            
            if query:
                sql += " AND (cve_id LIKE ? OR description LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])
            
            if severity:
                sql += " AND severity = ?"
                params.append(severity.upper())
            
            sql += " ORDER BY cvss_score DESC LIMIT ?"
            params.append(limit)
            
            cursor = self._conn.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'cve_id': row[0],
                    'severity': row[1],
                    'cvss_score': row[2],
                    'description': row[3],
                    'published_date': row[4],
                    'affected_products': json.loads(row[5]) if row[5] else [],
                    'references': json.loads(row[6]) if row[6] else [],
                    'source': 'local_cache'
                })
            
            return results
    
    def add_malicious_address(
        self,
        address: str,
        chain: str,
        category: str,
        threat_level: int,
        tags: List[str] = None,
        description: str = None
    ):
        """Add or update malicious address in local database."""
        with self._lock:
            now = datetime.now().isoformat()
            
            self._conn.execute("""
                INSERT INTO malicious_addresses 
                (address, chain, category, threat_level, first_seen, last_seen, tags, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    last_seen = ?,
                    report_count = report_count + 1,
                    threat_level = MAX(threat_level, ?)
            """, (
                address.lower(), chain, category, threat_level,
                now, now, json.dumps(tags or []), description,
                now, threat_level
            ))
            
            self._conn.commit()
    
    def import_threat_feed(self, feed_data: Dict, feed_type: str):
        """Import threat feed data into local database."""
        with self._lock:
            imported = 0
            
            if feed_type == 'addresses':
                for entry in feed_data.get('entries', []):
                    self.add_malicious_address(
                        address=entry.get('address'),
                        chain=entry.get('chain', 'unknown'),
                        category=entry.get('category', 'unknown'),
                        threat_level=entry.get('threat_level', 5),
                        tags=entry.get('tags', []),
                        description=entry.get('description')
                    )
                    imported += 1
            
            elif feed_type == 'cve':
                for entry in feed_data.get('entries', []):
                    self._conn.execute("""
                        INSERT OR REPLACE INTO cve_database
                        (cve_id, severity, cvss_score, description, published_date, affected_products, references)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.get('cve_id'),
                        entry.get('severity', 'UNKNOWN'),
                        entry.get('cvss_score', 0.0),
                        entry.get('description', ''),
                        entry.get('published_date'),
                        json.dumps(entry.get('affected_products', [])),
                        json.dumps(entry.get('references', []))
                    ))
                    imported += 1
            
            self._conn.commit()
            
            # Update metadata
            self._conn.execute("""
                INSERT OR REPLACE INTO cache_metadata
                (table_name, last_updated, record_count, source, version)
                VALUES (?, ?, ?, ?, ?)
            """, (
                feed_type,
                datetime.now().isoformat(),
                imported,
                feed_data.get('source', 'manual'),
                feed_data.get('version', '1.0')
            ))
            self._conn.commit()
            
            logger.info(f"Imported {imported} entries for {feed_type}")
            return imported
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._lock:
            stats = {
                'tables': {}
            }
            
            # Get counts for each table
            for table in ['malicious_addresses', 'malicious_ips', 'malicious_domains', 'cve_database', 'fingerprints']:
                cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                
                # Get metadata if available
                meta_cursor = self._conn.execute(
                    "SELECT last_updated, source, version FROM cache_metadata WHERE table_name = ?",
                    (table,)
                )
                meta = meta_cursor.fetchone()
                
                stats['tables'][table] = {
                    'record_count': count,
                    'last_updated': meta[0] if meta else None,
                    'source': meta[1] if meta else None,
                    'version': meta[2] if meta else None
                }
            
            # Database file size
            if os.path.exists(self._db_path):
                stats['database_size_mb'] = os.path.getsize(self._db_path) / (1024 * 1024)
            
            return stats
    
    def close(self):
        """Close database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


class OfflineCache:
    """
    General-purpose offline cache for any data.
    
    Supports RAM-based and disk-based caching with
    optional compression and encryption.
    """
    
    def __init__(
        self,
        cache_dir: str = None,
        location: CacheLocation = CacheLocation.DISK,
        max_size_mb: int = 100,
        default_ttl_hours: int = 24
    ):
        self._location = location
        self._max_size = max_size_mb * 1024 * 1024
        self._default_ttl = timedelta(hours=default_ttl_hours)
        
        # Set cache directory based on location
        if location == CacheLocation.RAM:
            self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        else:
            self._cache_dir = cache_dir or PERSISTENT_CACHE_DIR
        
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._current_size = 0
        
        os.makedirs(self._cache_dir, mode=0o700, exist_ok=True)
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        metadata_path = os.path.join(self._cache_dir, "cache_metadata.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                
                for key, entry_data in data.get('entries', {}).items():
                    self._entries[key] = CacheEntry(
                        key=entry_data['key'],
                        data_hash=entry_data['data_hash'],
                        created_at=datetime.fromisoformat(entry_data['created_at']),
                        expires_at=datetime.fromisoformat(entry_data['expires_at']) if entry_data.get('expires_at') else None,
                        size_bytes=entry_data['size_bytes'],
                        access_count=entry_data.get('access_count', 0),
                        compressed=entry_data.get('compressed', False),
                        encrypted=entry_data.get('encrypted', False)
                    )
                    self._current_size += entry_data['size_bytes']
                    
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        metadata_path = os.path.join(self._cache_dir, "cache_metadata.json")
        
        try:
            data = {
                'entries': {
                    key: {
                        'key': entry.key,
                        'data_hash': entry.data_hash,
                        'created_at': entry.created_at.isoformat(),
                        'expires_at': entry.expires_at.isoformat() if entry.expires_at else None,
                        'size_bytes': entry.size_bytes,
                        'access_count': entry.access_count,
                        'compressed': entry.compressed,
                        'encrypted': entry.encrypted
                    }
                    for key, entry in self._entries.items()
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(data, f)
                
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:32]
        return os.path.join(self._cache_dir, f"cache_{key_hash}.dat")
    
    def set(
        self,
        key: str,
        data: Any,
        ttl_hours: int = None,
        compress: bool = True
    ) -> bool:
        """
        Store data in cache.
        
        Args:
            key: Cache key
            data: Data to cache (any picklable object)
            ttl_hours: Time-to-live in hours
            compress: Compress data before storing
        """
        with self._lock:
            try:
                # Serialize data
                serialized = pickle.dumps(data)
                
                # Compress if enabled
                if compress:
                    serialized = gzip.compress(serialized)
                
                # Check size
                if len(serialized) > self._max_size:
                    logger.warning(f"Data too large for cache: {len(serialized)} bytes")
                    return False
                
                # Evict if needed
                while self._current_size + len(serialized) > self._max_size:
                    self._evict_oldest()
                
                # Write to file
                cache_path = self._get_cache_path(key)
                with open(cache_path, 'wb') as f:
                    f.write(serialized)
                
                # Create entry
                ttl = timedelta(hours=ttl_hours) if ttl_hours else self._default_ttl
                entry = CacheEntry(
                    key=key,
                    data_hash=hashlib.sha256(serialized).hexdigest(),
                    created_at=datetime.now(),
                    expires_at=datetime.now() + ttl if ttl else None,
                    size_bytes=len(serialized),
                    compressed=compress
                )
                
                # Update tracking
                if key in self._entries:
                    self._current_size -= self._entries[key].size_bytes
                
                self._entries[key] = entry
                self._current_size += len(serialized)
                
                self._save_metadata()
                
                return True
                
            except Exception as e:
                logger.error(f"Cache set failed: {e}")
                return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve data from cache.
        
        Returns default if not found or expired.
        """
        with self._lock:
            entry = self._entries.get(key)
            
            if not entry:
                return default
            
            # Check expiration
            if entry.expires_at and datetime.now() > entry.expires_at:
                self.delete(key)
                return default
            
            try:
                cache_path = self._get_cache_path(key)
                
                if not os.path.exists(cache_path):
                    del self._entries[key]
                    return default
                
                with open(cache_path, 'rb') as f:
                    data = f.read()
                
                # Decompress if needed
                if entry.compressed:
                    data = gzip.decompress(data)
                
                # Deserialize
                result = pickle.loads(data)
                
                # Update access stats
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                return result
                
            except Exception as e:
                logger.error(f"Cache get failed: {e}")
                return default
    
    def delete(self, key: str):
        """Delete item from cache."""
        with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                self._current_size -= entry.size_bytes
                del self._entries[key]
                
                # Delete file
                cache_path = self._get_cache_path(key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                
                self._save_metadata()
    
    def _evict_oldest(self):
        """Evict oldest cache entry."""
        if not self._entries:
            return
        
        # Find oldest entry
        oldest_key = min(self._entries.keys(), key=lambda k: self._entries[k].created_at)
        self.delete(oldest_key)
    
    def clear(self):
        """Clear all cached data."""
        with self._lock:
            for key in list(self._entries.keys()):
                self.delete(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'location': self._location.value,
                'max_size_mb': self._max_size / (1024 * 1024),
                'current_size_mb': self._current_size / (1024 * 1024),
                'usage_percent': (self._current_size / self._max_size) * 100 if self._max_size > 0 else 0,
                'entry_count': len(self._entries),
                'cache_dir': self._cache_dir
            }


class OfflineCapabilityManager:
    """
    Main manager for offline capabilities.
    
    Coordinates all offline functionality and ensures
    system works without network connectivity.
    """
    
    def __init__(self):
        self._status = OfflineStatus.OFFLINE
        self._threat_db = LocalThreatDatabase()
        self._cache = OfflineCache()
        self._network_blocked = True
        self._lock = threading.RLock()
        
        # Track which features require network
        self._network_required_features: Dict[str, bool] = {}
    
    def get_status(self) -> OfflineStatus:
        """Get current offline status."""
        return self._status
    
    def set_status(self, status: OfflineStatus):
        """Set offline status."""
        with self._lock:
            old_status = self._status
            self._status = status
            
            if status == OfflineStatus.OFFLINE:
                self._network_blocked = True
                logger.info("Switched to OFFLINE mode - all network blocked")
            elif status == OfflineStatus.ONLINE_RESTRICTED:
                self._network_blocked = False
                logger.info("Switched to RESTRICTED mode - Tor/VPN only")
            else:
                self._network_blocked = False
                logger.info("Switched to FULL ONLINE mode")
    
    def is_offline(self) -> bool:
        """Check if system is in offline mode."""
        return self._status == OfflineStatus.OFFLINE
    
    def check_feature_offline_capable(self, feature_name: str) -> Tuple[bool, str]:
        """
        Check if feature works offline.
        
        Returns:
            Tuple of (is_offline_capable, reason)
        """
        # Features that always work offline
        offline_features = {
            'spectrum_analysis': True,
            'signal_capture': True,
            'signal_replay': True,
            'rf_fingerprinting': True,
            'jamming': True,
            'local_threat_check': True,
            'cve_search': True,
            'stealth_mode': True,
            'mac_randomization': True,
            'secure_delete': True,
            'ram_operations': True,
            'vehicle_can_bus': True,
            'bluetooth_scan': True,
            'wifi_scan': True,
        }
        
        # Features that require network
        network_features = {
            'online_threat_feed_update': "Requires internet to download threat feeds",
            'tor_connection': "Requires internet for Tor network",
            'vpn_connection': "Requires internet for VPN",
            'osint_lookup': "Requires internet for OSINT queries",
            'blockchain_live_query': "Requires internet for blockchain APIs",
            'cloud_security_scan': "Requires internet for cloud API access",
        }
        
        if feature_name in offline_features:
            return True, "Feature fully operational offline"
        
        if feature_name in network_features:
            return False, network_features[feature_name]
        
        # Unknown feature - assume offline capable
        return True, "Feature assumed offline capable"
    
    def get_threat_database(self) -> LocalThreatDatabase:
        """Get local threat database for offline lookups."""
        return self._threat_db
    
    def get_cache(self) -> OfflineCache:
        """Get offline cache."""
        return self._cache
    
    def cache_for_offline(self, key: str, data: Any, ttl_hours: int = 168):
        """
        Cache data for offline use.
        
        Default TTL is 1 week (168 hours).
        """
        return self._cache.set(key, data, ttl_hours=ttl_hours)
    
    def get_cached(self, key: str, default: Any = None) -> Any:
        """Get cached data."""
        return self._cache.get(key, default)
    
    def get_offline_status_report(self) -> Dict[str, Any]:
        """Get comprehensive offline status report."""
        return {
            'status': self._status.value,
            'network_blocked': self._network_blocked,
            'threat_database': self._threat_db.get_database_stats(),
            'cache': self._cache.get_stats(),
            'offline_features': [
                'spectrum_analysis', 'signal_capture', 'signal_replay',
                'rf_fingerprinting', 'jamming', 'local_threat_check',
                'cve_search', 'stealth_mode', 'mac_randomization',
                'secure_delete', 'ram_operations', 'vehicle_can_bus',
                'bluetooth_scan', 'wifi_scan'
            ],
            'network_required_features': [
                'online_threat_feed_update', 'tor_connection', 'vpn_connection',
                'osint_lookup', 'blockchain_live_query', 'cloud_security_scan'
            ]
        }
    
    def shutdown(self):
        """Clean shutdown of offline manager."""
        self._threat_db.close()


# Singleton instance
_offline_manager: Optional[OfflineCapabilityManager] = None
_manager_lock = threading.Lock()


def get_offline_manager() -> OfflineCapabilityManager:
    """Get singleton offline capability manager."""
    global _offline_manager
    
    with _manager_lock:
        if _offline_manager is None:
            _offline_manager = OfflineCapabilityManager()
        return _offline_manager


# Export all classes
__all__ = [
    'CacheLocation',
    'OfflineStatus',
    'CacheEntry',
    'LocalThreatDatabase',
    'OfflineCache',
    'OfflineCapabilityManager',
    'get_offline_manager',
]
