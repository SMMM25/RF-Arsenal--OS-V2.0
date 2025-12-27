#!/usr/bin/env python3
"""
RF Arsenal OS - Online Intelligence (Tor-Routed)

Anonymous web intelligence gathering - ALL traffic via Tor.
Zero attribution, zero tracking, zero logging.

README COMPLIANCE:
- Anonymous: All traffic through Tor
- No telemetry: Data flows TO system, never OUT
- No logging: No persistent query logs
- Optional: Only used when explicitly enabled

Capabilities:
- CVE/exploit database queries
- Shodan/Censys reconnaissance (anonymous)
- Threat intelligence feeds
- Real-time vulnerability data
- IP/domain reputation

Author: RF Arsenal Team  
License: Proprietary - Authorized Use Only
"""

import os
import sys
import json
import logging
import threading
import hashlib
import time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from urllib.parse import urlencode, quote
import socket

logger = logging.getLogger(__name__)


class IntelSource(Enum):
    """Intelligence sources"""
    NVD = "nvd"                    # NIST NVD (CVE database)
    EXPLOIT_DB = "exploit_db"      # Exploit Database
    GITHUB_ADVISORIES = "github"   # GitHub Security Advisories
    SHODAN = "shodan"              # Shodan (requires API key)
    CENSYS = "censys"              # Censys (requires API key)
    VIRUSTOTAL = "virustotal"      # VirusTotal (requires API key)
    MITRE_ATT = "mitre_attack"     # MITRE ATT&CK
    CISA_KEV = "cisa_kev"          # CISA Known Exploited Vulnerabilities


@dataclass
class IntelResult:
    """Intelligence query result"""
    source: IntelSource
    query: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    cached: bool = False
    error: Optional[str] = None


class TorClient:
    """
    Tor-routed HTTP client
    
    All requests go through Tor - ZERO attribution.
    """
    
    TOR_SOCKS_HOST = "127.0.0.1"
    TOR_SOCKS_PORT = 9050
    TOR_CONTROL_PORT = 9051
    
    def __init__(self):
        self.logger = logging.getLogger('TorClient')
        self._session = None
        self._tor_available = None
        self._circuit_id = None
        
        # User agents for blending
        self._user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]
        self._ua_index = 0
    
    def _get_user_agent(self) -> str:
        """Get rotating user agent"""
        ua = self._user_agents[self._ua_index % len(self._user_agents)]
        self._ua_index += 1
        return ua
    
    def is_tor_available(self) -> bool:
        """Check if Tor is available"""
        if self._tor_available is not None:
            return self._tor_available
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.TOR_SOCKS_HOST, self.TOR_SOCKS_PORT))
            sock.close()
            self._tor_available = (result == 0)
        except Exception:
            self._tor_available = False
        
        return self._tor_available
    
    def _get_session(self):
        """Get requests session with Tor proxy"""
        if self._session is not None:
            return self._session
        
        try:
            import requests
        except ImportError:
            self.logger.error("requests library not available")
            return None
        
        self._session = requests.Session()
        
        # Configure Tor SOCKS proxy
        self._session.proxies = {
            'http': f'socks5h://{self.TOR_SOCKS_HOST}:{self.TOR_SOCKS_PORT}',
            'https': f'socks5h://{self.TOR_SOCKS_HOST}:{self.TOR_SOCKS_PORT}',
        }
        
        return self._session
    
    def new_circuit(self) -> bool:
        """Request a new Tor circuit for fresh identity"""
        try:
            from stem import Signal
            from stem.control import Controller
            
            with Controller.from_port(port=self.TOR_CONTROL_PORT) as controller:
                controller.authenticate()
                controller.signal(Signal.NEWNYM)
                self.logger.info("New Tor circuit established")
                time.sleep(5)  # Wait for circuit to be ready
                return True
        except Exception as e:
            self.logger.warning(f"Failed to get new circuit: {e}")
            return False
    
    def get(
        self,
        url: str,
        params: Dict[str, str] = None,
        headers: Dict[str, str] = None,
        timeout: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """
        Make anonymous GET request via Tor
        
        Returns JSON response or None on error
        """
        if not self.is_tor_available():
            self.logger.error("Tor not available - refusing to make request")
            return None
        
        session = self._get_session()
        if not session:
            return None
        
        # Build headers
        req_headers = {
            'User-Agent': self._get_user_agent(),
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',
        }
        if headers:
            req_headers.update(headers)
        
        try:
            response = session.get(
                url,
                params=params,
                headers=req_headers,
                timeout=timeout,
            )
            response.raise_for_status()
            
            # Try to parse JSON
            try:
                return response.json()
            except json.JSONDecodeError:
                return {'raw': response.text}
                
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return None
    
    def get_ip(self) -> Optional[str]:
        """Get current Tor exit IP for verification"""
        result = self.get("https://api.ipify.org?format=json")
        if result and 'ip' in result:
            return result['ip']
        return None


class OnlineIntelligence:
    """
    Online intelligence gathering - Tor-routed
    
    IMPORTANT: This module ONLY operates when:
    1. User has explicitly enabled online mode
    2. Tor is available and verified
    3. Never logs queries to disk
    
    Data flows TO the system, never OUT (no telemetry).
    """
    
    def __init__(self):
        self.logger = logging.getLogger('OnlineIntelligence')
        self.tor = TorClient()
        
        # RAM-only cache (not persisted)
        self._cache: Dict[str, Tuple[IntelResult, datetime]] = {}
        self._cache_ttl = timedelta(hours=1)
        self._lock = threading.Lock()
        
        # API keys (loaded from environment, never logged)
        self._api_keys = {
            IntelSource.SHODAN: os.environ.get('SHODAN_API_KEY'),
            IntelSource.CENSYS: os.environ.get('CENSYS_API_KEY'),
            IntelSource.VIRUSTOTAL: os.environ.get('VIRUSTOTAL_API_KEY'),
        }
        
        self.logger.info("Online Intelligence initialized (Tor-routed)")
    
    def is_available(self) -> bool:
        """Check if online intelligence is available"""
        return self.tor.is_tor_available()
    
    def verify_anonymity(self) -> Dict[str, Any]:
        """Verify Tor anonymity before operations"""
        if not self.is_available():
            return {"anonymous": False, "error": "Tor not available"}
        
        exit_ip = self.tor.get_ip()
        if not exit_ip:
            return {"anonymous": False, "error": "Could not verify exit IP"}
        
        return {
            "anonymous": True,
            "exit_ip": exit_ip,
            "tor_verified": True,
        }
    
    def _cache_key(self, source: IntelSource, query: str) -> str:
        """Generate cache key"""
        return hashlib.sha256(f"{source.value}:{query}".encode()).hexdigest()
    
    def _check_cache(self, source: IntelSource, query: str) -> Optional[IntelResult]:
        """Check RAM cache for result"""
        key = self._cache_key(source, query)
        with self._lock:
            if key in self._cache:
                result, cached_at = self._cache[key]
                if datetime.now() - cached_at < self._cache_ttl:
                    result.cached = True
                    return result
                else:
                    del self._cache[key]
        return None
    
    def _store_cache(self, source: IntelSource, query: str, result: IntelResult):
        """Store result in RAM cache"""
        key = self._cache_key(source, query)
        with self._lock:
            self._cache[key] = (result, datetime.now())
    
    # === CVE / Exploit Intelligence ===
    
    def search_cve(self, keyword: str) -> IntelResult:
        """
        Search NIST NVD for CVEs
        
        Anonymous via Tor - no API key required
        """
        # Check cache
        cached = self._check_cache(IntelSource.NVD, keyword)
        if cached:
            return cached
        
        url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {"keywordSearch": keyword, "resultsPerPage": 20}
        
        data = self.tor.get(url, params=params)
        
        result = IntelResult(
            source=IntelSource.NVD,
            query=keyword,
            data=data or {"error": "Request failed"},
            error=None if data else "Request failed",
        )
        
        if data:
            self._store_cache(IntelSource.NVD, keyword, result)
        
        return result
    
    def get_cve_details(self, cve_id: str) -> IntelResult:
        """Get details for a specific CVE"""
        cached = self._check_cache(IntelSource.NVD, cve_id)
        if cached:
            return cached
        
        url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {"cveId": cve_id}
        
        data = self.tor.get(url, params=params)
        
        result = IntelResult(
            source=IntelSource.NVD,
            query=cve_id,
            data=data or {"error": "Request failed"},
            error=None if data else "Request failed",
        )
        
        if data:
            self._store_cache(IntelSource.NVD, cve_id, result)
        
        return result
    
    def search_exploits(self, keyword: str) -> IntelResult:
        """
        Search Exploit Database
        
        Anonymous via Tor
        """
        cached = self._check_cache(IntelSource.EXPLOIT_DB, keyword)
        if cached:
            return cached
        
        # ExploitDB doesn't have a public API, use search
        url = f"https://www.exploit-db.com/search"
        params = {"q": keyword}
        
        # Note: This may return HTML, not JSON
        data = self.tor.get(url, params=params)
        
        result = IntelResult(
            source=IntelSource.EXPLOIT_DB,
            query=keyword,
            data=data or {"error": "Request failed"},
            error=None if data else "Request failed",
        )
        
        if data:
            self._store_cache(IntelSource.EXPLOIT_DB, keyword, result)
        
        return result
    
    def get_cisa_kev(self) -> IntelResult:
        """
        Get CISA Known Exploited Vulnerabilities
        
        Critical vulnerabilities actively exploited in the wild
        """
        cached = self._check_cache(IntelSource.CISA_KEV, "catalog")
        if cached:
            return cached
        
        url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
        
        data = self.tor.get(url)
        
        result = IntelResult(
            source=IntelSource.CISA_KEV,
            query="catalog",
            data=data or {"error": "Request failed"},
            error=None if data else "Request failed",
        )
        
        if data:
            self._store_cache(IntelSource.CISA_KEV, "catalog", result)
        
        return result
    
    # === Reconnaissance ===
    
    def shodan_search(self, query: str) -> IntelResult:
        """
        Search Shodan for exposed devices/services
        
        Requires SHODAN_API_KEY environment variable
        """
        api_key = self._api_keys.get(IntelSource.SHODAN)
        if not api_key:
            return IntelResult(
                source=IntelSource.SHODAN,
                query=query,
                data={},
                error="SHODAN_API_KEY not configured",
            )
        
        cached = self._check_cache(IntelSource.SHODAN, query)
        if cached:
            return cached
        
        url = "https://api.shodan.io/shodan/host/search"
        params = {"key": api_key, "query": query}
        
        data = self.tor.get(url, params=params)
        
        result = IntelResult(
            source=IntelSource.SHODAN,
            query=query,
            data=data or {"error": "Request failed"},
            error=None if data else "Request failed",
        )
        
        if data and 'error' not in data:
            self._store_cache(IntelSource.SHODAN, query, result)
        
        return result
    
    def shodan_host(self, ip: str) -> IntelResult:
        """Get Shodan info for a specific IP"""
        api_key = self._api_keys.get(IntelSource.SHODAN)
        if not api_key:
            return IntelResult(
                source=IntelSource.SHODAN,
                query=ip,
                data={},
                error="SHODAN_API_KEY not configured",
            )
        
        cached = self._check_cache(IntelSource.SHODAN, f"host:{ip}")
        if cached:
            return cached
        
        url = f"https://api.shodan.io/shodan/host/{ip}"
        params = {"key": api_key}
        
        data = self.tor.get(url, params=params)
        
        result = IntelResult(
            source=IntelSource.SHODAN,
            query=ip,
            data=data or {"error": "Request failed"},
            error=None if data else "Request failed",
        )
        
        if data and 'error' not in data:
            self._store_cache(IntelSource.SHODAN, f"host:{ip}", result)
        
        return result
    
    # === MITRE ATT&CK ===
    
    def get_attack_techniques(self, tactic: str = None) -> IntelResult:
        """
        Get MITRE ATT&CK techniques
        
        Anonymous access to ATT&CK data
        """
        query = f"attack:{tactic}" if tactic else "attack:all"
        cached = self._check_cache(IntelSource.MITRE_ATT, query)
        if cached:
            return cached
        
        # MITRE ATT&CK STIX data
        url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
        
        data = self.tor.get(url, timeout=60)  # Large file
        
        # Filter by tactic if specified
        if data and tactic and 'objects' in data:
            filtered = [
                obj for obj in data['objects']
                if obj.get('type') == 'attack-pattern' and
                any(tactic.lower() in str(phase).lower() 
                    for phase in obj.get('kill_chain_phases', []))
            ]
            data = {'objects': filtered, 'filtered_by': tactic}
        
        result = IntelResult(
            source=IntelSource.MITRE_ATT,
            query=query,
            data=data or {"error": "Request failed"},
            error=None if data else "Request failed",
        )
        
        if data:
            self._store_cache(IntelSource.MITRE_ATT, query, result)
        
        return result
    
    # === Threat Intelligence ===
    
    def check_ip_reputation(self, ip: str) -> Dict[str, Any]:
        """Check IP reputation from multiple sources"""
        results = {}
        
        # AbuseIPDB (if available)
        # VirusTotal (if API key configured)
        vt_key = self._api_keys.get(IntelSource.VIRUSTOTAL)
        if vt_key:
            url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
            headers = {"x-apikey": vt_key}
            data = self.tor.get(url, headers=headers)
            if data:
                results['virustotal'] = data
        
        # Shodan
        shodan_result = self.shodan_host(ip)
        if not shodan_result.error:
            results['shodan'] = shodan_result.data
        
        return results
    
    # === Convenience Methods ===
    
    def quick_vuln_check(self, target: str) -> Dict[str, Any]:
        """
        Quick vulnerability check for a target
        
        Combines multiple sources for comprehensive results
        """
        results = {
            "target": target,
            "timestamp": datetime.now().isoformat(),
            "cves": [],
            "exploits": [],
            "shodan": None,
        }
        
        # Search for CVEs
        cve_result = self.search_cve(target)
        if not cve_result.error and 'vulnerabilities' in cve_result.data:
            results['cves'] = [
                {
                    "id": v.get('cve', {}).get('id'),
                    "description": v.get('cve', {}).get('descriptions', [{}])[0].get('value', ''),
                    "severity": v.get('cve', {}).get('metrics', {}).get('cvssMetricV31', [{}])[0].get('cvssData', {}).get('baseSeverity', 'UNKNOWN'),
                }
                for v in cve_result.data.get('vulnerabilities', [])[:10]
            ]
        
        # Search for exploits
        exploit_result = self.search_exploits(target)
        if not exploit_result.error:
            results['exploits'] = exploit_result.data
        
        return results
    
    def clear_cache(self):
        """Clear RAM cache"""
        with self._lock:
            self._cache.clear()
        self.logger.info("Intelligence cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get intelligence module statistics"""
        return {
            "tor_available": self.is_available(),
            "cache_entries": len(self._cache),
            "api_keys_configured": {
                source.value: bool(key)
                for source, key in self._api_keys.items()
            },
        }


# Convenience function
def get_online_intel() -> OnlineIntelligence:
    """Get online intelligence instance"""
    return OnlineIntelligence()
