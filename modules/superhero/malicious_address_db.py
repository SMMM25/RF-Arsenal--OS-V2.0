#!/usr/bin/env python3
"""
RF Arsenal OS - SUPERHERO Module
Malicious Address Database

A comprehensive database system for tracking and categorizing known malicious
cryptocurrency addresses. Aggregates data from public sources only.

STEALTH COMPLIANCE:
- All operations through proxy chains
- RAM-only data handling option
- No telemetry or logging
- Offline capability for analysis
- Data sourced from PUBLIC sources only

LEGAL COMPLIANCE:
- No unauthorized access to any systems
- All data from publicly available sources
- Analysis results are informational only
- No actions taken against addresses

Author: RF Arsenal Security Team
License: Authorized Use Only
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import sqlite3
import os
from abc import ABC, abstractmethod


class ThreatLevel(Enum):
    """Threat level classification for addresses."""
    CRITICAL = "critical"       # Active scam/hack, high volume
    HIGH = "high"              # Known malicious, confirmed reports
    MEDIUM = "medium"          # Suspected malicious, multiple flags
    LOW = "low"                # Single flag or unconfirmed
    WATCH = "watch"            # Under observation
    UNKNOWN = "unknown"        # Not in database


class AddressCategory(Enum):
    """Categories of malicious activity."""
    SCAM = "scam"                      # General scams
    PHISHING = "phishing"              # Phishing attacks
    RANSOMWARE = "ransomware"          # Ransomware payments
    DARKNET_MARKET = "darknet_market"  # Darknet marketplace
    MIXER = "mixer"                    # Mixing service
    EXCHANGE_HACK = "exchange_hack"    # Stolen from exchange
    DEFI_EXPLOIT = "defi_exploit"      # DeFi protocol exploit
    RUG_PULL = "rug_pull"              # Project rug pull
    PONZI = "ponzi"                    # Ponzi scheme
    SANCTIONS = "sanctions"            # Sanctioned entity
    TERRORISM = "terrorism"            # Terror financing
    MONEY_LAUNDERING = "money_laundering"
    NFT_SCAM = "nft_scam"              # NFT-related scam
    FAKE_ICO = "fake_ico"              # Fake ICO/token sale
    DUST_ATTACK = "dust_attack"        # Dust attack source
    HONEYPOT = "honeypot"              # Honeypot contract
    OTHER = "other"


class DataSource(Enum):
    """Public data sources for address intelligence."""
    ETHERSCAN = "etherscan"
    BLOCKCHAIN_INFO = "blockchain_info"
    CHAINALYSIS_ALERTS = "chainalysis_alerts"
    OFAC_SDN = "ofac_sdn"             # OFAC Sanctions list
    FBI_ALERTS = "fbi_alerts"
    SCAM_SNIFFER = "scam_sniffer"
    CRYPTO_SCAM_DB = "crypto_scam_db"
    COMMUNITY_REPORTS = "community_reports"
    SOCIAL_MEDIA = "social_media"
    SECURITY_FIRMS = "security_firms"
    MANUAL_ENTRY = "manual_entry"


class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    BSC = "bsc"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    SOLANA = "solana"
    TRON = "tron"
    FANTOM = "fantom"


@dataclass
class AddressReport:
    """Individual report about a malicious address."""
    report_id: str
    source: DataSource
    report_date: datetime
    description: str
    evidence_links: List[str] = field(default_factory=list)
    victim_count: int = 0
    amount_stolen_usd: float = 0.0
    verified: bool = False
    verifier: Optional[str] = None


@dataclass
class MaliciousAddress:
    """Record of a known malicious address."""
    address: str
    chain: Chain
    threat_level: ThreatLevel
    categories: List[AddressCategory]
    first_seen: datetime
    last_updated: datetime
    
    # Reports and evidence
    reports: List[AddressReport] = field(default_factory=list)
    total_stolen_usd: float = 0.0
    known_victims: int = 0
    
    # Related addresses
    related_addresses: List[str] = field(default_factory=list)
    cluster_id: Optional[str] = None
    
    # Labels and tags
    labels: List[str] = field(default_factory=list)
    entity_name: Optional[str] = None
    
    # Activity metrics
    active: bool = True
    last_activity: Optional[datetime] = None
    transaction_count: int = 0
    
    # Metadata
    confidence_score: float = 0.0  # 0-100
    notes: str = ""
    
    def calculate_threat_level(self) -> ThreatLevel:
        """Calculate threat level based on available data."""
        score = 0
        
        # Based on stolen amount
        if self.total_stolen_usd > 10_000_000:
            score += 40
        elif self.total_stolen_usd > 1_000_000:
            score += 30
        elif self.total_stolen_usd > 100_000:
            score += 20
        elif self.total_stolen_usd > 10_000:
            score += 10
        
        # Based on victim count
        if self.known_victims > 1000:
            score += 25
        elif self.known_victims > 100:
            score += 15
        elif self.known_victims > 10:
            score += 10
        
        # Based on report count
        score += min(len(self.reports) * 5, 20)
        
        # Based on categories
        if AddressCategory.TERRORISM in self.categories:
            score += 30
        if AddressCategory.SANCTIONS in self.categories:
            score += 25
        if AddressCategory.RANSOMWARE in self.categories:
            score += 20
        
        # Based on activity
        if self.active:
            score += 10
        
        # Determine threat level
        if score >= 80:
            return ThreatLevel.CRITICAL
        elif score >= 60:
            return ThreatLevel.HIGH
        elif score >= 40:
            return ThreatLevel.MEDIUM
        elif score >= 20:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.WATCH


@dataclass
class AddressCluster:
    """Cluster of related addresses (same entity)."""
    cluster_id: str
    addresses: List[str]
    entity_name: Optional[str]
    categories: List[AddressCategory]
    total_stolen_usd: float
    threat_level: ThreatLevel
    first_seen: datetime
    last_updated: datetime
    notes: str = ""


@dataclass
class SecurityProfile:
    """Security profile analysis for an address."""
    address: str
    chain: Chain
    analyzed_at: datetime
    
    # Wallet type detection
    wallet_type: str = "unknown"  # EOA, contract, multisig, etc.
    is_contract: bool = False
    contract_verified: bool = False
    contract_name: Optional[str] = None
    
    # Security features
    has_timelock: bool = False
    timelock_duration: Optional[int] = None  # seconds
    is_multisig: bool = False
    multisig_threshold: Optional[Tuple[int, int]] = None  # (required, total)
    has_guardians: bool = False
    guardian_count: int = 0
    
    # Access patterns
    uses_mixer: bool = False
    uses_bridge: bool = False
    uses_dex: bool = False
    
    # Activity analysis
    transaction_frequency: str = "unknown"  # low, medium, high, very_high
    typical_transaction_size: float = 0.0
    unusual_patterns: List[str] = field(default_factory=list)
    
    # Risk indicators
    risk_score: float = 0.0  # 0-100
    risk_factors: List[str] = field(default_factory=list)


class MaliciousAddressDatabase:
    """
    Database for tracking known malicious cryptocurrency addresses.
    
    STEALTH COMPLIANCE:
    - Can operate entirely in RAM (no disk persistence)
    - No external network calls in offline mode
    - All data from public sources only
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        ram_only: bool = True
    ):
        """
        Initialize the malicious address database.
        
        Args:
            db_path: Path to SQLite database (None for RAM-only)
            ram_only: If True, operate entirely in RAM
        """
        self.ram_only = ram_only
        self.db_path = db_path if not ram_only else ":memory:"
        
        # RAM storage
        self._addresses: Dict[str, MaliciousAddress] = {}
        self._clusters: Dict[str, AddressCluster] = {}
        self._security_profiles: Dict[str, SecurityProfile] = {}
        
        # Indexes for fast lookup
        self._category_index: Dict[AddressCategory, Set[str]] = {
            cat: set() for cat in AddressCategory
        }
        self._chain_index: Dict[Chain, Set[str]] = {
            chain: set() for chain in Chain
        }
        self._threat_index: Dict[ThreatLevel, Set[str]] = {
            level: set() for level in ThreatLevel
        }
        
        # Initialize database if not RAM-only
        if not ram_only and db_path:
            self._init_sqlite_db()
    
    def _init_sqlite_db(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS addresses (
                address TEXT PRIMARY KEY,
                chain TEXT NOT NULL,
                threat_level TEXT NOT NULL,
                categories TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                total_stolen_usd REAL DEFAULT 0,
                known_victims INTEGER DEFAULT 0,
                entity_name TEXT,
                cluster_id TEXT,
                active INTEGER DEFAULT 1,
                confidence_score REAL DEFAULT 0,
                notes TEXT
            );
            
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                address TEXT NOT NULL,
                source TEXT NOT NULL,
                report_date TEXT NOT NULL,
                description TEXT,
                evidence_links TEXT,
                victim_count INTEGER DEFAULT 0,
                amount_stolen_usd REAL DEFAULT 0,
                verified INTEGER DEFAULT 0,
                FOREIGN KEY (address) REFERENCES addresses(address)
            );
            
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id TEXT PRIMARY KEY,
                addresses TEXT NOT NULL,
                entity_name TEXT,
                categories TEXT NOT NULL,
                total_stolen_usd REAL DEFAULT 0,
                threat_level TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                notes TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_address_chain ON addresses(chain);
            CREATE INDEX IF NOT EXISTS idx_address_threat ON addresses(threat_level);
            CREATE INDEX IF NOT EXISTS idx_reports_address ON reports(address);
        """)
        
        conn.commit()
        conn.close()
    
    def add_address(
        self,
        address: str,
        chain: Chain,
        categories: List[AddressCategory],
        source: DataSource,
        description: str,
        threat_level: Optional[ThreatLevel] = None,
        stolen_amount: float = 0.0,
        victim_count: int = 0,
        evidence_links: Optional[List[str]] = None,
        entity_name: Optional[str] = None,
        labels: Optional[List[str]] = None
    ) -> MaliciousAddress:
        """
        Add a new malicious address to the database.
        
        Args:
            address: The cryptocurrency address
            chain: Blockchain network
            categories: Categories of malicious activity
            source: Data source for this entry
            description: Description of the malicious activity
            threat_level: Optional override for threat level
            stolen_amount: Estimated stolen amount in USD
            victim_count: Number of known victims
            evidence_links: Links to evidence/reports
            entity_name: Name of the malicious entity
            labels: Additional labels/tags
            
        Returns:
            Created MaliciousAddress object
        """
        now = datetime.now()
        address_lower = address.lower()
        
        # Check if address already exists
        if address_lower in self._addresses:
            return self.update_address(
                address_lower,
                categories=categories,
                source=source,
                description=description,
                stolen_amount=stolen_amount,
                victim_count=victim_count,
                evidence_links=evidence_links
            )
        
        # Create report
        report = AddressReport(
            report_id=hashlib.sha256(
                f"{address_lower}{now.isoformat()}{source.value}".encode()
            ).hexdigest()[:16],
            source=source,
            report_date=now,
            description=description,
            evidence_links=evidence_links or [],
            victim_count=victim_count,
            amount_stolen_usd=stolen_amount
        )
        
        # Create address entry
        mal_address = MaliciousAddress(
            address=address_lower,
            chain=chain,
            threat_level=threat_level or ThreatLevel.UNKNOWN,
            categories=categories,
            first_seen=now,
            last_updated=now,
            reports=[report],
            total_stolen_usd=stolen_amount,
            known_victims=victim_count,
            entity_name=entity_name,
            labels=labels or []
        )
        
        # Calculate threat level if not provided
        if not threat_level:
            mal_address.threat_level = mal_address.calculate_threat_level()
        
        # Store in RAM
        self._addresses[address_lower] = mal_address
        
        # Update indexes
        for category in categories:
            self._category_index[category].add(address_lower)
        self._chain_index[chain].add(address_lower)
        self._threat_index[mal_address.threat_level].add(address_lower)
        
        # Persist to SQLite if not RAM-only
        if not self.ram_only:
            self._persist_address(mal_address)
        
        return mal_address
    
    def update_address(
        self,
        address: str,
        categories: Optional[List[AddressCategory]] = None,
        source: Optional[DataSource] = None,
        description: Optional[str] = None,
        stolen_amount: float = 0.0,
        victim_count: int = 0,
        evidence_links: Optional[List[str]] = None,
        active: Optional[bool] = None
    ) -> Optional[MaliciousAddress]:
        """Update an existing address entry."""
        address_lower = address.lower()
        
        if address_lower not in self._addresses:
            return None
        
        mal_address = self._addresses[address_lower]
        now = datetime.now()
        
        # Add new report if source and description provided
        if source and description:
            report = AddressReport(
                report_id=hashlib.sha256(
                    f"{address_lower}{now.isoformat()}{source.value}".encode()
                ).hexdigest()[:16],
                source=source,
                report_date=now,
                description=description,
                evidence_links=evidence_links or [],
                victim_count=victim_count,
                amount_stolen_usd=stolen_amount
            )
            mal_address.reports.append(report)
        
        # Update categories
        if categories:
            for cat in categories:
                if cat not in mal_address.categories:
                    mal_address.categories.append(cat)
                    self._category_index[cat].add(address_lower)
        
        # Update metrics
        mal_address.total_stolen_usd += stolen_amount
        mal_address.known_victims += victim_count
        mal_address.last_updated = now
        
        if active is not None:
            mal_address.active = active
        
        # Recalculate threat level
        old_threat = mal_address.threat_level
        mal_address.threat_level = mal_address.calculate_threat_level()
        
        # Update threat index if changed
        if old_threat != mal_address.threat_level:
            self._threat_index[old_threat].discard(address_lower)
            self._threat_index[mal_address.threat_level].add(address_lower)
        
        return mal_address
    
    def lookup_address(self, address: str) -> Optional[MaliciousAddress]:
        """
        Look up an address in the database.
        
        Args:
            address: Address to look up
            
        Returns:
            MaliciousAddress if found, None otherwise
        """
        return self._addresses.get(address.lower())
    
    def check_address(self, address: str) -> Dict[str, Any]:
        """
        Quick check if an address is known malicious.
        
        Args:
            address: Address to check
            
        Returns:
            Dictionary with check results
        """
        address_lower = address.lower()
        mal_address = self._addresses.get(address_lower)
        
        if not mal_address:
            return {
                "address": address,
                "known_malicious": False,
                "threat_level": ThreatLevel.UNKNOWN.value,
                "message": "Address not found in malicious database"
            }
        
        return {
            "address": address,
            "known_malicious": True,
            "threat_level": mal_address.threat_level.value,
            "categories": [cat.value for cat in mal_address.categories],
            "total_stolen_usd": mal_address.total_stolen_usd,
            "known_victims": mal_address.known_victims,
            "report_count": len(mal_address.reports),
            "entity_name": mal_address.entity_name,
            "active": mal_address.active,
            "first_seen": mal_address.first_seen.isoformat(),
            "last_updated": mal_address.last_updated.isoformat(),
            "confidence_score": mal_address.confidence_score
        }
    
    def batch_check(self, addresses: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Check multiple addresses at once.
        
        Args:
            addresses: List of addresses to check
            
        Returns:
            Dictionary mapping addresses to check results
        """
        return {addr: self.check_address(addr) for addr in addresses}
    
    def search_by_category(
        self,
        category: AddressCategory,
        chain: Optional[Chain] = None,
        min_threat_level: ThreatLevel = ThreatLevel.LOW,
        active_only: bool = False,
        limit: int = 100
    ) -> List[MaliciousAddress]:
        """
        Search addresses by category.
        
        Args:
            category: Category to search for
            chain: Optional chain filter
            min_threat_level: Minimum threat level
            active_only: Only return active addresses
            limit: Maximum results
            
        Returns:
            List of matching addresses
        """
        results = []
        threat_order = [
            ThreatLevel.CRITICAL,
            ThreatLevel.HIGH,
            ThreatLevel.MEDIUM,
            ThreatLevel.LOW,
            ThreatLevel.WATCH
        ]
        min_index = threat_order.index(min_threat_level)
        valid_threats = set(threat_order[:min_index + 1])
        
        for address_key in self._category_index[category]:
            if len(results) >= limit:
                break
            
            mal_address = self._addresses[address_key]
            
            # Apply filters
            if chain and mal_address.chain != chain:
                continue
            if mal_address.threat_level not in valid_threats:
                continue
            if active_only and not mal_address.active:
                continue
            
            results.append(mal_address)
        
        # Sort by threat level
        results.sort(
            key=lambda x: threat_order.index(x.threat_level)
        )
        
        return results
    
    def search_by_entity(self, entity_name: str) -> List[MaliciousAddress]:
        """Search addresses by entity name."""
        entity_lower = entity_name.lower()
        results = []
        
        for mal_address in self._addresses.values():
            if mal_address.entity_name and entity_lower in mal_address.entity_name.lower():
                results.append(mal_address)
        
        return results
    
    def get_related_addresses(self, address: str) -> List[MaliciousAddress]:
        """Get addresses related to the given address."""
        mal_address = self.lookup_address(address)
        if not mal_address:
            return []
        
        related = []
        
        # Check direct relations
        for related_addr in mal_address.related_addresses:
            related_entry = self._addresses.get(related_addr.lower())
            if related_entry:
                related.append(related_entry)
        
        # Check cluster
        if mal_address.cluster_id and mal_address.cluster_id in self._clusters:
            cluster = self._clusters[mal_address.cluster_id]
            for cluster_addr in cluster.addresses:
                if cluster_addr.lower() != address.lower():
                    cluster_entry = self._addresses.get(cluster_addr.lower())
                    if cluster_entry and cluster_entry not in related:
                        related.append(cluster_entry)
        
        return related
    
    def create_cluster(
        self,
        addresses: List[str],
        entity_name: Optional[str] = None,
        categories: Optional[List[AddressCategory]] = None,
        notes: str = ""
    ) -> AddressCluster:
        """
        Create a cluster of related addresses.
        
        Args:
            addresses: List of addresses in cluster
            entity_name: Name of the entity
            categories: Categories for the cluster
            notes: Additional notes
            
        Returns:
            Created cluster
        """
        now = datetime.now()
        cluster_id = hashlib.sha256(
            f"cluster_{','.join(sorted(addresses))}_{now.isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Calculate cluster metrics
        total_stolen = 0.0
        all_categories = set(categories or [])
        highest_threat = ThreatLevel.WATCH
        
        threat_order = [
            ThreatLevel.WATCH,
            ThreatLevel.LOW,
            ThreatLevel.MEDIUM,
            ThreatLevel.HIGH,
            ThreatLevel.CRITICAL
        ]
        
        for addr in addresses:
            mal_address = self._addresses.get(addr.lower())
            if mal_address:
                total_stolen += mal_address.total_stolen_usd
                all_categories.update(mal_address.categories)
                
                if threat_order.index(mal_address.threat_level) > threat_order.index(highest_threat):
                    highest_threat = mal_address.threat_level
                
                # Update address with cluster ID
                mal_address.cluster_id = cluster_id
        
        cluster = AddressCluster(
            cluster_id=cluster_id,
            addresses=addresses,
            entity_name=entity_name,
            categories=list(all_categories),
            total_stolen_usd=total_stolen,
            threat_level=highest_threat,
            first_seen=now,
            last_updated=now,
            notes=notes
        )
        
        self._clusters[cluster_id] = cluster
        
        return cluster
    
    def analyze_security(self, address: str, chain: Chain) -> SecurityProfile:
        """
        Analyze security profile of an address.
        
        This analyzes publicly available on-chain data to determine
        the security characteristics of a wallet/contract.
        
        Args:
            address: Address to analyze
            chain: Blockchain network
            
        Returns:
            Security profile analysis
        """
        now = datetime.now()
        address_lower = address.lower()
        
        profile = SecurityProfile(
            address=address_lower,
            chain=chain,
            analyzed_at=now
        )
        
        # Basic detection based on address format
        if address_lower.startswith("0x"):
            # Ethereum-style address
            if len(address_lower) == 42:
                # Could be EOA or contract - would need on-chain check
                profile.wallet_type = "eoa_or_contract"
        
        # Check if we have this address in our database
        mal_address = self._addresses.get(address_lower)
        if mal_address:
            # Use known information
            profile.risk_score = 100 - mal_address.confidence_score
            profile.risk_factors.extend([
                f"Known malicious: {cat.value}"
                for cat in mal_address.categories
            ])
            
            if AddressCategory.MIXER in mal_address.categories:
                profile.uses_mixer = True
        
        # Calculate overall risk score
        if profile.risk_factors:
            profile.risk_score = min(100, len(profile.risk_factors) * 15)
        
        # Store profile
        self._security_profiles[address_lower] = profile
        
        return profile
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            "total_addresses": len(self._addresses),
            "total_clusters": len(self._clusters),
            "by_chain": {},
            "by_category": {},
            "by_threat_level": {},
            "total_stolen_usd": 0.0,
            "total_victims": 0,
            "active_threats": 0
        }
        
        for chain, addresses in self._chain_index.items():
            stats["by_chain"][chain.value] = len(addresses)
        
        for category, addresses in self._category_index.items():
            if addresses:
                stats["by_category"][category.value] = len(addresses)
        
        for threat, addresses in self._threat_index.items():
            stats["by_threat_level"][threat.value] = len(addresses)
        
        for mal_address in self._addresses.values():
            stats["total_stolen_usd"] += mal_address.total_stolen_usd
            stats["total_victims"] += mal_address.known_victims
            if mal_address.active:
                stats["active_threats"] += 1
        
        return stats
    
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export database to JSON format.
        
        Args:
            filepath: Optional file path to write to
            
        Returns:
            JSON string of database
        """
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "addresses": [],
            "clusters": []
        }
        
        for mal_address in self._addresses.values():
            addr_data = {
                "address": mal_address.address,
                "chain": mal_address.chain.value,
                "threat_level": mal_address.threat_level.value,
                "categories": [cat.value for cat in mal_address.categories],
                "first_seen": mal_address.first_seen.isoformat(),
                "last_updated": mal_address.last_updated.isoformat(),
                "total_stolen_usd": mal_address.total_stolen_usd,
                "known_victims": mal_address.known_victims,
                "entity_name": mal_address.entity_name,
                "cluster_id": mal_address.cluster_id,
                "active": mal_address.active,
                "confidence_score": mal_address.confidence_score,
                "labels": mal_address.labels,
                "report_count": len(mal_address.reports)
            }
            export_data["addresses"].append(addr_data)
        
        for cluster in self._clusters.values():
            cluster_data = {
                "cluster_id": cluster.cluster_id,
                "addresses": cluster.addresses,
                "entity_name": cluster.entity_name,
                "categories": [cat.value for cat in cluster.categories],
                "total_stolen_usd": cluster.total_stolen_usd,
                "threat_level": cluster.threat_level.value
            }
            export_data["clusters"].append(cluster_data)
        
        json_str = json.dumps(export_data, indent=2)
        
        if filepath and not self.ram_only:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def import_from_json(self, json_data: str) -> Dict[str, int]:
        """
        Import addresses from JSON data.
        
        Args:
            json_data: JSON string with address data
            
        Returns:
            Import statistics
        """
        data = json.loads(json_data)
        stats = {"imported": 0, "updated": 0, "errors": 0}
        
        for addr_data in data.get("addresses", []):
            try:
                chain = Chain(addr_data["chain"])
                categories = [AddressCategory(cat) for cat in addr_data["categories"]]
                threat_level = ThreatLevel(addr_data["threat_level"])
                
                if addr_data["address"].lower() in self._addresses:
                    self.update_address(
                        addr_data["address"],
                        categories=categories
                    )
                    stats["updated"] += 1
                else:
                    self.add_address(
                        address=addr_data["address"],
                        chain=chain,
                        categories=categories,
                        source=DataSource.MANUAL_ENTRY,
                        description="Imported from JSON",
                        threat_level=threat_level,
                        stolen_amount=addr_data.get("total_stolen_usd", 0),
                        victim_count=addr_data.get("known_victims", 0),
                        entity_name=addr_data.get("entity_name"),
                        labels=addr_data.get("labels", [])
                    )
                    stats["imported"] += 1
            except Exception as e:
                stats["errors"] += 1
        
        return stats
    
    def _persist_address(self, mal_address: MaliciousAddress) -> None:
        """Persist address to SQLite database."""
        if self.ram_only:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO addresses
            (address, chain, threat_level, categories, first_seen, last_updated,
             total_stolen_usd, known_victims, entity_name, cluster_id, active,
             confidence_score, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mal_address.address,
            mal_address.chain.value,
            mal_address.threat_level.value,
            json.dumps([cat.value for cat in mal_address.categories]),
            mal_address.first_seen.isoformat(),
            mal_address.last_updated.isoformat(),
            mal_address.total_stolen_usd,
            mal_address.known_victims,
            mal_address.entity_name,
            mal_address.cluster_id,
            1 if mal_address.active else 0,
            mal_address.confidence_score,
            mal_address.notes
        ))
        
        conn.commit()
        conn.close()
    
    def clear_database(self) -> None:
        """Clear all data from the database (RAM and disk)."""
        self._addresses.clear()
        self._clusters.clear()
        self._security_profiles.clear()
        
        for index in self._category_index.values():
            index.clear()
        for index in self._chain_index.values():
            index.clear()
        for index in self._threat_index.values():
            index.clear()
        
        if not self.ram_only and self.db_path and self.db_path != ":memory:":
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM addresses")
            cursor.execute("DELETE FROM reports")
            cursor.execute("DELETE FROM clusters")
            conn.commit()
            conn.close()


class AddressIntelligenceAggregator:
    """
    Aggregates address intelligence from multiple public sources.
    
    IMPORTANT: All data sources are PUBLIC. No unauthorized access.
    """
    
    def __init__(self, database: MaliciousAddressDatabase):
        self.database = database
        self._source_status: Dict[DataSource, bool] = {
            source: False for source in DataSource
        }
    
    def aggregate_from_ofac(self, ofac_data: List[Dict]) -> Dict[str, int]:
        """
        Import OFAC SDN list cryptocurrency addresses.
        
        Args:
            ofac_data: Parsed OFAC SDN data
            
        Returns:
            Import statistics
        """
        stats = {"added": 0, "updated": 0}
        
        for entry in ofac_data:
            if "digital_currency_addresses" in entry:
                for addr_info in entry["digital_currency_addresses"]:
                    address = addr_info.get("address", "")
                    currency = addr_info.get("currency", "").upper()
                    
                    # Map currency to chain
                    chain_map = {
                        "ETH": Chain.ETHEREUM,
                        "BTC": Chain.BITCOIN,
                        "BSC": Chain.BSC,
                        "XRP": Chain.TRON,  # Placeholder
                    }
                    chain = chain_map.get(currency, Chain.ETHEREUM)
                    
                    if address:
                        existing = self.database.lookup_address(address)
                        if existing:
                            self.database.update_address(
                                address,
                                categories=[AddressCategory.SANCTIONS],
                                source=DataSource.OFAC_SDN,
                                description=f"OFAC SDN List: {entry.get('name', 'Unknown')}"
                            )
                            stats["updated"] += 1
                        else:
                            self.database.add_address(
                                address=address,
                                chain=chain,
                                categories=[AddressCategory.SANCTIONS],
                                source=DataSource.OFAC_SDN,
                                description=f"OFAC SDN List: {entry.get('name', 'Unknown')}",
                                threat_level=ThreatLevel.CRITICAL,
                                entity_name=entry.get("name")
                            )
                            stats["added"] += 1
        
        self._source_status[DataSource.OFAC_SDN] = True
        return stats
    
    def aggregate_from_community_reports(
        self,
        reports: List[Dict]
    ) -> Dict[str, int]:
        """
        Import community-submitted scam reports.
        
        Args:
            reports: List of community reports
            
        Returns:
            Import statistics
        """
        stats = {"added": 0, "updated": 0, "skipped": 0}
        
        for report in reports:
            address = report.get("address", "")
            if not address:
                stats["skipped"] += 1
                continue
            
            # Validate report has minimum required info
            if not report.get("description"):
                stats["skipped"] += 1
                continue
            
            # Map category
            category_map = {
                "scam": AddressCategory.SCAM,
                "phishing": AddressCategory.PHISHING,
                "rugpull": AddressCategory.RUG_PULL,
                "rug pull": AddressCategory.RUG_PULL,
                "honeypot": AddressCategory.HONEYPOT,
                "fake": AddressCategory.FAKE_ICO,
                "ponzi": AddressCategory.PONZI,
            }
            
            report_type = report.get("type", "scam").lower()
            category = category_map.get(report_type, AddressCategory.SCAM)
            
            # Map chain
            chain_str = report.get("chain", "ethereum").lower()
            chain_map = {
                "ethereum": Chain.ETHEREUM,
                "eth": Chain.ETHEREUM,
                "bitcoin": Chain.BITCOIN,
                "btc": Chain.BITCOIN,
                "bsc": Chain.BSC,
                "binance": Chain.BSC,
                "polygon": Chain.POLYGON,
                "matic": Chain.POLYGON,
            }
            chain = chain_map.get(chain_str, Chain.ETHEREUM)
            
            existing = self.database.lookup_address(address)
            if existing:
                self.database.update_address(
                    address,
                    categories=[category],
                    source=DataSource.COMMUNITY_REPORTS,
                    description=report.get("description", "Community report"),
                    stolen_amount=report.get("amount_lost", 0),
                    victim_count=1,
                    evidence_links=report.get("evidence", [])
                )
                stats["updated"] += 1
            else:
                self.database.add_address(
                    address=address,
                    chain=chain,
                    categories=[category],
                    source=DataSource.COMMUNITY_REPORTS,
                    description=report.get("description", "Community report"),
                    stolen_amount=report.get("amount_lost", 0),
                    victim_count=1,
                    evidence_links=report.get("evidence", [])
                )
                stats["added"] += 1
        
        self._source_status[DataSource.COMMUNITY_REPORTS] = True
        return stats
    
    def get_source_status(self) -> Dict[str, bool]:
        """Get status of data sources."""
        return {source.value: status for source, status in self._source_status.items()}


# Convenience functions
def create_database(ram_only: bool = True) -> MaliciousAddressDatabase:
    """Create a new malicious address database."""
    return MaliciousAddressDatabase(ram_only=ram_only)


def quick_check(address: str, database: MaliciousAddressDatabase) -> Dict[str, Any]:
    """Quick check if an address is known malicious."""
    return database.check_address(address)


# Export all public classes and functions
__all__ = [
    'ThreatLevel',
    'AddressCategory',
    'DataSource',
    'Chain',
    'AddressReport',
    'MaliciousAddress',
    'AddressCluster',
    'SecurityProfile',
    'MaliciousAddressDatabase',
    'AddressIntelligenceAggregator',
    'create_database',
    'quick_check',
]
