#!/usr/bin/env python3
"""
RF Arsenal OS - SUPERHERO Identity Correlation Engine
======================================================

OSINT-based identity attribution from blockchain activity.

LEGAL NOTICE:
- All data from PUBLIC sources only
- ENS/domain lookups are public
- Social media profiles are public
- Leaked databases are publicly available
- No unauthorized access to any systems

Identity Sources:
- ENS (Ethereum Name Service)
- Unstoppable Domains
- NFT ownership (OpenSea profiles)
- Social media (Twitter, Discord, Telegram)
- Forum analysis (BitcoinTalk, Reddit)
- GitHub/code repositories
- Public breach databases
- Domain WHOIS records

Author: RF Arsenal Security Team
License: Authorized Use Only
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import secrets

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for identity leads."""
    CONFIRMED = "confirmed"       # 95%+ - Multiple corroborating sources
    HIGH = "high"                 # 80-95% - Strong evidence
    MEDIUM = "medium"             # 50-80% - Moderate evidence
    LOW = "low"                   # 25-50% - Weak evidence
    SPECULATIVE = "speculative"  # <25% - Needs verification
    
    @classmethod
    def from_score(cls, score: float) -> 'ConfidenceLevel':
        if score >= 0.95:
            return cls.CONFIRMED
        elif score >= 0.80:
            return cls.HIGH
        elif score >= 0.50:
            return cls.MEDIUM
        elif score >= 0.25:
            return cls.LOW
        return cls.SPECULATIVE


class LeadSource(Enum):
    """Sources for identity leads."""
    ENS = "ens"
    UNSTOPPABLE_DOMAINS = "unstoppable"
    NFT_PROFILE = "nft"
    OPENSEA = "opensea"
    TWITTER = "twitter"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    GITHUB = "github"
    REDDIT = "reddit"
    BITCOINTALK = "bitcointalk"
    DARKWEB_FORUM = "darkweb"
    BREACH_DATABASE = "breach"
    WHOIS = "whois"
    SSL_CERTIFICATE = "ssl"
    BLOCKCHAIN_MESSAGE = "message"
    DEFI_GOVERNANCE = "governance"
    EXCHANGE_LEAK = "exchange"
    EMAIL_CORRELATION = "email"
    PHONE_CORRELATION = "phone"
    IP_CORRELATION = "ip"


@dataclass
class IdentityLead:
    """Single piece of identity evidence."""
    lead_id: str
    source: LeadSource
    data_type: str  # email, name, phone, username, etc.
    value: str
    confidence: float
    timestamp: datetime
    wallet_address: str
    chain: str
    evidence_url: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'lead_id': self.lead_id,
            'source': self.source.value,
            'type': self.data_type,
            'value': self.value,
            'confidence': self.confidence,
            'confidence_level': ConfidenceLevel.from_score(self.confidence).value,
            'timestamp': self.timestamp.isoformat(),
            'wallet': self.wallet_address,
            'chain': self.chain,
            'evidence_url': self.evidence_url,
            'notes': self.notes,
            'verified': self.verified,
        }


@dataclass
class PersonProfile:
    """Aggregated identity profile."""
    profile_id: str
    
    # Core identity
    real_names: List[Tuple[str, float]] = field(default_factory=list)  # (name, confidence)
    usernames: List[Tuple[str, str, float]] = field(default_factory=list)  # (username, platform, confidence)
    emails: List[Tuple[str, float]] = field(default_factory=list)
    phones: List[Tuple[str, float]] = field(default_factory=list)
    
    # Wallets
    wallets: Dict[str, List[str]] = field(default_factory=dict)  # chain -> [addresses]
    
    # Online presence
    social_profiles: Dict[str, str] = field(default_factory=dict)  # platform -> url
    domains: List[str] = field(default_factory=list)
    
    # Physical
    locations: List[Tuple[str, float]] = field(default_factory=list)  # (location, confidence)
    timezones: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    
    # Photos
    photos: List[str] = field(default_factory=list)  # URLs
    
    # Evidence
    leads: List[IdentityLead] = field(default_factory=list)
    
    # Metadata
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    overall_confidence: float = 0.0
    
    def get_best_name(self) -> Optional[str]:
        """Get highest confidence real name."""
        if self.real_names:
            return max(self.real_names, key=lambda x: x[1])[0]
        return None
    
    def get_best_email(self) -> Optional[str]:
        """Get highest confidence email."""
        if self.emails:
            return max(self.emails, key=lambda x: x[1])[0]
        return None
    
    def add_lead(self, lead: IdentityLead):
        """Add lead and update profile."""
        self.leads.append(lead)
        
        # Update appropriate field based on data type
        if lead.data_type == "name":
            existing = [n for n, c in self.real_names if n.lower() == lead.value.lower()]
            if not existing:
                self.real_names.append((lead.value, lead.confidence))
            else:
                # Update confidence if higher
                self.real_names = [
                    (n, max(c, lead.confidence) if n.lower() == lead.value.lower() else c)
                    for n, c in self.real_names
                ]
        
        elif lead.data_type == "email":
            existing = [e for e, c in self.emails if e.lower() == lead.value.lower()]
            if not existing:
                self.emails.append((lead.value.lower(), lead.confidence))
        
        elif lead.data_type == "phone":
            existing = [p for p, c in self.phones if p == lead.value]
            if not existing:
                self.phones.append((lead.value, lead.confidence))
        
        elif lead.data_type == "username":
            platform = lead.raw_data.get("platform", lead.source.value)
            self.usernames.append((lead.value, platform, lead.confidence))
        
        elif lead.data_type == "location":
            self.locations.append((lead.value, lead.confidence))
        
        elif lead.data_type == "photo":
            if lead.evidence_url not in self.photos:
                self.photos.append(lead.evidence_url)
        
        # Update timestamps
        if not self.first_seen or lead.timestamp < self.first_seen:
            self.first_seen = lead.timestamp
        if not self.last_seen or lead.timestamp > self.last_seen:
            self.last_seen = lead.timestamp
        
        # Recalculate overall confidence
        self._calculate_confidence()
    
    def _calculate_confidence(self):
        """Calculate overall profile confidence."""
        if not self.leads:
            self.overall_confidence = 0.0
            return
        
        # Weight by number of corroborating sources
        source_types = set(lead.source for lead in self.leads)
        data_types_found = set(lead.data_type for lead in self.leads)
        
        base_confidence = sum(lead.confidence for lead in self.leads) / len(self.leads)
        
        # Bonus for multiple source types
        source_bonus = min(0.2, len(source_types) * 0.05)
        
        # Bonus for complete profile
        completeness_bonus = min(0.15, len(data_types_found) * 0.03)
        
        self.overall_confidence = min(1.0, base_confidence + source_bonus + completeness_bonus)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'profile_id': self.profile_id,
            'real_names': self.real_names,
            'usernames': self.usernames,
            'emails': self.emails,
            'phones': self.phones,
            'wallets': self.wallets,
            'social_profiles': self.social_profiles,
            'domains': self.domains,
            'locations': self.locations,
            'timezones': self.timezones,
            'languages': self.languages,
            'photos': self.photos,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'overall_confidence': self.overall_confidence,
            'confidence_level': ConfidenceLevel.from_score(self.overall_confidence).value,
            'leads_count': len(self.leads),
        }


class ENSResolver:
    """
    Ethereum Name Service resolution.
    
    ENS names are PUBLIC and often link wallets to identities.
    """
    
    def __init__(self):
        self._cache: Dict[str, Optional[str]] = {}
    
    async def resolve_address(self, address: str) -> Optional[str]:
        """Get ENS name for address (reverse resolution)."""
        address = address.lower()
        
        if address in self._cache:
            return self._cache[address]
        
        # In production, query ENS contract or API
        # ENS reverse records are PUBLIC blockchain data
        
        # Placeholder - real implementation would use web3.py
        self._cache[address] = None
        return None
    
    async def resolve_name(self, name: str) -> Optional[str]:
        """Get address for ENS name (forward resolution)."""
        name = name.lower()
        
        # In production, query ENS contract
        return None
    
    async def get_text_records(self, name: str) -> Dict[str, str]:
        """
        Get ENS text records.
        
        Common records that leak identity:
        - email
        - url
        - avatar
        - description
        - com.twitter
        - com.github
        - com.discord
        """
        # ENS text records are PUBLIC
        return {}


class SocialMediaAnalyzer:
    """
    Social media OSINT analysis.
    
    All data gathered from PUBLIC profiles only.
    """
    
    # Patterns to find wallet addresses in profiles
    WALLET_PATTERNS = [
        r'0x[a-fA-F0-9]{40}',  # Ethereum
        r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}',  # Bitcoin
        r'bc1[a-zA-HJ-NP-Z0-9]{39,59}',  # Bitcoin Bech32
    ]
    
    # Crypto-related keywords
    CRYPTO_KEYWORDS = [
        'eth', 'btc', 'crypto', 'nft', 'defi', 'web3',
        'solana', 'polygon', 'ethereum', 'bitcoin',
    ]
    
    async def search_twitter(self, query: str) -> List[Dict[str, Any]]:
        """
        Search Twitter for wallet mentions.
        
        PUBLIC tweets only - uses Twitter search.
        """
        results = []
        
        # In production, use Twitter API v2 (public search)
        # or scrape public tweets
        
        return results
    
    async def analyze_twitter_profile(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Analyze public Twitter profile.
        
        Extracts:
        - Bio (may contain wallet)
        - Name
        - Location
        - Website
        - Profile picture
        """
        # PUBLIC profile data only
        return None
    
    async def search_discord(self, wallet: str) -> List[Dict[str, Any]]:
        """
        Search public Discord servers for wallet mentions.
        
        Many crypto communities are PUBLIC.
        """
        return []
    
    async def search_telegram(self, wallet: str) -> List[Dict[str, Any]]:
        """
        Search public Telegram groups for wallet mentions.
        """
        return []
    
    async def search_github(self, wallet: str) -> List[Dict[str, Any]]:
        """
        Search GitHub for wallet in code/issues.
        
        GitHub is PUBLIC by default.
        """
        return []


class ForumAnalyzer:
    """
    Crypto forum OSINT analysis.
    
    Forums analyzed (PUBLIC posts only):
    - BitcoinTalk
    - Reddit (r/cryptocurrency, r/bitcoin, etc.)
    - Dark web forums (public sections)
    """
    
    async def search_bitcointalk(self, query: str) -> List[Dict[str, Any]]:
        """
        Search BitcoinTalk forum.
        
        BitcoinTalk profiles often include:
        - Bitcoin/ETH addresses for tips
        - PGP keys
        - Activity history
        """
        return []
    
    async def search_reddit(self, query: str, subreddits: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search Reddit for wallet mentions.
        
        Default subreddits:
        - r/cryptocurrency
        - r/bitcoin
        - r/ethereum
        - r/CryptoMarkets
        """
        if subreddits is None:
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'CryptoMarkets']
        
        return []
    
    async def analyze_reddit_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Analyze public Reddit user profile."""
        return None


class BreachDatabase:
    """
    Public breach database correlation.
    
    LEGAL: Uses publicly available breach compilations.
    Does NOT access private databases illegally.
    
    Sources:
    - Have I Been Pwned (API)
    - Public breach compilations
    - Leaked combo lists (publicly posted)
    """
    
    async def search_email(self, email: str) -> List[Dict[str, Any]]:
        """
        Check if email appears in public breaches.
        
        Uses Have I Been Pwned API (PUBLIC).
        """
        results = []
        
        # In production, query HIBP API
        # This is a PUBLIC service
        
        return results
    
    async def correlate_email_to_wallet(self, email: str) -> List[str]:
        """
        Try to find wallet addresses associated with email.
        
        Methods:
        - Exchange breach data (leaked KYC)
        - Forum registrations
        - Newsletter signups
        """
        return []
    
    async def search_username(self, username: str) -> List[Dict[str, Any]]:
        """Search for username in breach data."""
        return []


class NFTAnalyzer:
    """
    NFT ownership and profile analysis.
    
    All NFT data is PUBLIC on the blockchain.
    OpenSea profiles are PUBLIC.
    """
    
    async def get_nft_holdings(self, address: str) -> List[Dict[str, Any]]:
        """Get NFT holdings for address."""
        return []
    
    async def get_opensea_profile(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Get OpenSea profile for address.
        
        Profile may include:
        - Username
        - Bio
        - Twitter link
        - Website
        - Profile picture
        """
        return None
    
    async def analyze_nft_metadata(self, nfts: List[Dict]) -> List[Dict[str, Any]]:
        """
        Analyze NFT metadata for identity clues.
        
        Some NFTs contain creator identity in metadata.
        """
        return []


class IdentityEngine:
    """
    Main identity correlation engine.
    
    Aggregates OSINT from multiple sources to build identity profiles.
    """
    
    def __init__(self, proxy_manager=None):
        """
        Initialize identity engine.
        
        Args:
            proxy_manager: Optional proxy chain for stealth
        """
        self.proxy_manager = proxy_manager
        
        # Sub-analyzers
        self.ens = ENSResolver()
        self.social = SocialMediaAnalyzer()
        self.forums = ForumAnalyzer()
        self.breaches = BreachDatabase()
        self.nft = NFTAnalyzer()
        
        # Profile storage (RAM-only)
        self._profiles: Dict[str, PersonProfile] = {}
        self._wallet_to_profile: Dict[str, str] = {}
        
        # Statistics
        self._stats = {
            'wallets_analyzed': 0,
            'leads_found': 0,
            'profiles_created': 0,
            'identities_confirmed': 0,
        }
        
        logger.info("Identity Engine initialized")
    
    async def analyze_wallet(self, address: str, chain: str = "eth", 
                             deep_scan: bool = True) -> PersonProfile:
        """
        Full identity analysis for a wallet address.
        
        Args:
            address: Wallet address to analyze
            chain: Blockchain
            deep_scan: Whether to do extensive OSINT
        
        Returns:
            PersonProfile with all discovered identity leads
        """
        address = address.lower()
        self._stats['wallets_analyzed'] += 1
        
        # Check if already analyzed
        if address in self._wallet_to_profile:
            return self._profiles[self._wallet_to_profile[address]]
        
        # Create new profile
        profile = PersonProfile(
            profile_id=secrets.token_hex(8),
            wallets={chain: [address]},
        )
        
        # Run all analyzers
        leads = []
        
        # 1. ENS Resolution
        ens_leads = await self._analyze_ens(address, chain)
        leads.extend(ens_leads)
        
        # 2. NFT/OpenSea Profile
        nft_leads = await self._analyze_nft_profile(address, chain)
        leads.extend(nft_leads)
        
        if deep_scan:
            # 3. Social Media Search
            social_leads = await self._analyze_social_media(address, chain)
            leads.extend(social_leads)
            
            # 4. Forum Search
            forum_leads = await self._analyze_forums(address, chain)
            leads.extend(forum_leads)
            
            # 5. Breach Correlation (if we have emails)
            for lead in leads:
                if lead.data_type == "email":
                    breach_leads = await self._correlate_breaches(lead.value, address, chain)
                    leads.extend(breach_leads)
        
        # Add all leads to profile
        for lead in leads:
            profile.add_lead(lead)
            self._stats['leads_found'] += 1
        
        # Store profile
        self._profiles[profile.profile_id] = profile
        self._wallet_to_profile[address] = profile.profile_id
        self._stats['profiles_created'] += 1
        
        if profile.overall_confidence >= 0.80:
            self._stats['identities_confirmed'] += 1
        
        return profile
    
    async def _analyze_ens(self, address: str, chain: str) -> List[IdentityLead]:
        """Analyze ENS records for identity clues."""
        leads = []
        
        # Get ENS name
        ens_name = await self.ens.resolve_address(address)
        if ens_name:
            leads.append(IdentityLead(
                lead_id=secrets.token_hex(8),
                source=LeadSource.ENS,
                data_type="username",
                value=ens_name,
                confidence=0.90,
                timestamp=datetime.now(),
                wallet_address=address,
                chain=chain,
                evidence_url=f"https://app.ens.domains/name/{ens_name}",
                notes=f"ENS name owned by wallet",
                raw_data={"platform": "ens"},
            ))
            
            # Get text records
            records = await self.ens.get_text_records(ens_name)
            
            if records.get("email"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.ENS,
                    data_type="email",
                    value=records["email"],
                    confidence=0.85,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    notes="Email from ENS text record",
                ))
            
            if records.get("com.twitter"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.ENS,
                    data_type="username",
                    value=records["com.twitter"],
                    confidence=0.85,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    evidence_url=f"https://twitter.com/{records['com.twitter']}",
                    raw_data={"platform": "twitter"},
                ))
            
            if records.get("com.github"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.ENS,
                    data_type="username",
                    value=records["com.github"],
                    confidence=0.85,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    evidence_url=f"https://github.com/{records['com.github']}",
                    raw_data={"platform": "github"},
                ))
            
            if records.get("url"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.ENS,
                    data_type="website",
                    value=records["url"],
                    confidence=0.80,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    evidence_url=records["url"],
                ))
        
        return leads
    
    async def _analyze_nft_profile(self, address: str, chain: str) -> List[IdentityLead]:
        """Analyze NFT holdings and OpenSea profile."""
        leads = []
        
        # Get OpenSea profile
        profile = await self.nft.get_opensea_profile(address)
        if profile:
            if profile.get("username"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.OPENSEA,
                    data_type="username",
                    value=profile["username"],
                    confidence=0.75,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    evidence_url=f"https://opensea.io/{profile['username']}",
                    raw_data={"platform": "opensea"},
                ))
            
            if profile.get("twitter_username"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.OPENSEA,
                    data_type="username",
                    value=profile["twitter_username"],
                    confidence=0.80,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    evidence_url=f"https://twitter.com/{profile['twitter_username']}",
                    raw_data={"platform": "twitter"},
                    notes="Twitter linked on OpenSea profile",
                ))
            
            if profile.get("profile_img_url"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.OPENSEA,
                    data_type="photo",
                    value="profile_photo",
                    confidence=0.60,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    evidence_url=profile["profile_img_url"],
                    notes="Profile photo from OpenSea",
                ))
        
        return leads
    
    async def _analyze_social_media(self, address: str, chain: str) -> List[IdentityLead]:
        """Search social media for wallet mentions."""
        leads = []
        
        # Twitter search
        twitter_results = await self.social.search_twitter(address)
        for result in twitter_results:
            if result.get("username"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.TWITTER,
                    data_type="username",
                    value=result["username"],
                    confidence=0.70,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    evidence_url=result.get("url", ""),
                    raw_data={"platform": "twitter", "tweet": result.get("text", "")},
                    notes="Posted wallet address on Twitter",
                ))
        
        # GitHub search
        github_results = await self.social.search_github(address)
        for result in github_results:
            if result.get("username"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.GITHUB,
                    data_type="username",
                    value=result["username"],
                    confidence=0.80,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    evidence_url=result.get("url", ""),
                    raw_data={"platform": "github"},
                    notes="Wallet found in GitHub repo/issue",
                ))
        
        return leads
    
    async def _analyze_forums(self, address: str, chain: str) -> List[IdentityLead]:
        """Search forums for wallet mentions."""
        leads = []
        
        # BitcoinTalk
        btc_results = await self.forums.search_bitcointalk(address)
        for result in btc_results:
            if result.get("username"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.BITCOINTALK,
                    data_type="username",
                    value=result["username"],
                    confidence=0.75,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    evidence_url=result.get("profile_url", ""),
                    raw_data={"platform": "bitcointalk"},
                    notes="Wallet in BitcoinTalk signature/post",
                ))
        
        # Reddit
        reddit_results = await self.forums.search_reddit(address)
        for result in reddit_results:
            if result.get("author"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.REDDIT,
                    data_type="username",
                    value=result["author"],
                    confidence=0.65,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    evidence_url=result.get("url", ""),
                    raw_data={"platform": "reddit", "subreddit": result.get("subreddit", "")},
                    notes="Posted wallet on Reddit",
                ))
        
        return leads
    
    async def _correlate_breaches(self, email: str, address: str, chain: str) -> List[IdentityLead]:
        """Correlate email with breach databases."""
        leads = []
        
        # Check HIBP
        breaches = await self.breaches.search_email(email)
        for breach in breaches:
            leads.append(IdentityLead(
                lead_id=secrets.token_hex(8),
                source=LeadSource.BREACH_DATABASE,
                data_type="breach",
                value=breach.get("name", "Unknown"),
                confidence=0.50,  # Breach data is less reliable
                timestamp=datetime.now(),
                wallet_address=address,
                chain=chain,
                raw_data=breach,
                notes=f"Email found in {breach.get('name', 'unknown')} breach",
            ))
            
            # If breach contains additional data
            if breach.get("real_name"):
                leads.append(IdentityLead(
                    lead_id=secrets.token_hex(8),
                    source=LeadSource.BREACH_DATABASE,
                    data_type="name",
                    value=breach["real_name"],
                    confidence=0.60,
                    timestamp=datetime.now(),
                    wallet_address=address,
                    chain=chain,
                    notes=f"Name from {breach.get('name', 'unknown')} breach",
                ))
        
        return leads
    
    async def cross_reference_profiles(self, profiles: List[PersonProfile]) -> List[PersonProfile]:
        """
        Cross-reference profiles to find connections.
        
        Merges profiles that appear to be the same person.
        """
        merged = []
        
        # Find profiles with matching identifiers
        email_to_profiles: Dict[str, List[PersonProfile]] = defaultdict(list)
        username_to_profiles: Dict[str, List[PersonProfile]] = defaultdict(list)
        
        for profile in profiles:
            for email, _ in profile.emails:
                email_to_profiles[email.lower()].append(profile)
            
            for username, platform, _ in profile.usernames:
                key = f"{platform}:{username.lower()}"
                username_to_profiles[key].append(profile)
        
        # Merge profiles with shared identifiers
        processed = set()
        
        for profile in profiles:
            if profile.profile_id in processed:
                continue
            
            # Find all related profiles
            related = {profile}
            
            for email, _ in profile.emails:
                related.update(email_to_profiles[email.lower()])
            
            for username, platform, _ in profile.usernames:
                key = f"{platform}:{username.lower()}"
                related.update(username_to_profiles[key])
            
            # Merge all related into one
            if len(related) > 1:
                merged_profile = PersonProfile(profile_id=secrets.token_hex(8))
                
                for p in related:
                    for lead in p.leads:
                        merged_profile.add_lead(lead)
                    
                    for chain, addrs in p.wallets.items():
                        if chain not in merged_profile.wallets:
                            merged_profile.wallets[chain] = []
                        merged_profile.wallets[chain].extend(addrs)
                    
                    processed.add(p.profile_id)
                
                merged.append(merged_profile)
            else:
                merged.append(profile)
                processed.add(profile.profile_id)
        
        return merged
    
    def get_profile(self, profile_id: str) -> Optional[PersonProfile]:
        """Get profile by ID."""
        return self._profiles.get(profile_id)
    
    def get_profile_for_wallet(self, address: str) -> Optional[PersonProfile]:
        """Get profile for wallet address."""
        address = address.lower()
        if address in self._wallet_to_profile:
            return self._profiles.get(self._wallet_to_profile[address])
        return None
    
    def get_statistics(self) -> Dict[str, int]:
        """Get engine statistics."""
        return self._stats.copy()
    
    def clear_cache(self):
        """Clear all cached data (RAM-only compliance)."""
        self._profiles.clear()
        self._wallet_to_profile.clear()
        self.ens._cache.clear()
        logger.info("Identity Engine cache cleared")


# Convenience functions
async def identify_wallet_owner(address: str, chain: str = "eth") -> Dict[str, Any]:
    """
    Quick identity lookup for a wallet.
    
    Returns summary of discovered identity.
    """
    engine = IdentityEngine()
    profile = await engine.analyze_wallet(address, chain)
    
    return {
        'address': address,
        'best_name': profile.get_best_name(),
        'best_email': profile.get_best_email(),
        'confidence': profile.overall_confidence,
        'confidence_level': ConfidenceLevel.from_score(profile.overall_confidence).value,
        'usernames': [(u, p) for u, p, _ in profile.usernames],
        'social_profiles': profile.social_profiles,
        'leads_count': len(profile.leads),
    }


# Alias for compatibility
IdentityCorrelationEngine = IdentityEngine
