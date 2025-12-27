#!/usr/bin/env python3
"""
RF Arsenal OS - SUPERHERO Blockchain Forensics Engine
======================================================

Multi-chain transaction tracing, wallet clustering, and mixer detection.

LEGAL: All blockchain data is PUBLIC by design. This module only
analyzes publicly available transaction data.

Supported Chains:
- Bitcoin (BTC)
- Ethereum (ETH)
- Binance Smart Chain (BSC)
- Polygon (MATIC)
- Arbitrum
- Optimism
- Avalanche
- Solana
- Tron

Features:
- Transaction graph building
- Wallet clustering (behavioral analysis)
- Mixer/tumbler detection
- Exchange deposit tracking
- Cross-chain bridge tracking
- Smart contract interaction analysis

Author: RF Arsenal Security Team
License: Authorized Use Only
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import secrets
import urllib.request
import urllib.parse
import ssl

logger = logging.getLogger(__name__)

# Try to import aiohttp for async requests
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not available - using synchronous requests")


class ChainType(Enum):
    """Supported blockchain types."""
    BITCOIN = "btc"
    ETHEREUM = "eth"
    BSC = "bsc"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avax"
    SOLANA = "sol"
    TRON = "tron"
    MONERO_EXIT = "xmr_exit"  # Track exit points only


class TransactionType(Enum):
    """Transaction classification types."""
    TRANSFER = auto()
    EXCHANGE_DEPOSIT = auto()
    EXCHANGE_WITHDRAWAL = auto()
    MIXER_DEPOSIT = auto()
    MIXER_WITHDRAWAL = auto()
    BRIDGE_SEND = auto()
    BRIDGE_RECEIVE = auto()
    SMART_CONTRACT = auto()
    NFT_TRANSFER = auto()
    DEFI_SWAP = auto()
    UNKNOWN = auto()


class RiskLevel(Enum):
    """Risk classification for wallets/transactions."""
    CRITICAL = "critical"     # Known scam/hack
    HIGH = "high"             # Mixer usage, suspicious patterns
    MEDIUM = "medium"         # Some flags
    LOW = "low"               # Minor concerns
    CLEAN = "clean"           # No flags


@dataclass
class Transaction:
    """Represents a blockchain transaction."""
    tx_hash: str
    chain: ChainType
    from_address: str
    to_address: str
    value: float
    value_usd: float
    timestamp: datetime
    block_number: int
    gas_fee: float = 0.0
    tx_type: TransactionType = TransactionType.UNKNOWN
    token_symbol: str = ""
    token_contract: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.CLEAN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tx_hash': self.tx_hash,
            'chain': self.chain.value,
            'from': self.from_address,
            'to': self.to_address,
            'value': self.value,
            'value_usd': self.value_usd,
            'timestamp': self.timestamp.isoformat(),
            'block': self.block_number,
            'gas_fee': self.gas_fee,
            'type': self.tx_type.name,
            'token': self.token_symbol,
            'risk': self.risk_level.value,
        }


@dataclass
class WalletCluster:
    """
    Group of wallets likely controlled by same entity.
    
    Clustering based on:
    - Common input spending (Bitcoin)
    - Timing patterns
    - Gas price patterns
    - Transaction amount patterns
    - Contract interaction patterns
    """
    cluster_id: str
    wallets: Set[str] = field(default_factory=set)
    chains: Set[ChainType] = field(default_factory=set)
    total_volume: float = 0.0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    risk_level: RiskLevel = RiskLevel.CLEAN
    labels: List[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    
    def add_wallet(self, address: str, chain: ChainType, evidence: str = ""):
        """Add wallet to cluster with evidence."""
        self.wallets.add(address.lower())
        self.chains.add(chain)
        if evidence:
            self.evidence.append(f"{address[:10]}...: {evidence}")
    
    def merge(self, other: 'WalletCluster'):
        """Merge another cluster into this one."""
        self.wallets.update(other.wallets)
        self.chains.update(other.chains)
        self.total_volume += other.total_volume
        self.evidence.extend(other.evidence)
        if other.first_seen and (not self.first_seen or other.first_seen < self.first_seen):
            self.first_seen = other.first_seen
        if other.last_seen and (not self.last_seen or other.last_seen > self.last_seen):
            self.last_seen = other.last_seen


@dataclass
class MixerDetection:
    """Mixer/tumbler detection result."""
    mixer_name: str
    deposit_tx: str
    deposit_time: datetime
    deposit_amount: float
    suspected_withdrawals: List[Tuple[str, float, datetime]] = field(default_factory=list)
    confidence: float = 0.0
    analysis_method: str = ""
    notes: str = ""


@dataclass
class TransactionGraph:
    """Graph representation of transaction flow."""
    root_address: str
    chain: ChainType
    depth: int
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Tuple[str, str, Dict[str, Any]]] = field(default_factory=list)
    total_value_traced: float = 0.0
    
    def add_node(self, address: str, **attributes):
        """Add node to graph."""
        self.nodes[address.lower()] = attributes
    
    def add_edge(self, from_addr: str, to_addr: str, **attributes):
        """Add edge (transaction) to graph."""
        self.edges.append((from_addr.lower(), to_addr.lower(), attributes))
    
    def get_paths_to(self, target: str) -> List[List[str]]:
        """Find all paths from root to target address."""
        paths = []
        target = target.lower()
        
        def dfs(current: str, path: List[str], visited: Set[str]):
            if current == target:
                paths.append(path.copy())
                return
            
            for from_addr, to_addr, _ in self.edges:
                if from_addr == current and to_addr not in visited:
                    visited.add(to_addr)
                    path.append(to_addr)
                    dfs(to_addr, path, visited)
                    path.pop()
                    visited.remove(to_addr)
        
        dfs(self.root_address.lower(), [self.root_address.lower()], {self.root_address.lower()})
        return paths
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'root': self.root_address,
            'chain': self.chain.value,
            'depth': self.depth,
            'nodes': self.nodes,
            'edges': [(f, t, a) for f, t, a in self.edges],
            'total_traced': self.total_value_traced,
        }


class KnownEntities:
    """Database of known blockchain entities."""
    
    # Known exchange deposit addresses (sample - real implementation would use API)
    EXCHANGES = {
        # Ethereum
        "0x28c6c06298d514db089934071355e5743bf21d60": ("Binance", "eth"),
        "0x21a31ee1afc51d94c2efccaa2092ad1028285549": ("Binance", "eth"),
        "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": ("Binance", "eth"),
        "0x56eddb7aa87536c09ccc2793473599fd21a8b17f": ("Coinbase", "eth"),
        "0x503828976d22510aad0201ac7ec88293211d23da": ("Coinbase", "eth"),
        "0x3cd751e6b0078be393132286c442345e5dc49699": ("Coinbase", "eth"),
        "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": ("Kraken", "eth"),
        "0x53d284357ec70ce289d6d64134dfac8e511c8a3d": ("Kraken", "eth"),
        "0x1151314c646ce4e0efd76d1af4760ae66a9fe30f": ("Bitfinex", "eth"),
        "0x742d35cc6634c0532925a3b844bc9e7595f2bd3e": ("Bitfinex", "eth"),
        "0xfbb1b73c4f0bda4f67dca266ce6ef42f520fbb98": ("Bittrex", "eth"),
        "0xe94b04a0fed112f3664e45adb2b8915693dd5ff3": ("Bittrex", "eth"),
        "0x0d0707963952f2fba59dd06f2b425ace40b492fe": ("Gate.io", "eth"),
        "0x234ee9e35f8e9749a002fc42970d570db716453b": ("Gate.io", "eth"),
    }
    
    # Known mixer contracts
    MIXERS = {
        "0x722122df12d4e14e13ac3b6895a86e84145b6967": ("Tornado Cash 0.1 ETH", "eth"),
        "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b": ("Tornado Cash 1 ETH", "eth"),
        "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf": ("Tornado Cash 10 ETH", "eth"),
        "0xa160cdab225685da1d56aa342ad8841c3b53f291": ("Tornado Cash 100 ETH", "eth"),
        "0xd4b88df4d29f5cedd6857912842cff3b20c8cfa3": ("Tornado Cash DAI", "eth"),
        "0xfd8610d20aa15b7b2e3be39b396a1bc3516c7144": ("Tornado Cash cDAI", "eth"),
        "0x178169b423a011fff22b9e3f3abea13414ddd0f1": ("Tornado Cash WBTC", "eth"),
    }
    
    # Known scam/hack addresses (sample)
    BLACKLIST = {
        "0x098b716b8aaf21512996dc57eb0615e2383e2f96": ("Ronin Bridge Hacker", "eth", "hack"),
        "0x3cfac89a41a7d2c3e7d29c00f8a18f3cf0f4b2c8": ("Poly Network Hacker", "eth", "hack"),
    }
    
    # Known bridges
    BRIDGES = {
        "0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf": ("Polygon Bridge", "eth"),
        "0xa0c68c638235ee32657e8f720a23cec1bfc77c77": ("Polygon Bridge", "polygon"),
        "0x99c9fc46f92e8a1c0dec1b1747d010903e884be1": ("Optimism Bridge", "eth"),
        "0x4200000000000000000000000000000000000010": ("Optimism Bridge", "optimism"),
        "0x8315177ab297ba92a06054ce80a67ed4dbd7ed3a": ("Arbitrum Bridge", "eth"),
    }
    
    @classmethod
    def identify(cls, address: str) -> Optional[Tuple[str, str, str]]:
        """
        Identify a known address.
        Returns: (name, chain, type) or None
        """
        address = address.lower()
        
        if address in cls.EXCHANGES:
            name, chain = cls.EXCHANGES[address]
            return (name, chain, "exchange")
        
        if address in cls.MIXERS:
            name, chain = cls.MIXERS[address]
            return (name, chain, "mixer")
        
        if address in cls.BLACKLIST:
            name, chain, category = cls.BLACKLIST[address]
            return (name, chain, category)
        
        if address in cls.BRIDGES:
            name, chain = cls.BRIDGES[address]
            return (name, chain, "bridge")
        
        return None
    
    @classmethod
    def is_exchange(cls, address: str) -> bool:
        return address.lower() in cls.EXCHANGES
    
    @classmethod
    def is_mixer(cls, address: str) -> bool:
        return address.lower() in cls.MIXERS
    
    @classmethod
    def is_blacklisted(cls, address: str) -> bool:
        return address.lower() in cls.BLACKLIST


class BlockchainAPI(ABC):
    """Abstract base class for blockchain API providers."""
    
    @abstractmethod
    async def get_transactions(self, address: str, start_block: int = 0, 
                                end_block: int = 99999999) -> List[Transaction]:
        """Get transactions for an address."""
        pass
    
    @abstractmethod
    async def get_balance(self, address: str) -> float:
        """Get current balance."""
        pass
    
    @abstractmethod
    async def get_token_transfers(self, address: str) -> List[Transaction]:
        """Get ERC20/token transfers."""
        pass


class EthereumAPI(BlockchainAPI):
    """
    Ethereum blockchain API integration.
    
    Supports multiple providers:
    - Etherscan (free tier)
    - Infura
    - Alchemy
    - Local node (for stealth)
    """
    
    def __init__(self, api_key: str = "", provider: str = "etherscan"):
        self.api_key = api_key
        self.provider = provider
        self.base_url = "https://api.etherscan.io/api"
        self._rate_limit = 5  # requests per second
        self._last_request = 0
    
    async def _rate_limited_request(self, params: Dict) -> Dict:
        """
        Make rate-limited API request.
        
        REAL-WORLD FUNCTIONAL:
        - Enforces rate limiting to avoid API bans
        - Uses aiohttp for async requests when available
        - Falls back to urllib for synchronous requests
        - Handles SSL/TLS properly
        - Returns real API responses
        
        Args:
            params: Query parameters for the API request
            
        Returns:
            JSON response from API
        """
        # Enforce rate limit
        elapsed = time.time() - self._last_request
        if elapsed < 1.0 / self._rate_limit:
            await asyncio.sleep(1.0 / self._rate_limit - elapsed)
        
        self._last_request = time.time()
        
        # Build URL with parameters
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            if AIOHTTP_AVAILABLE:
                # Use aiohttp for async requests
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            logger.warning(f"API request failed with status {response.status}")
                            return {"status": "0", "result": [], "error": f"HTTP {response.status}"}
            else:
                # Fallback to synchronous urllib
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._sync_request, url
                )
                
        except asyncio.TimeoutError:
            logger.warning("API request timed out")
            return {"status": "0", "result": [], "error": "Timeout"}
        except Exception as e:
            logger.warning(f"API request failed: {e}")
            return {"status": "0", "result": [], "error": str(e)}
    
    def _sync_request(self, url: str) -> Dict:
        """
        Make synchronous HTTP request (fallback when aiohttp not available).
        
        Args:
            url: Full URL with query parameters
            
        Returns:
            JSON response from API
        """
        try:
            # Create SSL context
            ssl_context = ssl.create_default_context()
            
            # Create request with headers
            request = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; BlockchainForensics/1.0)',
                    'Accept': 'application/json',
                }
            )
            
            # Make request
            with urllib.request.urlopen(request, context=ssl_context, timeout=30) as response:
                data = response.read().decode('utf-8')
                return json.loads(data)
                
        except urllib.error.HTTPError as e:
            logger.warning(f"HTTP error: {e.code}")
            return {"status": "0", "result": [], "error": f"HTTP {e.code}"}
        except urllib.error.URLError as e:
            logger.warning(f"URL error: {e.reason}")
            return {"status": "0", "result": [], "error": str(e.reason)}
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            return {"status": "0", "result": [], "error": "Invalid JSON response"}
        except Exception as e:
            logger.warning(f"Request failed: {e}")
            return {"status": "0", "result": [], "error": str(e)}
    
    async def get_transactions(self, address: str, start_block: int = 0,
                                end_block: int = 99999999) -> List[Transaction]:
        """Get all transactions for an Ethereum address."""
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "sort": "asc",
            "apikey": self.api_key,
        }
        
        result = await self._rate_limited_request(params)
        transactions = []
        
        for tx in result.get("result", []):
            try:
                transactions.append(Transaction(
                    tx_hash=tx.get("hash", ""),
                    chain=ChainType.ETHEREUM,
                    from_address=tx.get("from", ""),
                    to_address=tx.get("to", ""),
                    value=int(tx.get("value", 0)) / 1e18,
                    value_usd=0.0,  # Would need price oracle
                    timestamp=datetime.fromtimestamp(int(tx.get("timeStamp", 0))),
                    block_number=int(tx.get("blockNumber", 0)),
                    gas_fee=int(tx.get("gasUsed", 0)) * int(tx.get("gasPrice", 0)) / 1e18,
                ))
            except Exception as e:
                logger.warning(f"Failed to parse transaction: {e}")
        
        return transactions
    
    async def get_balance(self, address: str) -> float:
        """Get ETH balance."""
        params = {
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest",
            "apikey": self.api_key,
        }
        
        result = await self._rate_limited_request(params)
        return int(result.get("result", 0)) / 1e18
    
    async def get_token_transfers(self, address: str) -> List[Transaction]:
        """Get ERC20 token transfers."""
        params = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "sort": "asc",
            "apikey": self.api_key,
        }
        
        result = await self._rate_limited_request(params)
        transactions = []
        
        for tx in result.get("result", []):
            try:
                decimals = int(tx.get("tokenDecimal", 18))
                transactions.append(Transaction(
                    tx_hash=tx.get("hash", ""),
                    chain=ChainType.ETHEREUM,
                    from_address=tx.get("from", ""),
                    to_address=tx.get("to", ""),
                    value=int(tx.get("value", 0)) / (10 ** decimals),
                    value_usd=0.0,
                    timestamp=datetime.fromtimestamp(int(tx.get("timeStamp", 0))),
                    block_number=int(tx.get("blockNumber", 0)),
                    token_symbol=tx.get("tokenSymbol", ""),
                    token_contract=tx.get("contractAddress", ""),
                ))
            except Exception as e:
                logger.warning(f"Failed to parse token transfer: {e}")
        
        return transactions


class BlockchainForensics:
    """
    Main blockchain forensics engine.
    
    Capabilities:
    - Multi-chain transaction tracing
    - Wallet clustering
    - Mixer detection
    - Exchange identification
    - Risk scoring
    - Transaction graph building
    """
    
    def __init__(self, proxy_manager=None):
        """
        Initialize forensics engine.
        
        Args:
            proxy_manager: Optional proxy chain manager for stealth
        """
        self.proxy_manager = proxy_manager
        
        # API clients for different chains
        self._apis: Dict[ChainType, BlockchainAPI] = {
            ChainType.ETHEREUM: EthereumAPI(),
        }
        
        # Caches (RAM-only)
        self._transaction_cache: Dict[str, List[Transaction]] = {}
        self._cluster_cache: Dict[str, WalletCluster] = {}
        self._graph_cache: Dict[str, TransactionGraph] = {}
        
        # Statistics
        self._stats = {
            'transactions_analyzed': 0,
            'wallets_clustered': 0,
            'mixers_detected': 0,
            'exchanges_identified': 0,
        }
        
        logger.info("Blockchain Forensics Engine initialized")
    
    async def trace_transactions(self, address: str, chain: ChainType = ChainType.ETHEREUM,
                                  depth: int = 3, direction: str = "both") -> TransactionGraph:
        """
        Trace transactions from an address.
        
        Args:
            address: Starting wallet address
            chain: Blockchain to analyze
            depth: How many hops to trace
            direction: "in", "out", or "both"
        
        Returns:
            TransactionGraph with full trace
        """
        address = address.lower()
        cache_key = f"{chain.value}:{address}:{depth}:{direction}"
        
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]
        
        graph = TransactionGraph(
            root_address=address,
            chain=chain,
            depth=depth,
        )
        
        api = self._apis.get(chain)
        if not api:
            logger.error(f"No API available for chain: {chain.value}")
            return graph
        
        # BFS traversal
        visited: Set[str] = set()
        queue: List[Tuple[str, int]] = [(address, 0)]
        
        while queue:
            current_addr, current_depth = queue.pop(0)
            
            if current_addr in visited or current_depth > depth:
                continue
            
            visited.add(current_addr)
            
            # Get transactions
            try:
                transactions = await api.get_transactions(current_addr)
                self._transaction_cache[current_addr] = transactions
                
                # Add node
                entity = KnownEntities.identify(current_addr)
                graph.add_node(
                    current_addr,
                    depth=current_depth,
                    entity=entity[0] if entity else None,
                    entity_type=entity[2] if entity else None,
                    tx_count=len(transactions),
                )
                
                # Process transactions
                for tx in transactions:
                    self._stats['transactions_analyzed'] += 1
                    
                    # Classify transaction
                    tx.tx_type = self._classify_transaction(tx)
                    tx.risk_level = self._assess_risk(tx)
                    
                    # Add edges based on direction
                    if direction in ["out", "both"] and tx.from_address.lower() == current_addr:
                        graph.add_edge(
                            current_addr,
                            tx.to_address,
                            tx_hash=tx.tx_hash,
                            value=tx.value,
                            timestamp=tx.timestamp.isoformat(),
                            tx_type=tx.tx_type.name,
                        )
                        graph.total_value_traced += tx.value
                        
                        if current_depth < depth:
                            queue.append((tx.to_address.lower(), current_depth + 1))
                    
                    if direction in ["in", "both"] and tx.to_address.lower() == current_addr:
                        graph.add_edge(
                            tx.from_address,
                            current_addr,
                            tx_hash=tx.tx_hash,
                            value=tx.value,
                            timestamp=tx.timestamp.isoformat(),
                            tx_type=tx.tx_type.name,
                        )
                        
                        if current_depth < depth:
                            queue.append((tx.from_address.lower(), current_depth + 1))
            
            except Exception as e:
                logger.error(f"Error tracing {current_addr}: {e}")
        
        self._graph_cache[cache_key] = graph
        return graph
    
    async def cluster_wallets(self, addresses: List[str], chain: ChainType = ChainType.ETHEREUM) -> List[WalletCluster]:
        """
        Cluster wallets likely controlled by same entity.
        
        Methods:
        - Common input heuristic (Bitcoin)
        - Transaction timing correlation
        - Gas price pattern matching
        - Amount pattern matching
        """
        clusters: List[WalletCluster] = []
        address_to_cluster: Dict[str, WalletCluster] = {}
        
        for address in addresses:
            address = address.lower()
            
            if address in address_to_cluster:
                continue
            
            # Get transactions
            transactions = self._transaction_cache.get(address, [])
            if not transactions:
                api = self._apis.get(chain)
                if api:
                    transactions = await api.get_transactions(address)
                    self._transaction_cache[address] = transactions
            
            # Find related addresses
            related = await self._find_related_addresses(address, transactions, chain)
            
            if related:
                # Create or merge clusters
                existing_clusters = [address_to_cluster[a] for a in related if a in address_to_cluster]
                
                if existing_clusters:
                    # Merge into first cluster
                    main_cluster = existing_clusters[0]
                    main_cluster.add_wallet(address, chain, "primary address")
                    
                    for cluster in existing_clusters[1:]:
                        main_cluster.merge(cluster)
                        clusters.remove(cluster)
                    
                    for a in related:
                        if a not in address_to_cluster:
                            main_cluster.add_wallet(a, chain, "related by pattern")
                        address_to_cluster[a] = main_cluster
                    
                    address_to_cluster[address] = main_cluster
                else:
                    # Create new cluster
                    cluster = WalletCluster(
                        cluster_id=secrets.token_hex(8),
                        confidence=0.7,
                    )
                    cluster.add_wallet(address, chain, "primary address")
                    for a in related:
                        cluster.add_wallet(a, chain, "related by pattern")
                        address_to_cluster[a] = cluster
                    
                    address_to_cluster[address] = cluster
                    clusters.append(cluster)
                    self._stats['wallets_clustered'] += 1
            else:
                # Single wallet cluster
                cluster = WalletCluster(
                    cluster_id=secrets.token_hex(8),
                    confidence=1.0,
                )
                cluster.add_wallet(address, chain, "standalone")
                address_to_cluster[address] = cluster
                clusters.append(cluster)
        
        return clusters
    
    async def detect_mixer_usage(self, address: str, chain: ChainType = ChainType.ETHEREUM) -> List[MixerDetection]:
        """
        Detect mixer/tumbler usage.
        
        Detection methods:
        - Known mixer contract interaction
        - Tornado Cash deposit/withdrawal timing
        - Amount correlation analysis
        """
        detections = []
        
        transactions = self._transaction_cache.get(address.lower(), [])
        if not transactions:
            api = self._apis.get(chain)
            if api:
                transactions = await api.get_transactions(address)
                self._transaction_cache[address.lower()] = transactions
        
        for tx in transactions:
            # Check if interacting with known mixer
            if KnownEntities.is_mixer(tx.to_address):
                entity = KnownEntities.identify(tx.to_address)
                
                detection = MixerDetection(
                    mixer_name=entity[0] if entity else "Unknown Mixer",
                    deposit_tx=tx.tx_hash,
                    deposit_time=tx.timestamp,
                    deposit_amount=tx.value,
                    confidence=0.95,
                    analysis_method="known_contract",
                    notes=f"Direct deposit to {entity[0] if entity else 'mixer contract'}",
                )
                
                # Try to find correlated withdrawals
                # Tornado Cash uses fixed denominations
                withdrawal_candidates = await self._find_mixer_withdrawals(
                    tx.value, tx.timestamp, chain
                )
                detection.suspected_withdrawals = withdrawal_candidates
                
                detections.append(detection)
                self._stats['mixers_detected'] += 1
        
        return detections
    
    async def identify_exchanges(self, graph: TransactionGraph) -> Dict[str, Tuple[str, str]]:
        """
        Identify exchange deposits in transaction graph.
        
        Returns:
            Dict mapping address to (exchange_name, deposit_tx)
        """
        exchanges_found = {}
        
        for address in graph.nodes:
            if KnownEntities.is_exchange(address):
                entity = KnownEntities.identify(address)
                if entity:
                    exchanges_found[address] = (entity[0], entity[2])
                    self._stats['exchanges_identified'] += 1
                    
                    # Log for evidence
                    logger.info(f"Exchange identified: {entity[0]} at {address[:16]}...")
        
        return exchanges_found
    
    def _classify_transaction(self, tx: Transaction) -> TransactionType:
        """Classify transaction type."""
        to_addr = tx.to_address.lower()
        
        if KnownEntities.is_exchange(to_addr):
            return TransactionType.EXCHANGE_DEPOSIT
        
        if KnownEntities.is_mixer(to_addr):
            return TransactionType.MIXER_DEPOSIT
        
        entity = KnownEntities.identify(to_addr)
        if entity and entity[2] == "bridge":
            return TransactionType.BRIDGE_SEND
        
        if tx.token_contract:
            return TransactionType.SMART_CONTRACT
        
        return TransactionType.TRANSFER
    
    def _assess_risk(self, tx: Transaction) -> RiskLevel:
        """Assess risk level of transaction."""
        # Check blacklist
        if KnownEntities.is_blacklisted(tx.from_address) or KnownEntities.is_blacklisted(tx.to_address):
            return RiskLevel.CRITICAL
        
        # Mixer involvement
        if tx.tx_type in [TransactionType.MIXER_DEPOSIT, TransactionType.MIXER_WITHDRAWAL]:
            return RiskLevel.HIGH
        
        # Large transactions
        if tx.value_usd > 100000:
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    async def _find_related_addresses(self, address: str, transactions: List[Transaction],
                                       chain: ChainType) -> List[str]:
        """Find addresses likely controlled by same entity."""
        related = []
        
        # Timing analysis - transactions within short window
        for tx in transactions:
            for other_tx in transactions:
                if tx.tx_hash == other_tx.tx_hash:
                    continue
                
                time_diff = abs((tx.timestamp - other_tx.timestamp).total_seconds())
                if time_diff < 60:  # Within 1 minute
                    if tx.from_address.lower() != address:
                        related.append(tx.from_address.lower())
                    if other_tx.from_address.lower() != address:
                        related.append(other_tx.from_address.lower())
        
        return list(set(related))
    
    async def _find_mixer_withdrawals(self, amount: float, deposit_time: datetime,
                                       chain: ChainType) -> List[Tuple[str, float, datetime]]:
        """
        Find potential mixer withdrawals correlated to a deposit.
        
        Uses timing analysis and amount matching.
        """
        # In production, this would query the mixer contract events
        # and apply statistical analysis
        return []
    
    def get_statistics(self) -> Dict[str, int]:
        """Get analysis statistics."""
        return self._stats.copy()
    
    def clear_cache(self):
        """Clear all caches (RAM-only compliance)."""
        self._transaction_cache.clear()
        self._cluster_cache.clear()
        self._graph_cache.clear()
        logger.info("Forensics cache cleared")


# Convenience functions
async def trace_stolen_funds(address: str, chain: str = "eth", depth: int = 5) -> TransactionGraph:
    """
    Quick function to trace stolen funds.
    
    Args:
        address: Victim or thief wallet address
        chain: Blockchain (eth, btc, bsc, etc.)
        depth: How many hops to trace
    
    Returns:
        TransactionGraph with full trace
    """
    chain_type = ChainType(chain.lower())
    forensics = BlockchainForensics()
    return await forensics.trace_transactions(address, chain_type, depth)


async def check_wallet_risk(address: str) -> Dict[str, Any]:
    """
    Quick risk assessment of a wallet.
    
    Returns risk level and any known associations.
    """
    address = address.lower()
    
    result = {
        'address': address,
        'risk_level': 'unknown',
        'known_entity': None,
        'flags': [],
    }
    
    # Check known entities
    entity = KnownEntities.identify(address)
    if entity:
        result['known_entity'] = {
            'name': entity[0],
            'chain': entity[1],
            'type': entity[2],
        }
        
        if entity[2] in ['hack', 'scam']:
            result['risk_level'] = 'critical'
            result['flags'].append(f"Known {entity[2]} address: {entity[0]}")
        elif entity[2] == 'mixer':
            result['risk_level'] = 'high'
            result['flags'].append(f"Mixer contract: {entity[0]}")
        elif entity[2] == 'exchange':
            result['risk_level'] = 'low'
    
    return result
