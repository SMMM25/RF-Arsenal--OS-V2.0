#!/usr/bin/env python3
"""
RF Arsenal OS - SUPERHERO Counter-Countermeasure Systems
=========================================================

Advanced techniques to trace funds through obfuscation methods.

LEGAL NOTICE:
- All analysis from PUBLIC blockchain data
- Statistical inference and pattern matching
- No unauthorized access to any systems

Capabilities:
- Mixer/tumbler partial tracing (Tornado Cash timing analysis)
- Chain-hopping tracker (bridge transaction correlation)
- Privacy coin exit detection (Monero → Exchange)
- Multi-wallet persona linking
- Decoy/fake identity detection

Author: RF Arsenal Security Team
License: Authorized Use Only
"""

import asyncio
import hashlib
import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import secrets

logger = logging.getLogger(__name__)


class ObfuscationMethod(Enum):
    """Types of obfuscation criminals use."""
    MIXER = "mixer"
    TUMBLER = "tumbler"
    CHAIN_HOP = "chain_hop"
    PRIVACY_COIN = "privacy_coin"
    COINJOIN = "coinjoin"
    MULTI_WALLET = "multi_wallet"
    TIME_DELAY = "time_delay"
    AMOUNT_SPLIT = "amount_split"
    PEEL_CHAIN = "peel_chain"


@dataclass
class MixerTrace:
    """Result of mixer tracing analysis."""
    mixer_name: str
    deposit_tx: str
    deposit_amount: float
    deposit_time: datetime
    probable_withdrawals: List[Dict[str, Any]]
    confidence: float
    method: str
    evidence: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mixer': self.mixer_name,
            'deposit_tx': self.deposit_tx,
            'deposit_amount': self.deposit_amount,
            'deposit_time': self.deposit_time.isoformat(),
            'withdrawals': self.probable_withdrawals,
            'confidence': self.confidence,
            'method': self.method,
            'evidence': self.evidence,
        }


@dataclass
class ChainHopTrace:
    """Result of cross-chain hop tracing."""
    source_chain: str
    source_tx: str
    source_amount: float
    destination_chain: str
    destination_tx: Optional[str]
    destination_amount: Optional[float]
    bridge_used: str
    time_delta: timedelta
    confidence: float
    evidence: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_chain': self.source_chain,
            'source_tx': self.source_tx,
            'source_amount': self.source_amount,
            'dest_chain': self.destination_chain,
            'dest_tx': self.destination_tx,
            'dest_amount': self.destination_amount,
            'bridge': self.bridge_used,
            'time_delta_seconds': self.time_delta.total_seconds(),
            'confidence': self.confidence,
            'evidence': self.evidence,
        }


@dataclass
class PersonaLink:
    """Link between wallets likely owned by same person."""
    wallet_a: str
    wallet_b: str
    chain: str
    confidence: float
    link_type: str  # timing, amount, gas, behavior, etc.
    evidence: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'wallet_a': self.wallet_a,
            'wallet_b': self.wallet_b,
            'chain': self.chain,
            'confidence': self.confidence,
            'link_type': self.link_type,
            'evidence': self.evidence,
        }


class MixerTracer:
    """
    Trace funds through cryptocurrency mixers.
    
    Techniques:
    1. Timing correlation - deposits and withdrawals within time window
    2. Amount matching - exact amounts or common splits
    3. Deposit pattern analysis - behavioral fingerprinting
    4. Withdrawal clustering - same entity withdrawing multiple times
    """
    
    # Known Tornado Cash denominations (ETH)
    TORNADO_DENOMINATIONS = [0.1, 1.0, 10.0, 100.0]
    
    # Typical mixing delay ranges (seconds)
    MIXING_DELAYS = {
        "tornado_fast": (300, 3600),       # 5 min to 1 hour
        "tornado_normal": (3600, 86400),   # 1 hour to 1 day
        "tornado_slow": (86400, 604800),   # 1 day to 1 week
    }
    
    def __init__(self):
        self._withdrawal_cache: Dict[str, List[Dict]] = {}
        self._stats = {
            'mixers_traced': 0,
            'funds_traced': 0.0,
        }
    
    async def trace_tornado_deposit(self, 
                                     deposit_tx: str,
                                     deposit_amount: float,
                                     deposit_time: datetime,
                                     withdrawal_candidates: List[Dict[str, Any]]) -> MixerTrace:
        """
        Trace a Tornado Cash deposit.
        
        Uses timing and amount correlation to find probable withdrawals.
        
        Args:
            deposit_tx: Deposit transaction hash
            deposit_amount: Amount deposited
            deposit_time: When deposit occurred
            withdrawal_candidates: List of potential withdrawal transactions
        
        Returns:
            MixerTrace with probable withdrawals
        """
        probable_withdrawals = []
        evidence = []
        
        # Filter withdrawals by amount (must match denomination)
        matching_amount = [
            w for w in withdrawal_candidates 
            if abs(w.get('amount', 0) - deposit_amount) < 0.001
        ]
        
        if not matching_amount:
            evidence.append(f"No withdrawals matching {deposit_amount} ETH denomination")
            return MixerTrace(
                mixer_name="Tornado Cash",
                deposit_tx=deposit_tx,
                deposit_amount=deposit_amount,
                deposit_time=deposit_time,
                probable_withdrawals=[],
                confidence=0.0,
                method="amount_matching",
                evidence=evidence,
            )
        
        # Filter by timing (must be after deposit)
        after_deposit = [
            w for w in matching_amount
            if w.get('timestamp') and w['timestamp'] > deposit_time
        ]
        
        # Score each withdrawal by timing
        scored = []
        for w in after_deposit:
            time_delta = (w['timestamp'] - deposit_time).total_seconds()
            
            # Score based on typical delay patterns
            # Most Tornado withdrawals happen 1-24 hours after deposit
            if 3600 <= time_delta <= 86400:
                time_score = 0.8
            elif 300 <= time_delta <= 3600:
                time_score = 0.5  # Suspiciously fast
            elif 86400 <= time_delta <= 604800:
                time_score = 0.6
            else:
                time_score = 0.3
            
            scored.append({
                'withdrawal': w,
                'score': time_score,
                'time_delta': time_delta,
            })
        
        # Sort by score and take top candidates
        scored.sort(key=lambda x: x['score'], reverse=True)
        
        for item in scored[:5]:  # Top 5 candidates
            w = item['withdrawal']
            probable_withdrawals.append({
                'tx_hash': w.get('tx_hash', ''),
                'amount': w.get('amount', 0),
                'timestamp': w.get('timestamp', datetime.now()).isoformat(),
                'to_address': w.get('to_address', ''),
                'probability': item['score'],
                'time_delta_hours': item['time_delta'] / 3600,
            })
        
        # Calculate overall confidence
        if probable_withdrawals:
            best_score = scored[0]['score'] if scored else 0
            confidence = best_score * 0.7  # Mixer tracing is inherently uncertain
            evidence.append(f"Found {len(probable_withdrawals)} probable withdrawals")
            evidence.append(f"Best match: {probable_withdrawals[0]['time_delta_hours']:.1f} hours after deposit")
        else:
            confidence = 0.1
            evidence.append("No strong withdrawal candidates found")
        
        self._stats['mixers_traced'] += 1
        self._stats['funds_traced'] += deposit_amount
        
        return MixerTrace(
            mixer_name="Tornado Cash",
            deposit_tx=deposit_tx,
            deposit_amount=deposit_amount,
            deposit_time=deposit_time,
            probable_withdrawals=probable_withdrawals,
            confidence=confidence,
            method="timing_correlation",
            evidence=evidence,
        )
    
    async def analyze_withdrawal_patterns(self, 
                                           withdrawals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze withdrawal patterns to cluster likely same-entity withdrawals.
        
        If someone withdraws multiple times, they often:
        - Use similar timing patterns
        - Use same gas prices
        - Withdraw to related addresses
        """
        clusters = []
        
        if len(withdrawals) < 2:
            return clusters
        
        # Group by timing pattern
        # Check for withdrawals at similar times of day
        hour_groups = defaultdict(list)
        for w in withdrawals:
            if w.get('timestamp'):
                hour = w['timestamp'].hour
                hour_groups[hour].append(w)
        
        # Suspicious: multiple withdrawals in same hour across days
        for hour, group in hour_groups.items():
            if len(group) >= 2:
                clusters.append({
                    'type': 'timing_pattern',
                    'hour': hour,
                    'withdrawals': group,
                    'confidence': min(0.7, len(group) * 0.2),
                    'evidence': f"{len(group)} withdrawals at hour {hour}",
                })
        
        # Group by gas price similarity
        gas_groups = defaultdict(list)
        for w in withdrawals:
            if w.get('gas_price'):
                # Round to nearest gwei
                gas_bucket = round(w['gas_price'] / 1e9)
                gas_groups[gas_bucket].append(w)
        
        for gas, group in gas_groups.items():
            if len(group) >= 2:
                clusters.append({
                    'type': 'gas_pattern',
                    'gas_gwei': gas,
                    'withdrawals': group,
                    'confidence': min(0.5, len(group) * 0.15),
                    'evidence': f"{len(group)} withdrawals at ~{gas} gwei",
                })
        
        return clusters


class ChainHopTracker:
    """
    Track funds across blockchain bridges.
    
    When criminals bridge from ETH → Polygon → BSC, we can:
    1. Monitor bridge contract events
    2. Correlate amounts and timing
    3. Follow the trail across chains
    """
    
    # Known bridge contracts
    BRIDGES = {
        "polygon_bridge": {
            "eth_contract": "0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf",
            "polygon_contract": "0xa0c68c638235ee32657e8f720a23cec1bfc77c77",
            "typical_delay": (300, 1800),  # 5-30 minutes
        },
        "arbitrum_bridge": {
            "eth_contract": "0x8315177ab297ba92a06054ce80a67ed4dbd7ed3a",
            "typical_delay": (600, 3600),  # 10-60 minutes
        },
        "optimism_bridge": {
            "eth_contract": "0x99c9fc46f92e8a1c0dec1b1747d010903e884be1",
            "typical_delay": (60, 600),  # 1-10 minutes for deposits
        },
        "wormhole": {
            "eth_contract": "0x98f3c9e6E3fAce36bAAD05FE09d375Ef1464288B",
            "typical_delay": (60, 300),
        },
    }
    
    def __init__(self):
        self._stats = {
            'hops_traced': 0,
            'chains_tracked': set(),
        }
    
    async def trace_bridge_transfer(self,
                                     source_tx: str,
                                     source_chain: str,
                                     source_amount: float,
                                     source_time: datetime,
                                     destination_chain: str,
                                     dest_candidates: List[Dict[str, Any]]) -> ChainHopTrace:
        """
        Trace a bridge transfer to destination chain.
        
        Args:
            source_tx: Source chain transaction hash
            source_chain: Source blockchain
            source_amount: Amount bridged
            source_time: When bridge was initiated
            destination_chain: Target blockchain
            dest_candidates: Candidate transactions on destination chain
        
        Returns:
            ChainHopTrace with best match
        """
        evidence = []
        
        # Identify bridge used
        bridge_used = self._identify_bridge(source_chain, destination_chain)
        if not bridge_used:
            bridge_used = "unknown_bridge"
        
        bridge_info = self.BRIDGES.get(bridge_used, {})
        min_delay, max_delay = bridge_info.get("typical_delay", (60, 3600))
        
        # Filter candidates by timing and amount
        matches = []
        for candidate in dest_candidates:
            if not candidate.get('timestamp') or not candidate.get('amount'):
                continue
            
            time_delta = (candidate['timestamp'] - source_time).total_seconds()
            amount_diff = abs(candidate['amount'] - source_amount) / source_amount if source_amount else 1
            
            # Must be after source tx
            if time_delta < 0:
                continue
            
            # Score based on timing and amount match
            timing_score = 0
            if min_delay <= time_delta <= max_delay:
                timing_score = 0.9
            elif time_delta <= max_delay * 2:
                timing_score = 0.6
            else:
                timing_score = 0.3
            
            amount_score = max(0, 1 - amount_diff * 10)  # Allow small fee differences
            
            total_score = timing_score * 0.6 + amount_score * 0.4
            
            if total_score > 0.3:
                matches.append({
                    'candidate': candidate,
                    'score': total_score,
                    'time_delta': timedelta(seconds=time_delta),
                    'amount_diff': amount_diff,
                })
        
        # Best match
        if matches:
            matches.sort(key=lambda x: x['score'], reverse=True)
            best = matches[0]
            
            evidence.append(f"Bridge: {bridge_used}")
            evidence.append(f"Time delta: {best['time_delta']}")
            evidence.append(f"Amount difference: {best['amount_diff']*100:.2f}%")
            
            self._stats['hops_traced'] += 1
            self._stats['chains_tracked'].add(source_chain)
            self._stats['chains_tracked'].add(destination_chain)
            
            return ChainHopTrace(
                source_chain=source_chain,
                source_tx=source_tx,
                source_amount=source_amount,
                destination_chain=destination_chain,
                destination_tx=best['candidate'].get('tx_hash'),
                destination_amount=best['candidate'].get('amount'),
                bridge_used=bridge_used,
                time_delta=best['time_delta'],
                confidence=best['score'],
                evidence=evidence,
            )
        
        evidence.append(f"No matching transactions found on {destination_chain}")
        
        return ChainHopTrace(
            source_chain=source_chain,
            source_tx=source_tx,
            source_amount=source_amount,
            destination_chain=destination_chain,
            destination_tx=None,
            destination_amount=None,
            bridge_used=bridge_used,
            time_delta=timedelta(0),
            confidence=0.1,
            evidence=evidence,
        )
    
    def _identify_bridge(self, source: str, dest: str) -> Optional[str]:
        """Identify which bridge was likely used."""
        # Simple matching - production would check actual contract addresses
        if source == "eth" and dest == "polygon":
            return "polygon_bridge"
        elif source == "eth" and dest == "arbitrum":
            return "arbitrum_bridge"
        elif source == "eth" and dest == "optimism":
            return "optimism_bridge"
        return None


class PrivacyCoinExitDetector:
    """
    Detect when funds exit privacy coins (Monero, Zcash) back to traceable chains.
    
    Monero is untraceable ON-CHAIN, but we can:
    1. Track deposits TO Monero (from exchanges)
    2. Track when they convert BACK to traceable coins
    3. Use timing/amount correlation
    """
    
    def __init__(self):
        self._stats = {
            'exit_points_detected': 0,
        }
    
    async def detect_monero_exit(self,
                                  entry_amount_usd: float,
                                  entry_time: datetime,
                                  exchange_deposits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Try to detect Monero exit points.
        
        Args:
            entry_amount_usd: USD value when entering Monero
            entry_time: When they bought Monero
            exchange_deposits: Deposits to exchanges (potential exit points)
        
        Returns:
            List of potential exit transactions
        """
        candidates = []
        
        for deposit in exchange_deposits:
            if not deposit.get('timestamp') or not deposit.get('amount_usd'):
                continue
            
            time_delta = (deposit['timestamp'] - entry_time).total_seconds()
            
            # Must be after entry
            if time_delta < 0:
                continue
            
            # Typical holding time: few hours to few weeks
            if time_delta > 30 * 86400:  # More than 30 days
                continue
            
            # Amount should be similar (allowing for price changes)
            amount_ratio = deposit['amount_usd'] / entry_amount_usd if entry_amount_usd else 0
            if 0.5 < amount_ratio < 2.0:  # Within 50-200% of original
                score = 0
                
                # Timing score
                if 3600 <= time_delta <= 86400:
                    score += 0.3  # 1-24 hours - quick flip
                elif time_delta <= 7 * 86400:
                    score += 0.5  # Within a week - typical
                else:
                    score += 0.2
                
                # Amount score
                if 0.9 < amount_ratio < 1.1:
                    score += 0.4
                elif 0.7 < amount_ratio < 1.3:
                    score += 0.2
                
                candidates.append({
                    'deposit': deposit,
                    'score': score,
                    'time_delta_hours': time_delta / 3600,
                    'amount_ratio': amount_ratio,
                })
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        self._stats['exit_points_detected'] += len(candidates)
        
        return candidates[:10]  # Top 10 candidates


class WalletPersonaLinker:
    """
    Link multiple wallets to same persona.
    
    Methods:
    - Timing pattern correlation
    - Gas price patterns
    - Amount patterns (same amounts between wallets)
    - Contract interaction patterns
    - NFT transfer patterns
    - Token holding patterns
    """
    
    def __init__(self):
        self._link_cache: Dict[str, List[PersonaLink]] = {}
        self._stats = {
            'wallets_linked': 0,
            'high_confidence_links': 0,
        }
    
    async def find_linked_wallets(self,
                                   wallet: str,
                                   transactions: List[Dict[str, Any]],
                                   all_wallets: List[str],
                                   all_transactions: Dict[str, List[Dict]]) -> List[PersonaLink]:
        """
        Find wallets likely controlled by same person.
        
        Args:
            wallet: Primary wallet to analyze
            transactions: Transactions for primary wallet
            all_wallets: All wallets to check against
            all_transactions: Transactions for all wallets
        
        Returns:
            List of PersonaLinks to related wallets
        """
        links = []
        wallet = wallet.lower()
        
        if not transactions:
            return links
        
        # Extract patterns from primary wallet
        primary_patterns = self._extract_patterns(transactions)
        
        for other in all_wallets:
            other = other.lower()
            if other == wallet:
                continue
            
            other_txs = all_transactions.get(other, [])
            if not other_txs:
                continue
            
            other_patterns = self._extract_patterns(other_txs)
            
            # Compare patterns
            confidence = 0
            evidence = []
            link_types = []
            
            # 1. Timing correlation
            timing_score = self._compare_timing(primary_patterns, other_patterns)
            if timing_score > 0.6:
                confidence += timing_score * 0.3
                evidence.append(f"Similar activity times (score: {timing_score:.2f})")
                link_types.append("timing")
            
            # 2. Gas price correlation
            gas_score = self._compare_gas(primary_patterns, other_patterns)
            if gas_score > 0.6:
                confidence += gas_score * 0.2
                evidence.append(f"Similar gas prices (score: {gas_score:.2f})")
                link_types.append("gas")
            
            # 3. Direct transfers between wallets
            direct_transfers = [
                tx for tx in transactions
                if tx.get('to_address', '').lower() == other or 
                   tx.get('from_address', '').lower() == other
            ]
            if direct_transfers:
                confidence += min(0.3, len(direct_transfers) * 0.1)
                evidence.append(f"{len(direct_transfers)} direct transfers between wallets")
                link_types.append("direct_transfer")
            
            # 4. Funding source (same wallet funded both)
            # Would need to trace back funding
            
            if confidence >= 0.3:
                link = PersonaLink(
                    wallet_a=wallet,
                    wallet_b=other,
                    chain="eth",  # Could be dynamic
                    confidence=min(0.95, confidence),
                    link_type=",".join(link_types),
                    evidence=evidence,
                )
                links.append(link)
                self._stats['wallets_linked'] += 1
                
                if confidence >= 0.7:
                    self._stats['high_confidence_links'] += 1
        
        # Sort by confidence
        links.sort(key=lambda x: x.confidence, reverse=True)
        
        return links
    
    def _extract_patterns(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Extract behavioral patterns from transactions."""
        patterns = {
            'hours': defaultdict(int),
            'days': defaultdict(int),
            'gas_prices': [],
            'amounts': [],
            'contracts': set(),
        }
        
        for tx in transactions:
            if tx.get('timestamp'):
                patterns['hours'][tx['timestamp'].hour] += 1
                patterns['days'][tx['timestamp'].weekday()] += 1
            
            if tx.get('gas_price'):
                patterns['gas_prices'].append(tx['gas_price'])
            
            if tx.get('value'):
                patterns['amounts'].append(tx['value'])
            
            if tx.get('to_address'):
                patterns['contracts'].add(tx['to_address'].lower())
        
        return patterns
    
    def _compare_timing(self, p1: Dict, p2: Dict) -> float:
        """Compare timing patterns."""
        if not p1['hours'] or not p2['hours']:
            return 0
        
        # Cosine similarity of hourly activity
        hours1 = [p1['hours'].get(h, 0) for h in range(24)]
        hours2 = [p2['hours'].get(h, 0) for h in range(24)]
        
        dot_product = sum(a * b for a, b in zip(hours1, hours2))
        norm1 = math.sqrt(sum(a * a for a in hours1))
        norm2 = math.sqrt(sum(a * a for a in hours2))
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def _compare_gas(self, p1: Dict, p2: Dict) -> float:
        """Compare gas price patterns."""
        if not p1['gas_prices'] or not p2['gas_prices']:
            return 0
        
        avg1 = statistics.mean(p1['gas_prices'])
        avg2 = statistics.mean(p2['gas_prices'])
        
        if avg1 == 0 or avg2 == 0:
            return 0
        
        # How similar are average gas prices
        ratio = min(avg1, avg2) / max(avg1, avg2)
        return ratio


class CounterMeasureAnalyzer:
    """
    Main counter-countermeasure analysis engine.
    
    Orchestrates all sub-analyzers to trace funds through obfuscation.
    """
    
    def __init__(self, proxy_manager=None):
        self.proxy_manager = proxy_manager
        
        self.mixer_tracer = MixerTracer()
        self.chain_hopper = ChainHopTracker()
        self.privacy_detector = PrivacyCoinExitDetector()
        self.persona_linker = WalletPersonaLinker()
        
        self._stats = {
            'analyses_performed': 0,
            'obfuscation_detected': 0,
        }
    
    async def analyze_obfuscation(self,
                                   wallet: str,
                                   transactions: List[Dict[str, Any]],
                                   related_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Full obfuscation analysis.
        
        Args:
            wallet: Wallet to analyze
            transactions: Transaction history
            related_data: Additional context (other wallets, cross-chain data)
        
        Returns:
            Complete analysis of obfuscation techniques used
        """
        self._stats['analyses_performed'] += 1
        
        results = {
            'wallet': wallet,
            'mixer_usage': [],
            'chain_hops': [],
            'privacy_coin_exits': [],
            'linked_wallets': [],
            'obfuscation_methods_detected': [],
        }
        
        related_data = related_data or {}
        
        # 1. Check for mixer usage
        mixer_txs = [tx for tx in transactions if self._is_mixer_interaction(tx)]
        for tx in mixer_txs:
            trace = await self.mixer_tracer.trace_tornado_deposit(
                tx.get('tx_hash', ''),
                tx.get('value', 0),
                tx.get('timestamp', datetime.now()),
                related_data.get('mixer_withdrawals', []),
            )
            if trace.confidence > 0.3:
                results['mixer_usage'].append(trace.to_dict())
                self._stats['obfuscation_detected'] += 1
        
        if results['mixer_usage']:
            results['obfuscation_methods_detected'].append(ObfuscationMethod.MIXER.value)
        
        # 2. Check for chain hopping
        bridge_txs = [tx for tx in transactions if self._is_bridge_interaction(tx)]
        for tx in bridge_txs:
            for dest_chain in ['polygon', 'arbitrum', 'optimism', 'bsc']:
                trace = await self.chain_hopper.trace_bridge_transfer(
                    tx.get('tx_hash', ''),
                    'eth',
                    tx.get('value', 0),
                    tx.get('timestamp', datetime.now()),
                    dest_chain,
                    related_data.get(f'{dest_chain}_transactions', []),
                )
                if trace.confidence > 0.3:
                    results['chain_hops'].append(trace.to_dict())
        
        if results['chain_hops']:
            results['obfuscation_methods_detected'].append(ObfuscationMethod.CHAIN_HOP.value)
        
        # 3. Link related wallets
        all_wallets = related_data.get('related_wallets', [])
        all_txs = related_data.get('all_transactions', {})
        
        links = await self.persona_linker.find_linked_wallets(
            wallet, transactions, all_wallets, all_txs
        )
        results['linked_wallets'] = [link.to_dict() for link in links]
        
        if results['linked_wallets']:
            results['obfuscation_methods_detected'].append(ObfuscationMethod.MULTI_WALLET.value)
        
        return results
    
    def _is_mixer_interaction(self, tx: Dict) -> bool:
        """Check if transaction interacts with known mixer."""
        to_addr = tx.get('to_address', '').lower()
        
        # Tornado Cash contracts
        tornado_contracts = [
            '0x722122df12d4e14e13ac3b6895a86e84145b6967',
            '0xd90e2f925da726b50c4ed8d0fb90ad053324f31b',
            '0x910cbd523d972eb0a6f4cae4618ad62622b39dbf',
            '0xa160cdab225685da1d56aa342ad8841c3b53f291',
        ]
        
        return to_addr in tornado_contracts
    
    def _is_bridge_interaction(self, tx: Dict) -> bool:
        """Check if transaction interacts with known bridge."""
        to_addr = tx.get('to_address', '').lower()
        
        bridge_contracts = [
            '0x40ec5b33f54e0e8a33a975908c5ba1c14e5bbbdf',  # Polygon
            '0x8315177ab297ba92a06054ce80a67ed4dbd7ed3a',  # Arbitrum
            '0x99c9fc46f92e8a1c0dec1b1747d010903e884be1',  # Optimism
        ]
        
        return to_addr in bridge_contracts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            'counter_measure': self._stats,
            'mixer_tracer': self.mixer_tracer._stats,
            'chain_hopper': dict(self.chain_hopper._stats),
            'persona_linker': self.persona_linker._stats,
        }


# Alias for compatibility
CounterMeasureSystem = CounterMeasureAnalyzer
