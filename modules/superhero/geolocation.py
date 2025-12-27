#!/usr/bin/env python3
"""
RF Arsenal OS - SUPERHERO Geolocation & Behavioral Analysis
============================================================

Estimate criminal location and behavioral patterns from blockchain data.

LEGAL NOTICE:
- All analysis from PUBLIC blockchain data
- Transaction timing is PUBLIC
- No active tracking or surveillance
- Statistical inference only

Methods:
- Transaction timing → Timezone inference
- Gas price patterns → Regional exchange correlation
- Language analysis from linked communications
- Activity pattern profiling

Author: RF Arsenal Security Team
License: Authorized Use Only
"""

import asyncio
import hashlib
import json
import logging
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import secrets

logger = logging.getLogger(__name__)


# Timezone to region mapping
TIMEZONE_REGIONS = {
    -12: ["Baker Island", "Howland Island"],
    -11: ["American Samoa", "Niue"],
    -10: ["Hawaii", "Cook Islands", "Tahiti"],
    -9: ["Alaska", "Gambier Islands"],
    -8: ["Pacific Time (US)", "Baja California", "Yukon"],
    -7: ["Mountain Time (US)", "Arizona"],
    -6: ["Central Time (US)", "Mexico City", "Costa Rica"],
    -5: ["Eastern Time (US)", "Colombia", "Peru", "Cuba"],
    -4: ["Atlantic Time", "Venezuela", "Bolivia", "Dominican Republic"],
    -3: ["Brazil (Brasília)", "Argentina", "Uruguay", "Chile"],
    -2: ["South Georgia", "Fernando de Noronha"],
    -1: ["Azores", "Cape Verde"],
    0: ["UK", "Portugal", "Iceland", "Ghana", "Morocco"],
    1: ["Central Europe", "Nigeria", "Algeria", "Tunisia"],
    2: ["Eastern Europe", "Egypt", "South Africa", "Israel"],
    3: ["Moscow", "Turkey", "Saudi Arabia", "Kenya", "Iraq"],
    4: ["UAE", "Oman", "Azerbaijan", "Georgia"],
    5: ["Pakistan", "Uzbekistan", "Maldives"],
    5.5: ["India", "Sri Lanka"],
    6: ["Bangladesh", "Kazakhstan", "Bhutan"],
    7: ["Thailand", "Vietnam", "Indonesia (West)", "Cambodia"],
    8: ["China", "Singapore", "Malaysia", "Philippines", "Hong Kong", "Taiwan"],
    9: ["Japan", "Korea", "Indonesia (East)"],
    9.5: ["Australia (Central)"],
    10: ["Australia (East)", "Papua New Guinea", "Guam"],
    11: ["Solomon Islands", "New Caledonia"],
    12: ["New Zealand", "Fiji", "Marshall Islands"],
}

# Common active hours by lifestyle
LIFESTYLE_PATTERNS = {
    "professional": {"start": 8, "end": 22, "peak": [9, 10, 14, 15, 16]},
    "night_owl": {"start": 14, "end": 4, "peak": [22, 23, 0, 1, 2]},
    "trader": {"start": 6, "end": 20, "peak": [9, 10, 14, 15]},  # Market hours
    "global_trader": {"start": 0, "end": 24, "peak": []},  # 24/7
}


@dataclass
class TimezoneAnalysis:
    """Timezone inference result."""
    estimated_offset: float  # UTC offset
    confidence: float
    possible_regions: List[str]
    evidence: List[str]
    sample_size: int
    active_hours_utc: List[int]
    quiet_hours_utc: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'utc_offset': self.estimated_offset,
            'confidence': self.confidence,
            'regions': self.possible_regions,
            'evidence': self.evidence,
            'sample_size': self.sample_size,
            'active_hours': self.active_hours_utc,
            'quiet_hours': self.quiet_hours_utc,
        }


@dataclass
class BehavioralPattern:
    """Behavioral analysis result."""
    lifestyle_type: str
    activity_days: Dict[str, float]  # day -> activity level
    peak_hours: List[int]
    average_tx_per_day: float
    consistency_score: float  # How regular are their patterns
    anomalies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'lifestyle': self.lifestyle_type,
            'activity_by_day': self.activity_days,
            'peak_hours': self.peak_hours,
            'avg_tx_per_day': self.average_tx_per_day,
            'consistency': self.consistency_score,
            'anomalies': self.anomalies,
        }


@dataclass
class LocationEstimate:
    """Geographic location estimate."""
    region: str
    country: Optional[str]
    timezone: str
    confidence: float
    evidence: List[str]
    language_hints: List[str]
    currency_hints: List[str]
    exchange_hints: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'region': self.region,
            'country': self.country,
            'timezone': self.timezone,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'language_hints': self.language_hints,
            'currency_hints': self.currency_hints,
            'exchange_hints': self.exchange_hints,
        }


@dataclass
class IPLeak:
    """IP address leak from blockchain activity."""
    ip_address: str
    source: str  # node_broadcast, exchange_login, etc.
    timestamp: datetime
    confidence: float
    geolocation: Optional[Dict[str, Any]] = None


class TransactionTimingAnalyzer:
    """
    Analyze transaction timing to infer timezone.
    
    Method: People transact during waking hours.
    If we see consistent activity 14:00-02:00 UTC,
    that suggests UTC+8 to UTC+10 timezone.
    """
    
    def __init__(self):
        self._hour_weights = [0.0] * 24  # Weight for each UTC hour
    
    def analyze_timestamps(self, timestamps: List[datetime]) -> TimezoneAnalysis:
        """
        Analyze transaction timestamps to estimate timezone.
        
        Args:
            timestamps: List of transaction times (should be UTC)
        
        Returns:
            TimezoneAnalysis with estimated timezone
        """
        if len(timestamps) < 10:
            return TimezoneAnalysis(
                estimated_offset=0,
                confidence=0.0,
                possible_regions=["Insufficient data"],
                evidence=["Need at least 10 transactions"],
                sample_size=len(timestamps),
                active_hours_utc=[],
                quiet_hours_utc=[],
            )
        
        # Count transactions per hour (UTC)
        hour_counts = Counter()
        for ts in timestamps:
            hour_counts[ts.hour] += 1
        
        # Find active and quiet periods
        total = sum(hour_counts.values())
        hourly_rates = {h: hour_counts[h] / total for h in range(24)}
        
        # Find the "night" period (consecutive hours with low activity)
        # This gives us their sleep time
        avg_rate = 1 / 24
        quiet_hours = [h for h, rate in hourly_rates.items() if rate < avg_rate * 0.5]
        active_hours = [h for h, rate in hourly_rates.items() if rate > avg_rate * 1.5]
        
        # Estimate timezone based on when they're quiet
        # Assume people sleep roughly 2-7 AM local time
        if quiet_hours:
            # Find the center of quiet period
            quiet_center = self._find_hour_center(quiet_hours)
            
            # If quiet at hour X UTC, and we assume they sleep at ~4 AM local,
            # then their timezone offset is X - 4
            estimated_offset = (quiet_center - 4) % 24
            if estimated_offset > 12:
                estimated_offset -= 24
            
            confidence = min(0.9, len(timestamps) / 100 * 0.3 + len(quiet_hours) / 8 * 0.4)
        else:
            # No clear quiet period - might be global trader or VPN user
            estimated_offset = 0
            confidence = 0.2
        
        # Get possible regions
        regions = TIMEZONE_REGIONS.get(estimated_offset, ["Unknown"])
        
        # Build evidence
        evidence = [
            f"Analyzed {len(timestamps)} transactions",
            f"Quiet hours (UTC): {sorted(quiet_hours)}",
            f"Active hours (UTC): {sorted(active_hours)}",
        ]
        
        return TimezoneAnalysis(
            estimated_offset=estimated_offset,
            confidence=confidence,
            possible_regions=regions,
            evidence=evidence,
            sample_size=len(timestamps),
            active_hours_utc=sorted(active_hours),
            quiet_hours_utc=sorted(quiet_hours),
        )
    
    def _find_hour_center(self, hours: List[int]) -> int:
        """Find center of a (possibly wrapping) hour range."""
        if not hours:
            return 12
        
        hours = sorted(hours)
        
        # Check if it wraps around midnight
        gaps = []
        for i in range(len(hours)):
            next_i = (i + 1) % len(hours)
            gap = (hours[next_i] - hours[i]) % 24
            gaps.append((hours[i], gap))
        
        # Find largest gap (this is NOT the quiet period)
        largest_gap = max(gaps, key=lambda x: x[1])
        
        # Center is opposite to largest gap
        start_of_quiet = (largest_gap[0] + largest_gap[1]) % 24
        end_of_quiet = largest_gap[0]
        
        center = (start_of_quiet + (end_of_quiet - start_of_quiet) % 24 // 2) % 24
        return center


class GasPriceAnalyzer:
    """
    Analyze gas price patterns for regional hints.
    
    Different regions have different gas price sensitivity:
    - Asia tends to use higher gas (more time-sensitive)
    - Europe/US often wait for lower gas
    """
    
    # Regional gas price tendencies (relative to average)
    REGIONAL_GAS_PATTERNS = {
        "asia": 1.2,      # Willing to pay more
        "us_east": 0.9,   # Cost conscious
        "us_west": 0.85,  # More cost conscious
        "europe": 0.95,   # Moderate
    }
    
    def analyze_gas_prices(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze gas price patterns.
        
        Args:
            transactions: List with 'gas_price' and 'timestamp' fields
        
        Returns:
            Analysis of gas usage patterns
        """
        if not transactions:
            return {"hint": None, "confidence": 0}
        
        # Calculate average gas price by hour
        hour_gas = defaultdict(list)
        for tx in transactions:
            if 'gas_price' in tx and 'timestamp' in tx:
                hour = tx['timestamp'].hour
                hour_gas[hour].append(tx['gas_price'])
        
        # Analyze patterns
        if not hour_gas:
            return {"hint": None, "confidence": 0}
        
        avg_gas = statistics.mean([g for prices in hour_gas.values() for g in prices])
        
        # Check if they transact during high or low gas periods
        # This can hint at their urgency/region
        
        return {
            "average_gas": avg_gas,
            "gas_by_hour": {h: statistics.mean(prices) for h, prices in hour_gas.items()},
            "hint": None,  # Would need more sophisticated analysis
            "confidence": 0.3,
        }


class LanguageAnalyzer:
    """
    Analyze language from linked content.
    
    Sources:
    - ENS name patterns
    - NFT metadata
    - Social media posts
    - Forum posts
    """
    
    # Language detection patterns
    LANGUAGE_PATTERNS = {
        "chinese": [r'[\u4e00-\u9fff]', r'[\u3400-\u4dbf]'],
        "japanese": [r'[\u3040-\u309f]', r'[\u30a0-\u30ff]'],
        "korean": [r'[\uac00-\ud7af]', r'[\u1100-\u11ff]'],
        "russian": [r'[\u0400-\u04ff]'],
        "arabic": [r'[\u0600-\u06ff]'],
        "hebrew": [r'[\u0590-\u05ff]'],
        "thai": [r'[\u0e00-\u0e7f]'],
        "hindi": [r'[\u0900-\u097f]'],
    }
    
    def detect_language(self, text: str) -> List[Tuple[str, float]]:
        """
        Detect language from text.
        
        Returns list of (language, confidence) tuples.
        """
        if not text:
            return []
        
        detected = []
        
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    confidence = min(1.0, len(matches) / len(text) * 10)
                    detected.append((lang, confidence))
                    break
        
        # Check for English (Latin characters)
        latin_matches = re.findall(r'[a-zA-Z]+', text)
        if latin_matches and not detected:
            detected.append(("english_or_european", 0.5))
        
        return detected
    
    def analyze_ens_name(self, name: str) -> Dict[str, Any]:
        """
        Analyze ENS name for language/cultural hints.
        """
        languages = self.detect_language(name)
        
        hints = {
            "detected_languages": languages,
            "possible_regions": [],
        }
        
        for lang, conf in languages:
            if lang == "chinese":
                hints["possible_regions"].extend(["China", "Taiwan", "Hong Kong", "Singapore"])
            elif lang == "japanese":
                hints["possible_regions"].append("Japan")
            elif lang == "korean":
                hints["possible_regions"].append("South Korea")
            elif lang == "russian":
                hints["possible_regions"].extend(["Russia", "Ukraine", "Belarus"])
            elif lang == "arabic":
                hints["possible_regions"].extend(["UAE", "Saudi Arabia", "Egypt"])
        
        return hints


class ExchangePatternAnalyzer:
    """
    Analyze exchange usage patterns for regional hints.
    
    Different regions prefer different exchanges:
    - Binance: Global, but strong in Asia
    - Coinbase: US dominant
    - Kraken: US/Europe
    - FTX: Was global (defunct)
    - OKX: Asia
    - Bitfinex: Global, large traders
    """
    
    EXCHANGE_REGIONS = {
        "binance": ["Global", "Asia", "Europe"],
        "coinbase": ["United States"],
        "kraken": ["United States", "Europe"],
        "gemini": ["United States"],
        "okx": ["Asia", "China"],
        "huobi": ["Asia", "China"],
        "kucoin": ["Asia"],
        "gate.io": ["Asia", "Global"],
        "bitstamp": ["Europe"],
        "bitfinex": ["Global"],
        "upbit": ["South Korea"],
        "bithumb": ["South Korea"],
        "bitflyer": ["Japan"],
    }
    
    def analyze_exchange_usage(self, exchanges_used: List[str]) -> Dict[str, Any]:
        """
        Analyze which exchanges were used to infer region.
        """
        if not exchanges_used:
            return {"hint": None, "confidence": 0}
        
        region_votes = Counter()
        
        for exchange in exchanges_used:
            exchange_lower = exchange.lower()
            for known, regions in self.EXCHANGE_REGIONS.items():
                if known in exchange_lower:
                    for region in regions:
                        region_votes[region] += 1
        
        if not region_votes:
            return {"hint": None, "confidence": 0}
        
        most_common = region_votes.most_common(3)
        
        return {
            "likely_regions": [r for r, _ in most_common],
            "region_scores": dict(region_votes),
            "confidence": min(0.7, len(exchanges_used) / 5 * 0.3),
        }


class GeolocationAnalyzer:
    """
    Main geolocation analysis engine.
    
    Combines multiple signals to estimate criminal location.
    """
    
    def __init__(self):
        self.timing = TransactionTimingAnalyzer()
        self.gas = GasPriceAnalyzer()
        self.language = LanguageAnalyzer()
        self.exchange = ExchangePatternAnalyzer()
        
        # Statistics
        self._stats = {
            'analyses_performed': 0,
            'high_confidence_locations': 0,
        }
    
    async def analyze_location(self, 
                                transactions: List[Dict[str, Any]],
                                identity_leads: List[Dict[str, Any]] = None,
                                exchanges_used: List[str] = None) -> LocationEstimate:
        """
        Full location analysis.
        
        Args:
            transactions: List of transactions with timestamps
            identity_leads: Identity leads (may contain language clues)
            exchanges_used: List of exchange names interacted with
        
        Returns:
            LocationEstimate with best guess at location
        """
        self._stats['analyses_performed'] += 1
        
        evidence = []
        region_scores = Counter()
        
        # 1. Timing analysis
        if transactions:
            timestamps = [tx.get('timestamp') for tx in transactions if tx.get('timestamp')]
            timing_result = self.timing.analyze_timestamps(timestamps)
            
            if timing_result.confidence > 0.5:
                for region in timing_result.possible_regions:
                    region_scores[region] += timing_result.confidence * 2
                evidence.append(f"Timing analysis suggests UTC{timing_result.estimated_offset:+.1f}")
        
        # 2. Language analysis from identity leads
        language_hints = []
        if identity_leads:
            for lead in identity_leads:
                if lead.get('value'):
                    lang_result = self.language.detect_language(lead['value'])
                    for lang, conf in lang_result:
                        language_hints.append(lang)
                        
                        # Add region scores based on language
                        lang_regions = self.language.analyze_ens_name(lead['value'])
                        for region in lang_regions.get('possible_regions', []):
                            region_scores[region] += conf
        
        # 3. Exchange analysis
        exchange_hints = []
        if exchanges_used:
            exchange_result = self.exchange.analyze_exchange_usage(exchanges_used)
            for region in exchange_result.get('likely_regions', []):
                region_scores[region] += exchange_result.get('confidence', 0)
            exchange_hints = exchanges_used
        
        # 4. Gas price analysis
        if transactions:
            gas_result = self.gas.analyze_gas_prices(transactions)
            if gas_result.get('hint'):
                evidence.append(f"Gas pattern: {gas_result['hint']}")
        
        # Determine best estimate
        if region_scores:
            best_region, best_score = region_scores.most_common(1)[0]
            total_score = sum(region_scores.values())
            confidence = min(0.9, best_score / total_score if total_score > 0 else 0)
            
            # Try to narrow down to country
            country = self._region_to_country(best_region)
            timezone = self._region_to_timezone(best_region)
        else:
            best_region = "Unknown"
            country = None
            timezone = "UTC"
            confidence = 0.1
        
        if confidence >= 0.7:
            self._stats['high_confidence_locations'] += 1
        
        return LocationEstimate(
            region=best_region,
            country=country,
            timezone=timezone,
            confidence=confidence,
            evidence=evidence,
            language_hints=list(set(language_hints)),
            currency_hints=[],  # Could analyze stablecoin preferences
            exchange_hints=exchange_hints,
        )
    
    def analyze_behavioral_pattern(self, transactions: List[Dict[str, Any]]) -> BehavioralPattern:
        """
        Analyze behavioral patterns from transaction history.
        """
        if not transactions:
            return BehavioralPattern(
                lifestyle_type="unknown",
                activity_days={},
                peak_hours=[],
                average_tx_per_day=0,
                consistency_score=0,
                anomalies=[],
            )
        
        # Activity by day of week
        day_counts = Counter()
        hour_counts = Counter()
        
        for tx in transactions:
            ts = tx.get('timestamp')
            if ts:
                day_counts[ts.strftime('%A')] += 1
                hour_counts[ts.hour] += 1
        
        total = sum(day_counts.values())
        activity_days = {day: count / total for day, count in day_counts.items()}
        
        # Peak hours
        avg_hour_count = total / 24
        peak_hours = [h for h, c in hour_counts.items() if c > avg_hour_count * 1.5]
        
        # Determine lifestyle type
        if all(hour_counts[h] > 0 for h in range(24)):
            lifestyle = "global_trader"
        elif any(hour_counts[h] > avg_hour_count * 2 for h in range(22, 24)) or \
             any(hour_counts[h] > avg_hour_count * 2 for h in range(0, 4)):
            lifestyle = "night_owl"
        else:
            lifestyle = "professional"
        
        # Calculate consistency
        if len(transactions) >= 7:
            # Check if activity pattern is consistent week to week
            week_patterns = defaultdict(list)
            for tx in transactions:
                ts = tx.get('timestamp')
                if ts:
                    week_num = ts.isocalendar()[1]
                    week_patterns[week_num].append(ts.hour)
            
            if len(week_patterns) >= 2:
                # Compare patterns across weeks
                consistency = 0.7  # Placeholder
            else:
                consistency = 0.5
        else:
            consistency = 0.3
        
        # Detect anomalies
        anomalies = []
        if activity_days:
            avg_activity = statistics.mean(activity_days.values())
            for day, activity in activity_days.items():
                if activity < avg_activity * 0.3:
                    anomalies.append(f"Unusually low activity on {day}")
        
        return BehavioralPattern(
            lifestyle_type=lifestyle,
            activity_days=activity_days,
            peak_hours=sorted(peak_hours),
            average_tx_per_day=total / max(1, len(set(tx.get('timestamp', datetime.now()).date() for tx in transactions if tx.get('timestamp')))),
            consistency_score=consistency,
            anomalies=anomalies,
        )
    
    def _region_to_country(self, region: str) -> Optional[str]:
        """Try to narrow region to specific country."""
        # Simple mapping for common regions
        single_country_regions = {
            "Japan": "Japan",
            "South Korea": "South Korea",
            "India": "India",
            "Brazil": "Brazil",
            "UK": "United Kingdom",
            "Iceland": "Iceland",
        }
        return single_country_regions.get(region)
    
    def _region_to_timezone(self, region: str) -> str:
        """Get timezone string for region."""
        for offset, regions in TIMEZONE_REGIONS.items():
            if region in regions:
                if offset >= 0:
                    return f"UTC+{offset}"
                return f"UTC{offset}"
        return "UTC"
    
    def get_statistics(self) -> Dict[str, int]:
        """Get analysis statistics."""
        return self._stats.copy()


# Convenience function
async def estimate_location(transactions: List[Dict], 
                            identity_leads: List[Dict] = None) -> Dict[str, Any]:
    """
    Quick location estimation.
    
    Args:
        transactions: Transaction history
        identity_leads: Optional identity leads
    
    Returns:
        Location estimate summary
    """
    analyzer = GeolocationAnalyzer()
    result = await analyzer.analyze_location(transactions, identity_leads)
    return result.to_dict()
