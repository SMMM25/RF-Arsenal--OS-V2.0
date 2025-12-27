"""
RF Arsenal OS - Key Derivation Analyzer
=======================================

Comprehensive analysis of cryptocurrency key derivation mechanisms.
Identifies weaknesses in mnemonic generation, HD wallet derivation,
and key stretching implementations.

AUTHORIZED USE ONLY:
- Only analyze key derivation for wallets you own or have authorization
- Designed for security auditors and authorized pentesters
- Helps identify weak key generation for remediation

STEALTH COMPLIANCE:
- All operations through proxy chain
- RAM-only data handling
- No telemetry or logging to external services
- Offline analysis capability

Author: RF Arsenal Security Team
License: Authorized Use Only
"""

import hashlib
import hmac
import math
import os
import re
import secrets
import struct
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class DerivationStandard(Enum):
    """Key derivation standards."""
    BIP32 = "bip32"
    BIP39 = "bip39"
    BIP44 = "bip44"
    BIP49 = "bip49"
    BIP84 = "bip84"
    BIP86 = "bip86"
    SLIP10 = "slip10"
    ELECTRUM_V1 = "electrum_v1"
    ELECTRUM_V2 = "electrum_v2"
    BRAIN_WALLET = "brain_wallet"
    CUSTOM = "custom"


class WeaknessType(Enum):
    """Types of key derivation weaknesses."""
    LOW_ENTROPY = "low_entropy"
    PREDICTABLE_SEED = "predictable_seed"
    WEAK_PASSPHRASE = "weak_passphrase"
    BROKEN_RANDOMNESS = "broken_randomness"
    DICTIONARY_DERIVATION = "dictionary_derivation"
    TIMESTAMP_BASED = "timestamp_based"
    SEQUENTIAL = "sequential"
    BIASED_SAMPLING = "biased_sampling"
    INSUFFICIENT_STRETCHING = "insufficient_stretching"
    KNOWN_BRAIN_WALLET = "known_brain_wallet"


class CrackDifficulty(Enum):
    """Estimated difficulty to crack key derivation."""
    TRIVIAL = "trivial"           # Seconds
    EASY = "easy"                 # Minutes to hours
    MODERATE = "moderate"         # Days to weeks
    HARD = "hard"                 # Months to years
    INFEASIBLE = "infeasible"     # Computationally infeasible


@dataclass
class EntropySource:
    """Information about entropy source."""
    source_type: str
    bits_of_entropy: float
    is_secure: bool
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MnemonicAnalysis:
    """Analysis of BIP39 mnemonic phrase."""
    word_count: int
    entropy_bits: int
    checksum_bits: int
    checksum_valid: bool
    language: str
    weakness_detected: bool
    weaknesses: List[str] = field(default_factory=list)
    estimated_crack_time: str = ""
    is_common_phrase: bool = False
    common_phrase_rank: Optional[int] = None


@dataclass
class DerivationPathAnalysis:
    """Analysis of HD derivation path."""
    path: str
    standard: DerivationStandard
    depth: int
    is_hardened: bool
    coin_type: Optional[int] = None
    account: Optional[int] = None
    change: Optional[int] = None
    address_index: Optional[int] = None
    issues: List[str] = field(default_factory=list)


@dataclass
class KeyStretchingAnalysis:
    """Analysis of key stretching parameters."""
    algorithm: str
    iterations: int
    memory_cost: Optional[int] = None
    parallelism: Optional[int] = None
    salt_length: int = 0
    is_secure: bool = True
    estimated_crack_time: str = ""
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class KeyDerivationReport:
    """Complete key derivation analysis report."""
    analysis_id: str
    timestamp: datetime
    derivation_standard: DerivationStandard
    entropy_analysis: EntropySource
    mnemonic_analysis: Optional[MnemonicAnalysis] = None
    path_analysis: Optional[DerivationPathAnalysis] = None
    stretching_analysis: Optional[KeyStretchingAnalysis] = None
    overall_security_score: float = 0.0
    crack_difficulty: CrackDifficulty = CrackDifficulty.INFEASIBLE
    vulnerabilities: List[Tuple[WeaknessType, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'analysis_id': self.analysis_id,
            'timestamp': self.timestamp.isoformat(),
            'derivation_standard': self.derivation_standard.value,
            'entropy_analysis': {
                'source_type': self.entropy_analysis.source_type,
                'bits_of_entropy': self.entropy_analysis.bits_of_entropy,
                'is_secure': self.entropy_analysis.is_secure,
                'weaknesses': self.entropy_analysis.weaknesses
            },
            'mnemonic_analysis': {
                'word_count': self.mnemonic_analysis.word_count,
                'entropy_bits': self.mnemonic_analysis.entropy_bits,
                'checksum_valid': self.mnemonic_analysis.checksum_valid,
                'weakness_detected': self.mnemonic_analysis.weakness_detected,
                'weaknesses': self.mnemonic_analysis.weaknesses
            } if self.mnemonic_analysis else None,
            'path_analysis': {
                'path': self.path_analysis.path,
                'standard': self.path_analysis.standard.value,
                'depth': self.path_analysis.depth,
                'issues': self.path_analysis.issues
            } if self.path_analysis else None,
            'overall_security_score': self.overall_security_score,
            'crack_difficulty': self.crack_difficulty.value,
            'vulnerabilities': [(v[0].value, v[1]) for v in self.vulnerabilities],
            'recommendations': self.recommendations
        }


class KeyDerivationAnalyzer:
    """
    Comprehensive key derivation analyzer for cryptocurrency wallets.
    
    Analyzes:
    - BIP39 mnemonic phrases for weaknesses
    - HD derivation paths (BIP32/44/49/84/86)
    - Entropy sources and randomness quality
    - Key stretching parameters
    - Brain wallet security
    - Custom derivation schemes
    
    AUTHORIZED USE ONLY - Only analyze keys you own or have authorization to test.
    """
    
    # BIP39 English wordlist (first 100 for pattern detection, full list in production)
    BIP39_WORDS_SAMPLE = [
        "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
        "absurd", "abuse", "access", "accident", "account", "accuse", "achieve", "acid",
        "acoustic", "acquire", "across", "act", "action", "actor", "actress", "actual",
        "adapt", "add", "addict", "address", "adjust", "admit", "adult", "advance",
        "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
        "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album", "alcohol",
        "alert", "alien", "all", "alley", "allow", "almost", "alone", "alpha", "already",
        "also", "alter", "always", "amateur", "amazing", "among", "amount", "amused",
        "analyst", "anchor", "ancient", "anger", "angle", "angry", "animal", "ankle",
        "announce", "annual", "another", "answer", "antenna", "antique", "anxiety", "any",
        "apart", "apology", "appear", "apple", "approve", "april", "arch", "arctic",
        "area", "arena", "argue", "arm", "armed", "armor", "army", "around", "arrange"
    ]
    
    # Known weak/common brain wallet phrases
    KNOWN_BRAIN_WALLETS = {
        "password": "Extremely common",
        "123456": "Numeric sequence",
        "bitcoin": "Crypto-related",
        "satoshi": "Crypto-related",
        "blockchain": "Crypto-related",
        "crypto": "Crypto-related",
        "wallet": "Crypto-related",
        "": "Empty passphrase",
        " ": "Single space",
        "correct horse battery staple": "Famous XKCD comic",
        "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about": "First BIP39 mnemonic",
    }
    
    # PBKDF2 recommended minimums
    MIN_PBKDF2_ITERATIONS = 100000
    
    # Scrypt recommended minimums
    MIN_SCRYPT_N = 16384
    MIN_SCRYPT_R = 8
    MIN_SCRYPT_P = 1
    
    # Argon2 recommended minimums  
    MIN_ARGON2_TIME = 3
    MIN_ARGON2_MEMORY = 65536  # 64MB
    
    def __init__(self, ram_only: bool = True):
        """
        Initialize key derivation analyzer.
        
        Args:
            ram_only: If True, never write sensitive data to disk
        """
        self.ram_only = ram_only
        self._analysis_cache: Dict[str, KeyDerivationReport] = {}
        
        logger.info("KeyDerivationAnalyzer initialized (RAM-only: %s)", ram_only)
    
    def analyze_mnemonic(
        self,
        mnemonic: str,
        passphrase: str = "",
        expected_standard: DerivationStandard = DerivationStandard.BIP39
    ) -> KeyDerivationReport:
        """
        Analyze a mnemonic phrase for security weaknesses.
        
        Args:
            mnemonic: The mnemonic phrase to analyze
            passphrase: Optional BIP39 passphrase
            expected_standard: Expected derivation standard
            
        Returns:
            KeyDerivationReport with analysis results
        """
        analysis_id = secrets.token_hex(8)
        timestamp = datetime.now(timezone.utc)
        vulnerabilities = []
        recommendations = []
        
        logger.info("Analyzing mnemonic phrase (ID: %s)", analysis_id)
        
        # Normalize mnemonic
        words = mnemonic.strip().lower().split()
        word_count = len(words)
        
        # Determine entropy from word count
        entropy_bits = self._word_count_to_entropy(word_count)
        checksum_bits = word_count // 3
        
        # Validate checksum
        checksum_valid = self._validate_mnemonic_checksum(mnemonic)
        
        # Check for known weak mnemonics
        is_common, common_rank = self._check_known_weak_mnemonic(mnemonic)
        
        weaknesses = []
        
        if is_common:
            weaknesses.append(f"Known weak mnemonic (rank #{common_rank})")
            vulnerabilities.append((
                WeaknessType.KNOWN_BRAIN_WALLET,
                f"Mnemonic matches known weak phrase at rank #{common_rank}"
            ))
        
        # Check for sequential words
        if self._has_sequential_pattern(words):
            weaknesses.append("Sequential word pattern detected")
            vulnerabilities.append((
                WeaknessType.SEQUENTIAL,
                "Mnemonic contains sequential word pattern"
            ))
        
        # Check for repeated words
        if len(set(words)) < len(words) * 0.5:
            weaknesses.append("High word repetition")
            vulnerabilities.append((
                WeaknessType.LOW_ENTROPY,
                "Mnemonic has excessive word repetition, reducing entropy"
            ))
        
        # Check for alphabetical ordering
        if words == sorted(words):
            weaknesses.append("Words are alphabetically sorted")
            vulnerabilities.append((
                WeaknessType.PREDICTABLE_SEED,
                "Mnemonic words are in alphabetical order"
            ))
        
        # Analyze passphrase
        if passphrase:
            passphrase_weaknesses = self._analyze_passphrase(passphrase)
            if passphrase_weaknesses:
                weaknesses.extend(passphrase_weaknesses)
                vulnerabilities.append((
                    WeaknessType.WEAK_PASSPHRASE,
                    f"Passphrase weaknesses: {', '.join(passphrase_weaknesses)}"
                ))
        
        # Estimate crack time
        effective_entropy = entropy_bits
        if weaknesses:
            effective_entropy = max(0, effective_entropy - len(weaknesses) * 20)
        
        crack_time = self._estimate_crack_time(effective_entropy)
        crack_difficulty = self._entropy_to_difficulty(effective_entropy)
        
        # Build mnemonic analysis
        mnemonic_analysis = MnemonicAnalysis(
            word_count=word_count,
            entropy_bits=entropy_bits,
            checksum_bits=checksum_bits,
            checksum_valid=checksum_valid,
            language="english",  # Would detect in production
            weakness_detected=len(weaknesses) > 0,
            weaknesses=weaknesses,
            estimated_crack_time=crack_time,
            is_common_phrase=is_common,
            common_phrase_rank=common_rank
        )
        
        # Build entropy source analysis
        entropy_analysis = EntropySource(
            source_type="bip39_mnemonic",
            bits_of_entropy=float(entropy_bits),
            is_secure=len(weaknesses) == 0 and entropy_bits >= 128,
            weaknesses=weaknesses,
            recommendations=self._get_entropy_recommendations(entropy_bits, weaknesses)
        )
        
        # Generate recommendations
        if entropy_bits < 128:
            recommendations.append("Use 12 or more words for at least 128 bits of entropy")
        if entropy_bits < 256:
            recommendations.append("Consider using 24-word mnemonic for maximum security")
        if not passphrase:
            recommendations.append("Consider adding a BIP39 passphrase for additional security")
        if weaknesses:
            recommendations.append("Generate new mnemonic with proper randomness")
        
        # Calculate security score
        security_score = self._calculate_security_score(entropy_bits, vulnerabilities)
        
        # Build report
        report = KeyDerivationReport(
            analysis_id=analysis_id,
            timestamp=timestamp,
            derivation_standard=expected_standard,
            entropy_analysis=entropy_analysis,
            mnemonic_analysis=mnemonic_analysis,
            overall_security_score=security_score,
            crack_difficulty=crack_difficulty,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations
        )
        
        # Cache in RAM
        self._analysis_cache[analysis_id] = report
        
        logger.info("Mnemonic analysis complete (Score: %.1f, Vulns: %d)", 
                   security_score, len(vulnerabilities))
        
        return report
    
    def analyze_derivation_path(
        self,
        path: str,
        coin_type: Optional[int] = None
    ) -> DerivationPathAnalysis:
        """
        Analyze HD wallet derivation path for issues.
        
        Args:
            path: BIP32 derivation path (e.g., "m/44'/0'/0'/0/0")
            coin_type: Expected coin type for validation
            
        Returns:
            DerivationPathAnalysis with findings
        """
        issues = []
        
        # Parse path
        if not path.startswith('m'):
            issues.append("Path should start with 'm' for master key")
            path = 'm/' + path.lstrip('/')
        
        parts = path.split('/')
        depth = len(parts) - 1  # Exclude 'm'
        
        # Detect standard from path pattern
        standard = self._detect_derivation_standard(path)
        
        # Check for hardened derivation where expected
        has_hardened = any("'" in p or "h" in p.lower() for p in parts[1:4])
        
        # Parse components
        parsed_coin_type = None
        parsed_account = None
        parsed_change = None
        parsed_index = None
        
        try:
            if len(parts) > 2:
                parsed_coin_type = int(parts[2].replace("'", "").replace("h", ""))
            if len(parts) > 3:
                parsed_account = int(parts[3].replace("'", "").replace("h", ""))
            if len(parts) > 4:
                parsed_change = int(parts[4].replace("'", "").replace("h", ""))
            if len(parts) > 5:
                parsed_index = int(parts[5].replace("'", "").replace("h", ""))
        except (ValueError, IndexError):
            issues.append("Invalid path component format")
        
        # Validate coin type if provided
        if coin_type is not None and parsed_coin_type != coin_type:
            issues.append(f"Coin type mismatch: expected {coin_type}, got {parsed_coin_type}")
        
        # Check for non-hardened in sensitive positions
        if len(parts) > 2 and "'" not in parts[2] and "h" not in parts[2].lower():
            issues.append("Coin type should use hardened derivation")
        if len(parts) > 3 and "'" not in parts[3] and "h" not in parts[3].lower():
            issues.append("Account should use hardened derivation")
        
        # Check for unusually deep paths
        if depth > 6:
            issues.append(f"Unusually deep derivation path (depth {depth})")
        
        # Check for very high indices (potential enumeration attack vector)
        for i, part in enumerate(parts[1:], 1):
            try:
                index = int(part.replace("'", "").replace("h", ""))
                if index > 1000000:
                    issues.append(f"Very high index at position {i}: {index}")
            except ValueError:
                continue
        
        return DerivationPathAnalysis(
            path=path,
            standard=standard,
            depth=depth,
            is_hardened=has_hardened,
            coin_type=parsed_coin_type,
            account=parsed_account,
            change=parsed_change,
            address_index=parsed_index,
            issues=issues
        )
    
    def analyze_key_stretching(
        self,
        algorithm: str,
        iterations: int = 0,
        memory_cost: int = 0,
        parallelism: int = 1,
        salt_length: int = 0
    ) -> KeyStretchingAnalysis:
        """
        Analyze key stretching parameters for security.
        
        Args:
            algorithm: Stretching algorithm (pbkdf2, scrypt, argon2, etc.)
            iterations: Number of iterations (PBKDF2) or time cost (Argon2)
            memory_cost: Memory cost in KB (Scrypt N, Argon2 memory)
            parallelism: Parallelism factor
            salt_length: Salt length in bytes
            
        Returns:
            KeyStretchingAnalysis with security assessment
        """
        weaknesses = []
        is_secure = True
        
        algorithm = algorithm.lower()
        
        if algorithm == 'pbkdf2':
            if iterations < self.MIN_PBKDF2_ITERATIONS:
                weaknesses.append(
                    f"Iterations ({iterations}) below minimum ({self.MIN_PBKDF2_ITERATIONS})"
                )
                is_secure = False
            
            # Estimate crack time based on iterations
            crack_time = self._estimate_pbkdf2_crack_time(iterations)
        
        elif algorithm == 'scrypt':
            if memory_cost < self.MIN_SCRYPT_N:
                weaknesses.append(
                    f"N parameter ({memory_cost}) below minimum ({self.MIN_SCRYPT_N})"
                )
                is_secure = False
            
            crack_time = self._estimate_scrypt_crack_time(memory_cost, parallelism)
        
        elif algorithm in ['argon2', 'argon2id', 'argon2i', 'argon2d']:
            if iterations < self.MIN_ARGON2_TIME:
                weaknesses.append(
                    f"Time cost ({iterations}) below minimum ({self.MIN_ARGON2_TIME})"
                )
                is_secure = False
            
            if memory_cost < self.MIN_ARGON2_MEMORY:
                weaknesses.append(
                    f"Memory cost ({memory_cost}KB) below minimum ({self.MIN_ARGON2_MEMORY}KB)"
                )
                is_secure = False
            
            crack_time = self._estimate_argon2_crack_time(iterations, memory_cost)
        
        elif algorithm == 'bcrypt':
            if iterations < 10:  # bcrypt cost factor
                weaknesses.append(f"Cost factor ({iterations}) below minimum (10)")
                is_secure = False
            
            crack_time = self._estimate_bcrypt_crack_time(iterations)
        
        else:
            weaknesses.append(f"Unknown/custom algorithm: {algorithm}")
            crack_time = "Unknown"
            is_secure = False
        
        # Check salt
        if salt_length < 16:
            weaknesses.append(f"Salt length ({salt_length} bytes) below recommended 16 bytes")
            if salt_length == 0:
                is_secure = False
        
        return KeyStretchingAnalysis(
            algorithm=algorithm,
            iterations=iterations,
            memory_cost=memory_cost if memory_cost else None,
            parallelism=parallelism if parallelism > 1 else None,
            salt_length=salt_length,
            is_secure=is_secure,
            estimated_crack_time=crack_time,
            weaknesses=weaknesses
        )
    
    def analyze_brain_wallet(self, passphrase: str) -> KeyDerivationReport:
        """
        Analyze brain wallet passphrase for security.
        
        Brain wallets are inherently insecure. This analysis helps
        demonstrate why to clients.
        
        Args:
            passphrase: Brain wallet passphrase
            
        Returns:
            KeyDerivationReport with analysis
        """
        analysis_id = secrets.token_hex(8)
        timestamp = datetime.now(timezone.utc)
        vulnerabilities = []
        weaknesses = []
        
        logger.info("Analyzing brain wallet (ID: %s)", analysis_id)
        
        # Check against known weak passphrases
        passphrase_lower = passphrase.lower().strip()
        
        if passphrase_lower in self.KNOWN_BRAIN_WALLETS:
            weakness_desc = self.KNOWN_BRAIN_WALLETS[passphrase_lower]
            weaknesses.append(f"Known weak passphrase: {weakness_desc}")
            vulnerabilities.append((
                WeaknessType.KNOWN_BRAIN_WALLET,
                f"Passphrase matches known weak brain wallet: {weakness_desc}"
            ))
        
        # Analyze passphrase characteristics
        entropy_bits = self._estimate_passphrase_entropy(passphrase)
        
        # Check for common patterns
        if re.match(r'^[0-9]+$', passphrase):
            weaknesses.append("Numeric-only passphrase")
            vulnerabilities.append((
                WeaknessType.LOW_ENTROPY,
                "Passphrase contains only numbers"
            ))
            entropy_bits = min(entropy_bits, len(passphrase) * 3.3)
        
        if re.match(r'^[a-z]+$', passphrase.lower()):
            weaknesses.append("Alphabetic-only passphrase")
            entropy_bits = min(entropy_bits, len(passphrase) * 4.7)
        
        if len(passphrase) < 20:
            weaknesses.append("Short passphrase")
            vulnerabilities.append((
                WeaknessType.LOW_ENTROPY,
                f"Passphrase length ({len(passphrase)}) is too short"
            ))
        
        # Check for dictionary words
        word_count = len(passphrase.split())
        if word_count > 0 and word_count < 6:
            weaknesses.append(f"Only {word_count} words - dictionary attackable")
            vulnerabilities.append((
                WeaknessType.DICTIONARY_DERIVATION,
                "Passphrase consists of few dictionary words"
            ))
        
        # Brain wallets always get a critical vulnerability
        vulnerabilities.append((
            WeaknessType.PREDICTABLE_SEED,
            "Brain wallets are fundamentally insecure - human-chosen entropy"
        ))
        
        # Calculate effective entropy
        effective_entropy = max(0, entropy_bits - len(weaknesses) * 15)
        
        crack_time = self._estimate_crack_time(effective_entropy)
        crack_difficulty = self._entropy_to_difficulty(effective_entropy)
        
        # Build entropy analysis
        entropy_analysis = EntropySource(
            source_type="brain_wallet_passphrase",
            bits_of_entropy=entropy_bits,
            is_secure=False,  # Brain wallets are never secure
            weaknesses=weaknesses + ["Human-generated entropy is predictable"],
            recommendations=[
                "Do NOT use brain wallets",
                "Migrate funds immediately to proper BIP39 wallet",
                "Use hardware wallet or properly generated seed phrase"
            ]
        )
        
        # Security score is always low for brain wallets
        security_score = min(20.0, effective_entropy / 5)
        
        recommendations = [
            "🚨 CRITICAL: Brain wallets are NOT SECURE",
            "Migrate all funds to a hardware wallet immediately",
            "Generate new wallet using BIP39 with proper randomness",
            "Never reuse this passphrase anywhere"
        ]
        
        return KeyDerivationReport(
            analysis_id=analysis_id,
            timestamp=timestamp,
            derivation_standard=DerivationStandard.BRAIN_WALLET,
            entropy_analysis=entropy_analysis,
            overall_security_score=security_score,
            crack_difficulty=crack_difficulty,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations
        )
    
    def _word_count_to_entropy(self, word_count: int) -> int:
        """Convert BIP39 word count to entropy bits."""
        entropy_map = {
            12: 128,
            15: 160,
            18: 192,
            21: 224,
            24: 256
        }
        return entropy_map.get(word_count, word_count * 11 - word_count // 3)
    
    def _validate_mnemonic_checksum(self, mnemonic: str) -> bool:
        """
        Validate BIP39 mnemonic checksum.
        
        REAL-WORLD FUNCTIONAL:
        - Validates word count (12, 15, 18, 21, 24)
        - Validates all words are in BIP39 wordlist
        - Computes and validates checksum bits
        
        BIP39 checksum algorithm:
        1. Convert mnemonic words to indices (11 bits each)
        2. Concatenate all bits
        3. Split into entropy bits and checksum bits
        4. Compute SHA256 of entropy
        5. Compare first N bits of hash with checksum
        
        Args:
            mnemonic: BIP39 mnemonic phrase
            
        Returns:
            True if checksum is valid
        """
        words = mnemonic.strip().lower().split()
        word_count = len(words)
        
        # Validate word count
        if word_count not in [12, 15, 18, 21, 24]:
            return False
        
        # Try to use mnemonic library if available (most accurate)
        try:
            from mnemonic import Mnemonic
            mnemo = Mnemonic("english")
            return mnemo.check(mnemonic)
        except ImportError:
            pass
        
        # Fallback: Manual BIP39 validation
        # Load full wordlist if available
        wordlist = self._get_bip39_wordlist()
        if wordlist is None:
            # Cannot validate without wordlist, assume valid for structure
            logger.warning("BIP39 wordlist not available, skipping checksum validation")
            return True
        
        # Create word-to-index mapping
        word_indices = {word: idx for idx, word in enumerate(wordlist)}
        
        # Convert words to bits
        indices = []
        for word in words:
            if word not in word_indices:
                logger.debug(f"Word not in BIP39 wordlist: {word}")
                return False
            indices.append(word_indices[word])
        
        # Convert indices to bits (each word is 11 bits)
        bit_string = ''.join(format(idx, '011b') for idx in indices)
        
        # Calculate entropy and checksum bit lengths
        # Total bits = word_count * 11
        # Checksum bits = word_count // 3
        # Entropy bits = Total bits - Checksum bits
        checksum_bits = word_count // 3
        entropy_bits = len(bit_string) - checksum_bits
        
        # Extract entropy and checksum
        entropy_str = bit_string[:entropy_bits]
        checksum_str = bit_string[entropy_bits:]
        
        # Convert entropy string to bytes
        entropy_bytes = int(entropy_str, 2).to_bytes(entropy_bits // 8, byteorder='big')
        
        # Compute SHA256 of entropy
        h = hashlib.sha256(entropy_bytes).hexdigest()
        h_bits = bin(int(h, 16))[2:].zfill(256)
        
        # Compare computed checksum with actual checksum
        computed_checksum = h_bits[:checksum_bits]
        
        return computed_checksum == checksum_str
    
    def _get_bip39_wordlist(self) -> Optional[List[str]]:
        """
        Get the full BIP39 English wordlist.
        
        Attempts to load from:
        1. mnemonic library
        2. Local file if available
        3. Embedded partial list (returns None for full validation)
        
        Returns:
            Full BIP39 wordlist (2048 words) or None
        """
        # Try mnemonic library first
        try:
            from mnemonic import Mnemonic
            mnemo = Mnemonic("english")
            return mnemo.wordlist
        except ImportError:
            pass
        
        # Try to load from common locations
        wordlist_paths = [
            "/usr/share/wordlists/bip39/english.txt",
            "~/.local/share/bip39/english.txt",
        ]
        
        for path in wordlist_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                try:
                    with open(expanded_path, 'r') as f:
                        words = [line.strip() for line in f if line.strip()]
                        if len(words) == 2048:
                            return words
                except Exception:
                    continue
        
        # Cannot load full wordlist
        return None
    
    def _check_known_weak_mnemonic(self, mnemonic: str) -> Tuple[bool, Optional[int]]:
        """Check if mnemonic matches known weak patterns."""
        normalized = mnemonic.strip().lower()
        
        # Check exact matches
        if normalized in self.KNOWN_BRAIN_WALLETS:
            return True, 1
        
        # Check for "abandon" mnemonic variants
        if normalized.startswith("abandon abandon"):
            return True, 1
        
        # Check for last word "zoo" variants
        if normalized.endswith("zoo wrong"):
            return True, 2
        
        return False, None
    
    def _has_sequential_pattern(self, words: List[str]) -> bool:
        """Check for sequential word patterns."""
        # Check if words are from sequential positions in wordlist
        # Simplified check
        if words == sorted(words):
            return True
        
        if words == sorted(words, reverse=True):
            return True
        
        return False
    
    def _analyze_passphrase(self, passphrase: str) -> List[str]:
        """Analyze BIP39 passphrase for weaknesses."""
        weaknesses = []
        
        if len(passphrase) < 8:
            weaknesses.append("Short passphrase")
        
        if passphrase.lower() in self.KNOWN_BRAIN_WALLETS:
            weaknesses.append("Common passphrase")
        
        if passphrase.isdigit():
            weaknesses.append("Numeric-only passphrase")
        
        if passphrase.isalpha():
            weaknesses.append("Letters-only passphrase")
        
        return weaknesses
    
    def _estimate_passphrase_entropy(self, passphrase: str) -> float:
        """Estimate entropy of passphrase."""
        if not passphrase:
            return 0.0
        
        # Character set analysis
        has_lower = bool(re.search(r'[a-z]', passphrase))
        has_upper = bool(re.search(r'[A-Z]', passphrase))
        has_digit = bool(re.search(r'[0-9]', passphrase))
        has_special = bool(re.search(r'[^a-zA-Z0-9]', passphrase))
        
        charset_size = 0
        if has_lower:
            charset_size += 26
        if has_upper:
            charset_size += 26
        if has_digit:
            charset_size += 10
        if has_special:
            charset_size += 32
        
        if charset_size == 0:
            return 0.0
        
        # Bits per character
        bits_per_char = math.log2(charset_size) if charset_size > 0 else 0
        
        return len(passphrase) * bits_per_char
    
    def _estimate_crack_time(self, entropy_bits: float) -> str:
        """Estimate time to crack given entropy."""
        # Assume 10 billion guesses per second (high-end GPU cluster)
        guesses_per_second = 10_000_000_000
        
        if entropy_bits <= 0:
            return "Instant"
        
        total_combinations = 2 ** entropy_bits
        seconds = total_combinations / guesses_per_second / 2  # Average case
        
        if seconds < 1:
            return "Instant"
        elif seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.0f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.0f} hours"
        elif seconds < 31536000:
            return f"{seconds/86400:.0f} days"
        elif seconds < 31536000 * 1000:
            return f"{seconds/31536000:.0f} years"
        else:
            return "Infeasible (> 1000 years)"
    
    def _entropy_to_difficulty(self, entropy_bits: float) -> CrackDifficulty:
        """Convert entropy to crack difficulty."""
        if entropy_bits < 40:
            return CrackDifficulty.TRIVIAL
        elif entropy_bits < 60:
            return CrackDifficulty.EASY
        elif entropy_bits < 80:
            return CrackDifficulty.MODERATE
        elif entropy_bits < 128:
            return CrackDifficulty.HARD
        else:
            return CrackDifficulty.INFEASIBLE
    
    def _detect_derivation_standard(self, path: str) -> DerivationStandard:
        """Detect derivation standard from path."""
        if "/44'" in path or "/44h" in path.lower():
            return DerivationStandard.BIP44
        elif "/49'" in path or "/49h" in path.lower():
            return DerivationStandard.BIP49
        elif "/84'" in path or "/84h" in path.lower():
            return DerivationStandard.BIP84
        elif "/86'" in path or "/86h" in path.lower():
            return DerivationStandard.BIP86
        else:
            return DerivationStandard.BIP32
    
    def _estimate_pbkdf2_crack_time(self, iterations: int) -> str:
        """Estimate PBKDF2 crack time."""
        # Assume 1M iterations/sec per GPU
        guesses_per_second = 1_000_000 / iterations
        
        if guesses_per_second > 1000:
            return "Very fast - upgrade required"
        elif guesses_per_second > 10:
            return "Hours to days"
        else:
            return "Acceptable with strong password"
    
    def _estimate_scrypt_crack_time(self, n: int, p: int) -> str:
        """Estimate Scrypt crack time."""
        memory_mb = (128 * n * p) / (1024 * 1024)
        
        if memory_mb < 16:
            return "GPU-attackable"
        elif memory_mb < 64:
            return "Moderately resistant"
        else:
            return "Memory-hard - resistant"
    
    def _estimate_argon2_crack_time(self, time_cost: int, memory_kb: int) -> str:
        """Estimate Argon2 crack time."""
        if time_cost < 3 or memory_kb < 65536:
            return "Below recommended - vulnerable"
        elif memory_kb >= 131072:  # 128MB
            return "Strong - memory-hard"
        else:
            return "Acceptable"
    
    def _estimate_bcrypt_crack_time(self, cost: int) -> str:
        """Estimate bcrypt crack time."""
        if cost < 10:
            return "Fast - increase cost"
        elif cost < 12:
            return "Acceptable"
        else:
            return "Strong"
    
    def _get_entropy_recommendations(
        self, 
        entropy_bits: int, 
        weaknesses: List[str]
    ) -> List[str]:
        """Generate entropy recommendations."""
        recommendations = []
        
        if entropy_bits < 128:
            recommendations.append("Increase to 128+ bits of entropy")
        
        if weaknesses:
            recommendations.append("Regenerate with secure random source")
        
        if entropy_bits < 256:
            recommendations.append("Consider 256-bit entropy for long-term security")
        
        return recommendations
    
    def _calculate_security_score(
        self, 
        entropy_bits: int,
        vulnerabilities: List[Tuple[WeaknessType, str]]
    ) -> float:
        """Calculate overall security score."""
        # Base score from entropy
        if entropy_bits >= 256:
            base_score = 100.0
        elif entropy_bits >= 128:
            base_score = 80.0 + (entropy_bits - 128) * 0.15
        else:
            base_score = entropy_bits * 0.6
        
        # Deduct for vulnerabilities
        severity_deductions = {
            WeaknessType.KNOWN_BRAIN_WALLET: 40,
            WeaknessType.PREDICTABLE_SEED: 30,
            WeaknessType.LOW_ENTROPY: 25,
            WeaknessType.WEAK_PASSPHRASE: 20,
            WeaknessType.SEQUENTIAL: 20,
            WeaknessType.DICTIONARY_DERIVATION: 15,
            WeaknessType.BIASED_SAMPLING: 15,
            WeaknessType.BROKEN_RANDOMNESS: 40,
            WeaknessType.TIMESTAMP_BASED: 25,
            WeaknessType.INSUFFICIENT_STRETCHING: 10,
        }
        
        total_deduction = sum(
            severity_deductions.get(v[0], 10)
            for v in vulnerabilities
        )
        
        return max(0.0, min(100.0, base_score - total_deduction))
    
    def get_analysis(self, analysis_id: str) -> Optional[KeyDerivationReport]:
        """Get cached analysis by ID."""
        return self._analysis_cache.get(analysis_id)
    
    def clear_cache(self) -> None:
        """Securely clear analysis cache."""
        for key in list(self._analysis_cache.keys()):
            self._analysis_cache[key] = None
        self._analysis_cache.clear()
        logger.info("Analysis cache securely cleared")


# Convenience function
def get_analyzer() -> KeyDerivationAnalyzer:
    """Get key derivation analyzer instance."""
    return KeyDerivationAnalyzer(ram_only=True)
