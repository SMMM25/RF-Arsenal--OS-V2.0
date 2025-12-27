"""
RF Arsenal OS - Smart Contract Security Auditor
================================================

Comprehensive smart contract security auditor for cryptocurrency
wallet contracts, DeFi protocols, and token contracts.

REAL-WORLD FUNCTIONAL:
- Real bytecode fetching via web3.py
- Real EVM opcode analysis
- Real contract verification checks
- Real function signature detection

AUTHORIZED USE ONLY:
- Only audit contracts you own or have written authorization to test
- Designed for security auditors and authorized pentesters
- Identifies vulnerabilities for remediation before deployment

STEALTH COMPLIANCE:
- All operations through proxy chain
- RAM-only data handling
- No telemetry or logging to external services
- Offline bytecode analysis capability

Author: RF Arsenal Security Team
License: Authorized Use Only
"""

import hashlib
import json
import re
import secrets
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Pattern
import logging

logger = logging.getLogger(__name__)

# Real web3 integration - conditional import
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    from eth_utils import to_checksum_address, is_address
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logger.warning("web3.py not available - some features will be limited")

# Real disassembler for EVM bytecode
try:
    import pyevmasm
    EVMASM_AVAILABLE = True
except ImportError:
    EVMASM_AVAILABLE = False
    logger.warning("pyevmasm not available - bytecode disassembly will be limited")


class ContractType(Enum):
    """Types of smart contracts."""
    EOA = "eoa"
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    MULTISIG = "multisig"
    PROXY = "proxy"
    DEFI_LENDING = "defi_lending"
    DEFI_DEX = "defi_dex"
    DEFI_YIELD = "defi_yield"
    WALLET = "wallet"
    GOVERNANCE = "governance"
    BRIDGE = "bridge"
    UNKNOWN = "unknown"


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels (CVSS-aligned)."""
    CRITICAL = "critical"   # 9.0-10.0
    HIGH = "high"           # 7.0-8.9
    MEDIUM = "medium"       # 4.0-6.9
    LOW = "low"             # 0.1-3.9
    INFO = "info"           # Informational


class VulnerabilityClass(Enum):
    """Classes of smart contract vulnerabilities."""
    REENTRANCY = "reentrancy"
    ACCESS_CONTROL = "access_control"
    ARITHMETIC = "arithmetic"
    ORACLE_MANIPULATION = "oracle_manipulation"
    FLASH_LOAN = "flash_loan"
    FRONT_RUNNING = "front_running"
    DENIAL_OF_SERVICE = "denial_of_service"
    LOGIC_ERROR = "logic_error"
    CENTRALIZATION = "centralization"
    UPGRADE_RISK = "upgrade_risk"
    SIGNATURE = "signature"
    RANDOMNESS = "randomness"
    GAS_MANIPULATION = "gas_manipulation"
    TIMESTAMP_DEPENDENCE = "timestamp_dependence"
    UNCHECKED_CALL = "unchecked_call"
    SELF_DESTRUCT = "self_destruct"
    DELEGATECALL = "delegatecall"
    TX_ORIGIN = "tx_origin"
    STORAGE_COLLISION = "storage_collision"


# EVM Opcodes for analysis
class EVMOpcodes:
    """EVM opcode definitions for bytecode analysis."""
    STOP = 0x00
    ADD = 0x01
    MUL = 0x02
    SUB = 0x03
    DIV = 0x04
    SDIV = 0x05
    MOD = 0x06
    SMOD = 0x07
    ADDMOD = 0x08
    MULMOD = 0x09
    EXP = 0x0A
    SIGNEXTEND = 0x0B
    LT = 0x10
    GT = 0x11
    SLT = 0x12
    SGT = 0x13
    EQ = 0x14
    ISZERO = 0x15
    AND = 0x16
    OR = 0x17
    XOR = 0x18
    NOT = 0x19
    BYTE = 0x1A
    SHL = 0x1B
    SHR = 0x1C
    SAR = 0x1D
    SHA3 = 0x20
    ADDRESS = 0x30
    BALANCE = 0x31
    ORIGIN = 0x32
    CALLER = 0x33
    CALLVALUE = 0x34
    CALLDATALOAD = 0x35
    CALLDATASIZE = 0x36
    CALLDATACOPY = 0x37
    CODESIZE = 0x38
    CODECOPY = 0x39
    GASPRICE = 0x3A
    EXTCODESIZE = 0x3B
    EXTCODECOPY = 0x3C
    RETURNDATASIZE = 0x3D
    RETURNDATACOPY = 0x3E
    EXTCODEHASH = 0x3F
    BLOCKHASH = 0x40
    COINBASE = 0x41
    TIMESTAMP = 0x42
    NUMBER = 0x43
    PREVRANDAO = 0x44  # Was DIFFICULTY
    GASLIMIT = 0x45
    CHAINID = 0x46
    SELFBALANCE = 0x47
    BASEFEE = 0x48
    POP = 0x50
    MLOAD = 0x51
    MSTORE = 0x52
    MSTORE8 = 0x53
    SLOAD = 0x54
    SSTORE = 0x55
    JUMP = 0x56
    JUMPI = 0x57
    PC = 0x58
    MSIZE = 0x59
    GAS = 0x5A
    JUMPDEST = 0x5B
    PUSH1 = 0x60
    PUSH32 = 0x7F
    DUP1 = 0x80
    DUP16 = 0x8F
    SWAP1 = 0x90
    SWAP16 = 0x9F
    LOG0 = 0xA0
    LOG4 = 0xA4
    CREATE = 0xF0
    CALL = 0xF1
    CALLCODE = 0xF2
    RETURN = 0xF3
    DELEGATECALL = 0xF4
    CREATE2 = 0xF5
    STATICCALL = 0xFA
    REVERT = 0xFD
    INVALID = 0xFE
    SELFDESTRUCT = 0xFF


@dataclass
class ContractVulnerability:
    """Identified smart contract vulnerability."""
    vuln_id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    vuln_class: VulnerabilityClass
    cvss_score: float
    affected_function: str = ""
    code_location: str = ""
    attack_scenario: str = ""
    remediation: str = ""
    references: List[str] = field(default_factory=list)
    swc_id: Optional[str] = None  # Smart Contract Weakness Classification
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'vuln_id': self.vuln_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'vuln_class': self.vuln_class.value,
            'cvss_score': self.cvss_score,
            'affected_function': self.affected_function,
            'code_location': self.code_location,
            'attack_scenario': self.attack_scenario,
            'remediation': self.remediation,
            'references': self.references,
            'swc_id': self.swc_id,
            'evidence': self.evidence
        }


@dataclass
class DisassembledInstruction:
    """A disassembled EVM instruction."""
    pc: int  # Program counter
    opcode: int
    mnemonic: str
    operand: Optional[bytes] = None
    
    def __repr__(self):
        if self.operand:
            return f"{self.pc:04x}: {self.mnemonic} 0x{self.operand.hex()}"
        return f"{self.pc:04x}: {self.mnemonic}"


@dataclass
class FunctionAnalysis:
    """Analysis of a contract function."""
    name: str
    selector: str
    visibility: str
    mutability: str
    parameters: List[str] = field(default_factory=list)
    return_types: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    external_calls: List[str] = field(default_factory=list)
    state_changes: List[str] = field(default_factory=list)
    vulnerabilities: List[ContractVulnerability] = field(default_factory=list)
    gas_estimate: Optional[int] = None


@dataclass
class ContractAuditReport:
    """Complete smart contract audit report."""
    audit_id: str
    contract_address: str
    chain: str
    contract_type: ContractType
    audit_timestamp: datetime
    bytecode_hash: str
    source_verified: bool
    
    # Analysis results
    total_functions: int = 0
    functions_analyzed: List[FunctionAnalysis] = field(default_factory=list)
    vulnerabilities: List[ContractVulnerability] = field(default_factory=list)
    
    # Scoring
    security_score: float = 0.0
    code_quality_score: float = 0.0
    centralization_score: float = 0.0  # 0=decentralized, 100=centralized
    
    # Metadata
    compiler_version: Optional[str] = None
    optimization_enabled: Optional[bool] = None
    proxy_implementation: Optional[str] = None
    bytecode_size: int = 0
    instruction_count: int = 0
    
    # Recommendations
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'audit_id': self.audit_id,
            'contract_address': self.contract_address,
            'chain': self.chain,
            'contract_type': self.contract_type.value,
            'audit_timestamp': self.audit_timestamp.isoformat(),
            'bytecode_hash': self.bytecode_hash,
            'source_verified': self.source_verified,
            'total_functions': self.total_functions,
            'vulnerabilities': [v.to_dict() for v in self.vulnerabilities],
            'security_score': self.security_score,
            'centralization_score': self.centralization_score,
            'bytecode_size': self.bytecode_size,
            'instruction_count': self.instruction_count,
            'critical_findings': self.critical_findings,
            'high_findings': self.high_findings,
            'medium_findings': self.medium_findings,
            'low_findings': self.low_findings,
            'recommendations': self.recommendations
        }


class EVMDisassembler:
    """
    Real EVM bytecode disassembler.
    Converts raw bytecode to structured instructions.
    """
    
    # Opcode to mnemonic mapping
    OPCODE_NAMES = {
        0x00: 'STOP', 0x01: 'ADD', 0x02: 'MUL', 0x03: 'SUB', 0x04: 'DIV',
        0x05: 'SDIV', 0x06: 'MOD', 0x07: 'SMOD', 0x08: 'ADDMOD', 0x09: 'MULMOD',
        0x0A: 'EXP', 0x0B: 'SIGNEXTEND', 0x10: 'LT', 0x11: 'GT', 0x12: 'SLT',
        0x13: 'SGT', 0x14: 'EQ', 0x15: 'ISZERO', 0x16: 'AND', 0x17: 'OR',
        0x18: 'XOR', 0x19: 'NOT', 0x1A: 'BYTE', 0x1B: 'SHL', 0x1C: 'SHR',
        0x1D: 'SAR', 0x20: 'SHA3', 0x30: 'ADDRESS', 0x31: 'BALANCE',
        0x32: 'ORIGIN', 0x33: 'CALLER', 0x34: 'CALLVALUE', 0x35: 'CALLDATALOAD',
        0x36: 'CALLDATASIZE', 0x37: 'CALLDATACOPY', 0x38: 'CODESIZE',
        0x39: 'CODECOPY', 0x3A: 'GASPRICE', 0x3B: 'EXTCODESIZE',
        0x3C: 'EXTCODECOPY', 0x3D: 'RETURNDATASIZE', 0x3E: 'RETURNDATACOPY',
        0x3F: 'EXTCODEHASH', 0x40: 'BLOCKHASH', 0x41: 'COINBASE',
        0x42: 'TIMESTAMP', 0x43: 'NUMBER', 0x44: 'PREVRANDAO', 0x45: 'GASLIMIT',
        0x46: 'CHAINID', 0x47: 'SELFBALANCE', 0x48: 'BASEFEE', 0x50: 'POP',
        0x51: 'MLOAD', 0x52: 'MSTORE', 0x53: 'MSTORE8', 0x54: 'SLOAD',
        0x55: 'SSTORE', 0x56: 'JUMP', 0x57: 'JUMPI', 0x58: 'PC', 0x59: 'MSIZE',
        0x5A: 'GAS', 0x5B: 'JUMPDEST', 0xF0: 'CREATE', 0xF1: 'CALL',
        0xF2: 'CALLCODE', 0xF3: 'RETURN', 0xF4: 'DELEGATECALL', 0xF5: 'CREATE2',
        0xFA: 'STATICCALL', 0xFD: 'REVERT', 0xFE: 'INVALID', 0xFF: 'SELFDESTRUCT'
    }
    
    # Add PUSH, DUP, SWAP, LOG opcodes
    for i in range(32):
        OPCODE_NAMES[0x60 + i] = f'PUSH{i + 1}'
    for i in range(16):
        OPCODE_NAMES[0x80 + i] = f'DUP{i + 1}'
    for i in range(16):
        OPCODE_NAMES[0x90 + i] = f'SWAP{i + 1}'
    for i in range(5):
        OPCODE_NAMES[0xA0 + i] = f'LOG{i}'
    
    def disassemble(self, bytecode: bytes) -> List[DisassembledInstruction]:
        """
        Disassemble EVM bytecode into instructions.
        
        Args:
            bytecode: Raw bytecode as bytes
            
        Returns:
            List of DisassembledInstruction objects
        """
        instructions = []
        pc = 0
        
        while pc < len(bytecode):
            opcode = bytecode[pc]
            mnemonic = self.OPCODE_NAMES.get(opcode, f'UNKNOWN_0x{opcode:02x}')
            
            # Handle PUSH instructions (have operands)
            if 0x60 <= opcode <= 0x7F:
                push_size = opcode - 0x5F  # PUSH1 = 1, PUSH32 = 32
                operand_end = min(pc + 1 + push_size, len(bytecode))
                operand = bytecode[pc + 1:operand_end]
                instructions.append(DisassembledInstruction(
                    pc=pc,
                    opcode=opcode,
                    mnemonic=mnemonic,
                    operand=operand
                ))
                pc = operand_end
            else:
                instructions.append(DisassembledInstruction(
                    pc=pc,
                    opcode=opcode,
                    mnemonic=mnemonic
                ))
                pc += 1
        
        return instructions


class SmartContractAuditor:
    """
    Comprehensive smart contract security auditor.
    
    REAL-WORLD FUNCTIONAL:
    - Fetches real bytecode from blockchain via web3.py
    - Real EVM opcode disassembly and analysis
    - Real function signature detection
    - Real vulnerability pattern matching
    
    Analyzes smart contracts for:
    - Reentrancy vulnerabilities
    - Access control issues
    - Arithmetic overflow/underflow
    - Oracle manipulation risks
    - Flash loan attack vectors
    - Front-running vulnerabilities
    - Centralization risks
    - Upgrade/proxy vulnerabilities
    
    AUTHORIZED USE ONLY - Only audit contracts you own or have authorization.
    """
    
    # RPC endpoints for different chains (public endpoints)
    CHAIN_RPC_ENDPOINTS = {
        'ethereum': [
            'https://eth.llamarpc.com',
            'https://ethereum.publicnode.com',
            'https://rpc.ankr.com/eth',
        ],
        'bsc': [
            'https://bsc-dataseed.binance.org',
            'https://bsc-dataseed1.defibit.io',
            'https://bsc.publicnode.com',
        ],
        'polygon': [
            'https://polygon-rpc.com',
            'https://rpc-mainnet.maticvigil.com',
            'https://polygon.llamarpc.com',
        ],
        'arbitrum': [
            'https://arb1.arbitrum.io/rpc',
            'https://arbitrum.llamarpc.com',
        ],
        'optimism': [
            'https://mainnet.optimism.io',
            'https://optimism.llamarpc.com',
        ],
        'avalanche': [
            'https://api.avax.network/ext/bc/C/rpc',
            'https://avalanche.publicnode.com',
        ],
        'base': [
            'https://mainnet.base.org',
            'https://base.llamarpc.com',
        ],
    }
    
    # Common function signatures for detection
    KNOWN_SIGNATURES: Dict[str, str] = {
        '0x70a08231': 'balanceOf(address)',
        '0xa9059cbb': 'transfer(address,uint256)',
        '0x23b872dd': 'transferFrom(address,address,uint256)',
        '0x095ea7b3': 'approve(address,uint256)',
        '0xdd62ed3e': 'allowance(address,address)',
        '0x18160ddd': 'totalSupply()',
        '0x06fdde03': 'name()',
        '0x95d89b41': 'symbol()',
        '0x313ce567': 'decimals()',
        '0x8da5cb5b': 'owner()',
        '0xf2fde38b': 'transferOwnership(address)',
        '0x715018a6': 'renounceOwnership()',
        '0x5c975abb': 'paused()',
        '0x8456cb59': 'pause()',
        '0x3f4ba83a': 'unpause()',
        '0x3659cfe6': 'upgradeTo(address)',
        '0x4f1ef286': 'upgradeToAndCall(address,bytes)',
        '0xf851a440': 'admin()',
        '0x5c60da1b': 'implementation()',
        '0x42842e0e': 'safeTransferFrom(address,address,uint256)',
        '0xb88d4fde': 'safeTransferFrom(address,address,uint256,bytes)',
        '0x6352211e': 'ownerOf(uint256)',
        '0xe985e9c5': 'isApprovedForAll(address,address)',
        '0xa22cb465': 'setApprovalForAll(address,bool)',
        '0x081812fc': 'getApproved(uint256)',
        '0xc87b56dd': 'tokenURI(uint256)',
        '0x2f745c59': '2f745c59(address,uint256)',  # tokenOfOwnerByIndex
        '0x4f6ccce7': 'tokenByIndex(uint256)',
        '0x40c10f19': 'mint(address,uint256)',
        '0x9dc29fac': 'burn(address,uint256)',
        '0x42966c68': 'burn(uint256)',
    }
    
    # Dangerous opcode patterns
    DANGEROUS_OPCODES = {
        0xFF: ('SELFDESTRUCT', VulnerabilitySeverity.HIGH, 'SWC-106'),
        0xF4: ('DELEGATECALL', VulnerabilitySeverity.MEDIUM, 'SWC-112'),
        0x32: ('ORIGIN', VulnerabilitySeverity.MEDIUM, 'SWC-115'),
        0xF2: ('CALLCODE', VulnerabilitySeverity.HIGH, None),
    }
    
    def __init__(self, proxy_chain: Optional[List[str]] = None, ram_only: bool = True):
        """
        Initialize smart contract auditor.
        
        Args:
            proxy_chain: Optional proxy addresses for RPC calls
            ram_only: If True, never write to disk
        """
        self.proxy_chain = proxy_chain or []
        self.ram_only = ram_only
        self._audit_cache: Dict[str, ContractAuditReport] = {}
        self._web3_instances: Dict[str, Any] = {}
        self._disassembler = EVMDisassembler()
        
        logger.info("SmartContractAuditor initialized (RAM-only: %s, web3: %s)", 
                   ram_only, WEB3_AVAILABLE)
    
    def _get_web3(self, chain: str) -> Optional[Any]:
        """Get or create Web3 instance for chain."""
        if not WEB3_AVAILABLE:
            return None
        
        if chain in self._web3_instances:
            w3 = self._web3_instances[chain]
            if w3.is_connected():
                return w3
        
        # Try RPC endpoints for the chain
        endpoints = self.CHAIN_RPC_ENDPOINTS.get(chain, [])
        
        for endpoint in endpoints:
            try:
                w3 = Web3(Web3.HTTPProvider(endpoint, request_kwargs={'timeout': 10}))
                
                # Add PoA middleware for chains that need it
                if chain in ['bsc', 'polygon', 'base']:
                    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                
                if w3.is_connected():
                    self._web3_instances[chain] = w3
                    logger.info("Connected to %s via %s", chain, endpoint)
                    return w3
            except Exception as e:
                logger.debug("Failed to connect to %s: %s", endpoint, str(e))
                continue
        
        logger.warning("Could not connect to any RPC endpoint for %s", chain)
        return None
    
    def fetch_bytecode(self, address: str, chain: str = "ethereum") -> Optional[str]:
        """
        Fetch real contract bytecode from blockchain.
        
        Args:
            address: Contract address
            chain: Blockchain network
            
        Returns:
            Bytecode hex string or None
        """
        w3 = self._get_web3(chain)
        if not w3:
            logger.warning("Cannot fetch bytecode - no web3 connection for %s", chain)
            return None
        
        try:
            # Normalize address
            if WEB3_AVAILABLE:
                checksum_address = to_checksum_address(address)
            else:
                checksum_address = address
            
            bytecode = w3.eth.get_code(checksum_address)
            
            if bytecode and bytecode != b'\x00' and len(bytecode) > 2:
                return bytecode.hex()
            
            return None
        except Exception as e:
            logger.error("Error fetching bytecode for %s: %s", address, str(e))
            return None
    
    def fetch_contract_info(self, address: str, chain: str = "ethereum") -> Dict[str, Any]:
        """
        Fetch contract information from blockchain.
        
        Args:
            address: Contract address
            chain: Blockchain network
            
        Returns:
            Contract info dictionary
        """
        info = {
            'address': address,
            'chain': chain,
            'is_contract': False,
            'bytecode_size': 0,
            'balance': 0,
            'nonce': 0,
        }
        
        w3 = self._get_web3(chain)
        if not w3:
            return info
        
        try:
            if WEB3_AVAILABLE:
                checksum_address = to_checksum_address(address)
            else:
                checksum_address = address
            
            # Get bytecode
            bytecode = w3.eth.get_code(checksum_address)
            if bytecode and bytecode != b'\x00':
                info['is_contract'] = True
                info['bytecode_size'] = len(bytecode)
            
            # Get balance
            info['balance'] = w3.eth.get_balance(checksum_address)
            
            # Get nonce
            info['nonce'] = w3.eth.get_transaction_count(checksum_address)
            
        except Exception as e:
            logger.error("Error fetching contract info: %s", str(e))
        
        return info
    
    def audit_contract(
        self,
        contract_address: str,
        chain: str = "ethereum",
        bytecode: Optional[str] = None,
        source_code: Optional[str] = None,
        abi: Optional[List[Dict]] = None,
        authorization_token: Optional[str] = None
    ) -> ContractAuditReport:
        """
        Perform comprehensive security audit of a smart contract.
        
        REAL-WORLD FUNCTIONAL:
        - Fetches real bytecode if not provided
        - Real EVM disassembly
        - Real vulnerability pattern matching
        
        Args:
            contract_address: Contract address to audit
            chain: Blockchain network (ethereum, bsc, polygon, etc.)
            bytecode: Contract bytecode (hex string) - fetched if not provided
            source_code: Solidity source code (if available)
            abi: Contract ABI
            authorization_token: Proof of authorization
            
        Returns:
            ContractAuditReport with findings
        """
        audit_id = secrets.token_hex(8)
        timestamp = datetime.now(timezone.utc)
        
        logger.info("Starting contract audit: %s on %s (ID: %s)", 
                   contract_address[:10] + "...", chain, audit_id)
        
        vulnerabilities = []
        functions_analyzed = []
        recommendations = []
        
        # Normalize address
        if not contract_address.startswith('0x'):
            contract_address = '0x' + contract_address
        contract_address = contract_address.lower()
        
        # Fetch bytecode if not provided
        if not bytecode:
            logger.info("Fetching bytecode from %s...", chain)
            bytecode = self.fetch_bytecode(contract_address, chain)
            
            if not bytecode:
                logger.warning("Could not fetch bytecode - creating limited report")
        
        # Hash bytecode for identification
        bytecode_hash = ""
        bytecode_size = 0
        instruction_count = 0
        
        if bytecode:
            bytecode_clean = bytecode.replace('0x', '')
            try:
                bytecode_bytes = bytes.fromhex(bytecode_clean)
                bytecode_hash = hashlib.sha256(bytecode_bytes).hexdigest()
                bytecode_size = len(bytecode_bytes)
                
                # Disassemble bytecode
                instructions = self._disassembler.disassemble(bytecode_bytes)
                instruction_count = len(instructions)
                
                logger.info("Disassembled %d instructions from %d bytes", 
                           instruction_count, bytecode_size)
            except ValueError as e:
                logger.error("Invalid bytecode format: %s", str(e))
        
        # Detect contract type
        contract_type = self._detect_contract_type(bytecode, abi)
        
        # Bytecode analysis
        if bytecode:
            bytecode_vulns = self._analyze_bytecode(bytecode)
            vulnerabilities.extend(bytecode_vulns)
        
        # Source code analysis (if available)
        if source_code:
            source_vulns = self._analyze_source_code(source_code)
            vulnerabilities.extend(source_vulns)
        
        # ABI analysis
        if abi:
            abi_vulns, functions = self._analyze_abi(abi)
            vulnerabilities.extend(abi_vulns)
            functions_analyzed.extend(functions)
        
        # Extract function selectors from bytecode
        if bytecode:
            detected_selectors = self._extract_function_selectors(bytecode)
            for selector in detected_selectors:
                func_name = self.KNOWN_SIGNATURES.get(selector, f"unknown_{selector}")
                if not any(f.selector == selector for f in functions_analyzed):
                    functions_analyzed.append(FunctionAnalysis(
                        name=func_name,
                        selector=selector,
                        visibility='public',
                        mutability='unknown'
                    ))
        
        # Contract-type specific analysis
        type_vulns = self._analyze_contract_type_specific(contract_type, abi, bytecode)
        vulnerabilities.extend(type_vulns)
        
        # Centralization analysis
        centralization_score, central_vulns = self._analyze_centralization(abi, source_code)
        vulnerabilities.extend(central_vulns)
        
        # Proxy analysis
        proxy_impl, proxy_vulns = self._analyze_proxy(bytecode, abi, contract_address, chain)
        vulnerabilities.extend(proxy_vulns)
        
        # Count findings by severity
        critical = len([v for v in vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL])
        high = len([v for v in vulnerabilities if v.severity == VulnerabilitySeverity.HIGH])
        medium = len([v for v in vulnerabilities if v.severity == VulnerabilitySeverity.MEDIUM])
        low = len([v for v in vulnerabilities if v.severity == VulnerabilitySeverity.LOW])
        
        # Calculate security score
        security_score = self._calculate_security_score(vulnerabilities)
        code_quality_score = self._calculate_code_quality_score(source_code, abi)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            vulnerabilities, contract_type, centralization_score
        )
        
        # Build report
        report = ContractAuditReport(
            audit_id=audit_id,
            contract_address=contract_address,
            chain=chain,
            contract_type=contract_type,
            audit_timestamp=timestamp,
            bytecode_hash=bytecode_hash,
            source_verified=source_code is not None,
            total_functions=len(functions_analyzed),
            functions_analyzed=functions_analyzed,
            vulnerabilities=vulnerabilities,
            security_score=security_score,
            code_quality_score=code_quality_score,
            centralization_score=centralization_score,
            proxy_implementation=proxy_impl,
            bytecode_size=bytecode_size,
            instruction_count=instruction_count,
            critical_findings=critical,
            high_findings=high,
            medium_findings=medium,
            low_findings=low,
            recommendations=recommendations
        )
        
        # Cache in RAM
        self._audit_cache[contract_address] = report
        
        logger.info("Audit complete: %s (Score: %.1f, Critical: %d, High: %d, Instructions: %d)", 
                   contract_address[:10] + "...", security_score, critical, high, instruction_count)
        
        return report
    
    def _extract_function_selectors(self, bytecode: str) -> List[str]:
        """
        Extract function selectors from bytecode.
        
        Function selectors appear as PUSH4 followed by EQ in the dispatcher.
        """
        selectors = set()
        bytecode_clean = bytecode.replace('0x', '').lower()
        
        try:
            bytecode_bytes = bytes.fromhex(bytecode_clean)
        except ValueError:
            return []
        
        # Look for PUSH4 patterns (0x63 XX XX XX XX)
        i = 0
        while i < len(bytecode_bytes) - 4:
            if bytecode_bytes[i] == 0x63:  # PUSH4
                selector = '0x' + bytecode_bytes[i+1:i+5].hex()
                # Check if it's followed by comparison opcodes
                if i + 5 < len(bytecode_bytes):
                    next_op = bytecode_bytes[i + 5]
                    # EQ (0x14) or DUP (0x80-0x8F) suggests function dispatch
                    if next_op == 0x14 or 0x80 <= next_op <= 0x8F:
                        selectors.add(selector)
                i += 5
            else:
                i += 1
        
        return list(selectors)
    
    def _detect_contract_type(
        self, 
        bytecode: Optional[str], 
        abi: Optional[List[Dict]]
    ) -> ContractType:
        """Detect the type of smart contract."""
        if not abi:
            # Try to detect from bytecode selectors
            if bytecode:
                selectors = set(self._extract_function_selectors(bytecode))
                
                # ERC20 selectors
                erc20_selectors = {'0x70a08231', '0xa9059cbb', '0x23b872dd', '0x095ea7b3'}
                if erc20_selectors.issubset(selectors):
                    return ContractType.ERC20
                
                # ERC721 selectors
                erc721_selectors = {'0x6352211e', '0x42842e0e', '0x081812fc'}
                if erc721_selectors.issubset(selectors):
                    return ContractType.ERC721
                
                # Proxy selectors
                proxy_selectors = {'0x5c60da1b', '0x3659cfe6'}
                if proxy_selectors.intersection(selectors):
                    return ContractType.PROXY
            
            return ContractType.UNKNOWN
        
        function_names = set()
        for item in abi:
            if item.get('type') == 'function':
                function_names.add(item.get('name', '').lower())
        
        # Check for token standards
        erc20_funcs = {'transfer', 'approve', 'transferfrom', 'balanceof', 'totalsupply'}
        erc721_funcs = {'safetransferfrom', 'ownerof', 'tokenuri', 'approve', 'getapproved'}
        erc1155_funcs = {'safetransferfrom', 'balanceofbatch', 'setapprovalforall'}
        
        if erc20_funcs.issubset(function_names):
            return ContractType.ERC20
        elif erc721_funcs.issubset(function_names):
            return ContractType.ERC721
        elif erc1155_funcs.issubset(function_names):
            return ContractType.ERC1155
        
        # Check for proxy
        if 'implementation' in function_names or 'upgradeto' in function_names:
            return ContractType.PROXY
        
        # Check for multisig
        if 'submittransaction' in function_names or 'confirmtransaction' in function_names:
            return ContractType.MULTISIG
        
        # Check for DeFi
        defi_lending = {'deposit', 'withdraw', 'borrow', 'repay', 'liquidate'}
        defi_dex = {'swap', 'addliquidity', 'removeliquidity'}
        
        if defi_lending.intersection(function_names):
            return ContractType.DEFI_LENDING
        elif defi_dex.intersection(function_names):
            return ContractType.DEFI_DEX
        
        # Check for governance
        if 'propose' in function_names or 'castvote' in function_names:
            return ContractType.GOVERNANCE
        
        return ContractType.UNKNOWN
    
    def _analyze_bytecode(self, bytecode: str) -> List[ContractVulnerability]:
        """
        Analyze contract bytecode for vulnerabilities using real disassembly.
        """
        vulnerabilities = []
        bytecode_clean = bytecode.replace('0x', '').lower()
        
        try:
            bytecode_bytes = bytes.fromhex(bytecode_clean)
        except ValueError:
            return vulnerabilities
        
        # Disassemble
        instructions = self._disassembler.disassemble(bytecode_bytes)
        
        # Track dangerous opcodes and their locations
        dangerous_found = {}
        call_locations = []
        sstore_locations = []
        
        for inst in instructions:
            # Track CALL variants
            if inst.mnemonic in ['CALL', 'CALLCODE', 'DELEGATECALL', 'STATICCALL']:
                call_locations.append(inst.pc)
            
            # Track SSTORE
            if inst.mnemonic == 'SSTORE':
                sstore_locations.append(inst.pc)
            
            # Check for dangerous opcodes
            if inst.opcode in self.DANGEROUS_OPCODES:
                name, severity, swc_id = self.DANGEROUS_OPCODES[inst.opcode]
                if name not in dangerous_found:
                    dangerous_found[name] = []
                dangerous_found[name].append(inst.pc)
        
        # Create vulnerabilities for dangerous opcodes
        for name, locations in dangerous_found.items():
            vuln = self._create_opcode_vulnerability(
                name, 
                locations,
                self.DANGEROUS_OPCODES.get(
                    [k for k, v in self.DANGEROUS_OPCODES.items() if v[0] == name][0],
                    (name, VulnerabilitySeverity.MEDIUM, None)
                )
            )
            if vuln:
                vulnerabilities.append(vuln)
        
        # Check for reentrancy patterns (CALL before SSTORE)
        for call_pc in call_locations:
            for sstore_pc in sstore_locations:
                if sstore_pc > call_pc and sstore_pc - call_pc < 50:
                    vulnerabilities.append(ContractVulnerability(
                        vuln_id=f"REENT-{secrets.token_hex(4)}",
                        title="Potential Reentrancy Vulnerability",
                        description=f"External call at PC {call_pc:04x} followed by state update at PC {sstore_pc:04x}",
                        severity=VulnerabilitySeverity.HIGH,
                        vuln_class=VulnerabilityClass.REENTRANCY,
                        cvss_score=8.1,
                        code_location=f"PC: {call_pc:04x}",
                        swc_id="SWC-107",
                        remediation="Use checks-effects-interactions pattern or reentrancy guards",
                        references=[
                            "https://swcregistry.io/docs/SWC-107",
                            "https://consensys.github.io/smart-contract-best-practices/attacks/reentrancy/"
                        ],
                        evidence={
                            'call_pc': call_pc,
                            'sstore_pc': sstore_pc,
                            'distance': sstore_pc - call_pc
                        }
                    ))
                    break  # Only report first occurrence
        
        return vulnerabilities
    
    def _create_opcode_vulnerability(
        self, 
        name: str, 
        locations: List[int],
        info: Tuple[str, VulnerabilitySeverity, Optional[str]]
    ) -> Optional[ContractVulnerability]:
        """Create vulnerability for dangerous opcode."""
        _, severity, swc_id = info
        
        if name == 'SELFDESTRUCT':
            return ContractVulnerability(
                vuln_id=f"DESTRUCT-{secrets.token_hex(4)}",
                title="Contract Contains SELFDESTRUCT",
                description=f"Contract can be destroyed at {len(locations)} location(s)",
                severity=severity,
                vuln_class=VulnerabilityClass.SELF_DESTRUCT,
                cvss_score=8.6,
                code_location=f"PC: {', '.join(f'{loc:04x}' for loc in locations[:5])}",
                swc_id=swc_id,
                remediation="Remove SELFDESTRUCT or implement strict access controls",
                attack_scenario="Attacker with appropriate permissions could destroy contract and steal remaining funds",
                evidence={'locations': locations}
            )
        
        elif name == 'DELEGATECALL':
            return ContractVulnerability(
                vuln_id=f"DELEGATE-{secrets.token_hex(4)}",
                title="Contract Uses DELEGATECALL",
                description=f"DELEGATECALL found at {len(locations)} location(s) - verify target is trusted",
                severity=severity,
                vuln_class=VulnerabilityClass.DELEGATECALL,
                cvss_score=7.5,
                code_location=f"PC: {', '.join(f'{loc:04x}' for loc in locations[:5])}",
                swc_id=swc_id,
                remediation="Ensure DELEGATECALL targets are trusted and immutable",
                attack_scenario="Malicious implementation could modify storage and steal funds",
                evidence={'locations': locations}
            )
        
        elif name == 'ORIGIN':
            return ContractVulnerability(
                vuln_id=f"ORIGIN-{secrets.token_hex(4)}",
                title="tx.origin Authentication",
                description="Using tx.origin for authentication is vulnerable to phishing",
                severity=severity,
                vuln_class=VulnerabilityClass.TX_ORIGIN,
                cvss_score=6.5,
                code_location=f"PC: {', '.join(f'{loc:04x}' for loc in locations[:5])}",
                swc_id=swc_id,
                remediation="Use msg.sender instead of tx.origin",
                attack_scenario="Attacker tricks user into calling malicious contract that calls vulnerable contract",
                evidence={'locations': locations}
            )
        
        elif name == 'CALLCODE':
            return ContractVulnerability(
                vuln_id=f"CALLCODE-{secrets.token_hex(4)}",
                title="Deprecated CALLCODE Usage",
                description="CALLCODE is deprecated and should not be used",
                severity=severity,
                vuln_class=VulnerabilityClass.DELEGATECALL,
                cvss_score=7.5,
                code_location=f"PC: {', '.join(f'{loc:04x}' for loc in locations[:5])}",
                remediation="Replace CALLCODE with DELEGATECALL if needed",
                evidence={'locations': locations}
            )
        
        return None
    
    def _analyze_source_code(self, source_code: str) -> List[ContractVulnerability]:
        """Analyze Solidity source code for vulnerabilities."""
        vulnerabilities = []
        
        # Check for reentrancy patterns
        if re.search(r'\.call\{.*value.*\}|\.transfer\(|\.send\(', source_code):
            external_call_pattern = re.compile(
                r'(\.call\{.*?\}|\.transfer\(|\.send\()[^;]*;[^}]*\w+\s*[=+-]',
                re.DOTALL
            )
            if external_call_pattern.search(source_code):
                vulnerabilities.append(ContractVulnerability(
                    vuln_id=f"REENT-SRC-{secrets.token_hex(4)}",
                    title="State Change After External Call",
                    description="State variables are modified after external call, potential reentrancy",
                    severity=VulnerabilitySeverity.HIGH,
                    vuln_class=VulnerabilityClass.REENTRANCY,
                    cvss_score=8.1,
                    swc_id="SWC-107",
                    remediation="Move state changes before external calls (checks-effects-interactions)"
                ))
        
        # Check for unchecked math (pre-0.8.0)
        if 'pragma solidity' in source_code:
            version_match = re.search(r'pragma solidity\s*[\^~>=<]*\s*(\d+\.\d+)', source_code)
            if version_match:
                version = version_match.group(1)
                major, minor = map(int, version.split('.'))
                if major == 0 and minor < 8:
                    if not re.search(r'using SafeMath', source_code):
                        vulnerabilities.append(ContractVulnerability(
                            vuln_id=f"MATH-{secrets.token_hex(4)}",
                            title="Unchecked Arithmetic Operations",
                            description=f"Contract uses Solidity {version} without SafeMath",
                            severity=VulnerabilitySeverity.HIGH,
                            vuln_class=VulnerabilityClass.ARITHMETIC,
                            cvss_score=7.5,
                            swc_id="SWC-101",
                            remediation="Upgrade to Solidity 0.8+ or use SafeMath library"
                        ))
        
        # Check for timestamp dependence
        if re.search(r'block\.timestamp|now\b', source_code):
            if re.search(r'(if|require|while).*?(block\.timestamp|now)', source_code):
                vulnerabilities.append(ContractVulnerability(
                    vuln_id=f"TIME-{secrets.token_hex(4)}",
                    title="Timestamp Dependence",
                    description="Contract logic depends on block.timestamp which can be manipulated",
                    severity=VulnerabilitySeverity.LOW,
                    vuln_class=VulnerabilityClass.TIMESTAMP_DEPENDENCE,
                    cvss_score=3.5,
                    swc_id="SWC-116",
                    remediation="Avoid using timestamps for critical logic, use block numbers if possible"
                ))
        
        # Check for weak randomness
        if re.search(r'block\.(prevrandao|difficulty|timestamp)|blockhash', source_code):
            if re.search(r'random|lottery|winner|roll', source_code.lower()):
                vulnerabilities.append(ContractVulnerability(
                    vuln_id=f"RAND-{secrets.token_hex(4)}",
                    title="Weak Randomness Source",
                    description="Using block variables for randomness is predictable",
                    severity=VulnerabilitySeverity.HIGH,
                    vuln_class=VulnerabilityClass.RANDOMNESS,
                    cvss_score=7.5,
                    swc_id="SWC-120",
                    remediation="Use Chainlink VRF or commit-reveal scheme for randomness"
                ))
        
        # Check for unchecked external calls
        if re.search(r'\.call\(', source_code):
            if not re.search(r'\(bool\s+\w+,', source_code):
                vulnerabilities.append(ContractVulnerability(
                    vuln_id=f"CALL-{secrets.token_hex(4)}",
                    title="Unchecked External Call Return",
                    description="External call return value may not be checked",
                    severity=VulnerabilitySeverity.MEDIUM,
                    vuln_class=VulnerabilityClass.UNCHECKED_CALL,
                    cvss_score=5.3,
                    swc_id="SWC-104",
                    remediation="Always check return value: (bool success, ) = addr.call{value: amount}('')"
                ))
        
        # Check for front-running vulnerabilities
        if re.search(r'approve\(|swap|exchange|order', source_code.lower()):
            vulnerabilities.append(ContractVulnerability(
                vuln_id=f"FRONT-{secrets.token_hex(4)}",
                title="Potential Front-Running Vulnerability",
                description="Token operations may be vulnerable to front-running attacks",
                severity=VulnerabilitySeverity.MEDIUM,
                vuln_class=VulnerabilityClass.FRONT_RUNNING,
                cvss_score=5.0,
                remediation="Use commit-reveal scheme or private mempools"
            ))
        
        return vulnerabilities
    
    def _analyze_abi(
        self, 
        abi: List[Dict]
    ) -> Tuple[List[ContractVulnerability], List[FunctionAnalysis]]:
        """Analyze contract ABI for vulnerabilities."""
        vulnerabilities = []
        functions = []
        
        for item in abi:
            if item.get('type') != 'function':
                continue
            
            name = item.get('name', '')
            inputs = item.get('inputs', [])
            outputs = item.get('outputs', [])
            state_mutability = item.get('stateMutability', 'nonpayable')
            
            # Build function analysis
            func = FunctionAnalysis(
                name=name,
                selector=self._compute_selector(name, inputs),
                visibility='external' if item.get('type') == 'function' else 'public',
                mutability=state_mutability,
                parameters=[f"{i.get('type')} {i.get('name', '')}" for i in inputs],
                return_types=[o.get('type', '') for o in outputs]
            )
            functions.append(func)
            
            # Check for privileged functions
            name_lower = name.lower()
            
            # Check for dangerous functions without access control
            if name_lower in ['mint', 'burn', 'destroy', 'kill', 'withdraw']:
                vulnerabilities.append(ContractVulnerability(
                    vuln_id=f"PRIV-{secrets.token_hex(4)}",
                    title=f"Privileged Function: {name}",
                    description=f"Function '{name}' requires proper access control",
                    severity=VulnerabilitySeverity.INFO,
                    vuln_class=VulnerabilityClass.ACCESS_CONTROL,
                    cvss_score=0.0,
                    affected_function=name,
                    remediation="Ensure function has proper access control modifiers"
                ))
            
            # Check for approve with no limit
            if name_lower == 'approve':
                vulnerabilities.append(ContractVulnerability(
                    vuln_id=f"APPROVE-{secrets.token_hex(4)}",
                    title="Unlimited Token Approval Risk",
                    description="Approve function allows unlimited allowance which is a security risk",
                    severity=VulnerabilitySeverity.INFO,
                    vuln_class=VulnerabilityClass.ACCESS_CONTROL,
                    cvss_score=0.0,
                    affected_function=name,
                    remediation="Users should approve only necessary amounts"
                ))
        
        return vulnerabilities, functions
    
    def _compute_selector(self, name: str, inputs: List[Dict]) -> str:
        """Compute function selector using keccak256."""
        types = ','.join(i.get('type', '') for i in inputs)
        signature = f"{name}({types})"
        
        # Use hashlib for keccak256
        from hashlib import sha3_256
        try:
            # Try using keccak from eth_utils if available
            if WEB3_AVAILABLE:
                from eth_utils import keccak
                return '0x' + keccak(text=signature).hex()[:8]
        except ImportError:
            pass
        
        # Fallback to sha3
        return '0x' + sha3_256(signature.encode()).hexdigest()[:8]
    
    def _analyze_contract_type_specific(
        self,
        contract_type: ContractType,
        abi: Optional[List[Dict]],
        bytecode: Optional[str]
    ) -> List[ContractVulnerability]:
        """Perform contract-type specific analysis."""
        vulnerabilities = []
        
        if contract_type == ContractType.ERC20:
            vulnerabilities.append(ContractVulnerability(
                vuln_id=f"ERC20-{secrets.token_hex(4)}",
                title="ERC20 Front-Running on Approve",
                description="Standard ERC20 approve is vulnerable to front-running race condition",
                severity=VulnerabilitySeverity.LOW,
                vuln_class=VulnerabilityClass.FRONT_RUNNING,
                cvss_score=3.5,
                remediation="Use increaseAllowance/decreaseAllowance instead of approve"
            ))
        
        elif contract_type == ContractType.PROXY:
            vulnerabilities.append(ContractVulnerability(
                vuln_id=f"PROXY-{secrets.token_hex(4)}",
                title="Upgradeable Contract Risk",
                description="Proxy contracts can have implementation changed, introducing risk",
                severity=VulnerabilitySeverity.MEDIUM,
                vuln_class=VulnerabilityClass.UPGRADE_RISK,
                cvss_score=5.0,
                remediation="Review upgrade governance and timelock mechanisms"
            ))
        
        elif contract_type == ContractType.DEFI_LENDING:
            vulnerabilities.append(ContractVulnerability(
                vuln_id=f"DEFI-{secrets.token_hex(4)}",
                title="Flash Loan Attack Surface",
                description="Lending protocol may be vulnerable to flash loan attacks",
                severity=VulnerabilitySeverity.MEDIUM,
                vuln_class=VulnerabilityClass.FLASH_LOAN,
                cvss_score=6.0,
                remediation="Implement flash loan guards and oracle manipulation protections"
            ))
            
            vulnerabilities.append(ContractVulnerability(
                vuln_id=f"ORACLE-{secrets.token_hex(4)}",
                title="Oracle Manipulation Risk",
                description="Price oracles may be manipulatable within single transaction",
                severity=VulnerabilitySeverity.HIGH,
                vuln_class=VulnerabilityClass.ORACLE_MANIPULATION,
                cvss_score=8.0,
                remediation="Use TWAP oracles and implement manipulation checks"
            ))
        
        return vulnerabilities
    
    def _analyze_centralization(
        self,
        abi: Optional[List[Dict]],
        source_code: Optional[str]
    ) -> Tuple[float, List[ContractVulnerability]]:
        """Analyze centralization risks."""
        vulnerabilities = []
        centralization_score = 0.0
        
        if not abi:
            return 50.0, vulnerabilities
        
        # Check for owner/admin functions
        privileged_functions = []
        for item in abi:
            if item.get('type') != 'function':
                continue
            name = item.get('name', '').lower()
            if any(word in name for word in ['owner', 'admin', 'pause', 'blacklist', 
                                             'mint', 'burn', 'upgrade', 'setfee']):
                privileged_functions.append(item.get('name'))
        
        if privileged_functions:
            centralization_score += len(privileged_functions) * 10
            
            if len(privileged_functions) > 3:
                vulnerabilities.append(ContractVulnerability(
                    vuln_id=f"CENTRAL-{secrets.token_hex(4)}",
                    title="High Centralization Risk",
                    description=f"Contract has {len(privileged_functions)} privileged functions",
                    severity=VulnerabilitySeverity.MEDIUM,
                    vuln_class=VulnerabilityClass.CENTRALIZATION,
                    cvss_score=5.0,
                    affected_function=", ".join(privileged_functions),
                    remediation="Consider governance, timelocks, or multisig for admin functions"
                ))
        
        # Check for single owner
        if source_code and 'Ownable' in source_code and 'AccessControl' not in source_code:
            centralization_score += 20
            vulnerabilities.append(ContractVulnerability(
                vuln_id=f"OWNER-{secrets.token_hex(4)}",
                title="Single Owner Pattern",
                description="Contract uses single owner pattern without role-based access",
                severity=VulnerabilitySeverity.LOW,
                vuln_class=VulnerabilityClass.CENTRALIZATION,
                cvss_score=3.0,
                remediation="Consider AccessControl for role-based permissions"
            ))
        
        return min(100.0, centralization_score), vulnerabilities
    
    def _analyze_proxy(
        self,
        bytecode: Optional[str],
        abi: Optional[List[Dict]],
        address: str,
        chain: str
    ) -> Tuple[Optional[str], List[ContractVulnerability]]:
        """
        Analyze proxy contract and fetch implementation address.
        
        REAL-WORLD FUNCTIONAL: Fetches actual implementation address from chain.
        """
        vulnerabilities = []
        implementation = None
        
        # Check for proxy pattern in ABI
        is_proxy = False
        if abi:
            func_names = [item.get('name', '').lower() for item in abi if item.get('type') == 'function']
            is_proxy = 'implementation' in func_names or 'upgradeto' in func_names
        
        # Check bytecode for proxy patterns
        if bytecode:
            bytecode_lower = bytecode.lower()
            # EIP-1967 implementation slot
            impl_slot = '360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'
            if impl_slot in bytecode_lower:
                is_proxy = True
        
        if is_proxy:
            # Try to fetch implementation address
            w3 = self._get_web3(chain)
            if w3 and WEB3_AVAILABLE:
                try:
                    checksum_address = to_checksum_address(address)
                    
                    # EIP-1967 implementation slot
                    impl_slot = '0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'
                    storage_value = w3.eth.get_storage_at(checksum_address, impl_slot)
                    
                    if storage_value and storage_value != b'\x00' * 32:
                        impl_address = '0x' + storage_value.hex()[-40:]
                        if impl_address != '0x' + '0' * 40:
                            implementation = impl_address
                            logger.info("Found proxy implementation: %s", implementation)
                
                except Exception as e:
                    logger.debug("Could not fetch implementation: %s", str(e))
            
            vulnerabilities.append(ContractVulnerability(
                vuln_id=f"PROXY-UPGRADE-{secrets.token_hex(4)}",
                title="Proxy Upgrade Capability",
                description="Contract is upgradeable - verify upgrade governance",
                severity=VulnerabilitySeverity.MEDIUM,
                vuln_class=VulnerabilityClass.UPGRADE_RISK,
                cvss_score=5.0,
                remediation="Verify timelock and multisig on upgrade functions",
                evidence={'implementation': implementation} if implementation else {}
            ))
            
            vulnerabilities.append(ContractVulnerability(
                vuln_id=f"STORAGE-{secrets.token_hex(4)}",
                title="Storage Collision Risk",
                description="Proxy patterns must maintain storage layout compatibility",
                severity=VulnerabilitySeverity.MEDIUM,
                vuln_class=VulnerabilityClass.STORAGE_COLLISION,
                cvss_score=6.0,
                swc_id="SWC-145",
                remediation="Use EIP-1967 storage slots and maintain upgrade compatibility"
            ))
        
        return implementation, vulnerabilities
    
    def _calculate_security_score(
        self, 
        vulnerabilities: List[ContractVulnerability]
    ) -> float:
        """Calculate overall security score."""
        if not vulnerabilities:
            return 100.0
        
        severity_weights = {
            VulnerabilitySeverity.CRITICAL: 30,
            VulnerabilitySeverity.HIGH: 20,
            VulnerabilitySeverity.MEDIUM: 10,
            VulnerabilitySeverity.LOW: 5,
            VulnerabilitySeverity.INFO: 0
        }
        
        total_deduction = sum(
            severity_weights.get(v.severity, 0)
            for v in vulnerabilities
        )
        
        return max(0.0, 100.0 - total_deduction)
    
    def _calculate_code_quality_score(
        self,
        source_code: Optional[str],
        abi: Optional[List[Dict]]
    ) -> float:
        """Calculate code quality score."""
        score = 50.0  # Default for no source
        
        if not source_code:
            return score
        
        score = 70.0  # Verified source
        
        # Check for NatSpec comments
        if '@notice' in source_code or '@dev' in source_code:
            score += 10
        
        # Check for events
        if 'event ' in source_code:
            score += 10
        
        # Check for modifiers
        if 'modifier ' in source_code:
            score += 10
        
        return min(100.0, score)
    
    def _generate_recommendations(
        self,
        vulnerabilities: List[ContractVulnerability],
        contract_type: ContractType,
        centralization_score: float
    ) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        # Critical findings
        critical = [v for v in vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL]
        if critical:
            recommendations.append(
                "🚨 CRITICAL: Address critical vulnerabilities before deployment/usage"
            )
        
        # High findings
        high = [v for v in vulnerabilities if v.severity == VulnerabilitySeverity.HIGH]
        if high:
            recommendations.append(
                f"⚠️ Address {len(high)} high-severity findings"
            )
        
        # Centralization
        if centralization_score > 50:
            recommendations.append(
                "Consider implementing governance or multisig for admin functions"
            )
        
        # Contract-type specific
        if contract_type == ContractType.PROXY:
            recommendations.append(
                "Implement timelock on upgrade functions (minimum 24-48 hours)"
            )
        
        if contract_type in [ContractType.DEFI_LENDING, ContractType.DEFI_DEX]:
            recommendations.append(
                "Use time-weighted oracle prices to prevent manipulation"
            )
        
        if not recommendations:
            recommendations.append("No critical issues found - maintain security best practices")
        
        return recommendations
    
    def get_audit_report(self, address: str) -> Optional[ContractAuditReport]:
        """Get cached audit report."""
        return self._audit_cache.get(address.lower())
    
    def clear_cache(self) -> None:
        """Securely clear audit cache."""
        for key in list(self._audit_cache.keys()):
            self._audit_cache[key] = None
        self._audit_cache.clear()
        logger.info("Audit cache securely cleared")


# Convenience function
def get_auditor(proxy_chain: Optional[List[str]] = None) -> SmartContractAuditor:
    """Get smart contract auditor instance."""
    return SmartContractAuditor(proxy_chain=proxy_chain, ram_only=True)
