"""
RF Arsenal OS - SUPERHERO Module
================================

Blockchain Intelligence & Identity Attribution Suite

Legal forensic analysis and OSINT-based identity attribution
for tracking stolen cryptocurrency and identifying criminals.

LEGAL NOTICE:
- All data gathered from PUBLIC sources only
- No unauthorized access to any systems
- Court-ready evidence with full chain of custody
- Designed for law enforcement collaboration

Components:
- Blockchain Forensics Engine (multi-chain tracing)
- Identity Correlation Engine (OSINT attribution)
- Geolocation & Behavioral Analysis
- Counter-Countermeasure Systems
- Evidence Dossier Generator

CRYPTOCURRENCY SECURITY ASSESSMENT & RECOVERY TOOLKIT:
- Wallet Security Scanner (authorized vulnerability assessment)
- Key Derivation Analyzer (weakness analysis)
- Smart Contract Auditor (exploit detection)
- Recovery Toolkit (client-owned wallet recovery)
- Malicious Address Database (known threat tracking)
- Authority Report Generator (law enforcement ready)

STEALTH COMPLIANCE:
- All operations through proxy chain
- RAM-only data handling
- No telemetry or logging
- Offline capability for analysis
"""

from .blockchain_forensics import (
    BlockchainForensics,
    ChainType,
    TransactionType,
    RiskLevel,
    Transaction,
    WalletCluster,
    MixerDetection,
    TransactionGraph,
)

from .identity_engine import (
    IdentityEngine,
    IdentityCorrelationEngine,
    IdentityLead,
    ConfidenceLevel,
    LeadSource,
    PersonProfile,
)

from .geolocation import (
    GeolocationAnalyzer,
    LocationEstimate,
    TimezoneAnalysis,
    BehavioralPattern,
)

from .counter_measures import (
    CounterMeasureAnalyzer,
    CounterMeasureSystem,
    MixerTracer,
    ChainHopTracker,
    PrivacyCoinExitDetector,
    WalletPersonaLinker,
    MixerTrace,
    ChainHopTrace,
    PersonaLink,
)

from .dossier_generator import (
    DossierGenerator,
    DossierFormat,
    ClassificationLevel,
    EvidenceType,
    EvidenceItem,
    IdentityAttribute,
    SuspectProfile,
    InvestigationSummary,
    IdentityDossier,
)

from .superhero_core import (
    SuperheroEngine,
    Investigation,
    InvestigationStatus,
    InvestigationTarget,
    Alert,
    AlertPriority,
    MonitoredAddress,
    SupportedChain,
    get_engine,
)

# Cryptocurrency Security Assessment & Recovery Toolkit
from .wallet_security_scanner import (
    WalletSecurityScanner,
    Vulnerability as WalletVulnerability,
    VulnerabilitySeverity,
    WalletType as ScannerWalletType,
    WalletSecurityProfile,
    get_scanner as create_scanner,
)

from .key_derivation_analyzer import (
    KeyDerivationAnalyzer,
    DerivationPathAnalysis as DerivationPath,
    CrackDifficulty as KeyStrength,
    KeyDerivationReport as DerivationAnalysis,
    get_analyzer as create_analyzer,
)

from .smart_contract_auditor import (
    SmartContractAuditor,
    ContractVulnerability,
    VulnerabilityClass as VulnerabilityType,
    ContractAuditReport as ContractAuditResult,
    VulnerabilitySeverity as AuditSeverity,
    get_auditor as create_auditor,
)

from .recovery_toolkit import (
    RecoveryToolkit,
    RecoveryMethod,
    RecoveryStatus,
    WalletType as RecoveryWalletType,
    AuthorizationLevel,
    AuthorizationDocument,
    RecoveryAttempt,
    RecoveredCredentials,
    SeedPhraseReconstructor,
    PasswordRecoveryEngine,
    MultisigRecoveryHandler,
    HardwareWalletRecoveryTool,
    create_recovery_toolkit,
    create_authorization,
)

from .malicious_address_db import (
    MaliciousAddressDatabase,
    MaliciousAddress,
    AddressCluster,
    SecurityProfile,
    ThreatLevel,
    AddressCategory,
    DataSource,
    Chain,
    AddressIntelligenceAggregator,
    create_database,
    quick_check,
)

from .authority_report_generator import (
    AuthorityReportGenerator,
    AuthorityReport,
    ReportType,
    EvidenceType as ReportEvidenceType,
    ClassificationLevel as ReportClassificationLevel,
    ChainOfCustodyAction,
    EvidenceItem as ReportEvidenceItem,
    TransactionEvidence,
    AddressEvidence,
    IdentityEvidence,
    CaseInformation,
    ReportSection,
    create_report_generator,
    generate_quick_report,
)

__all__ = [
    # Blockchain Forensics
    'BlockchainForensics',
    'ChainType',
    'TransactionType',
    'RiskLevel',
    'Transaction',
    'WalletCluster',
    'MixerDetection',
    'TransactionGraph',
    # Identity Engine
    'IdentityEngine',
    'IdentityCorrelationEngine',
    'IdentityLead',
    'ConfidenceLevel',
    'LeadSource',
    'PersonProfile',
    # Geolocation
    'GeolocationAnalyzer',
    'LocationEstimate',
    'TimezoneAnalysis',
    'BehavioralPattern',
    # Counter Measures
    'CounterMeasureAnalyzer',
    'CounterMeasureSystem',
    'MixerTracer',
    'ChainHopTracker',
    'PrivacyCoinExitDetector',
    'WalletPersonaLinker',
    'MixerTrace',
    'ChainHopTrace',
    'PersonaLink',
    # Dossier
    'DossierGenerator',
    'DossierFormat',
    'ClassificationLevel',
    'EvidenceType',
    'EvidenceItem',
    'IdentityAttribute',
    'SuspectProfile',
    'InvestigationSummary',
    'IdentityDossier',
    # Core
    'SuperheroEngine',
    'Investigation',
    'InvestigationStatus',
    'InvestigationTarget',
    'Alert',
    'AlertPriority',
    'MonitoredAddress',
    'SupportedChain',
    'get_engine',
    # Wallet Security Scanner
    'WalletSecurityScanner',
    'WalletVulnerability',
    'VulnerabilitySeverity',
    'ScannerWalletType',
    'WalletSecurityProfile',
    'create_scanner',
    # Key Derivation Analyzer
    'KeyDerivationAnalyzer',
    'DerivationPath',
    'KeyStrength',
    'DerivationAnalysis',
    'create_analyzer',
    # Smart Contract Auditor
    'SmartContractAuditor',
    'ContractVulnerability',
    'VulnerabilityType',
    'ContractAuditResult',
    'AuditSeverity',
    'create_auditor',
    # Recovery Toolkit
    'RecoveryToolkit',
    'RecoveryMethod',
    'RecoveryStatus',
    'RecoveryWalletType',
    'AuthorizationLevel',
    'AuthorizationDocument',
    'RecoveryAttempt',
    'RecoveredCredentials',
    'SeedPhraseReconstructor',
    'PasswordRecoveryEngine',
    'MultisigRecoveryHandler',
    'HardwareWalletRecoveryTool',
    'create_recovery_toolkit',
    'create_authorization',
    # Malicious Address Database
    'MaliciousAddressDatabase',
    'MaliciousAddress',
    'AddressCluster',
    'SecurityProfile',
    'ThreatLevel',
    'AddressCategory',
    'DataSource',
    'Chain',
    'AddressIntelligenceAggregator',
    'create_database',
    'quick_check',
    # Authority Report Generator
    'AuthorityReportGenerator',
    'AuthorityReport',
    'ReportType',
    'ReportEvidenceType',
    'ReportClassificationLevel',
    'ChainOfCustodyAction',
    'ReportEvidenceItem',
    'TransactionEvidence',
    'AddressEvidence',
    'IdentityEvidence',
    'CaseInformation',
    'ReportSection',
    'create_report_generator',
    'generate_quick_report',
]

__version__ = '2.0.0'
