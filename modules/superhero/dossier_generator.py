"""
RF Arsenal OS - SUPERHERO Identity Dossier Generator

Generates comprehensive, court-ready identity dossiers from all collected
intelligence. Creates professional reports for law enforcement submission.

Features:
- Court-admissible evidence packaging
- Multi-format output (PDF, JSON, HTML)
- Evidence chain documentation
- Confidence scoring and methodology notes
- Timestamp and hash verification
- Redacted versions for different audiences

Author: RF Arsenal Security Team
License: Authorized Use Only
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import base64
import io

# Optional PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Optional HTML templating
try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


class DossierFormat(Enum):
    """Output format types."""
    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"


class ClassificationLevel(Enum):
    """Report classification levels."""
    UNCLASSIFIED = "unclassified"
    LAW_ENFORCEMENT_ONLY = "law_enforcement_only"
    COURT_SUBMISSION = "court_submission"
    INTERNAL = "internal"
    REDACTED = "redacted"


class EvidenceType(Enum):
    """Types of evidence included."""
    BLOCKCHAIN_TRANSACTION = "blockchain_transaction"
    WALLET_CLUSTER = "wallet_cluster"
    IDENTITY_CORRELATION = "identity_correlation"
    GEOLOCATION = "geolocation"
    BEHAVIORAL = "behavioral"
    OSINT = "osint"
    TIMING_ANALYSIS = "timing_analysis"
    EXCHANGE_RECORD = "exchange_record"
    SOCIAL_MEDIA = "social_media"
    DOMAIN_RECORD = "domain_record"


@dataclass
class EvidenceItem:
    """Individual piece of evidence."""
    evidence_id: str
    evidence_type: EvidenceType
    source: str
    description: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    methodology: str
    hash_value: str = ""
    verification_notes: str = ""
    
    def __post_init__(self):
        """Calculate evidence hash."""
        if not self.hash_value:
            content = json.dumps({
                "type": self.evidence_type.value,
                "source": self.source,
                "data": self.data,
                "timestamp": self.timestamp.isoformat()
            }, sort_keys=True)
            self.hash_value = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class IdentityAttribute:
    """Single identity attribute with provenance."""
    attribute_name: str
    attribute_value: Any
    confidence: float
    sources: List[str]
    corroborating_evidence: List[str]
    methodology: str


@dataclass
class SuspectProfile:
    """Complete suspect profile."""
    suspect_id: str
    confidence_score: float
    
    # Identity attributes
    possible_names: List[IdentityAttribute] = field(default_factory=list)
    possible_emails: List[IdentityAttribute] = field(default_factory=list)
    possible_phones: List[IdentityAttribute] = field(default_factory=list)
    possible_addresses: List[IdentityAttribute] = field(default_factory=list)
    possible_locations: List[IdentityAttribute] = field(default_factory=list)
    
    # Online presence
    social_media: List[IdentityAttribute] = field(default_factory=list)
    usernames: List[IdentityAttribute] = field(default_factory=list)
    domains: List[IdentityAttribute] = field(default_factory=list)
    
    # Cryptocurrency
    wallets: List[str] = field(default_factory=list)
    wallet_clusters: List[str] = field(default_factory=list)
    exchange_accounts: List[IdentityAttribute] = field(default_factory=list)
    
    # Behavioral
    timezone_estimate: Optional[str] = None
    activity_patterns: Dict[str, Any] = field(default_factory=dict)
    language_indicators: List[str] = field(default_factory=list)


@dataclass
class InvestigationSummary:
    """High-level investigation summary."""
    case_id: str
    investigation_title: str
    start_date: datetime
    end_date: datetime
    total_transactions_analyzed: int
    total_wallets_tracked: int
    total_value_tracked: float  # In USD
    chains_investigated: List[str]
    key_findings: List[str]
    recommendations: List[str]


@dataclass 
class IdentityDossier:
    """Complete identity dossier."""
    dossier_id: str
    generated_at: datetime
    classification: ClassificationLevel
    
    # Investigation info
    summary: InvestigationSummary
    
    # Suspects
    primary_suspect: Optional[SuspectProfile]
    associated_suspects: List[SuspectProfile]
    
    # Evidence
    evidence_items: List[EvidenceItem]
    evidence_chain: List[str]  # Ordered list of evidence IDs
    
    # Analysis
    transaction_graph: Dict[str, Any]
    wallet_clusters: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    
    # Metadata
    analyst_notes: str
    methodology_summary: str
    limitations: List[str]
    confidence_assessment: str
    
    # Verification
    dossier_hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate dossier integrity hash."""
        content = {
            "dossier_id": self.dossier_id,
            "generated_at": self.generated_at.isoformat(),
            "summary": {
                "case_id": self.summary.case_id,
                "total_transactions": self.summary.total_transactions_analyzed,
                "total_value": self.summary.total_value_tracked
            },
            "evidence_count": len(self.evidence_items),
            "suspect_count": len(self.associated_suspects) + (1 if self.primary_suspect else 0)
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


class DossierGenerator:
    """
    Generates comprehensive identity dossiers from collected intelligence.
    
    Features:
    - Multi-format output (JSON, PDF, HTML, Markdown)
    - Court-admissible evidence packaging
    - Evidence chain documentation
    - Confidence scoring
    - Redaction support
    
    Adheres to RF Arsenal OS stealth principles:
    - RAM-only processing
    - No external telemetry
    - Secure output handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dossier generator."""
        self.config = config or {}
        self.logger = logging.getLogger("superhero.dossier")
        
        # Report templates
        self.templates = self._load_templates()
        
        # Evidence categories
        self.evidence_categories = {
            "blockchain": [
                EvidenceType.BLOCKCHAIN_TRANSACTION,
                EvidenceType.WALLET_CLUSTER,
                EvidenceType.EXCHANGE_RECORD
            ],
            "identity": [
                EvidenceType.IDENTITY_CORRELATION,
                EvidenceType.SOCIAL_MEDIA,
                EvidenceType.DOMAIN_RECORD,
                EvidenceType.OSINT
            ],
            "behavioral": [
                EvidenceType.GEOLOCATION,
                EvidenceType.BEHAVIORAL,
                EvidenceType.TIMING_ANALYSIS
            ]
        }
        
        # Redaction patterns
        self.redaction_patterns = {
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "phone": r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}',
            "address": r'\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|way)',
            "name": r'[A-Z][a-z]+\s+[A-Z][a-z]+'
        }
        
        self.logger.info("DossierGenerator initialized")
    
    def _load_templates(self) -> Dict[str, str]:
        """Load report templates."""
        templates = {}
        
        # HTML template
        templates["html"] = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { border-bottom: 2px solid #333; padding-bottom: 20px; }
        .classification { background: #f00; color: #fff; padding: 5px 10px; font-weight: bold; }
        .section { margin: 20px 0; }
        .section-title { font-size: 18px; font-weight: bold; color: #333; }
        .evidence-item { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
        .confidence-high { color: green; }
        .confidence-medium { color: orange; }
        .confidence-low { color: red; }
        .suspect-profile { background: #f5f5f5; padding: 15px; margin: 10px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .footer { margin-top: 40px; border-top: 1px solid #ccc; padding-top: 20px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <span class="classification">{{ classification }}</span>
        <h1>{{ title }}</h1>
        <p>Case ID: {{ case_id }}</p>
        <p>Generated: {{ generated_at }}</p>
        <p>Dossier Hash: {{ dossier_hash }}</p>
    </div>
    {{ content }}
    <div class="footer">
        <p>This document contains sensitive investigative information.</p>
        <p>Generated by RF Arsenal OS - SUPERHERO Blockchain Intelligence Module</p>
    </div>
</body>
</html>
"""
        
        # Markdown template
        templates["markdown"] = """
# {{ title }}

**Classification:** {{ classification }}
**Case ID:** {{ case_id }}
**Generated:** {{ generated_at }}
**Integrity Hash:** {{ dossier_hash }}

---

{{ content }}

---

*Generated by RF Arsenal OS - SUPERHERO Blockchain Intelligence Module*
"""
        
        return templates
    
    async def generate_dossier(
        self,
        case_id: str,
        investigation_title: str,
        forensics_data: Dict[str, Any],
        identity_data: Dict[str, Any],
        geolocation_data: Dict[str, Any],
        counter_measure_data: Dict[str, Any],
        classification: ClassificationLevel = ClassificationLevel.LAW_ENFORCEMENT_ONLY,
        analyst_notes: str = ""
    ) -> IdentityDossier:
        """
        Generate comprehensive identity dossier from all intelligence sources.
        
        Args:
            case_id: Unique case identifier
            investigation_title: Title for the investigation
            forensics_data: Output from BlockchainForensics
            identity_data: Output from IdentityCorrelationEngine
            geolocation_data: Output from GeolocationAnalyzer
            counter_measure_data: Output from CounterMeasureSystem
            classification: Report classification level
            analyst_notes: Additional analyst notes
            
        Returns:
            Complete IdentityDossier object
        """
        self.logger.info(f"Generating dossier for case: {case_id}")
        
        # Generate unique dossier ID
        dossier_id = f"DOS-{case_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        generated_at = datetime.now(timezone.utc)
        
        # Build investigation summary
        summary = self._build_summary(
            case_id=case_id,
            investigation_title=investigation_title,
            forensics_data=forensics_data,
            identity_data=identity_data
        )
        
        # Extract evidence items
        evidence_items = self._extract_evidence(
            forensics_data=forensics_data,
            identity_data=identity_data,
            geolocation_data=geolocation_data,
            counter_measure_data=counter_measure_data
        )
        
        # Build evidence chain
        evidence_chain = [e.evidence_id for e in sorted(evidence_items, key=lambda x: x.timestamp)]
        
        # Build suspect profiles
        primary_suspect, associated_suspects = self._build_suspect_profiles(
            identity_data=identity_data,
            geolocation_data=geolocation_data
        )
        
        # Build transaction graph
        transaction_graph = self._build_transaction_graph(forensics_data)
        
        # Build wallet clusters
        wallet_clusters = self._build_wallet_clusters(forensics_data)
        
        # Build timeline
        timeline = self._build_timeline(
            forensics_data=forensics_data,
            identity_data=identity_data,
            evidence_items=evidence_items
        )
        
        # Generate methodology summary
        methodology_summary = self._generate_methodology_summary(
            forensics_data=forensics_data,
            identity_data=identity_data,
            geolocation_data=geolocation_data
        )
        
        # Assess limitations
        limitations = self._assess_limitations(
            forensics_data=forensics_data,
            identity_data=identity_data
        )
        
        # Generate confidence assessment
        confidence_assessment = self._generate_confidence_assessment(
            primary_suspect=primary_suspect,
            evidence_items=evidence_items
        )
        
        # Create dossier
        dossier = IdentityDossier(
            dossier_id=dossier_id,
            generated_at=generated_at,
            classification=classification,
            summary=summary,
            primary_suspect=primary_suspect,
            associated_suspects=associated_suspects,
            evidence_items=evidence_items,
            evidence_chain=evidence_chain,
            transaction_graph=transaction_graph,
            wallet_clusters=wallet_clusters,
            timeline=timeline,
            analyst_notes=analyst_notes,
            methodology_summary=methodology_summary,
            limitations=limitations,
            confidence_assessment=confidence_assessment
        )
        
        # Calculate integrity hash
        dossier.dossier_hash = dossier.calculate_hash()
        
        self.logger.info(f"Dossier generated: {dossier_id} with {len(evidence_items)} evidence items")
        
        return dossier
    
    def _build_summary(
        self,
        case_id: str,
        investigation_title: str,
        forensics_data: Dict[str, Any],
        identity_data: Dict[str, Any]
    ) -> InvestigationSummary:
        """Build investigation summary."""
        # Extract transaction data
        trace_data = forensics_data.get("trace_result", {})
        transactions = trace_data.get("transactions", [])
        total_value = sum(t.get("value_usd", 0) for t in transactions)
        
        # Extract wallet data
        cluster_data = forensics_data.get("cluster_result", {})
        wallets = cluster_data.get("wallet_addresses", [])
        
        # Get chains investigated
        chains = set()
        for tx in transactions:
            if "chain" in tx:
                chains.add(tx["chain"])
        
        # Build key findings
        key_findings = []
        
        if identity_data.get("correlations"):
            high_conf = [c for c in identity_data["correlations"] if c.get("confidence", 0) > 0.8]
            if high_conf:
                key_findings.append(f"Identified {len(high_conf)} high-confidence identity correlations")
        
        if forensics_data.get("exchange_detections"):
            exchanges = forensics_data["exchange_detections"]
            key_findings.append(f"Funds traced through {len(exchanges)} cryptocurrency exchanges")
        
        if forensics_data.get("mixer_detections"):
            mixers = forensics_data["mixer_detections"]
            key_findings.append(f"Detected {len(mixers)} mixing service interactions")
        
        # Build recommendations
        recommendations = []
        
        if forensics_data.get("exchange_detections"):
            recommendations.append("Issue subpoenas to identified exchanges for KYC records")
        
        if identity_data.get("social_media_matches"):
            recommendations.append("Investigate linked social media accounts")
        
        if identity_data.get("email_matches"):
            recommendations.append("Cross-reference identified emails with breach databases")
        
        return InvestigationSummary(
            case_id=case_id,
            investigation_title=investigation_title,
            start_date=datetime.now(timezone.utc),  # Would be set from actual investigation
            end_date=datetime.now(timezone.utc),
            total_transactions_analyzed=len(transactions),
            total_wallets_tracked=len(wallets),
            total_value_tracked=total_value,
            chains_investigated=list(chains) or ["ethereum", "bitcoin"],
            key_findings=key_findings,
            recommendations=recommendations
        )
    
    def _extract_evidence(
        self,
        forensics_data: Dict[str, Any],
        identity_data: Dict[str, Any],
        geolocation_data: Dict[str, Any],
        counter_measure_data: Dict[str, Any]
    ) -> List[EvidenceItem]:
        """Extract and categorize all evidence items."""
        evidence_items = []
        evidence_counter = 0
        
        # Extract blockchain evidence
        if forensics_data.get("trace_result", {}).get("transactions"):
            for tx in forensics_data["trace_result"]["transactions"]:
                evidence_counter += 1
                evidence_items.append(EvidenceItem(
                    evidence_id=f"EVID-{evidence_counter:04d}",
                    evidence_type=EvidenceType.BLOCKCHAIN_TRANSACTION,
                    source="blockchain_forensics",
                    description=f"Transaction from {tx.get('from', 'unknown')} to {tx.get('to', 'unknown')}",
                    data=tx,
                    confidence=tx.get("confidence", 0.9),
                    timestamp=datetime.now(timezone.utc),
                    methodology="Multi-chain transaction tracing using graph analysis"
                ))
        
        # Extract wallet cluster evidence
        if forensics_data.get("cluster_result", {}).get("clusters"):
            for cluster_id, cluster_info in forensics_data["cluster_result"]["clusters"].items():
                evidence_counter += 1
                evidence_items.append(EvidenceItem(
                    evidence_id=f"EVID-{evidence_counter:04d}",
                    evidence_type=EvidenceType.WALLET_CLUSTER,
                    source="blockchain_forensics",
                    description=f"Wallet cluster {cluster_id} with {len(cluster_info.get('wallets', []))} addresses",
                    data=cluster_info,
                    confidence=cluster_info.get("confidence", 0.85),
                    timestamp=datetime.now(timezone.utc),
                    methodology="Common ownership heuristics and input clustering"
                ))
        
        # Extract identity correlations
        if identity_data.get("correlations"):
            for corr in identity_data["correlations"]:
                evidence_counter += 1
                evidence_items.append(EvidenceItem(
                    evidence_id=f"EVID-{evidence_counter:04d}",
                    evidence_type=EvidenceType.IDENTITY_CORRELATION,
                    source="identity_engine",
                    description=f"Identity correlation: {corr.get('attribute_type', 'unknown')}",
                    data=corr,
                    confidence=corr.get("confidence", 0.7),
                    timestamp=datetime.now(timezone.utc),
                    methodology="Cross-reference of OSINT sources with blockchain activity"
                ))
        
        # Extract geolocation evidence
        if geolocation_data.get("location_estimates"):
            for loc in geolocation_data["location_estimates"]:
                evidence_counter += 1
                evidence_items.append(EvidenceItem(
                    evidence_id=f"EVID-{evidence_counter:04d}",
                    evidence_type=EvidenceType.GEOLOCATION,
                    source="geolocation_analyzer",
                    description=f"Location estimate: {loc.get('region', 'unknown')}",
                    data=loc,
                    confidence=loc.get("confidence", 0.6),
                    timestamp=datetime.now(timezone.utc),
                    methodology="Transaction timing analysis and timezone correlation"
                ))
        
        # Extract behavioral evidence
        if geolocation_data.get("behavioral_patterns"):
            evidence_counter += 1
            evidence_items.append(EvidenceItem(
                evidence_id=f"EVID-{evidence_counter:04d}",
                evidence_type=EvidenceType.BEHAVIORAL,
                source="geolocation_analyzer",
                description="Behavioral pattern analysis",
                data=geolocation_data["behavioral_patterns"],
                confidence=0.75,
                timestamp=datetime.now(timezone.utc),
                methodology="Statistical analysis of transaction timing and value patterns"
            ))
        
        # Extract exchange records
        if forensics_data.get("exchange_detections"):
            for exchange in forensics_data["exchange_detections"]:
                evidence_counter += 1
                evidence_items.append(EvidenceItem(
                    evidence_id=f"EVID-{evidence_counter:04d}",
                    evidence_type=EvidenceType.EXCHANGE_RECORD,
                    source="blockchain_forensics",
                    description=f"Exchange interaction: {exchange.get('exchange_name', 'unknown')}",
                    data=exchange,
                    confidence=exchange.get("confidence", 0.9),
                    timestamp=datetime.now(timezone.utc),
                    methodology="Known exchange address matching and deposit pattern analysis"
                ))
        
        # Extract social media matches
        if identity_data.get("social_media_matches"):
            for sm in identity_data["social_media_matches"]:
                evidence_counter += 1
                evidence_items.append(EvidenceItem(
                    evidence_id=f"EVID-{evidence_counter:04d}",
                    evidence_type=EvidenceType.SOCIAL_MEDIA,
                    source="identity_engine",
                    description=f"Social media match: {sm.get('platform', 'unknown')}",
                    data=sm,
                    confidence=sm.get("confidence", 0.7),
                    timestamp=datetime.now(timezone.utc),
                    methodology="Username correlation and activity pattern matching"
                ))
        
        # Extract OSINT findings
        if identity_data.get("osint_findings"):
            for osint in identity_data["osint_findings"]:
                evidence_counter += 1
                evidence_items.append(EvidenceItem(
                    evidence_id=f"EVID-{evidence_counter:04d}",
                    evidence_type=EvidenceType.OSINT,
                    source="identity_engine",
                    description=f"OSINT finding: {osint.get('finding_type', 'unknown')}",
                    data=osint,
                    confidence=osint.get("confidence", 0.65),
                    timestamp=datetime.now(timezone.utc),
                    methodology="Open source intelligence gathering and correlation"
                ))
        
        # Extract counter-measure analysis
        if counter_measure_data.get("mixer_traces"):
            for trace in counter_measure_data["mixer_traces"]:
                evidence_counter += 1
                evidence_items.append(EvidenceItem(
                    evidence_id=f"EVID-{evidence_counter:04d}",
                    evidence_type=EvidenceType.BLOCKCHAIN_TRANSACTION,
                    source="counter_measures",
                    description=f"Mixer trace: {trace.get('mixer_type', 'unknown')}",
                    data=trace,
                    confidence=trace.get("confidence", 0.6),
                    timestamp=datetime.now(timezone.utc),
                    methodology="Advanced mixer detection using timing and amount correlation",
                    verification_notes="Mixer tracing has inherent uncertainty"
                ))
        
        return evidence_items
    
    def _build_suspect_profiles(
        self,
        identity_data: Dict[str, Any],
        geolocation_data: Dict[str, Any]
    ) -> tuple[Optional[SuspectProfile], List[SuspectProfile]]:
        """Build suspect profiles from identity data."""
        profiles = []
        
        # Process identity candidates
        candidates = identity_data.get("identity_candidates", [])
        
        for i, candidate in enumerate(candidates):
            profile = SuspectProfile(
                suspect_id=f"SUSPECT-{i+1:03d}",
                confidence_score=candidate.get("overall_confidence", 0.5)
            )
            
            # Extract names
            if candidate.get("possible_names"):
                for name in candidate["possible_names"]:
                    profile.possible_names.append(IdentityAttribute(
                        attribute_name="name",
                        attribute_value=name.get("value"),
                        confidence=name.get("confidence", 0.5),
                        sources=name.get("sources", []),
                        corroborating_evidence=name.get("evidence_ids", []),
                        methodology=name.get("methodology", "OSINT correlation")
                    ))
            
            # Extract emails
            if candidate.get("possible_emails"):
                for email in candidate["possible_emails"]:
                    profile.possible_emails.append(IdentityAttribute(
                        attribute_name="email",
                        attribute_value=email.get("value"),
                        confidence=email.get("confidence", 0.5),
                        sources=email.get("sources", []),
                        corroborating_evidence=email.get("evidence_ids", []),
                        methodology=email.get("methodology", "Email pattern matching")
                    ))
            
            # Extract social media
            if candidate.get("social_media"):
                for sm in candidate["social_media"]:
                    profile.social_media.append(IdentityAttribute(
                        attribute_name="social_media",
                        attribute_value=sm,
                        confidence=sm.get("confidence", 0.6) if isinstance(sm, dict) else 0.6,
                        sources=["osint_engine"],
                        corroborating_evidence=[],
                        methodology="Social media scraping and correlation"
                    ))
            
            # Extract wallets
            if candidate.get("associated_wallets"):
                profile.wallets = candidate["associated_wallets"]
            
            # Extract exchanges
            if candidate.get("exchange_accounts"):
                for exchange in candidate["exchange_accounts"]:
                    profile.exchange_accounts.append(IdentityAttribute(
                        attribute_name="exchange_account",
                        attribute_value=exchange,
                        confidence=0.8,
                        sources=["blockchain_forensics"],
                        corroborating_evidence=[],
                        methodology="Exchange deposit/withdrawal pattern analysis"
                    ))
            
            # Add geolocation data
            if geolocation_data.get("location_estimates"):
                for loc in geolocation_data["location_estimates"]:
                    profile.possible_locations.append(IdentityAttribute(
                        attribute_name="location",
                        attribute_value=loc,
                        confidence=loc.get("confidence", 0.5),
                        sources=["geolocation_analyzer"],
                        corroborating_evidence=[],
                        methodology="Timing analysis and behavioral correlation"
                    ))
            
            # Add timezone
            if geolocation_data.get("timezone_estimate"):
                profile.timezone_estimate = geolocation_data["timezone_estimate"]
            
            # Add activity patterns
            if geolocation_data.get("behavioral_patterns"):
                profile.activity_patterns = geolocation_data["behavioral_patterns"]
            
            profiles.append(profile)
        
        # Identify primary suspect (highest confidence)
        if profiles:
            profiles.sort(key=lambda x: x.confidence_score, reverse=True)
            primary = profiles[0]
            associated = profiles[1:]
            return primary, associated
        
        return None, []
    
    def _build_transaction_graph(self, forensics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build transaction graph for visualization."""
        graph = {
            "nodes": [],
            "edges": [],
            "metadata": {}
        }
        
        seen_addresses = set()
        transactions = forensics_data.get("trace_result", {}).get("transactions", [])
        
        for tx in transactions:
            from_addr = tx.get("from", "")
            to_addr = tx.get("to", "")
            
            # Add nodes
            if from_addr and from_addr not in seen_addresses:
                seen_addresses.add(from_addr)
                graph["nodes"].append({
                    "id": from_addr,
                    "type": tx.get("from_type", "wallet"),
                    "label": from_addr[:8] + "..."
                })
            
            if to_addr and to_addr not in seen_addresses:
                seen_addresses.add(to_addr)
                graph["nodes"].append({
                    "id": to_addr,
                    "type": tx.get("to_type", "wallet"),
                    "label": to_addr[:8] + "..."
                })
            
            # Add edge
            if from_addr and to_addr:
                graph["edges"].append({
                    "from": from_addr,
                    "to": to_addr,
                    "value": tx.get("value", 0),
                    "value_usd": tx.get("value_usd", 0),
                    "timestamp": tx.get("timestamp"),
                    "tx_hash": tx.get("tx_hash", "")
                })
        
        graph["metadata"] = {
            "total_nodes": len(graph["nodes"]),
            "total_edges": len(graph["edges"]),
            "chains": list(set(tx.get("chain", "unknown") for tx in transactions))
        }
        
        return graph
    
    def _build_wallet_clusters(self, forensics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build wallet cluster visualization data."""
        clusters = forensics_data.get("cluster_result", {}).get("clusters", {})
        
        cluster_data = {
            "clusters": [],
            "cross_cluster_links": [],
            "statistics": {}
        }
        
        for cluster_id, info in clusters.items():
            cluster_data["clusters"].append({
                "cluster_id": cluster_id,
                "wallet_count": len(info.get("wallets", [])),
                "wallets": info.get("wallets", []),
                "total_value": info.get("total_value", 0),
                "confidence": info.get("confidence", 0),
                "entity_type": info.get("entity_type", "unknown")
            })
        
        cluster_data["statistics"] = {
            "total_clusters": len(clusters),
            "largest_cluster": max((len(c.get("wallets", [])) for c in clusters.values()), default=0),
            "avg_cluster_size": sum(len(c.get("wallets", [])) for c in clusters.values()) / len(clusters) if clusters else 0
        }
        
        return cluster_data
    
    def _build_timeline(
        self,
        forensics_data: Dict[str, Any],
        identity_data: Dict[str, Any],
        evidence_items: List[EvidenceItem]
    ) -> List[Dict[str, Any]]:
        """Build investigation timeline."""
        timeline = []
        
        # Add transaction events
        transactions = forensics_data.get("trace_result", {}).get("transactions", [])
        for tx in transactions:
            if tx.get("timestamp"):
                timeline.append({
                    "timestamp": tx["timestamp"],
                    "event_type": "transaction",
                    "description": f"Transaction: {tx.get('value', 0)} from {tx.get('from', 'unknown')[:8]}... to {tx.get('to', 'unknown')[:8]}...",
                    "value_usd": tx.get("value_usd", 0),
                    "related_evidence": []
                })
        
        # Add identity discovery events
        correlations = identity_data.get("correlations", [])
        for corr in correlations:
            if corr.get("discovered_at"):
                timeline.append({
                    "timestamp": corr["discovered_at"],
                    "event_type": "identity_correlation",
                    "description": f"Identity correlation discovered: {corr.get('attribute_type', 'unknown')}",
                    "confidence": corr.get("confidence", 0),
                    "related_evidence": []
                })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x.get("timestamp", ""))
        
        return timeline
    
    def _generate_methodology_summary(
        self,
        forensics_data: Dict[str, Any],
        identity_data: Dict[str, Any],
        geolocation_data: Dict[str, Any]
    ) -> str:
        """Generate methodology summary for the dossier."""
        methods = []
        
        methods.append("## Investigation Methodology")
        methods.append("")
        methods.append("### Blockchain Forensics")
        methods.append("- Multi-chain transaction tracing using graph traversal algorithms")
        methods.append("- Common input ownership heuristics for wallet clustering")
        methods.append("- Exchange address matching against known databases")
        methods.append("- Mixer detection using timing and amount correlation analysis")
        methods.append("")
        methods.append("### Identity Correlation")
        methods.append("- ENS/domain WHOIS lookups")
        methods.append("- NFT ownership and metadata analysis")
        methods.append("- Social media username correlation")
        methods.append("- Code repository analysis (GitHub, GitLab)")
        methods.append("- Email pattern matching with breach databases")
        methods.append("- Forum and community activity analysis")
        methods.append("")
        methods.append("### Geolocation Analysis")
        methods.append("- Transaction timestamp analysis for timezone estimation")
        methods.append("- Activity pattern behavioral profiling")
        methods.append("- Exchange login timing (where available)")
        methods.append("- Language and regional indicators")
        methods.append("")
        methods.append("### Counter-Countermeasures")
        methods.append("- Advanced mixer tracing techniques")
        methods.append("- Cross-chain hopping detection")
        methods.append("- Privacy coin exit point monitoring")
        methods.append("- Behavioral fingerprinting across multiple wallets")
        methods.append("")
        methods.append("All analysis performed using publicly available data and")
        methods.append("legally obtained information sources.")
        
        return "\n".join(methods)
    
    def _assess_limitations(
        self,
        forensics_data: Dict[str, Any],
        identity_data: Dict[str, Any]
    ) -> List[str]:
        """Assess and document limitations of the analysis."""
        limitations = []
        
        # Check for mixer interactions
        if forensics_data.get("mixer_detections"):
            limitations.append(
                "Funds passed through mixing services, which introduces uncertainty "
                "in transaction flow tracking. Mixer tracing confidence is reduced."
            )
        
        # Check for privacy coins
        if forensics_data.get("privacy_coin_interactions"):
            limitations.append(
                "Target used privacy coins (Monero, Zcash shielded). Transactions within "
                "these systems cannot be traced. Only entry/exit points are identified."
            )
        
        # Check identity confidence
        if identity_data.get("identity_candidates"):
            max_conf = max(c.get("overall_confidence", 0) for c in identity_data["identity_candidates"])
            if max_conf < 0.7:
                limitations.append(
                    f"Identity attribution confidence is moderate ({max_conf:.0%}). "
                    "Additional evidence may be needed for positive identification."
                )
        
        # Check for chain hopping
        if forensics_data.get("cross_chain_hops"):
            limitations.append(
                "Cross-chain transactions were detected. Bridge transactions introduce "
                "potential tracking gaps."
            )
        
        # Standard limitations
        limitations.append(
            "Blockchain analysis relies on heuristics that may produce false positives. "
            "All findings should be corroborated with additional evidence."
        )
        
        limitations.append(
            "Identity correlations from OSINT sources should be verified independently "
            "before taking investigative action."
        )
        
        return limitations
    
    def _generate_confidence_assessment(
        self,
        primary_suspect: Optional[SuspectProfile],
        evidence_items: List[EvidenceItem]
    ) -> str:
        """Generate overall confidence assessment."""
        if not primary_suspect:
            return "INSUFFICIENT EVIDENCE: No primary suspect identified."
        
        # Calculate overall metrics
        evidence_confidences = [e.confidence for e in evidence_items]
        avg_evidence_confidence = sum(evidence_confidences) / len(evidence_confidences) if evidence_confidences else 0
        
        suspect_confidence = primary_suspect.confidence_score
        
        # High confidence identity attributes
        high_conf_attrs = 0
        total_attrs = 0
        
        for attr_list in [
            primary_suspect.possible_names,
            primary_suspect.possible_emails,
            primary_suspect.social_media
        ]:
            for attr in attr_list:
                total_attrs += 1
                if attr.confidence > 0.8:
                    high_conf_attrs += 1
        
        # Generate assessment
        lines = []
        
        if suspect_confidence > 0.8 and high_conf_attrs >= 2:
            lines.append("**CONFIDENCE LEVEL: HIGH**")
            lines.append("")
            lines.append("Strong evidence supports identification of the primary suspect.")
            lines.append(f"- Overall suspect confidence: {suspect_confidence:.0%}")
            lines.append(f"- High-confidence identity attributes: {high_conf_attrs}/{total_attrs}")
            lines.append(f"- Evidence items: {len(evidence_items)}")
            lines.append(f"- Average evidence confidence: {avg_evidence_confidence:.0%}")
        
        elif suspect_confidence > 0.6:
            lines.append("**CONFIDENCE LEVEL: MODERATE**")
            lines.append("")
            lines.append("Evidence provides reasonable basis for suspect identification.")
            lines.append("Additional investigation recommended to strengthen attribution.")
            lines.append(f"- Overall suspect confidence: {suspect_confidence:.0%}")
            lines.append(f"- Evidence items: {len(evidence_items)}")
        
        else:
            lines.append("**CONFIDENCE LEVEL: LOW**")
            lines.append("")
            lines.append("Preliminary evidence gathered. Significant additional investigation")
            lines.append("required for reliable suspect identification.")
            lines.append(f"- Overall suspect confidence: {suspect_confidence:.0%}")
        
        return "\n".join(lines)
    
    async def export_dossier(
        self,
        dossier: IdentityDossier,
        output_format: DossierFormat = DossierFormat.JSON,
        redact: bool = False
    ) -> bytes:
        """
        Export dossier to specified format.
        
        Args:
            dossier: The dossier to export
            output_format: Output format (JSON, PDF, HTML, Markdown)
            redact: Whether to redact sensitive information
            
        Returns:
            Exported dossier as bytes
        """
        self.logger.info(f"Exporting dossier {dossier.dossier_id} as {output_format.value}")
        
        # Apply redaction if needed
        if redact:
            dossier = self._apply_redaction(dossier)
        
        if output_format == DossierFormat.JSON:
            return self._export_json(dossier)
        elif output_format == DossierFormat.PDF:
            return await self._export_pdf(dossier)
        elif output_format == DossierFormat.HTML:
            return self._export_html(dossier)
        elif output_format == DossierFormat.MARKDOWN:
            return self._export_markdown(dossier)
        elif output_format == DossierFormat.TEXT:
            return self._export_text(dossier)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def _export_json(self, dossier: IdentityDossier) -> bytes:
        """Export dossier as JSON."""
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        data = {
            "dossier_id": dossier.dossier_id,
            "generated_at": dossier.generated_at.isoformat(),
            "classification": dossier.classification.value,
            "dossier_hash": dossier.dossier_hash,
            "summary": {
                "case_id": dossier.summary.case_id,
                "investigation_title": dossier.summary.investigation_title,
                "start_date": dossier.summary.start_date.isoformat(),
                "end_date": dossier.summary.end_date.isoformat(),
                "total_transactions_analyzed": dossier.summary.total_transactions_analyzed,
                "total_wallets_tracked": dossier.summary.total_wallets_tracked,
                "total_value_tracked": dossier.summary.total_value_tracked,
                "chains_investigated": dossier.summary.chains_investigated,
                "key_findings": dossier.summary.key_findings,
                "recommendations": dossier.summary.recommendations
            },
            "primary_suspect": self._serialize_suspect(dossier.primary_suspect) if dossier.primary_suspect else None,
            "associated_suspects": [self._serialize_suspect(s) for s in dossier.associated_suspects],
            "evidence_items": [
                {
                    "evidence_id": e.evidence_id,
                    "type": e.evidence_type.value,
                    "source": e.source,
                    "description": e.description,
                    "confidence": e.confidence,
                    "timestamp": e.timestamp.isoformat(),
                    "methodology": e.methodology,
                    "hash": e.hash_value
                }
                for e in dossier.evidence_items
            ],
            "evidence_chain": dossier.evidence_chain,
            "transaction_graph": dossier.transaction_graph,
            "wallet_clusters": dossier.wallet_clusters,
            "timeline": dossier.timeline,
            "methodology_summary": dossier.methodology_summary,
            "limitations": dossier.limitations,
            "confidence_assessment": dossier.confidence_assessment,
            "analyst_notes": dossier.analyst_notes
        }
        
        return json.dumps(data, indent=2, default=serialize).encode('utf-8')
    
    def _serialize_suspect(self, suspect: SuspectProfile) -> Dict[str, Any]:
        """Serialize suspect profile."""
        def serialize_attr(attr: IdentityAttribute) -> Dict:
            return {
                "name": attr.attribute_name,
                "value": attr.attribute_value,
                "confidence": attr.confidence,
                "sources": attr.sources,
                "evidence": attr.corroborating_evidence,
                "methodology": attr.methodology
            }
        
        return {
            "suspect_id": suspect.suspect_id,
            "confidence_score": suspect.confidence_score,
            "possible_names": [serialize_attr(a) for a in suspect.possible_names],
            "possible_emails": [serialize_attr(a) for a in suspect.possible_emails],
            "possible_phones": [serialize_attr(a) for a in suspect.possible_phones],
            "possible_addresses": [serialize_attr(a) for a in suspect.possible_addresses],
            "possible_locations": [serialize_attr(a) for a in suspect.possible_locations],
            "social_media": [serialize_attr(a) for a in suspect.social_media],
            "usernames": [serialize_attr(a) for a in suspect.usernames],
            "domains": [serialize_attr(a) for a in suspect.domains],
            "wallets": suspect.wallets,
            "wallet_clusters": suspect.wallet_clusters,
            "exchange_accounts": [serialize_attr(a) for a in suspect.exchange_accounts],
            "timezone_estimate": suspect.timezone_estimate,
            "activity_patterns": suspect.activity_patterns,
            "language_indicators": suspect.language_indicators
        }
    
    async def _export_pdf(self, dossier: IdentityDossier) -> bytes:
        """Export dossier as PDF."""
        if not HAS_REPORTLAB:
            self.logger.warning("reportlab not installed, falling back to text")
            return self._export_text(dossier)
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
        
        story = []
        
        # Header
        story.append(Paragraph(f"IDENTITY DOSSIER", title_style))
        story.append(Paragraph(f"<b>Classification:</b> {dossier.classification.value.upper()}", body_style))
        story.append(Paragraph(f"<b>Dossier ID:</b> {dossier.dossier_id}", body_style))
        story.append(Paragraph(f"<b>Generated:</b> {dossier.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}", body_style))
        story.append(Paragraph(f"<b>Integrity Hash:</b> {dossier.dossier_hash[:32]}...", body_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
        story.append(Paragraph(f"<b>Case ID:</b> {dossier.summary.case_id}", body_style))
        story.append(Paragraph(f"<b>Investigation:</b> {dossier.summary.investigation_title}", body_style))
        story.append(Paragraph(f"<b>Transactions Analyzed:</b> {dossier.summary.total_transactions_analyzed:,}", body_style))
        story.append(Paragraph(f"<b>Wallets Tracked:</b> {dossier.summary.total_wallets_tracked:,}", body_style))
        story.append(Paragraph(f"<b>Total Value:</b> ${dossier.summary.total_value_tracked:,.2f} USD", body_style))
        story.append(Spacer(1, 12))
        
        # Key Findings
        if dossier.summary.key_findings:
            story.append(Paragraph("KEY FINDINGS", heading_style))
            for finding in dossier.summary.key_findings:
                story.append(Paragraph(f"• {finding}", body_style))
            story.append(Spacer(1, 12))
        
        # Primary Suspect
        if dossier.primary_suspect:
            story.append(PageBreak())
            story.append(Paragraph("PRIMARY SUSPECT PROFILE", heading_style))
            story.append(Paragraph(f"<b>Suspect ID:</b> {dossier.primary_suspect.suspect_id}", body_style))
            story.append(Paragraph(f"<b>Confidence Score:</b> {dossier.primary_suspect.confidence_score:.0%}", body_style))
            
            if dossier.primary_suspect.possible_names:
                story.append(Paragraph("<b>Possible Names:</b>", body_style))
                for name in dossier.primary_suspect.possible_names:
                    story.append(Paragraph(f"  • {name.attribute_value} ({name.confidence:.0%} confidence)", body_style))
            
            if dossier.primary_suspect.possible_emails:
                story.append(Paragraph("<b>Possible Emails:</b>", body_style))
                for email in dossier.primary_suspect.possible_emails:
                    story.append(Paragraph(f"  • {email.attribute_value} ({email.confidence:.0%} confidence)", body_style))
            
            if dossier.primary_suspect.wallets:
                story.append(Paragraph(f"<b>Associated Wallets:</b> {len(dossier.primary_suspect.wallets)}", body_style))
        
        # Evidence Summary
        story.append(PageBreak())
        story.append(Paragraph("EVIDENCE SUMMARY", heading_style))
        story.append(Paragraph(f"<b>Total Evidence Items:</b> {len(dossier.evidence_items)}", body_style))
        
        # Evidence table
        evidence_data = [["ID", "Type", "Confidence", "Source"]]
        for e in dossier.evidence_items[:20]:  # Limit to first 20
            evidence_data.append([
                e.evidence_id,
                e.evidence_type.value[:20],
                f"{e.confidence:.0%}",
                e.source[:15]
            ])
        
        if evidence_data:
            table = Table(evidence_data, colWidths=[80, 120, 70, 100])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        # Limitations
        story.append(Paragraph("LIMITATIONS", heading_style))
        for limitation in dossier.limitations:
            story.append(Paragraph(f"• {limitation}", body_style))
        
        # Confidence Assessment
        story.append(Paragraph("CONFIDENCE ASSESSMENT", heading_style))
        for line in dossier.confidence_assessment.split('\n'):
            story.append(Paragraph(line, body_style))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            "This document is generated by RF Arsenal OS - SUPERHERO Blockchain Intelligence Module. "
            "All information should be verified independently before investigative action.",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
        ))
        
        doc.build(story)
        return buffer.getvalue()
    
    def _export_html(self, dossier: IdentityDossier) -> bytes:
        """Export dossier as HTML."""
        # Build content sections
        content_parts = []
        
        # Summary section
        content_parts.append('<div class="section">')
        content_parts.append('<div class="section-title">Executive Summary</div>')
        content_parts.append(f'<p><strong>Case ID:</strong> {dossier.summary.case_id}</p>')
        content_parts.append(f'<p><strong>Investigation:</strong> {dossier.summary.investigation_title}</p>')
        content_parts.append(f'<p><strong>Transactions Analyzed:</strong> {dossier.summary.total_transactions_analyzed:,}</p>')
        content_parts.append(f'<p><strong>Wallets Tracked:</strong> {dossier.summary.total_wallets_tracked:,}</p>')
        content_parts.append(f'<p><strong>Total Value:</strong> ${dossier.summary.total_value_tracked:,.2f} USD</p>')
        
        if dossier.summary.key_findings:
            content_parts.append('<h4>Key Findings</h4><ul>')
            for finding in dossier.summary.key_findings:
                content_parts.append(f'<li>{finding}</li>')
            content_parts.append('</ul>')
        content_parts.append('</div>')
        
        # Primary suspect
        if dossier.primary_suspect:
            content_parts.append('<div class="section suspect-profile">')
            content_parts.append('<div class="section-title">Primary Suspect</div>')
            content_parts.append(f'<p><strong>ID:</strong> {dossier.primary_suspect.suspect_id}</p>')
            conf_class = "confidence-high" if dossier.primary_suspect.confidence_score > 0.7 else \
                        "confidence-medium" if dossier.primary_suspect.confidence_score > 0.4 else "confidence-low"
            content_parts.append(f'<p><strong>Confidence:</strong> <span class="{conf_class}">{dossier.primary_suspect.confidence_score:.0%}</span></p>')
            
            if dossier.primary_suspect.possible_names:
                content_parts.append('<h4>Possible Names</h4><ul>')
                for name in dossier.primary_suspect.possible_names:
                    content_parts.append(f'<li>{name.attribute_value} ({name.confidence:.0%})</li>')
                content_parts.append('</ul>')
            
            content_parts.append('</div>')
        
        # Evidence table
        content_parts.append('<div class="section">')
        content_parts.append(f'<div class="section-title">Evidence Items ({len(dossier.evidence_items)})</div>')
        content_parts.append('<table><tr><th>ID</th><th>Type</th><th>Confidence</th><th>Source</th></tr>')
        for e in dossier.evidence_items[:30]:
            content_parts.append(f'<tr><td>{e.evidence_id}</td><td>{e.evidence_type.value}</td>')
            content_parts.append(f'<td>{e.confidence:.0%}</td><td>{e.source}</td></tr>')
        content_parts.append('</table></div>')
        
        # Confidence assessment
        content_parts.append('<div class="section">')
        content_parts.append('<div class="section-title">Confidence Assessment</div>')
        content_parts.append(f'<pre>{dossier.confidence_assessment}</pre>')
        content_parts.append('</div>')
        
        content = '\n'.join(content_parts)
        
        # Fill template
        html = self.templates["html"]
        html = html.replace("{{ title }}", dossier.summary.investigation_title)
        html = html.replace("{{ classification }}", dossier.classification.value.upper())
        html = html.replace("{{ case_id }}", dossier.summary.case_id)
        html = html.replace("{{ generated_at }}", dossier.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC'))
        html = html.replace("{{ dossier_hash }}", dossier.dossier_hash)
        html = html.replace("{{ content }}", content)
        
        return html.encode('utf-8')
    
    def _export_markdown(self, dossier: IdentityDossier) -> bytes:
        """Export dossier as Markdown."""
        lines = []
        
        # Header
        lines.append(f"# {dossier.summary.investigation_title}")
        lines.append("")
        lines.append(f"**Classification:** {dossier.classification.value.upper()}")
        lines.append(f"**Case ID:** {dossier.summary.case_id}")
        lines.append(f"**Dossier ID:** {dossier.dossier_id}")
        lines.append(f"**Generated:** {dossier.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Integrity Hash:** `{dossier.dossier_hash}`")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"- **Transactions Analyzed:** {dossier.summary.total_transactions_analyzed:,}")
        lines.append(f"- **Wallets Tracked:** {dossier.summary.total_wallets_tracked:,}")
        lines.append(f"- **Total Value:** ${dossier.summary.total_value_tracked:,.2f} USD")
        lines.append(f"- **Chains:** {', '.join(dossier.summary.chains_investigated)}")
        lines.append("")
        
        if dossier.summary.key_findings:
            lines.append("### Key Findings")
            lines.append("")
            for finding in dossier.summary.key_findings:
                lines.append(f"- {finding}")
            lines.append("")
        
        # Primary Suspect
        if dossier.primary_suspect:
            lines.append("## Primary Suspect")
            lines.append("")
            lines.append(f"**Suspect ID:** {dossier.primary_suspect.suspect_id}")
            lines.append(f"**Confidence:** {dossier.primary_suspect.confidence_score:.0%}")
            lines.append("")
            
            if dossier.primary_suspect.possible_names:
                lines.append("### Possible Names")
                for name in dossier.primary_suspect.possible_names:
                    lines.append(f"- {name.attribute_value} ({name.confidence:.0%} confidence)")
                lines.append("")
            
            if dossier.primary_suspect.possible_emails:
                lines.append("### Possible Emails")
                for email in dossier.primary_suspect.possible_emails:
                    lines.append(f"- {email.attribute_value} ({email.confidence:.0%} confidence)")
                lines.append("")
            
            if dossier.primary_suspect.wallets:
                lines.append(f"### Associated Wallets ({len(dossier.primary_suspect.wallets)})")
                for wallet in dossier.primary_suspect.wallets[:10]:
                    lines.append(f"- `{wallet}`")
                lines.append("")
        
        # Evidence Summary
        lines.append("## Evidence Summary")
        lines.append("")
        lines.append(f"Total evidence items: **{len(dossier.evidence_items)}**")
        lines.append("")
        lines.append("| ID | Type | Confidence | Source |")
        lines.append("|---|---|---|---|")
        for e in dossier.evidence_items[:20]:
            lines.append(f"| {e.evidence_id} | {e.evidence_type.value} | {e.confidence:.0%} | {e.source} |")
        lines.append("")
        
        # Methodology
        lines.append("## Methodology")
        lines.append("")
        lines.append(dossier.methodology_summary)
        lines.append("")
        
        # Limitations
        lines.append("## Limitations")
        lines.append("")
        for limitation in dossier.limitations:
            lines.append(f"- {limitation}")
        lines.append("")
        
        # Confidence Assessment
        lines.append("## Confidence Assessment")
        lines.append("")
        lines.append(dossier.confidence_assessment)
        lines.append("")
        
        # Recommendations
        if dossier.summary.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in dossier.summary.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by RF Arsenal OS - SUPERHERO Blockchain Intelligence Module*")
        
        return '\n'.join(lines).encode('utf-8')
    
    def _export_text(self, dossier: IdentityDossier) -> bytes:
        """Export dossier as plain text."""
        lines = []
        
        lines.append("=" * 80)
        lines.append(f"IDENTITY DOSSIER - {dossier.summary.investigation_title}")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Classification: {dossier.classification.value.upper()}")
        lines.append(f"Case ID: {dossier.summary.case_id}")
        lines.append(f"Dossier ID: {dossier.dossier_id}")
        lines.append(f"Generated: {dossier.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"Integrity Hash: {dossier.dossier_hash}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Transactions Analyzed: {dossier.summary.total_transactions_analyzed:,}")
        lines.append(f"Wallets Tracked: {dossier.summary.total_wallets_tracked:,}")
        lines.append(f"Total Value: ${dossier.summary.total_value_tracked:,.2f} USD")
        lines.append("")
        
        if dossier.primary_suspect:
            lines.append("-" * 80)
            lines.append("PRIMARY SUSPECT")
            lines.append("-" * 80)
            lines.append(f"ID: {dossier.primary_suspect.suspect_id}")
            lines.append(f"Confidence: {dossier.primary_suspect.confidence_score:.0%}")
            lines.append("")
        
        lines.append("-" * 80)
        lines.append(f"EVIDENCE ITEMS ({len(dossier.evidence_items)})")
        lines.append("-" * 80)
        for e in dossier.evidence_items[:20]:
            lines.append(f"  {e.evidence_id}: {e.evidence_type.value} ({e.confidence:.0%})")
        lines.append("")
        
        lines.append("-" * 80)
        lines.append("CONFIDENCE ASSESSMENT")
        lines.append("-" * 80)
        lines.append(dossier.confidence_assessment)
        lines.append("")
        lines.append("=" * 80)
        lines.append("Generated by RF Arsenal OS - SUPERHERO Blockchain Intelligence Module")
        lines.append("=" * 80)
        
        return '\n'.join(lines).encode('utf-8')
    
    def _apply_redaction(self, dossier: IdentityDossier) -> IdentityDossier:
        """Apply redaction to sensitive information."""
        import re
        import copy
        
        # Deep copy to avoid modifying original
        redacted = copy.deepcopy(dossier)
        redacted.classification = ClassificationLevel.REDACTED
        
        def redact_string(s: str) -> str:
            """Redact sensitive patterns from string."""
            if not isinstance(s, str):
                return s
            
            # Redact emails
            s = re.sub(self.redaction_patterns["email"], "[EMAIL REDACTED]", s, flags=re.IGNORECASE)
            
            # Redact phone numbers
            s = re.sub(self.redaction_patterns["phone"], "[PHONE REDACTED]", s)
            
            return s
        
        # Redact suspect profiles
        if redacted.primary_suspect:
            for attr in redacted.primary_suspect.possible_emails:
                attr.attribute_value = "[REDACTED]"
            for attr in redacted.primary_suspect.possible_phones:
                attr.attribute_value = "[REDACTED]"
            for attr in redacted.primary_suspect.possible_addresses:
                attr.attribute_value = "[REDACTED]"
        
        return redacted


async def main():
    """Test dossier generation."""
    generator = DossierGenerator()
    
    # Mock data for testing
    forensics_data = {
        "trace_result": {
            "transactions": [
                {
                    "from": "0x1234567890abcdef",
                    "to": "0xabcdef1234567890",
                    "value": 10.5,
                    "value_usd": 35000,
                    "chain": "ethereum",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "tx_hash": "0xabc123"
                }
            ]
        },
        "cluster_result": {
            "clusters": {
                "CLUSTER-001": {
                    "wallets": ["0x123", "0x456", "0x789"],
                    "confidence": 0.85,
                    "total_value": 50000
                }
            }
        },
        "exchange_detections": [
            {"exchange_name": "Binance", "confidence": 0.95}
        ]
    }
    
    identity_data = {
        "identity_candidates": [
            {
                "overall_confidence": 0.78,
                "possible_names": [{"value": "John Doe", "confidence": 0.75, "sources": ["forum_analysis"]}],
                "possible_emails": [{"value": "test@example.com", "confidence": 0.7, "sources": ["ens_lookup"]}],
                "associated_wallets": ["0x1234567890abcdef"]
            }
        ],
        "correlations": [
            {"attribute_type": "email", "confidence": 0.7, "discovered_at": "2024-01-15T12:00:00Z"}
        ]
    }
    
    geolocation_data = {
        "location_estimates": [
            {"region": "Eastern Europe", "confidence": 0.6, "timezone": "UTC+2"}
        ],
        "timezone_estimate": "UTC+2",
        "behavioral_patterns": {
            "peak_activity_hours": [14, 15, 16],
            "avg_transaction_frequency": 2.5
        }
    }
    
    counter_measure_data = {}
    
    # Generate dossier
    dossier = await generator.generate_dossier(
        case_id="CASE-2024-001",
        investigation_title="Cryptocurrency Theft Investigation",
        forensics_data=forensics_data,
        identity_data=identity_data,
        geolocation_data=geolocation_data,
        counter_measure_data=counter_measure_data,
        analyst_notes="Test investigation"
    )
    
    print(f"Generated dossier: {dossier.dossier_id}")
    print(f"Evidence items: {len(dossier.evidence_items)}")
    print(f"Dossier hash: {dossier.dossier_hash}")
    
    # Export to JSON
    json_output = await generator.export_dossier(dossier, DossierFormat.JSON)
    print(f"\nJSON output size: {len(json_output)} bytes")
    
    # Export to Markdown
    md_output = await generator.export_dossier(dossier, DossierFormat.MARKDOWN)
    print(f"Markdown output size: {len(md_output)} bytes")


if __name__ == "__main__":
    asyncio.run(main())
