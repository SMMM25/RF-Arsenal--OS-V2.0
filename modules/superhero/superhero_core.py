"""
RF Arsenal OS - SUPERHERO Core Engine

Central orchestration engine for the SUPERHERO Blockchain Intelligence Module.
Coordinates all sub-systems to provide comprehensive forensic investigation
and identity attribution capabilities.

Features:
- Unified investigation workflow
- Multi-chain support (ETH, BTC, BSC, Polygon, etc.)
- Real-time monitoring and alerts
- Law enforcement integration
- Court-ready evidence generation

Author: RF Arsenal Security Team
License: Authorized Use Only
"""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable
from pathlib import Path
import uuid

# Import sub-modules
from .blockchain_forensics import BlockchainForensics
from .identity_engine import IdentityCorrelationEngine
from .geolocation import GeolocationAnalyzer
from .counter_measures import CounterMeasureSystem
from .dossier_generator import DossierGenerator, IdentityDossier, DossierFormat, ClassificationLevel


class InvestigationStatus(Enum):
    """Investigation status states."""
    CREATED = "created"
    INITIALIZING = "initializing"
    COLLECTING = "collecting"
    ANALYZING = "analyzing"
    CORRELATING = "correlating"
    GENERATING_DOSSIER = "generating_dossier"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SupportedChain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    BSC = "bsc"
    POLYGON = "polygon"
    AVALANCHE = "avalanche"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    SOLANA = "solana"
    TRON = "tron"
    FANTOM = "fantom"


@dataclass
class InvestigationTarget:
    """Target for investigation."""
    target_id: str
    target_type: str  # "wallet", "transaction", "entity"
    value: str  # Address, tx hash, or entity name
    chain: Optional[SupportedChain] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Investigation:
    """Active investigation object."""
    investigation_id: str
    case_id: str
    title: str
    description: str
    created_at: datetime
    updated_at: datetime
    status: InvestigationStatus
    
    # Targets
    targets: List[InvestigationTarget] = field(default_factory=list)
    
    # Results
    forensics_results: Dict[str, Any] = field(default_factory=dict)
    identity_results: Dict[str, Any] = field(default_factory=dict)
    geolocation_results: Dict[str, Any] = field(default_factory=dict)
    counter_measure_results: Dict[str, Any] = field(default_factory=dict)
    
    # Dossier
    dossier: Optional[IdentityDossier] = None
    
    # Metadata
    analyst_notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Progress tracking
    progress: float = 0.0  # 0.0 to 1.0
    current_phase: str = ""
    errors: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """Real-time alert."""
    alert_id: str
    investigation_id: str
    priority: AlertPriority
    alert_type: str
    title: str
    description: str
    data: Dict[str, Any]
    created_at: datetime
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None


@dataclass
class MonitoredAddress:
    """Address being monitored for activity."""
    address: str
    chain: SupportedChain
    investigation_id: str
    alert_on_activity: bool
    last_activity: Optional[datetime] = None
    activity_count: int = 0


class SuperheroEngine:
    """
    SUPERHERO Core Engine - Orchestrates blockchain forensics and identity attribution.
    
    This engine coordinates:
    - Blockchain Forensics: Transaction tracing, cluster analysis
    - Identity Correlation: OSINT-based identity attribution
    - Geolocation Analysis: Location and behavioral profiling
    - Counter-Measures: Mixer tracing, chain-hop detection
    - Dossier Generation: Court-ready evidence packaging
    
    Adheres to RF Arsenal OS principles:
    - Stealth-first operation
    - No external telemetry
    - RAM-only for sensitive data
    - Full proxy chain support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SUPERHERO Engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger("superhero.engine")
        
        # Initialize sub-systems
        self.forensics = BlockchainForensics(config.get("forensics", {}))
        self.identity = IdentityCorrelationEngine(config.get("identity", {}))
        self.geolocation = GeolocationAnalyzer(config.get("geolocation", {}))
        self.counter_measures = CounterMeasureSystem(config.get("counter_measures", {}))
        self.dossier_gen = DossierGenerator(config.get("dossier", {}))
        
        # Active investigations (RAM-only)
        self._investigations: Dict[str, Investigation] = {}
        
        # Alerts
        self._alerts: Dict[str, Alert] = {}
        
        # Monitored addresses
        self._monitored: Dict[str, MonitoredAddress] = {}
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Callbacks
        self._alert_callbacks: List[Callable] = []
        self._progress_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            "investigations_created": 0,
            "investigations_completed": 0,
            "dossiers_generated": 0,
            "alerts_generated": 0,
            "addresses_monitored": 0,
            "total_transactions_traced": 0,
            "total_identities_correlated": 0
        }
        
        self.logger.info("SUPERHERO Engine initialized")
    
    # ============================================================
    # Investigation Management
    # ============================================================
    
    async def create_investigation(
        self,
        case_id: str,
        title: str,
        description: str = "",
        targets: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None
    ) -> Investigation:
        """
        Create a new investigation.
        
        Args:
            case_id: External case identifier
            title: Investigation title
            description: Investigation description
            targets: Initial targets (wallets, transactions)
            tags: Tags for categorization
            
        Returns:
            New Investigation object
        """
        investigation_id = f"INV-{uuid.uuid4().hex[:12].upper()}"
        now = datetime.now(timezone.utc)
        
        investigation = Investigation(
            investigation_id=investigation_id,
            case_id=case_id,
            title=title,
            description=description,
            created_at=now,
            updated_at=now,
            status=InvestigationStatus.CREATED,
            tags=tags or []
        )
        
        # Add initial targets
        if targets:
            for target in targets:
                inv_target = InvestigationTarget(
                    target_id=f"TGT-{uuid.uuid4().hex[:8].upper()}",
                    target_type=target.get("type", "wallet"),
                    value=target.get("value", ""),
                    chain=SupportedChain(target.get("chain", "ethereum")) if target.get("chain") else None,
                    metadata=target.get("metadata", {})
                )
                investigation.targets.append(inv_target)
        
        self._investigations[investigation_id] = investigation
        self.stats["investigations_created"] += 1
        
        self.logger.info(f"Created investigation: {investigation_id} ({title})")
        
        return investigation
    
    async def add_target(
        self,
        investigation_id: str,
        target_type: str,
        value: str,
        chain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> InvestigationTarget:
        """Add target to investigation."""
        if investigation_id not in self._investigations:
            raise ValueError(f"Investigation not found: {investigation_id}")
        
        investigation = self._investigations[investigation_id]
        
        target = InvestigationTarget(
            target_id=f"TGT-{uuid.uuid4().hex[:8].upper()}",
            target_type=target_type,
            value=value,
            chain=SupportedChain(chain) if chain else None,
            metadata=metadata or {}
        )
        
        investigation.targets.append(target)
        investigation.updated_at = datetime.now(timezone.utc)
        
        self.logger.info(f"Added target {target.target_id} to investigation {investigation_id}")
        
        return target
    
    async def run_investigation(
        self,
        investigation_id: str,
        phases: Optional[List[str]] = None
    ) -> Investigation:
        """
        Run full investigation workflow.
        
        Args:
            investigation_id: Investigation to run
            phases: Specific phases to run (all if None)
            
        Returns:
            Updated Investigation with results
        """
        if investigation_id not in self._investigations:
            raise ValueError(f"Investigation not found: {investigation_id}")
        
        investigation = self._investigations[investigation_id]
        
        try:
            investigation.status = InvestigationStatus.INITIALIZING
            investigation.progress = 0.0
            self._notify_progress(investigation)
            
            phases = phases or ["forensics", "identity", "geolocation", "counter_measures", "dossier"]
            total_phases = len(phases)
            
            # Phase 1: Blockchain Forensics
            if "forensics" in phases:
                investigation.current_phase = "Blockchain Forensics"
                investigation.status = InvestigationStatus.COLLECTING
                self._notify_progress(investigation)
                
                investigation.forensics_results = await self._run_forensics(investigation)
                investigation.progress = 1 / total_phases
                self._notify_progress(investigation)
                
                self.stats["total_transactions_traced"] += len(
                    investigation.forensics_results.get("trace_result", {}).get("transactions", [])
                )
            
            # Phase 2: Identity Correlation
            if "identity" in phases:
                investigation.current_phase = "Identity Correlation"
                investigation.status = InvestigationStatus.ANALYZING
                self._notify_progress(investigation)
                
                investigation.identity_results = await self._run_identity(investigation)
                investigation.progress = 2 / total_phases
                self._notify_progress(investigation)
                
                self.stats["total_identities_correlated"] += len(
                    investigation.identity_results.get("identity_candidates", [])
                )
            
            # Phase 3: Geolocation Analysis
            if "geolocation" in phases:
                investigation.current_phase = "Geolocation Analysis"
                self._notify_progress(investigation)
                
                investigation.geolocation_results = await self._run_geolocation(investigation)
                investigation.progress = 3 / total_phases
                self._notify_progress(investigation)
            
            # Phase 4: Counter-Measure Analysis
            if "counter_measures" in phases:
                investigation.current_phase = "Counter-Measure Analysis"
                self._notify_progress(investigation)
                
                investigation.counter_measure_results = await self._run_counter_measures(investigation)
                investigation.progress = 4 / total_phases
                self._notify_progress(investigation)
            
            # Phase 5: Generate Dossier
            if "dossier" in phases:
                investigation.current_phase = "Generating Dossier"
                investigation.status = InvestigationStatus.GENERATING_DOSSIER
                self._notify_progress(investigation)
                
                investigation.dossier = await self.dossier_gen.generate_dossier(
                    case_id=investigation.case_id,
                    investigation_title=investigation.title,
                    forensics_data=investigation.forensics_results,
                    identity_data=investigation.identity_results,
                    geolocation_data=investigation.geolocation_results,
                    counter_measure_data=investigation.counter_measure_results,
                    analyst_notes=investigation.analyst_notes
                )
                investigation.progress = 1.0
                self.stats["dossiers_generated"] += 1
            
            investigation.status = InvestigationStatus.COMPLETED
            investigation.updated_at = datetime.now(timezone.utc)
            self.stats["investigations_completed"] += 1
            
            self._notify_progress(investigation)
            self.logger.info(f"Investigation {investigation_id} completed successfully")
            
        except Exception as e:
            investigation.status = InvestigationStatus.FAILED
            investigation.errors.append(str(e))
            investigation.updated_at = datetime.now(timezone.utc)
            self.logger.error(f"Investigation {investigation_id} failed: {e}")
            raise
        
        return investigation
    
    async def _run_forensics(self, investigation: Investigation) -> Dict[str, Any]:
        """Run blockchain forensics phase."""
        results = {
            "trace_result": {"transactions": [], "summary": {}},
            "cluster_result": {"clusters": {}, "statistics": {}},
            "exchange_detections": [],
            "mixer_detections": []
        }
        
        for target in investigation.targets:
            if target.target_type == "wallet":
                # Trace transactions from wallet
                chain = target.chain.value if target.chain else "ethereum"
                
                trace_result = await self.forensics.trace_transactions(
                    address=target.value,
                    chain=chain,
                    depth=3
                )
                
                if "transactions" in trace_result:
                    results["trace_result"]["transactions"].extend(trace_result["transactions"])
                
                # Cluster analysis
                cluster_result = await self.forensics.cluster_wallets(
                    addresses=[target.value],
                    chain=chain
                )
                
                if "clusters" in cluster_result:
                    results["cluster_result"]["clusters"].update(cluster_result["clusters"])
                
                # Exchange detection
                exchange_result = await self.forensics.identify_exchanges(
                    transactions=trace_result.get("transactions", [])
                )
                
                if exchange_result:
                    results["exchange_detections"].extend(exchange_result)
                
                # Mixer detection
                mixer_result = await self.forensics.detect_mixers(
                    transactions=trace_result.get("transactions", [])
                )
                
                if mixer_result:
                    results["mixer_detections"].extend(mixer_result)
        
        # Generate alert if significant findings
        if results["mixer_detections"]:
            await self._create_alert(
                investigation_id=investigation.investigation_id,
                priority=AlertPriority.HIGH,
                alert_type="mixer_detected",
                title="Mixer Service Detected",
                description=f"Detected {len(results['mixer_detections'])} mixer interactions",
                data={"mixers": results["mixer_detections"]}
            )
        
        return results
    
    async def _run_identity(self, investigation: Investigation) -> Dict[str, Any]:
        """Run identity correlation phase."""
        results = {
            "identity_candidates": [],
            "correlations": [],
            "social_media_matches": [],
            "email_matches": [],
            "osint_findings": []
        }
        
        # Get wallets from forensics
        wallets = set()
        for target in investigation.targets:
            if target.target_type == "wallet":
                wallets.add(target.value)
        
        # Add wallets from clusters
        clusters = investigation.forensics_results.get("cluster_result", {}).get("clusters", {})
        for cluster_id, cluster_info in clusters.items():
            wallets.update(cluster_info.get("wallets", []))
        
        # Run identity correlation for each wallet
        for wallet in list(wallets)[:50]:  # Limit to 50 for performance
            chain = "ethereum"  # Default
            
            identity_result = await self.identity.correlate_identity(
                wallet_address=wallet,
                chain=chain
            )
            
            if identity_result.get("identity_candidates"):
                results["identity_candidates"].extend(identity_result["identity_candidates"])
            
            if identity_result.get("correlations"):
                results["correlations"].extend(identity_result["correlations"])
            
            if identity_result.get("social_media_matches"):
                results["social_media_matches"].extend(identity_result["social_media_matches"])
        
        # Create alert for high-confidence identities
        high_conf_identities = [
            c for c in results["identity_candidates"]
            if c.get("overall_confidence", 0) > 0.8
        ]
        
        if high_conf_identities:
            await self._create_alert(
                investigation_id=investigation.investigation_id,
                priority=AlertPriority.CRITICAL,
                alert_type="identity_identified",
                title="High-Confidence Identity Match",
                description=f"Found {len(high_conf_identities)} high-confidence identity matches",
                data={"identities": high_conf_identities}
            )
        
        return results
    
    async def _run_geolocation(self, investigation: Investigation) -> Dict[str, Any]:
        """Run geolocation analysis phase."""
        results = {
            "location_estimates": [],
            "timezone_estimate": None,
            "behavioral_patterns": {}
        }
        
        # Get transactions for timing analysis
        transactions = investigation.forensics_results.get("trace_result", {}).get("transactions", [])
        
        if transactions:
            geo_result = await self.geolocation.analyze_location(
                transactions=transactions
            )
            
            if geo_result:
                results.update(geo_result)
        
        return results
    
    async def _run_counter_measures(self, investigation: Investigation) -> Dict[str, Any]:
        """Run counter-measure analysis phase."""
        results = {
            "mixer_traces": [],
            "chain_hop_traces": [],
            "privacy_coin_exits": [],
            "multi_wallet_links": []
        }
        
        # Get mixer detections
        mixer_detections = investigation.forensics_results.get("mixer_detections", [])
        
        if mixer_detections:
            for mixer in mixer_detections:
                trace_result = await self.counter_measures.trace_mixer(
                    mixer_info=mixer
                )
                if trace_result:
                    results["mixer_traces"].append(trace_result)
        
        # Detect chain hopping
        transactions = investigation.forensics_results.get("trace_result", {}).get("transactions", [])
        
        chain_hop_result = await self.counter_measures.detect_chain_hopping(
            transactions=transactions
        )
        
        if chain_hop_result:
            results["chain_hop_traces"].extend(chain_hop_result)
        
        return results
    
    # ============================================================
    # Quick Analysis Methods
    # ============================================================
    
    async def quick_trace(
        self,
        address: str,
        chain: str = "ethereum",
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Quick transaction trace without full investigation.
        
        Args:
            address: Wallet address to trace
            chain: Blockchain network
            depth: Trace depth
            
        Returns:
            Trace results
        """
        return await self.forensics.trace_transactions(
            address=address,
            chain=chain,
            depth=depth
        )
    
    async def quick_identity(
        self,
        address: str,
        chain: str = "ethereum"
    ) -> Dict[str, Any]:
        """
        Quick identity lookup without full investigation.
        
        Args:
            address: Wallet address
            chain: Blockchain network
            
        Returns:
            Identity correlation results
        """
        return await self.identity.correlate_identity(
            wallet_address=address,
            chain=chain
        )
    
    async def check_address(self, address: str) -> Dict[str, Any]:
        """
        Quick check of address against threat databases.
        
        Args:
            address: Wallet address to check
            
        Returns:
            Threat intelligence results
        """
        result = {
            "address": address,
            "is_flagged": False,
            "flags": [],
            "risk_score": 0,
            "known_entity": None
        }
        
        # Check against threat database
        threat_check = await self.forensics.check_threat_database(address)
        
        if threat_check:
            result.update(threat_check)
        
        return result
    
    # ============================================================
    # Monitoring & Alerts
    # ============================================================
    
    async def monitor_address(
        self,
        address: str,
        chain: str,
        investigation_id: str,
        alert_on_activity: bool = True
    ) -> MonitoredAddress:
        """
        Add address to monitoring.
        
        Args:
            address: Address to monitor
            chain: Blockchain network
            investigation_id: Associated investigation
            alert_on_activity: Generate alerts on activity
            
        Returns:
            MonitoredAddress object
        """
        monitored = MonitoredAddress(
            address=address,
            chain=SupportedChain(chain),
            investigation_id=investigation_id,
            alert_on_activity=alert_on_activity
        )
        
        key = f"{chain}:{address}"
        self._monitored[key] = monitored
        self.stats["addresses_monitored"] += 1
        
        self.logger.info(f"Now monitoring address: {address} on {chain}")
        
        return monitored
    
    async def stop_monitoring(self, address: str, chain: str) -> bool:
        """Stop monitoring an address."""
        key = f"{chain}:{address}"
        if key in self._monitored:
            del self._monitored[key]
            self.stats["addresses_monitored"] -= 1
            return True
        return False
    
    async def start_monitoring_loop(self, interval: int = 60):
        """Start background monitoring loop."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop(interval))
        self.logger.info("Monitoring loop started")
    
    async def stop_monitoring_loop(self):
        """Stop background monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Monitoring loop stopped")
    
    async def _monitoring_loop(self, interval: int):
        """Background monitoring loop."""
        while self._running:
            try:
                for key, monitored in self._monitored.items():
                    # Check for new activity
                    activity = await self.forensics.check_recent_activity(
                        address=monitored.address,
                        chain=monitored.chain.value,
                        since=monitored.last_activity
                    )
                    
                    if activity.get("new_transactions"):
                        monitored.activity_count += len(activity["new_transactions"])
                        monitored.last_activity = datetime.now(timezone.utc)
                        
                        if monitored.alert_on_activity:
                            await self._create_alert(
                                investigation_id=monitored.investigation_id,
                                priority=AlertPriority.HIGH,
                                alert_type="address_activity",
                                title=f"Activity Detected on Monitored Address",
                                description=f"New transactions detected for {monitored.address[:16]}...",
                                data=activity
                            )
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval)
    
    async def _create_alert(
        self,
        investigation_id: str,
        priority: AlertPriority,
        alert_type: str,
        title: str,
        description: str,
        data: Dict[str, Any]
    ) -> Alert:
        """Create new alert."""
        alert = Alert(
            alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
            investigation_id=investigation_id,
            priority=priority,
            alert_type=alert_type,
            title=title,
            description=description,
            data=data,
            created_at=datetime.now(timezone.utc)
        )
        
        self._alerts[alert.alert_id] = alert
        self.stats["alerts_generated"] += 1
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
        
        self.logger.info(f"Alert created: {alert.alert_id} - {title}")
        
        return alert
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts."""
        self._alert_callbacks.append(callback)
    
    def register_progress_callback(self, callback: Callable):
        """Register callback for progress updates."""
        self._progress_callbacks.append(callback)
    
    def _notify_progress(self, investigation: Investigation):
        """Notify progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(investigation)
            except Exception as e:
                self.logger.error(f"Progress callback error: {e}")
    
    # ============================================================
    # Dossier Export
    # ============================================================
    
    async def export_investigation(
        self,
        investigation_id: str,
        output_format: DossierFormat = DossierFormat.PDF,
        classification: ClassificationLevel = ClassificationLevel.LAW_ENFORCEMENT_ONLY,
        redact: bool = False
    ) -> bytes:
        """
        Export investigation dossier.
        
        Args:
            investigation_id: Investigation to export
            output_format: Export format (PDF, JSON, HTML, Markdown)
            classification: Classification level
            redact: Apply redaction
            
        Returns:
            Exported dossier as bytes
        """
        if investigation_id not in self._investigations:
            raise ValueError(f"Investigation not found: {investigation_id}")
        
        investigation = self._investigations[investigation_id]
        
        if not investigation.dossier:
            # Generate dossier if not exists
            investigation.dossier = await self.dossier_gen.generate_dossier(
                case_id=investigation.case_id,
                investigation_title=investigation.title,
                forensics_data=investigation.forensics_results,
                identity_data=investigation.identity_results,
                geolocation_data=investigation.geolocation_results,
                counter_measure_data=investigation.counter_measure_results,
                classification=classification,
                analyst_notes=investigation.analyst_notes
            )
        
        return await self.dossier_gen.export_dossier(
            dossier=investigation.dossier,
            output_format=output_format,
            redact=redact
        )
    
    # ============================================================
    # Investigation Management
    # ============================================================
    
    def get_investigation(self, investigation_id: str) -> Optional[Investigation]:
        """Get investigation by ID."""
        return self._investigations.get(investigation_id)
    
    def list_investigations(
        self,
        status: Optional[InvestigationStatus] = None,
        limit: int = 50
    ) -> List[Investigation]:
        """List investigations."""
        investigations = list(self._investigations.values())
        
        if status:
            investigations = [i for i in investigations if i.status == status]
        
        investigations.sort(key=lambda x: x.updated_at, reverse=True)
        
        return investigations[:limit]
    
    def get_alerts(
        self,
        investigation_id: Optional[str] = None,
        unacknowledged_only: bool = False,
        limit: int = 100
    ) -> List[Alert]:
        """Get alerts."""
        alerts = list(self._alerts.values())
        
        if investigation_id:
            alerts = [a for a in alerts if a.investigation_id == investigation_id]
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        
        return alerts[:limit]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].acknowledged = True
            self._alerts[alert_id].acknowledged_at = datetime.now(timezone.utc)
            return True
        return False
    
    async def add_analyst_notes(self, investigation_id: str, notes: str):
        """Add analyst notes to investigation."""
        if investigation_id in self._investigations:
            self._investigations[investigation_id].analyst_notes = notes
            self._investigations[investigation_id].updated_at = datetime.now(timezone.utc)
    
    # ============================================================
    # Status & Statistics
    # ============================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "status": "operational",
            "monitoring_active": self._running,
            "active_investigations": len([
                i for i in self._investigations.values()
                if i.status not in [InvestigationStatus.COMPLETED, InvestigationStatus.FAILED]
            ]),
            "completed_investigations": len([
                i for i in self._investigations.values()
                if i.status == InvestigationStatus.COMPLETED
            ]),
            "monitored_addresses": len(self._monitored),
            "pending_alerts": len([a for a in self._alerts.values() if not a.acknowledged]),
            "statistics": self.stats
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        return {
            **self.stats,
            "total_investigations": len(self._investigations),
            "total_alerts": len(self._alerts),
            "by_status": {
                status.value: len([
                    i for i in self._investigations.values()
                    if i.status == status
                ])
                for status in InvestigationStatus
            }
        }
    
    # ============================================================
    # Cleanup
    # ============================================================
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.stop_monitoring_loop()
        
        # Clear sensitive data (RAM-only principle)
        self._investigations.clear()
        self._alerts.clear()
        self._monitored.clear()
        
        self.logger.info("SUPERHERO Engine cleanup completed")
    
    def clear_investigation(self, investigation_id: str) -> bool:
        """
        Securely clear investigation data (RAM-only principle).
        
        Args:
            investigation_id: Investigation to clear
            
        Returns:
            True if cleared successfully
        """
        if investigation_id in self._investigations:
            # Clear associated alerts
            alert_ids_to_remove = [
                a.alert_id for a in self._alerts.values()
                if a.investigation_id == investigation_id
            ]
            for alert_id in alert_ids_to_remove:
                del self._alerts[alert_id]
            
            # Clear monitored addresses
            monitor_keys_to_remove = [
                k for k, v in self._monitored.items()
                if v.investigation_id == investigation_id
            ]
            for key in monitor_keys_to_remove:
                del self._monitored[key]
            
            # Clear investigation
            del self._investigations[investigation_id]
            
            self.logger.info(f"Investigation {investigation_id} cleared")
            return True
        
        return False


# Singleton instance
_engine: Optional[SuperheroEngine] = None


def get_engine(config: Optional[Dict[str, Any]] = None) -> SuperheroEngine:
    """Get or create SUPERHERO engine instance."""
    global _engine
    if _engine is None:
        _engine = SuperheroEngine(config)
    return _engine


async def main():
    """Test SUPERHERO engine."""
    engine = get_engine()
    
    # Create investigation
    investigation = await engine.create_investigation(
        case_id="CASE-2024-TEST",
        title="Test Investigation",
        description="Testing SUPERHERO engine",
        targets=[
            {"type": "wallet", "value": "0x1234567890abcdef1234567890abcdef12345678", "chain": "ethereum"}
        ]
    )
    
    print(f"Created investigation: {investigation.investigation_id}")
    
    # Get status
    status = engine.get_status()
    print(f"Engine status: {json.dumps(status, indent=2)}")
    
    # Cleanup
    await engine.cleanup()
    print("Engine cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
