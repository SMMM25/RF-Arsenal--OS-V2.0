"""
TEMPEST Shielding Effectiveness Module for RF Arsenal OS.

This module provides comprehensive shielding effectiveness monitoring,
testing, and certification management for TEMPEST compliance.

Features:
- Real-time shielding effectiveness measurement
- Multi-frequency shielding analysis
- IEEE 299/MIL-STD-461G compliant testing
- Zone penetration monitoring
- Shielding degradation alerts
- Certification tracking and management

Author: RF Arsenal Development Team
License: Proprietary - Security Sensitive
"""

import asyncio
import hashlib
import logging
import math
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import secrets
import numpy as np

from . import (
    TEMPESTLevel,
    ZoneClassification,
    ShieldingStatus,
    TEMPESTException,
    ShieldingFailure,
)

logger = logging.getLogger(__name__)


class ShieldingType(Enum):
    """Types of electromagnetic shielding."""
    
    ENCLOSURE = "enclosure"         # Full equipment enclosure
    ROOM = "room"                   # Shielded room/facility
    CABINET = "cabinet"             # Cabinet/rack shielding
    CABLE = "cable"                 # Cable shielding
    GASKET = "gasket"               # EMI gasket
    FILTER = "filter"               # EMI filter
    WINDOW = "window"               # Shielded window/viewport
    PENETRATION = "penetration"     # Penetration panel


class TestMethod(Enum):
    """Shielding effectiveness test methods."""
    
    IEEE_299 = "ieee_299"           # IEEE 299 standard
    MIL_STD_461G = "mil_std_461g"   # MIL-STD-461G
    NSA_65_6 = "nsa_65_6"           # NSA 65-6 (TEMPEST)
    CUSTOM = "custom"               # Custom test method


class MeasurementType(Enum):
    """Types of shielding measurements."""
    
    MAGNETIC_FIELD = "magnetic"     # Low frequency H-field
    PLANE_WAVE = "plane_wave"       # Far-field plane wave
    ELECTRIC_FIELD = "electric"     # Near-field E-field


@dataclass
class ShieldingRequirement:
    """Shielding effectiveness requirement specification."""
    
    frequency_hz: float
    min_effectiveness_db: float
    measurement_type: MeasurementType
    test_method: TestMethod
    notes: str = ""


@dataclass
class ShieldingMeasurement:
    """Single shielding effectiveness measurement."""
    
    timestamp: datetime
    frequency_hz: float
    effectiveness_db: float
    measurement_type: MeasurementType
    test_method: TestMethod
    location: str
    shielding_type: ShieldingType
    test_equipment: str
    environmental_conditions: Dict[str, Any]
    uncertainty_db: float
    compliant: bool
    notes: str = ""


@dataclass
class ShieldingProfile:
    """Complete shielding profile for a location."""
    
    location_id: str
    shielding_type: ShieldingType
    zone: ZoneClassification
    protection_level: TEMPESTLevel
    measurements: List[ShieldingMeasurement]
    requirements: List[ShieldingRequirement]
    last_certification: Optional[datetime]
    certification_expires: Optional[datetime]
    certification_authority: Optional[str]
    overall_compliance: bool
    degradation_detected: bool


class ShieldingRequirementSet:
    """
    Collection of shielding requirements for different protection levels.
    
    Provides standard requirements based on TEMPEST protection levels
    and zone classifications.
    """
    
    def __init__(self):
        self._requirements: Dict[
            Tuple[TEMPESTLevel, ZoneClassification],
            List[ShieldingRequirement]
        ] = {}
        self._load_standard_requirements()
    
    def _load_standard_requirements(self) -> None:
        """Load standard TEMPEST shielding requirements."""
        # Test frequencies per standards
        test_frequencies = [
            1e6, 10e6, 100e6, 200e6, 300e6, 400e6, 500e6,
            1e9, 2e9, 3e9, 4e9, 5e9, 6e9, 10e9, 18e9
        ]
        
        # Level A (NATO SDIP-27 Level A) - Highest protection
        level_a_requirements = []
        for freq in test_frequencies:
            if freq < 100e6:
                # Low frequency magnetic field
                min_se = 80.0
                meas_type = MeasurementType.MAGNETIC_FIELD
            elif freq < 1e9:
                # Mid frequency plane wave
                min_se = 100.0
                meas_type = MeasurementType.PLANE_WAVE
            else:
                # High frequency plane wave
                min_se = 100.0
                meas_type = MeasurementType.PLANE_WAVE
            
            level_a_requirements.append(ShieldingRequirement(
                frequency_hz=freq,
                min_effectiveness_db=min_se,
                measurement_type=meas_type,
                test_method=TestMethod.NSA_65_6,
                notes="NATO SDIP-27 Level A requirement"
            ))
        
        # Level B (NATO SDIP-27 Level B) - Medium protection
        level_b_requirements = []
        for freq in test_frequencies:
            if freq < 100e6:
                min_se = 60.0
                meas_type = MeasurementType.MAGNETIC_FIELD
            elif freq < 1e9:
                min_se = 80.0
                meas_type = MeasurementType.PLANE_WAVE
            else:
                min_se = 80.0
                meas_type = MeasurementType.PLANE_WAVE
            
            level_b_requirements.append(ShieldingRequirement(
                frequency_hz=freq,
                min_effectiveness_db=min_se,
                measurement_type=meas_type,
                test_method=TestMethod.IEEE_299,
                notes="NATO SDIP-27 Level B requirement"
            ))
        
        # Level C (NATO SDIP-27 Level C) - Basic protection
        level_c_requirements = []
        for freq in test_frequencies:
            if freq < 100e6:
                min_se = 40.0
                meas_type = MeasurementType.MAGNETIC_FIELD
            elif freq < 1e9:
                min_se = 60.0
                meas_type = MeasurementType.PLANE_WAVE
            else:
                min_se = 60.0
                meas_type = MeasurementType.PLANE_WAVE
            
            level_c_requirements.append(ShieldingRequirement(
                frequency_hz=freq,
                min_effectiveness_db=min_se,
                measurement_type=meas_type,
                test_method=TestMethod.IEEE_299,
                notes="NATO SDIP-27 Level C requirement"
            ))
        
        # Store requirements for all zone combinations
        for zone in ZoneClassification:
            self._requirements[(TEMPESTLevel.LEVEL_A, zone)] = level_a_requirements
            self._requirements[(TEMPESTLevel.LEVEL_B, zone)] = level_b_requirements
            self._requirements[(TEMPESTLevel.LEVEL_C, zone)] = level_c_requirements
            self._requirements[(TEMPESTLevel.UNPROTECTED, zone)] = []
    
    def get_requirements(
        self,
        level: TEMPESTLevel,
        zone: ZoneClassification
    ) -> List[ShieldingRequirement]:
        """Get shielding requirements for protection level and zone."""
        return self._requirements.get((level, zone), [])
    
    def add_custom_requirement(
        self,
        level: TEMPESTLevel,
        zone: ZoneClassification,
        requirement: ShieldingRequirement
    ) -> None:
        """Add a custom shielding requirement."""
        key = (level, zone)
        if key not in self._requirements:
            self._requirements[key] = []
        self._requirements[key].append(requirement)


class ShieldingEffectivenessCalculator:
    """
    Calculator for shielding effectiveness metrics.
    
    Provides methods for calculating SE from measurements
    and applying correction factors.
    """
    
    @staticmethod
    def calculate_se_db(
        reference_level_dbm: float,
        shielded_level_dbm: float
    ) -> float:
        """
        Calculate shielding effectiveness in dB.
        
        SE = Reference Level - Shielded Level
        
        Args:
            reference_level_dbm: Signal level without shielding
            shielded_level_dbm: Signal level with shielding
            
        Returns:
            Shielding effectiveness in dB
        """
        return reference_level_dbm - shielded_level_dbm
    
    @staticmethod
    def apply_antenna_factors(
        measured_se_db: float,
        tx_antenna_factor_db: float,
        rx_antenna_factor_db: float
    ) -> float:
        """
        Apply antenna correction factors to SE measurement.
        
        Args:
            measured_se_db: Raw measured SE
            tx_antenna_factor_db: Transmit antenna factor
            rx_antenna_factor_db: Receive antenna factor
            
        Returns:
            Corrected SE in dB
        """
        return measured_se_db + tx_antenna_factor_db + rx_antenna_factor_db
    
    @staticmethod
    def calculate_uncertainty(
        measurement_uncertainty_db: float,
        antenna_uncertainty_db: float,
        positioning_uncertainty_db: float,
        environmental_uncertainty_db: float
    ) -> float:
        """
        Calculate combined uncertainty (RSS method).
        
        Args:
            measurement_uncertainty_db: Receiver measurement uncertainty
            antenna_uncertainty_db: Antenna calibration uncertainty
            positioning_uncertainty_db: Antenna positioning uncertainty
            environmental_uncertainty_db: Environmental factor uncertainty
            
        Returns:
            Combined uncertainty in dB (95% confidence)
        """
        # Root Sum Square method
        rss = math.sqrt(
            measurement_uncertainty_db ** 2 +
            antenna_uncertainty_db ** 2 +
            positioning_uncertainty_db ** 2 +
            environmental_uncertainty_db ** 2
        )
        
        # Apply coverage factor k=2 for 95% confidence
        return rss * 2
    
    @staticmethod
    def interpolate_se(
        measurements: List[ShieldingMeasurement],
        target_frequency_hz: float
    ) -> Optional[float]:
        """
        Interpolate SE at a specific frequency from measurements.
        
        Args:
            measurements: List of measurements
            target_frequency_hz: Target frequency for interpolation
            
        Returns:
            Interpolated SE in dB or None if not possible
        """
        if not measurements:
            return None
        
        # Sort by frequency
        sorted_meas = sorted(measurements, key=lambda m: m.frequency_hz)
        
        # Find bounding measurements
        lower = None
        upper = None
        
        for m in sorted_meas:
            if m.frequency_hz <= target_frequency_hz:
                lower = m
            elif upper is None:
                upper = m
                break
        
        if lower is None and upper is None:
            return None
        
        if lower is None:
            return upper.effectiveness_db
        
        if upper is None:
            return lower.effectiveness_db
        
        # Linear interpolation in log-frequency space
        log_freq_lower = math.log10(lower.frequency_hz)
        log_freq_upper = math.log10(upper.frequency_hz)
        log_freq_target = math.log10(target_frequency_hz)
        
        # Interpolation factor
        factor = (log_freq_target - log_freq_lower) / (log_freq_upper - log_freq_lower)
        
        # Interpolated SE
        return lower.effectiveness_db + factor * (
            upper.effectiveness_db - lower.effectiveness_db
        )


class ShieldingMonitor:
    """
    Real-time shielding effectiveness monitor.
    
    Monitors shielding performance and detects degradation.
    """
    
    def __init__(
        self,
        protection_level: TEMPESTLevel = TEMPESTLevel.LEVEL_B,
        zone: ZoneClassification = ZoneClassification.ZONE_1
    ):
        self.protection_level = protection_level
        self.zone = zone
        
        self._requirements = ShieldingRequirementSet()
        self._calculator = ShieldingEffectivenessCalculator()
        
        # Monitoring state
        self._profiles: Dict[str, ShieldingProfile] = {}
        self._monitoring = False
        self._lock = threading.Lock()
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, ShieldingMeasurement], None]] = []
        
        # Degradation tracking
        self._baseline_measurements: Dict[str, Dict[float, float]] = {}
        self._degradation_threshold_db = 6.0  # Alert if SE drops by 6dB
    
    def register_location(
        self,
        location_id: str,
        shielding_type: ShieldingType,
        zone: Optional[ZoneClassification] = None
    ) -> None:
        """Register a location for shielding monitoring."""
        with self._lock:
            profile = ShieldingProfile(
                location_id=location_id,
                shielding_type=shielding_type,
                zone=zone or self.zone,
                protection_level=self.protection_level,
                measurements=[],
                requirements=self._requirements.get_requirements(
                    self.protection_level,
                    zone or self.zone
                ),
                last_certification=None,
                certification_expires=None,
                certification_authority=None,
                overall_compliance=True,
                degradation_detected=False
            )
            self._profiles[location_id] = profile
            self._baseline_measurements[location_id] = {}
    
    def add_measurement(
        self,
        location_id: str,
        measurement: ShieldingMeasurement
    ) -> Dict[str, Any]:
        """
        Add a shielding measurement and check compliance.
        
        Args:
            location_id: Location identifier
            measurement: Shielding measurement
            
        Returns:
            Compliance check results
        """
        with self._lock:
            if location_id not in self._profiles:
                raise ValueError(f"Unknown location: {location_id}")
            
            profile = self._profiles[location_id]
            profile.measurements.append(measurement)
            
            # Find applicable requirement
            applicable_req = None
            for req in profile.requirements:
                if abs(req.frequency_hz - measurement.frequency_hz) / req.frequency_hz < 0.01:
                    applicable_req = req
                    break
            
            # Check compliance
            compliant = True
            compliance_margin_db = 0.0
            
            if applicable_req:
                compliant = (
                    measurement.effectiveness_db >= 
                    applicable_req.min_effectiveness_db
                )
                compliance_margin_db = (
                    measurement.effectiveness_db - 
                    applicable_req.min_effectiveness_db
                )
            
            measurement.compliant = compliant
            
            # Check for degradation
            degraded = False
            degradation_db = 0.0
            
            if measurement.frequency_hz in self._baseline_measurements.get(location_id, {}):
                baseline = self._baseline_measurements[location_id][measurement.frequency_hz]
                degradation_db = baseline - measurement.effectiveness_db
                
                if degradation_db > self._degradation_threshold_db:
                    degraded = True
                    profile.degradation_detected = True
            else:
                # Set baseline
                self._baseline_measurements[location_id][measurement.frequency_hz] = (
                    measurement.effectiveness_db
                )
            
            # Update overall compliance
            profile.overall_compliance = all(
                m.compliant for m in profile.measurements[-20:]  # Last 20 measurements
            )
            
            # Trigger alerts if needed
            if not compliant or degraded:
                self._trigger_alerts(location_id, measurement)
            
            return {
                "compliant": compliant,
                "compliance_margin_db": compliance_margin_db,
                "degraded": degraded,
                "degradation_db": degradation_db,
                "applicable_requirement": (
                    applicable_req.min_effectiveness_db if applicable_req else None
                )
            }
    
    def register_alert_callback(
        self,
        callback: Callable[[str, ShieldingMeasurement], None]
    ) -> None:
        """Register callback for shielding alerts."""
        self._alert_callbacks.append(callback)
    
    def _trigger_alerts(
        self,
        location_id: str,
        measurement: ShieldingMeasurement
    ) -> None:
        """Trigger alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(location_id, measurement)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_profile(self, location_id: str) -> Optional[ShieldingProfile]:
        """Get shielding profile for a location."""
        with self._lock:
            return self._profiles.get(location_id)
    
    def get_all_profiles(self) -> List[ShieldingProfile]:
        """Get all shielding profiles."""
        with self._lock:
            return list(self._profiles.values())
    
    def get_status(self, location_id: str) -> Optional[ShieldingStatus]:
        """Get current shielding status for a location."""
        with self._lock:
            profile = self._profiles.get(location_id)
            if not profile:
                return None
            
            # Get most recent measurement
            if not profile.measurements:
                return None
            
            recent = profile.measurements[-1]
            
            return ShieldingStatus(
                timestamp=recent.timestamp,
                location=location_id,
                effectiveness_db=recent.effectiveness_db,
                frequency_range_hz=(
                    min(m.frequency_hz for m in profile.measurements),
                    max(m.frequency_hz for m in profile.measurements)
                ),
                test_method=recent.test_method.value,
                compliant=profile.overall_compliance,
                certification_expires=profile.certification_expires
            )
    
    def update_certification(
        self,
        location_id: str,
        certification_date: datetime,
        expires: datetime,
        authority: str
    ) -> None:
        """Update certification information for a location."""
        with self._lock:
            if location_id in self._profiles:
                profile = self._profiles[location_id]
                profile.last_certification = certification_date
                profile.certification_expires = expires
                profile.certification_authority = authority
    
    def check_certification_status(self) -> Dict[str, Dict[str, Any]]:
        """Check certification status for all locations."""
        status = {}
        now = datetime.utcnow()
        
        with self._lock:
            for loc_id, profile in self._profiles.items():
                if profile.certification_expires:
                    days_until_expiry = (
                        profile.certification_expires - now
                    ).days
                    expired = days_until_expiry < 0
                    expiring_soon = 0 <= days_until_expiry <= 30
                else:
                    days_until_expiry = None
                    expired = True
                    expiring_soon = False
                
                status[loc_id] = {
                    "certified": profile.last_certification is not None,
                    "certification_date": (
                        profile.last_certification.isoformat() 
                        if profile.last_certification else None
                    ),
                    "expires": (
                        profile.certification_expires.isoformat()
                        if profile.certification_expires else None
                    ),
                    "authority": profile.certification_authority,
                    "days_until_expiry": days_until_expiry,
                    "expired": expired,
                    "expiring_soon": expiring_soon,
                    "compliant": profile.overall_compliance,
                    "degradation_detected": profile.degradation_detected
                }
        
        return status


class ShieldingTestManager:
    """
    Manager for shielding effectiveness testing.
    
    Coordinates test procedures and validates results
    per applicable standards.
    """
    
    def __init__(self):
        self._test_procedures: Dict[TestMethod, Dict[str, Any]] = {}
        self._load_test_procedures()
        self._calculator = ShieldingEffectivenessCalculator()
    
    def _load_test_procedures(self) -> None:
        """Load standard test procedures."""
        # IEEE 299 procedure
        self._test_procedures[TestMethod.IEEE_299] = {
            "name": "IEEE 299 Shielding Effectiveness Test",
            "standard": "IEEE 299-2006",
            "frequency_ranges": [
                (9e3, 20e6, MeasurementType.MAGNETIC_FIELD),
                (20e6, 300e6, MeasurementType.PLANE_WAVE),
                (300e6, 18e9, MeasurementType.PLANE_WAVE)
            ],
            "antenna_positions": [
                "center", "corner", "edge"
            ],
            "num_measurements_per_freq": 4,
            "environmental_requirements": {
                "temperature_c": (15, 35),
                "humidity_percent": (20, 80)
            }
        }
        
        # MIL-STD-461G procedure
        self._test_procedures[TestMethod.MIL_STD_461G] = {
            "name": "MIL-STD-461G Shielding Effectiveness Test",
            "standard": "MIL-STD-461G (RE102, RS103)",
            "frequency_ranges": [
                (10e3, 18e9, MeasurementType.ELECTRIC_FIELD),
                (10e3, 18e9, MeasurementType.MAGNETIC_FIELD)
            ],
            "test_levels": {
                "navy_below_deck": 100.0,
                "navy_above_deck": 120.0,
                "army": 80.0,
                "air_force": 90.0
            },
            "antenna_distance_m": 1.0
        }
        
        # NSA 65-6 (TEMPEST) procedure
        self._test_procedures[TestMethod.NSA_65_6] = {
            "name": "NSA 65-6 TEMPEST Test",
            "standard": "NSA/CSS EPL",
            "classified": True,
            "zones": ["0", "1", "2", "3"],
            "test_categories": [
                "RED equipment",
                "BLACK equipment",
                "Interconnect"
            ]
        }
    
    def get_procedure(self, method: TestMethod) -> Dict[str, Any]:
        """Get test procedure details."""
        return self._test_procedures.get(method, {})
    
    def generate_test_plan(
        self,
        location_id: str,
        shielding_type: ShieldingType,
        protection_level: TEMPESTLevel,
        test_method: TestMethod
    ) -> Dict[str, Any]:
        """
        Generate a test plan for shielding effectiveness testing.
        
        Args:
            location_id: Location to test
            shielding_type: Type of shielding
            protection_level: Required protection level
            test_method: Test method to use
            
        Returns:
            Complete test plan
        """
        procedure = self._test_procedures.get(test_method, {})
        requirements = ShieldingRequirementSet()
        
        # Generate frequency list
        test_frequencies = []
        if "frequency_ranges" in procedure:
            for freq_range in procedure["frequency_ranges"]:
                start, end, meas_type = freq_range
                # Generate 5 points per decade
                log_start = math.log10(start)
                log_end = math.log10(end)
                decades = log_end - log_start
                num_points = max(int(decades * 5), 3)
                
                for i in range(num_points):
                    freq = 10 ** (log_start + i * (log_end - log_start) / (num_points - 1))
                    test_frequencies.append({
                        "frequency_hz": freq,
                        "measurement_type": meas_type.value
                    })
        
        # Get requirements
        req_list = requirements.get_requirements(
            protection_level, ZoneClassification.ZONE_1
        )
        
        # Build test plan
        test_plan = {
            "test_plan_id": hashlib.sha256(
                f"{location_id}-{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:16],
            "location_id": location_id,
            "shielding_type": shielding_type.value,
            "protection_level": protection_level.value,
            "test_method": test_method.value,
            "procedure": procedure.get("name", "Custom"),
            "standard": procedure.get("standard", "N/A"),
            "test_frequencies": test_frequencies,
            "requirements": [
                {
                    "frequency_hz": r.frequency_hz,
                    "min_effectiveness_db": r.min_effectiveness_db,
                    "measurement_type": r.measurement_type.value
                }
                for r in req_list
            ],
            "equipment_required": [
                "Signal generator with required frequency range",
                "Spectrum analyzer or field strength meter",
                "Calibrated antennas for each frequency range",
                "Reference antenna for baseline measurements",
                "Environmental monitoring equipment"
            ],
            "estimated_duration_hours": len(test_frequencies) * 0.5,
            "generated_date": datetime.utcnow().isoformat()
        }
        
        return test_plan
    
    def validate_measurements(
        self,
        measurements: List[ShieldingMeasurement],
        requirements: List[ShieldingRequirement]
    ) -> Dict[str, Any]:
        """
        Validate measurements against requirements.
        
        Args:
            measurements: List of measurements
            requirements: List of requirements
            
        Returns:
            Validation results
        """
        results = {
            "overall_pass": True,
            "total_requirements": len(requirements),
            "requirements_met": 0,
            "requirements_failed": 0,
            "details": [],
            "min_margin_db": float("inf"),
            "worst_frequency_hz": None
        }
        
        for req in requirements:
            # Find closest measurement to requirement frequency
            closest_meas = None
            min_freq_diff = float("inf")
            
            for meas in measurements:
                freq_diff = abs(meas.frequency_hz - req.frequency_hz)
                if freq_diff < min_freq_diff:
                    min_freq_diff = freq_diff
                    closest_meas = meas
            
            if closest_meas is None:
                results["requirements_failed"] += 1
                results["overall_pass"] = False
                results["details"].append({
                    "frequency_hz": req.frequency_hz,
                    "required_db": req.min_effectiveness_db,
                    "measured_db": None,
                    "margin_db": None,
                    "pass": False,
                    "reason": "No measurement available"
                })
                continue
            
            # Check if frequency match is acceptable (within 10%)
            if min_freq_diff / req.frequency_hz > 0.1:
                results["requirements_failed"] += 1
                results["overall_pass"] = False
                results["details"].append({
                    "frequency_hz": req.frequency_hz,
                    "required_db": req.min_effectiveness_db,
                    "measured_db": closest_meas.effectiveness_db,
                    "margin_db": None,
                    "pass": False,
                    "reason": "Frequency mismatch too large"
                })
                continue
            
            # Calculate margin
            margin = closest_meas.effectiveness_db - req.min_effectiveness_db
            passed = margin >= 0
            
            if passed:
                results["requirements_met"] += 1
            else:
                results["requirements_failed"] += 1
                results["overall_pass"] = False
            
            if margin < results["min_margin_db"]:
                results["min_margin_db"] = margin
                results["worst_frequency_hz"] = req.frequency_hz
            
            results["details"].append({
                "frequency_hz": req.frequency_hz,
                "required_db": req.min_effectiveness_db,
                "measured_db": closest_meas.effectiveness_db,
                "margin_db": margin,
                "pass": passed,
                "uncertainty_db": closest_meas.uncertainty_db
            })
        
        return results
    
    def generate_certificate(
        self,
        location_id: str,
        validation_results: Dict[str, Any],
        authority: str,
        validity_years: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Generate shielding effectiveness certificate.
        
        Args:
            location_id: Location identifier
            validation_results: Validation results
            authority: Certifying authority
            validity_years: Certificate validity in years
            
        Returns:
            Certificate data or None if validation failed
        """
        if not validation_results.get("overall_pass", False):
            return None
        
        now = datetime.utcnow()
        expires = now + timedelta(days=365 * validity_years)
        
        # Generate certificate ID
        cert_data = f"{location_id}-{authority}-{now.isoformat()}"
        cert_id = hashlib.sha256(cert_data.encode()).hexdigest()[:24].upper()
        
        certificate = {
            "certificate_id": cert_id,
            "location_id": location_id,
            "certifying_authority": authority,
            "issue_date": now.isoformat(),
            "expiration_date": expires.isoformat(),
            "validity_years": validity_years,
            "test_results_summary": {
                "total_requirements": validation_results["total_requirements"],
                "requirements_met": validation_results["requirements_met"],
                "min_margin_db": validation_results["min_margin_db"],
                "worst_frequency_hz": validation_results["worst_frequency_hz"]
            },
            "certification_statement": (
                f"This certifies that location '{location_id}' has been tested "
                f"and meets all specified shielding effectiveness requirements. "
                f"Certificate valid until {expires.strftime('%Y-%m-%d')}."
            ),
            "digital_signature": hashlib.sha512(
                f"{cert_id}-{authority}".encode()
            ).hexdigest()
        }
        
        return certificate


# Export public API
__all__ = [
    "ShieldingType",
    "TestMethod",
    "MeasurementType",
    "ShieldingRequirement",
    "ShieldingMeasurement",
    "ShieldingProfile",
    "ShieldingRequirementSet",
    "ShieldingEffectivenessCalculator",
    "ShieldingMonitor",
    "ShieldingTestManager",
]
