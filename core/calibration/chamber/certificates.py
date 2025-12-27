"""
Calibration Certificate Generation Module.

This module provides comprehensive calibration certificate generation
capabilities for RF chamber testing and equipment validation per
ISO 17025 and industry standards.

Features:
- ISO 17025 compliant certificate generation
- Digital signature and integrity verification
- Measurement uncertainty documentation
- Traceability chain documentation
- Certificate database management
- Automated expiry tracking and notifications
- Multi-format export (PDF, XML, JSON, CSV)
- Certificate validation and authentication

Standards Compliance:
- ISO/IEC 17025:2017: Testing and calibration laboratory requirements
- ISO 10012: Measurement management systems
- ILAC P14: ILAC Policy on measurement uncertainty
- ANSI/NCSL Z540.3: Requirements for calibration laboratories

Author: RF Arsenal Development Team
License: Proprietary - Calibration Sensitive
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import struct
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Certificate Enumerations
# ============================================================================

class CertificateType(Enum):
    """Types of calibration certificates."""
    
    FULL_CALIBRATION = "full_calibration"
    VERIFICATION = "verification"
    FUNCTIONAL_CHECK = "functional_check"
    LIMITED_CALIBRATION = "limited_calibration"
    ADJUSTMENT_ONLY = "adjustment_only"
    AS_FOUND_AS_LEFT = "as_found_as_left"
    TRACEABILITY = "traceability"
    ACCREDITED = "accredited"
    MANUFACTURER = "manufacturer"


class CertificateStatus(Enum):
    """Status of calibration certificates."""
    
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ISSUED = "issued"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"
    EXPIRED = "expired"
    ARCHIVED = "archived"


class AccreditationBody(Enum):
    """Recognized accreditation bodies."""
    
    NVLAP = "nvlap"            # National Voluntary Laboratory Accreditation Program
    A2LA = "a2la"              # American Association for Laboratory Accreditation
    ANAB = "anab"              # ANSI National Accreditation Board
    UKAS = "ukas"              # United Kingdom Accreditation Service
    DAkkS = "dakks"            # German Accreditation Body
    ILAC = "ilac"              # International Laboratory Accreditation Cooperation
    NIST = "nist"              # NIST Direct Calibration
    PTB = "ptb"                # Physikalisch-Technische Bundesanstalt
    NPL = "npl"                # National Physical Laboratory (UK)
    INTERNAL = "internal"      # Internal calibration (non-accredited)


class UncertaintyType(Enum):
    """Types of measurement uncertainty."""
    
    TYPE_A = "type_a"          # Statistical analysis of measurements
    TYPE_B = "type_b"          # Other means (manuals, certificates)
    COMBINED = "combined"       # Combined standard uncertainty
    EXPANDED = "expanded"       # Expanded uncertainty (k factor)
    COVERAGE = "coverage"       # Coverage probability


class SignatureAlgorithm(Enum):
    """Digital signature algorithms for certificates."""
    
    HMAC_SHA256 = "hmac_sha256"
    HMAC_SHA384 = "hmac_sha384"
    HMAC_SHA512 = "hmac_sha512"
    RSA_SHA256 = "rsa_sha256"
    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"
    ED25519 = "ed25519"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LaboratoryInfo:
    """Calibration laboratory information."""
    
    name: str
    address: str
    city: str
    state: str
    postal_code: str
    country: str
    phone: str = ""
    email: str = ""
    website: str = ""
    accreditation_number: str = ""
    accreditation_body: AccreditationBody = AccreditationBody.INTERNAL
    accreditation_scope: str = ""
    accreditation_expiry: Optional[datetime] = None
    quality_manager: str = ""
    technical_manager: str = ""
    logo_path: str = ""
    
    def is_accredited(self) -> bool:
        """Check if laboratory has valid accreditation."""
        if self.accreditation_body == AccreditationBody.INTERNAL:
            return False
        if not self.accreditation_number:
            return False
        if self.accreditation_expiry and datetime.now() > self.accreditation_expiry:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'address': self.address,
            'city': self.city,
            'state': self.state,
            'postal_code': self.postal_code,
            'country': self.country,
            'phone': self.phone,
            'email': self.email,
            'website': self.website,
            'accreditation_number': self.accreditation_number,
            'accreditation_body': self.accreditation_body.value,
            'accreditation_scope': self.accreditation_scope,
            'accreditation_expiry': self.accreditation_expiry.isoformat() if self.accreditation_expiry else None,
            'quality_manager': self.quality_manager,
            'technical_manager': self.technical_manager,
            'is_accredited': self.is_accredited()
        }


@dataclass
class EquipmentInfo:
    """Equipment under test information."""
    
    description: str
    manufacturer: str
    model: str
    serial_number: str
    asset_id: str = ""
    firmware_version: str = ""
    hardware_revision: str = ""
    owner: str = ""
    department: str = ""
    location: str = ""
    date_received: Optional[datetime] = None
    condition_received: str = "Good"
    accessories: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'description': self.description,
            'manufacturer': self.manufacturer,
            'model': self.model,
            'serial_number': self.serial_number,
            'asset_id': self.asset_id,
            'firmware_version': self.firmware_version,
            'hardware_revision': self.hardware_revision,
            'owner': self.owner,
            'department': self.department,
            'location': self.location,
            'date_received': self.date_received.isoformat() if self.date_received else None,
            'condition_received': self.condition_received,
            'accessories': self.accessories,
            'notes': self.notes
        }


@dataclass
class ReferenceStandard:
    """Reference standard used for calibration."""
    
    description: str
    manufacturer: str
    model: str
    serial_number: str
    certificate_number: str
    certificate_date: datetime
    certificate_expiry: datetime
    traceability: str
    uncertainty: float
    uncertainty_unit: str
    measurement_parameter: str
    measurement_range: str
    calibration_lab: str = ""
    calibration_body: AccreditationBody = AccreditationBody.NIST
    
    def is_valid(self) -> bool:
        """Check if reference standard certificate is valid."""
        return datetime.now() < self.certificate_expiry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'description': self.description,
            'manufacturer': self.manufacturer,
            'model': self.model,
            'serial_number': self.serial_number,
            'certificate_number': self.certificate_number,
            'certificate_date': self.certificate_date.isoformat(),
            'certificate_expiry': self.certificate_expiry.isoformat(),
            'traceability': self.traceability,
            'uncertainty': self.uncertainty,
            'uncertainty_unit': self.uncertainty_unit,
            'measurement_parameter': self.measurement_parameter,
            'measurement_range': self.measurement_range,
            'calibration_lab': self.calibration_lab,
            'calibration_body': self.calibration_body.value,
            'is_valid': self.is_valid()
        }


@dataclass
class UncertaintyComponent:
    """Individual uncertainty component in budget."""
    
    name: str
    source: str
    value: float
    unit: str
    distribution: str  # normal, rectangular, triangular, u-shaped
    divisor: float     # Distribution divisor
    sensitivity: float = 1.0
    degrees_of_freedom: float = float('inf')
    uncertainty_type: UncertaintyType = UncertaintyType.TYPE_B
    
    def standard_uncertainty(self) -> float:
        """Calculate standard uncertainty."""
        return (self.value / self.divisor) * abs(self.sensitivity)
    
    def variance_contribution(self) -> float:
        """Calculate variance contribution."""
        return self.standard_uncertainty() ** 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'source': self.source,
            'value': self.value,
            'unit': self.unit,
            'distribution': self.distribution,
            'divisor': self.divisor,
            'sensitivity': self.sensitivity,
            'degrees_of_freedom': self.degrees_of_freedom if self.degrees_of_freedom != float('inf') else 'infinite',
            'uncertainty_type': self.uncertainty_type.value,
            'standard_uncertainty': self.standard_uncertainty(),
            'variance_contribution': self.variance_contribution()
        }


@dataclass
class UncertaintyBudget:
    """Complete uncertainty budget for a measurement."""
    
    measurement_parameter: str
    components: List[UncertaintyComponent]
    coverage_factor: float = 2.0
    coverage_probability: float = 0.95
    unit: str = ""
    notes: str = ""
    
    def combined_uncertainty(self) -> float:
        """Calculate combined standard uncertainty."""
        variance_sum = sum(c.variance_contribution() for c in self.components)
        return np.sqrt(variance_sum)
    
    def expanded_uncertainty(self) -> float:
        """Calculate expanded uncertainty."""
        return self.combined_uncertainty() * self.coverage_factor
    
    def effective_degrees_of_freedom(self) -> float:
        """Calculate effective degrees of freedom using Welch-Satterthwaite."""
        combined_var = sum(c.variance_contribution() for c in self.components)
        if combined_var == 0:
            return float('inf')
        
        numerator = combined_var ** 2
        denominator = sum(
            c.variance_contribution() ** 2 / c.degrees_of_freedom
            for c in self.components
            if c.degrees_of_freedom != float('inf')
        )
        
        if denominator == 0:
            return float('inf')
        
        return numerator / denominator
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'measurement_parameter': self.measurement_parameter,
            'components': [c.to_dict() for c in self.components],
            'coverage_factor': self.coverage_factor,
            'coverage_probability': self.coverage_probability,
            'unit': self.unit,
            'notes': self.notes,
            'combined_uncertainty': self.combined_uncertainty(),
            'expanded_uncertainty': self.expanded_uncertainty(),
            'effective_degrees_of_freedom': self.effective_degrees_of_freedom()
        }


@dataclass
class MeasurementResult:
    """Individual measurement result."""
    
    parameter: str
    nominal_value: float
    measured_value: float
    unit: str
    tolerance_low: Optional[float] = None
    tolerance_high: Optional[float] = None
    uncertainty: Optional[float] = None
    uncertainty_budget: Optional[UncertaintyBudget] = None
    as_found: Optional[float] = None
    as_left: Optional[float] = None
    adjustment_made: bool = False
    test_point_id: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def error(self) -> float:
        """Calculate measurement error."""
        return self.measured_value - self.nominal_value
    
    def error_percent(self) -> float:
        """Calculate percentage error."""
        if self.nominal_value == 0:
            return 0.0
        return (self.error() / self.nominal_value) * 100
    
    def is_in_tolerance(self) -> bool:
        """Check if measurement is within tolerance."""
        if self.tolerance_low is not None and self.measured_value < self.tolerance_low:
            return False
        if self.tolerance_high is not None and self.measured_value > self.tolerance_high:
            return False
        return True
    
    def conformance_statement(self) -> str:
        """Generate conformance statement per ISO 17025."""
        if self.tolerance_low is None and self.tolerance_high is None:
            return "N/A - No tolerance specified"
        
        if self.is_in_tolerance():
            if self.uncertainty is not None:
                # Check if uncertainty affects conformance decision
                error = abs(self.error())
                guard_band = 0  # Simple acceptance
                
                if self.tolerance_high is not None:
                    margin_high = self.tolerance_high - self.measured_value
                    if margin_high < self.uncertainty:
                        return "PASS (conditional - uncertainty may affect conformance)"
                
                if self.tolerance_low is not None:
                    margin_low = self.measured_value - self.tolerance_low
                    if margin_low < self.uncertainty:
                        return "PASS (conditional - uncertainty may affect conformance)"
                
                return "PASS (conformance statement applies)"
            return "PASS"
        else:
            return "FAIL (out of tolerance)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'parameter': self.parameter,
            'nominal_value': self.nominal_value,
            'measured_value': self.measured_value,
            'unit': self.unit,
            'tolerance_low': self.tolerance_low,
            'tolerance_high': self.tolerance_high,
            'uncertainty': self.uncertainty,
            'uncertainty_budget': self.uncertainty_budget.to_dict() if self.uncertainty_budget else None,
            'as_found': self.as_found,
            'as_left': self.as_left,
            'adjustment_made': self.adjustment_made,
            'test_point_id': self.test_point_id,
            'conditions': self.conditions,
            'timestamp': self.timestamp.isoformat(),
            'error': self.error(),
            'error_percent': self.error_percent(),
            'is_in_tolerance': self.is_in_tolerance(),
            'conformance_statement': self.conformance_statement()
        }


@dataclass
class EnvironmentalConditions:
    """Environmental conditions during calibration."""
    
    temperature_c: float
    temperature_tolerance_c: float = 2.0
    humidity_percent: float = 50.0
    humidity_tolerance_percent: float = 10.0
    pressure_hpa: float = 1013.25
    pressure_tolerance_hpa: float = 50.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    notes: str = ""
    
    def is_within_specification(self) -> bool:
        """Check if conditions are within specification."""
        # Standard laboratory conditions: 23°C ± 2°C, 50% ± 10% RH
        if abs(self.temperature_c - 23.0) > self.temperature_tolerance_c:
            return False
        if abs(self.humidity_percent - 50.0) > self.humidity_tolerance_percent:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'temperature_c': self.temperature_c,
            'temperature_tolerance_c': self.temperature_tolerance_c,
            'humidity_percent': self.humidity_percent,
            'humidity_tolerance_percent': self.humidity_tolerance_percent,
            'pressure_hpa': self.pressure_hpa,
            'pressure_tolerance_hpa': self.pressure_tolerance_hpa,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'notes': self.notes,
            'is_within_specification': self.is_within_specification()
        }


@dataclass
class CalibrationProcedure:
    """Calibration procedure reference."""
    
    procedure_number: str
    title: str
    revision: str
    effective_date: datetime
    author: str = ""
    approver: str = ""
    scope: str = ""
    method: str = ""
    standards_referenced: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'procedure_number': self.procedure_number,
            'title': self.title,
            'revision': self.revision,
            'effective_date': self.effective_date.isoformat(),
            'author': self.author,
            'approver': self.approver,
            'scope': self.scope,
            'method': self.method,
            'standards_referenced': self.standards_referenced
        }


@dataclass 
class TechnicianInfo:
    """Technician/calibrator information."""
    
    name: str
    employee_id: str
    title: str = "Calibration Technician"
    qualifications: List[str] = field(default_factory=list)
    training_records: List[str] = field(default_factory=list)
    signature_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'employee_id': self.employee_id,
            'title': self.title,
            'qualifications': self.qualifications,
            'training_records': self.training_records,
            'signature_date': self.signature_date.isoformat() if self.signature_date else None
        }


# ============================================================================
# Certificate Data Structure
# ============================================================================

@dataclass
class CalibrationCertificate:
    """Complete calibration certificate."""
    
    # Certificate identification
    certificate_number: str
    certificate_type: CertificateType
    status: CertificateStatus = CertificateStatus.DRAFT
    
    # Dates
    calibration_date: datetime = field(default_factory=datetime.now)
    issue_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    calibration_interval_months: int = 12
    
    # Parties
    laboratory: Optional[LaboratoryInfo] = None
    equipment: Optional[EquipmentInfo] = None
    customer_info: Dict[str, Any] = field(default_factory=dict)
    
    # Calibration details
    procedures: List[CalibrationProcedure] = field(default_factory=list)
    reference_standards: List[ReferenceStandard] = field(default_factory=list)
    environmental_conditions: Optional[EnvironmentalConditions] = None
    measurement_results: List[MeasurementResult] = field(default_factory=list)
    
    # Personnel
    technician: Optional[TechnicianInfo] = None
    reviewer: Optional[TechnicianInfo] = None
    approver: Optional[TechnicianInfo] = None
    
    # Remarks and notes
    remarks: str = ""
    limitations: str = ""
    deviations: str = ""
    
    # Digital signature
    signature: Optional[bytes] = None
    signature_algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256
    certificate_hash: str = ""
    
    # Versioning
    version: int = 1
    previous_certificate: str = ""
    supersedes: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    
    def __post_init__(self):
        """Initialize certificate after creation."""
        if not self.certificate_number:
            self.certificate_number = self._generate_certificate_number()
        if self.expiry_date is None:
            self.expiry_date = self.calibration_date + timedelta(days=self.calibration_interval_months * 30)
    
    def _generate_certificate_number(self) -> str:
        """Generate unique certificate number."""
        timestamp = datetime.now().strftime("%Y%m%d")
        random_part = secrets.token_hex(4).upper()
        return f"CAL-{timestamp}-{random_part}"
    
    def overall_result(self) -> str:
        """Determine overall calibration result."""
        if not self.measurement_results:
            return "NO MEASUREMENTS"
        
        all_pass = all(r.is_in_tolerance() for r in self.measurement_results 
                      if r.tolerance_low is not None or r.tolerance_high is not None)
        
        has_tolerances = any(r.tolerance_low is not None or r.tolerance_high is not None 
                           for r in self.measurement_results)
        
        if not has_tolerances:
            return "MEASURED VALUES REPORTED"
        
        return "PASS" if all_pass else "FAIL"
    
    def is_valid(self) -> bool:
        """Check if certificate is currently valid."""
        if self.status not in [CertificateStatus.APPROVED, CertificateStatus.ISSUED]:
            return False
        if self.expiry_date and datetime.now() > self.expiry_date:
            return False
        return True
    
    def days_until_expiry(self) -> int:
        """Calculate days until certificate expires."""
        if self.expiry_date is None:
            return -1
        delta = self.expiry_date - datetime.now()
        return delta.days
    
    def calculate_hash(self) -> str:
        """Calculate certificate content hash."""
        content = json.dumps(self.to_dict(include_signature=False), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def sign(self, key: bytes) -> None:
        """Sign the certificate."""
        self.certificate_hash = self.calculate_hash()
        
        if self.signature_algorithm == SignatureAlgorithm.HMAC_SHA256:
            self.signature = hmac.new(key, self.certificate_hash.encode(), hashlib.sha256).digest()
        elif self.signature_algorithm == SignatureAlgorithm.HMAC_SHA384:
            self.signature = hmac.new(key, self.certificate_hash.encode(), hashlib.sha384).digest()
        elif self.signature_algorithm == SignatureAlgorithm.HMAC_SHA512:
            self.signature = hmac.new(key, self.certificate_hash.encode(), hashlib.sha512).digest()
        else:
            raise ValueError(f"Unsupported signature algorithm: {self.signature_algorithm}")
    
    def verify_signature(self, key: bytes) -> bool:
        """Verify certificate signature."""
        if self.signature is None:
            return False
        
        current_hash = self.calculate_hash()
        if current_hash != self.certificate_hash:
            return False
        
        if self.signature_algorithm == SignatureAlgorithm.HMAC_SHA256:
            expected = hmac.new(key, self.certificate_hash.encode(), hashlib.sha256).digest()
        elif self.signature_algorithm == SignatureAlgorithm.HMAC_SHA384:
            expected = hmac.new(key, self.certificate_hash.encode(), hashlib.sha384).digest()
        elif self.signature_algorithm == SignatureAlgorithm.HMAC_SHA512:
            expected = hmac.new(key, self.certificate_hash.encode(), hashlib.sha512).digest()
        else:
            return False
        
        return hmac.compare_digest(self.signature, expected)
    
    def to_dict(self, include_signature: bool = True) -> Dict[str, Any]:
        """Convert certificate to dictionary."""
        result = {
            'certificate_number': self.certificate_number,
            'certificate_type': self.certificate_type.value,
            'status': self.status.value,
            'calibration_date': self.calibration_date.isoformat(),
            'issue_date': self.issue_date.isoformat() if self.issue_date else None,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'calibration_interval_months': self.calibration_interval_months,
            'laboratory': self.laboratory.to_dict() if self.laboratory else None,
            'equipment': self.equipment.to_dict() if self.equipment else None,
            'customer_info': self.customer_info,
            'procedures': [p.to_dict() for p in self.procedures],
            'reference_standards': [r.to_dict() for r in self.reference_standards],
            'environmental_conditions': self.environmental_conditions.to_dict() if self.environmental_conditions else None,
            'measurement_results': [m.to_dict() for m in self.measurement_results],
            'technician': self.technician.to_dict() if self.technician else None,
            'reviewer': self.reviewer.to_dict() if self.reviewer else None,
            'approver': self.approver.to_dict() if self.approver else None,
            'remarks': self.remarks,
            'limitations': self.limitations,
            'deviations': self.deviations,
            'version': self.version,
            'previous_certificate': self.previous_certificate,
            'supersedes': self.supersedes,
            'overall_result': self.overall_result(),
            'is_valid': self.is_valid(),
            'days_until_expiry': self.days_until_expiry(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'created_by': self.created_by
        }
        
        if include_signature:
            result['signature'] = base64.b64encode(self.signature).decode() if self.signature else None
            result['signature_algorithm'] = self.signature_algorithm.value
            result['certificate_hash'] = self.certificate_hash
        
        return result


# ============================================================================
# Certificate Generator
# ============================================================================

class CertificateGenerator:
    """
    Generator for creating calibration certificates.
    
    Supports multiple output formats and templates with
    full ISO 17025 compliance features.
    """
    
    def __init__(
        self,
        laboratory: Optional[LaboratoryInfo] = None,
        signing_key: Optional[bytes] = None,
        template_dir: str = "",
        output_dir: str = ""
    ):
        """
        Initialize certificate generator.
        
        Args:
            laboratory: Default laboratory information
            signing_key: Key for digital signatures
            template_dir: Directory containing certificate templates
            output_dir: Directory for generated certificates
        """
        self.laboratory = laboratory
        self.signing_key = signing_key or secrets.token_bytes(32)
        self.template_dir = template_dir
        self.output_dir = output_dir
        
        # Certificate registry
        self._certificates: Dict[str, CalibrationCertificate] = {}
        self._certificate_counter = 0
        
        # Templates
        self._templates: Dict[str, str] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info("CertificateGenerator initialized")
    
    def create_certificate(
        self,
        equipment: EquipmentInfo,
        measurement_results: List[MeasurementResult],
        certificate_type: CertificateType = CertificateType.FULL_CALIBRATION,
        procedures: Optional[List[CalibrationProcedure]] = None,
        reference_standards: Optional[List[ReferenceStandard]] = None,
        environmental_conditions: Optional[EnvironmentalConditions] = None,
        technician: Optional[TechnicianInfo] = None,
        calibration_interval_months: int = 12,
        remarks: str = "",
        customer_info: Optional[Dict[str, Any]] = None
    ) -> CalibrationCertificate:
        """
        Create a new calibration certificate.
        
        Args:
            equipment: Equipment under test information
            measurement_results: List of measurement results
            certificate_type: Type of certificate
            procedures: Calibration procedures used
            reference_standards: Reference standards used
            environmental_conditions: Environmental conditions
            technician: Technician information
            calibration_interval_months: Calibration interval
            remarks: Additional remarks
            customer_info: Customer information
        
        Returns:
            Created calibration certificate
        """
        with self._lock:
            self._certificate_counter += 1
            
            certificate = CalibrationCertificate(
                certificate_number=self._generate_certificate_number(),
                certificate_type=certificate_type,
                laboratory=self.laboratory,
                equipment=equipment,
                measurement_results=measurement_results,
                procedures=procedures or [],
                reference_standards=reference_standards or [],
                environmental_conditions=environmental_conditions,
                technician=technician,
                calibration_interval_months=calibration_interval_months,
                remarks=remarks,
                customer_info=customer_info or {}
            )
            
            # Store certificate
            self._certificates[certificate.certificate_number] = certificate
            
            logger.info(f"Created certificate: {certificate.certificate_number}")
            return certificate
    
    def _generate_certificate_number(self) -> str:
        """Generate unique certificate number."""
        timestamp = datetime.now().strftime("%Y%m%d")
        counter = str(self._certificate_counter).zfill(4)
        lab_code = "RFA"  # RF Arsenal
        return f"{lab_code}-CAL-{timestamp}-{counter}"
    
    def sign_certificate(self, certificate: CalibrationCertificate) -> None:
        """
        Sign a certificate with digital signature.
        
        Args:
            certificate: Certificate to sign
        """
        certificate.sign(self.signing_key)
        certificate.status = CertificateStatus.APPROVED
        certificate.issue_date = datetime.now()
        logger.info(f"Signed certificate: {certificate.certificate_number}")
    
    def verify_certificate(self, certificate: CalibrationCertificate) -> bool:
        """
        Verify a certificate's digital signature.
        
        Args:
            certificate: Certificate to verify
        
        Returns:
            True if signature is valid
        """
        return certificate.verify_signature(self.signing_key)
    
    def export_json(self, certificate: CalibrationCertificate, filepath: str = "") -> str:
        """
        Export certificate to JSON format.
        
        Args:
            certificate: Certificate to export
            filepath: Optional file path
        
        Returns:
            JSON string
        """
        data = certificate.to_dict()
        json_str = json.dumps(data, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Exported certificate to: {filepath}")
        
        return json_str
    
    def export_xml(self, certificate: CalibrationCertificate, filepath: str = "") -> str:
        """
        Export certificate to XML format.
        
        Args:
            certificate: Certificate to export
            filepath: Optional file path
        
        Returns:
            XML string
        """
        def dict_to_xml(d: Dict[str, Any], root_name: str = "root") -> str:
            """Convert dictionary to XML string."""
            xml_parts = [f"<{root_name}>"]
            
            for key, value in d.items():
                if value is None:
                    xml_parts.append(f"  <{key}/>")
                elif isinstance(value, dict):
                    xml_parts.append(dict_to_xml(value, key))
                elif isinstance(value, list):
                    xml_parts.append(f"  <{key}>")
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            xml_parts.append(dict_to_xml(item, "item"))
                        else:
                            xml_parts.append(f"    <item>{self._xml_escape(str(item))}</item>")
                    xml_parts.append(f"  </{key}>")
                else:
                    xml_parts.append(f"  <{key}>{self._xml_escape(str(value))}</{key}>")
            
            xml_parts.append(f"</{root_name}>")
            return "\n".join(xml_parts)
        
        data = certificate.to_dict()
        xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_str += dict_to_xml(data, "CalibrationCertificate")
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(xml_str)
            logger.info(f"Exported certificate to: {filepath}")
        
        return xml_str
    
    def _xml_escape(self, text: str) -> str:
        """Escape special XML characters."""
        replacements = [
            ('&', '&amp;'),
            ('<', '&lt;'),
            ('>', '&gt;'),
            ('"', '&quot;'),
            ("'", '&apos;')
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text
    
    def export_csv(self, certificate: CalibrationCertificate, filepath: str = "") -> str:
        """
        Export measurement results to CSV format.
        
        Args:
            certificate: Certificate to export
            filepath: Optional file path
        
        Returns:
            CSV string
        """
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header information
        writer.writerow(["Certificate Information"])
        writer.writerow(["Certificate Number", certificate.certificate_number])
        writer.writerow(["Calibration Date", certificate.calibration_date.isoformat()])
        writer.writerow(["Equipment", f"{certificate.equipment.manufacturer} {certificate.equipment.model}" if certificate.equipment else ""])
        writer.writerow(["Serial Number", certificate.equipment.serial_number if certificate.equipment else ""])
        writer.writerow([])
        
        # Measurement results header
        writer.writerow([
            "Parameter",
            "Nominal Value",
            "Measured Value",
            "Unit",
            "Tolerance Low",
            "Tolerance High",
            "Error",
            "Error %",
            "Uncertainty",
            "In Tolerance",
            "Conformance"
        ])
        
        # Measurement data
        for result in certificate.measurement_results:
            writer.writerow([
                result.parameter,
                result.nominal_value,
                result.measured_value,
                result.unit,
                result.tolerance_low if result.tolerance_low is not None else "",
                result.tolerance_high if result.tolerance_high is not None else "",
                result.error(),
                f"{result.error_percent():.4f}",
                result.uncertainty if result.uncertainty else "",
                "Yes" if result.is_in_tolerance() else "No",
                result.conformance_statement()
            ])
        
        csv_str = output.getvalue()
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(csv_str)
            logger.info(f"Exported certificate to: {filepath}")
        
        return csv_str
    
    def generate_text_report(self, certificate: CalibrationCertificate) -> str:
        """
        Generate human-readable text report.
        
        Args:
            certificate: Certificate to report
        
        Returns:
            Text report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("CALIBRATION CERTIFICATE")
        lines.append("=" * 80)
        lines.append("")
        
        # Certificate information
        lines.append(f"Certificate Number: {certificate.certificate_number}")
        lines.append(f"Certificate Type:   {certificate.certificate_type.value.replace('_', ' ').title()}")
        lines.append(f"Status:             {certificate.status.value.replace('_', ' ').title()}")
        lines.append(f"Overall Result:     {certificate.overall_result()}")
        lines.append("")
        
        # Dates
        lines.append("-" * 40)
        lines.append("DATES")
        lines.append("-" * 40)
        lines.append(f"Calibration Date:   {certificate.calibration_date.strftime('%Y-%m-%d')}")
        if certificate.issue_date:
            lines.append(f"Issue Date:         {certificate.issue_date.strftime('%Y-%m-%d')}")
        if certificate.expiry_date:
            lines.append(f"Expiry Date:        {certificate.expiry_date.strftime('%Y-%m-%d')}")
            lines.append(f"Days Until Expiry:  {certificate.days_until_expiry()}")
        lines.append("")
        
        # Laboratory information
        if certificate.laboratory:
            lines.append("-" * 40)
            lines.append("CALIBRATION LABORATORY")
            lines.append("-" * 40)
            lines.append(f"Name:          {certificate.laboratory.name}")
            lines.append(f"Address:       {certificate.laboratory.address}")
            lines.append(f"               {certificate.laboratory.city}, {certificate.laboratory.state} {certificate.laboratory.postal_code}")
            if certificate.laboratory.is_accredited():
                lines.append(f"Accreditation: {certificate.laboratory.accreditation_body.value.upper()} #{certificate.laboratory.accreditation_number}")
            lines.append("")
        
        # Equipment information
        if certificate.equipment:
            lines.append("-" * 40)
            lines.append("EQUIPMENT UNDER TEST")
            lines.append("-" * 40)
            lines.append(f"Description:   {certificate.equipment.description}")
            lines.append(f"Manufacturer:  {certificate.equipment.manufacturer}")
            lines.append(f"Model:         {certificate.equipment.model}")
            lines.append(f"Serial Number: {certificate.equipment.serial_number}")
            if certificate.equipment.asset_id:
                lines.append(f"Asset ID:      {certificate.equipment.asset_id}")
            lines.append("")
        
        # Environmental conditions
        if certificate.environmental_conditions:
            lines.append("-" * 40)
            lines.append("ENVIRONMENTAL CONDITIONS")
            lines.append("-" * 40)
            ec = certificate.environmental_conditions
            lines.append(f"Temperature:   {ec.temperature_c:.1f} °C (±{ec.temperature_tolerance_c} °C)")
            lines.append(f"Humidity:      {ec.humidity_percent:.1f} % RH (±{ec.humidity_tolerance_percent} %)")
            lines.append(f"Pressure:      {ec.pressure_hpa:.1f} hPa")
            lines.append(f"Within Spec:   {'Yes' if ec.is_within_specification() else 'No'}")
            lines.append("")
        
        # Reference standards
        if certificate.reference_standards:
            lines.append("-" * 40)
            lines.append("REFERENCE STANDARDS")
            lines.append("-" * 40)
            for i, std in enumerate(certificate.reference_standards, 1):
                lines.append(f"{i}. {std.description}")
                lines.append(f"   Manufacturer:  {std.manufacturer} {std.model}")
                lines.append(f"   Serial Number: {std.serial_number}")
                lines.append(f"   Certificate:   {std.certificate_number}")
                lines.append(f"   Traceability:  {std.traceability}")
                lines.append(f"   Uncertainty:   ±{std.uncertainty} {std.uncertainty_unit}")
                lines.append("")
        
        # Measurement results
        lines.append("-" * 40)
        lines.append("MEASUREMENT RESULTS")
        lines.append("-" * 40)
        
        # Table header
        header = f"{'Parameter':<25} {'Nominal':>12} {'Measured':>12} {'Error':>10} {'Status':>12}"
        lines.append(header)
        lines.append("-" * len(header))
        
        for result in certificate.measurement_results:
            status = "PASS" if result.is_in_tolerance() else "FAIL"
            if result.tolerance_low is None and result.tolerance_high is None:
                status = "N/A"
            
            line = f"{result.parameter:<25} {result.nominal_value:>12.4f} {result.measured_value:>12.4f} {result.error():>10.4f} {status:>12}"
            lines.append(line)
        
        lines.append("")
        
        # Uncertainty information
        results_with_uncertainty = [r for r in certificate.measurement_results if r.uncertainty is not None]
        if results_with_uncertainty:
            lines.append("-" * 40)
            lines.append("MEASUREMENT UNCERTAINTY")
            lines.append("-" * 40)
            for result in results_with_uncertainty:
                lines.append(f"{result.parameter}: ±{result.uncertainty} {result.unit} (k=2, 95% confidence)")
            lines.append("")
        
        # Remarks
        if certificate.remarks:
            lines.append("-" * 40)
            lines.append("REMARKS")
            lines.append("-" * 40)
            lines.append(certificate.remarks)
            lines.append("")
        
        # Limitations
        if certificate.limitations:
            lines.append("-" * 40)
            lines.append("LIMITATIONS")
            lines.append("-" * 40)
            lines.append(certificate.limitations)
            lines.append("")
        
        # Personnel
        lines.append("-" * 40)
        lines.append("PERSONNEL")
        lines.append("-" * 40)
        if certificate.technician:
            lines.append(f"Calibrated By:  {certificate.technician.name} ({certificate.technician.employee_id})")
        if certificate.reviewer:
            lines.append(f"Reviewed By:    {certificate.reviewer.name} ({certificate.reviewer.employee_id})")
        if certificate.approver:
            lines.append(f"Approved By:    {certificate.approver.name} ({certificate.approver.employee_id})")
        lines.append("")
        
        # Signature verification
        if certificate.signature:
            lines.append("-" * 40)
            lines.append("DIGITAL SIGNATURE")
            lines.append("-" * 40)
            lines.append(f"Algorithm:  {certificate.signature_algorithm.value}")
            lines.append(f"Hash:       {certificate.certificate_hash[:32]}...")
            lines.append(f"Valid:      {'Yes' if self.verify_certificate(certificate) else 'No'}")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("END OF CERTIFICATE")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def get_certificate(self, certificate_number: str) -> Optional[CalibrationCertificate]:
        """
        Retrieve certificate by number.
        
        Args:
            certificate_number: Certificate number to retrieve
        
        Returns:
            Certificate if found, None otherwise
        """
        return self._certificates.get(certificate_number)
    
    def list_certificates(
        self,
        equipment_serial: str = "",
        status: Optional[CertificateStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[CalibrationCertificate]:
        """
        List certificates with optional filters.
        
        Args:
            equipment_serial: Filter by equipment serial number
            status: Filter by status
            start_date: Filter by calibration date start
            end_date: Filter by calibration date end
        
        Returns:
            List of matching certificates
        """
        results = []
        
        for cert in self._certificates.values():
            # Filter by equipment serial
            if equipment_serial and cert.equipment:
                if cert.equipment.serial_number != equipment_serial:
                    continue
            
            # Filter by status
            if status and cert.status != status:
                continue
            
            # Filter by date range
            if start_date and cert.calibration_date < start_date:
                continue
            if end_date and cert.calibration_date > end_date:
                continue
            
            results.append(cert)
        
        return sorted(results, key=lambda c: c.calibration_date, reverse=True)
    
    def get_expiring_certificates(self, days: int = 30) -> List[CalibrationCertificate]:
        """
        Get certificates expiring within specified days.
        
        Args:
            days: Number of days to check
        
        Returns:
            List of expiring certificates
        """
        results = []
        cutoff_date = datetime.now() + timedelta(days=days)
        
        for cert in self._certificates.values():
            if cert.status in [CertificateStatus.APPROVED, CertificateStatus.ISSUED]:
                if cert.expiry_date and cert.expiry_date <= cutoff_date:
                    results.append(cert)
        
        return sorted(results, key=lambda c: c.expiry_date)
    
    def revoke_certificate(self, certificate_number: str, reason: str = "") -> bool:
        """
        Revoke a certificate.
        
        Args:
            certificate_number: Certificate to revoke
            reason: Reason for revocation
        
        Returns:
            True if successfully revoked
        """
        cert = self._certificates.get(certificate_number)
        if cert:
            cert.status = CertificateStatus.REVOKED
            cert.remarks += f"\n\nREVOKED: {reason}" if reason else "\n\nREVOKED"
            cert.updated_at = datetime.now()
            logger.warning(f"Certificate revoked: {certificate_number}")
            return True
        return False


# ============================================================================
# Uncertainty Calculator
# ============================================================================

class UncertaintyCalculator:
    """
    Calculator for measurement uncertainty per GUM (Guide to Uncertainty in Measurement).
    
    Implements ISO/IEC Guide 98-3:2008 uncertainty evaluation methods.
    """
    
    @staticmethod
    def create_type_a_component(
        name: str,
        measurements: List[float],
        source: str = "Repeatability"
    ) -> UncertaintyComponent:
        """
        Create Type A uncertainty component from repeated measurements.
        
        Args:
            name: Component name
            measurements: List of repeated measurements
            source: Source description
        
        Returns:
            Type A uncertainty component
        """
        n = len(measurements)
        if n < 2:
            raise ValueError("Need at least 2 measurements for Type A evaluation")
        
        mean = np.mean(measurements)
        std = np.std(measurements, ddof=1)  # Sample standard deviation
        std_of_mean = std / np.sqrt(n)      # Standard deviation of mean
        
        return UncertaintyComponent(
            name=name,
            source=source,
            value=std_of_mean,
            unit="",  # Same as measurement
            distribution="normal",
            divisor=1.0,
            sensitivity=1.0,
            degrees_of_freedom=n - 1,
            uncertainty_type=UncertaintyType.TYPE_A
        )
    
    @staticmethod
    def create_type_b_component(
        name: str,
        value: float,
        distribution: str,
        source: str,
        unit: str = "",
        sensitivity: float = 1.0
    ) -> UncertaintyComponent:
        """
        Create Type B uncertainty component.
        
        Args:
            name: Component name
            value: Half-width of distribution
            distribution: Distribution type (normal, rectangular, triangular, u-shaped)
            source: Source description
            unit: Unit of measurement
            sensitivity: Sensitivity coefficient
        
        Returns:
            Type B uncertainty component
        """
        # Standard divisors for common distributions
        divisors = {
            "normal": 2.0,       # 95% coverage -> k=2
            "rectangular": np.sqrt(3),
            "triangular": np.sqrt(6),
            "u-shaped": np.sqrt(2),
            "uniform": np.sqrt(3)  # Same as rectangular
        }
        
        divisor = divisors.get(distribution.lower(), np.sqrt(3))
        
        return UncertaintyComponent(
            name=name,
            source=source,
            value=value,
            unit=unit,
            distribution=distribution,
            divisor=divisor,
            sensitivity=sensitivity,
            degrees_of_freedom=float('inf'),
            uncertainty_type=UncertaintyType.TYPE_B
        )
    
    @staticmethod
    def create_uncertainty_budget(
        parameter: str,
        components: List[UncertaintyComponent],
        coverage_factor: float = 2.0,
        coverage_probability: float = 0.95,
        unit: str = ""
    ) -> UncertaintyBudget:
        """
        Create complete uncertainty budget.
        
        Args:
            parameter: Measurement parameter name
            components: List of uncertainty components
            coverage_factor: Coverage factor (typically 2 for 95%)
            coverage_probability: Coverage probability
            unit: Unit of measurement
        
        Returns:
            Complete uncertainty budget
        """
        return UncertaintyBudget(
            measurement_parameter=parameter,
            components=components,
            coverage_factor=coverage_factor,
            coverage_probability=coverage_probability,
            unit=unit
        )
    
    @staticmethod
    def calculate_coverage_factor(
        confidence_level: float,
        degrees_of_freedom: float
    ) -> float:
        """
        Calculate coverage factor for given confidence level and DOF.
        
        Uses t-distribution for finite DOF, or standard normal for infinite DOF.
        
        Args:
            confidence_level: Desired confidence level (e.g., 0.95)
            degrees_of_freedom: Effective degrees of freedom
        
        Returns:
            Coverage factor k
        """
        from scipy import stats
        
        if degrees_of_freedom == float('inf') or degrees_of_freedom > 100:
            # Use normal distribution
            return stats.norm.ppf((1 + confidence_level) / 2)
        else:
            # Use t-distribution
            return stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)


# ============================================================================
# Traceability Chain Manager
# ============================================================================

class TraceabilityChainManager:
    """
    Manager for calibration traceability chains.
    
    Ensures and documents unbroken traceability to national/international standards.
    """
    
    def __init__(self):
        """Initialize traceability chain manager."""
        self._standards: Dict[str, ReferenceStandard] = {}
        self._chains: Dict[str, List[str]] = {}  # Equipment S/N -> chain of standard S/Ns
        
    def register_standard(self, standard: ReferenceStandard) -> None:
        """
        Register a reference standard.
        
        Args:
            standard: Reference standard to register
        """
        key = f"{standard.manufacturer}_{standard.model}_{standard.serial_number}"
        self._standards[key] = standard
        logger.info(f"Registered standard: {key}")
    
    def get_standard(
        self,
        manufacturer: str,
        model: str,
        serial_number: str
    ) -> Optional[ReferenceStandard]:
        """
        Get registered reference standard.
        
        Args:
            manufacturer: Standard manufacturer
            model: Standard model
            serial_number: Standard serial number
        
        Returns:
            Reference standard if found
        """
        key = f"{manufacturer}_{model}_{serial_number}"
        return self._standards.get(key)
    
    def establish_traceability_chain(
        self,
        equipment_serial: str,
        standard_chain: List[ReferenceStandard]
    ) -> bool:
        """
        Establish traceability chain for equipment.
        
        Args:
            equipment_serial: Equipment serial number
            standard_chain: Ordered list of reference standards from working to primary
        
        Returns:
            True if valid chain established
        """
        # Validate all standards in chain
        for std in standard_chain:
            if not std.is_valid():
                logger.warning(f"Invalid standard in chain: {std.serial_number}")
                return False
        
        # Store chain
        chain_keys = [f"{s.manufacturer}_{s.model}_{s.serial_number}" for s in standard_chain]
        self._chains[equipment_serial] = chain_keys
        
        logger.info(f"Established traceability chain for: {equipment_serial}")
        return True
    
    def get_traceability_chain(self, equipment_serial: str) -> List[ReferenceStandard]:
        """
        Get traceability chain for equipment.
        
        Args:
            equipment_serial: Equipment serial number
        
        Returns:
            List of reference standards in chain
        """
        chain_keys = self._chains.get(equipment_serial, [])
        return [self._standards[key] for key in chain_keys if key in self._standards]
    
    def verify_traceability(self, equipment_serial: str) -> Dict[str, Any]:
        """
        Verify traceability chain validity.
        
        Args:
            equipment_serial: Equipment serial number
        
        Returns:
            Verification results dictionary
        """
        chain = self.get_traceability_chain(equipment_serial)
        
        if not chain:
            return {
                'valid': False,
                'reason': 'No traceability chain established',
                'chain_length': 0
            }
        
        # Check all standards are valid
        invalid_standards = [s for s in chain if not s.is_valid()]
        if invalid_standards:
            return {
                'valid': False,
                'reason': 'Chain contains expired/invalid standards',
                'invalid_standards': [s.serial_number for s in invalid_standards],
                'chain_length': len(chain)
            }
        
        # Check chain reaches national/international standard
        top_level = chain[-1]
        accredited_bodies = [
            AccreditationBody.NIST,
            AccreditationBody.PTB,
            AccreditationBody.NPL,
            AccreditationBody.ILAC
        ]
        
        traceable_to_nmi = top_level.calibration_body in accredited_bodies
        
        return {
            'valid': True,
            'traceable_to_nmi': traceable_to_nmi,
            'top_level_body': top_level.calibration_body.value,
            'chain_length': len(chain),
            'combined_uncertainty': self._calculate_chain_uncertainty(chain)
        }
    
    def _calculate_chain_uncertainty(self, chain: List[ReferenceStandard]) -> float:
        """
        Calculate combined uncertainty through traceability chain.
        
        Args:
            chain: List of reference standards
        
        Returns:
            Combined uncertainty (RSS)
        """
        if not chain:
            return 0.0
        
        # Root sum square of uncertainties
        variances = [s.uncertainty ** 2 for s in chain]
        return np.sqrt(sum(variances))
    
    def generate_traceability_statement(self, equipment_serial: str) -> str:
        """
        Generate formal traceability statement.
        
        Args:
            equipment_serial: Equipment serial number
        
        Returns:
            Traceability statement string
        """
        chain = self.get_traceability_chain(equipment_serial)
        
        if not chain:
            return "Traceability: Not established"
        
        lines = [
            "TRACEABILITY STATEMENT",
            "=" * 40,
            "",
            "The calibration results are traceable to the International System",
            "of Units (SI) through an unbroken chain of calibrations:",
            ""
        ]
        
        for i, std in enumerate(chain, 1):
            lines.append(f"Level {i}:")
            lines.append(f"  Standard:     {std.description}")
            lines.append(f"  Certificate:  {std.certificate_number}")
            lines.append(f"  Calibrated by: {std.calibration_lab}")
            lines.append(f"  Traceability: {std.traceability}")
            lines.append(f"  Uncertainty:  ±{std.uncertainty} {std.uncertainty_unit}")
            lines.append("")
        
        verification = self.verify_traceability(equipment_serial)
        lines.append(f"Chain Valid: {'Yes' if verification['valid'] else 'No'}")
        if verification.get('traceable_to_nmi'):
            lines.append("Traceable to National Metrology Institute: Yes")
        
        return "\n".join(lines)


# ============================================================================
# Certificate Database
# ============================================================================

class CertificateDatabase:
    """
    Database for managing calibration certificates.
    
    Provides storage, retrieval, and query capabilities.
    """
    
    def __init__(self, storage_path: str = ""):
        """
        Initialize certificate database.
        
        Args:
            storage_path: Path for persistent storage
        """
        self.storage_path = storage_path
        self._certificates: Dict[str, CalibrationCertificate] = {}
        self._equipment_index: Dict[str, List[str]] = {}  # S/N -> certificate numbers
        self._date_index: Dict[str, List[str]] = {}  # YYYY-MM -> certificate numbers
        self._lock = threading.RLock()
        
        logger.info("CertificateDatabase initialized")
    
    def store(self, certificate: CalibrationCertificate) -> None:
        """
        Store certificate in database.
        
        Args:
            certificate: Certificate to store
        """
        with self._lock:
            cert_num = certificate.certificate_number
            self._certificates[cert_num] = certificate
            
            # Update equipment index
            if certificate.equipment:
                sn = certificate.equipment.serial_number
                if sn not in self._equipment_index:
                    self._equipment_index[sn] = []
                self._equipment_index[sn].append(cert_num)
            
            # Update date index
            date_key = certificate.calibration_date.strftime("%Y-%m")
            if date_key not in self._date_index:
                self._date_index[date_key] = []
            self._date_index[date_key].append(cert_num)
            
            logger.debug(f"Stored certificate: {cert_num}")
    
    def retrieve(self, certificate_number: str) -> Optional[CalibrationCertificate]:
        """
        Retrieve certificate by number.
        
        Args:
            certificate_number: Certificate number
        
        Returns:
            Certificate if found
        """
        return self._certificates.get(certificate_number)
    
    def find_by_equipment(self, serial_number: str) -> List[CalibrationCertificate]:
        """
        Find certificates for equipment.
        
        Args:
            serial_number: Equipment serial number
        
        Returns:
            List of certificates
        """
        cert_nums = self._equipment_index.get(serial_number, [])
        return [self._certificates[n] for n in cert_nums if n in self._certificates]
    
    def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[CalibrationCertificate]:
        """
        Find certificates in date range.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            List of certificates
        """
        results = []
        for cert in self._certificates.values():
            if start_date <= cert.calibration_date <= end_date:
                results.append(cert)
        return sorted(results, key=lambda c: c.calibration_date)
    
    def find_expiring(self, days: int = 30) -> List[CalibrationCertificate]:
        """
        Find certificates expiring within days.
        
        Args:
            days: Days until expiry
        
        Returns:
            List of expiring certificates
        """
        cutoff = datetime.now() + timedelta(days=days)
        results = []
        
        for cert in self._certificates.values():
            if cert.expiry_date and cert.expiry_date <= cutoff:
                if cert.status in [CertificateStatus.APPROVED, CertificateStatus.ISSUED]:
                    results.append(cert)
        
        return sorted(results, key=lambda c: c.expiry_date)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Statistics dictionary
        """
        total = len(self._certificates)
        by_status = {}
        by_type = {}
        by_result = {"PASS": 0, "FAIL": 0, "OTHER": 0}
        
        for cert in self._certificates.values():
            # Count by status
            status = cert.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            # Count by type
            cert_type = cert.certificate_type.value
            by_type[cert_type] = by_type.get(cert_type, 0) + 1
            
            # Count by result
            result = cert.overall_result()
            if result == "PASS":
                by_result["PASS"] += 1
            elif result == "FAIL":
                by_result["FAIL"] += 1
            else:
                by_result["OTHER"] += 1
        
        return {
            'total_certificates': total,
            'by_status': by_status,
            'by_type': by_type,
            'by_result': by_result,
            'unique_equipment': len(self._equipment_index),
            'expiring_30_days': len(self.find_expiring(30))
        }
    
    def export_database(self, filepath: str) -> None:
        """
        Export entire database to file.
        
        Args:
            filepath: Export file path
        """
        data = {
            'export_date': datetime.now().isoformat(),
            'certificates': [c.to_dict() for c in self._certificates.values()],
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported database to: {filepath}")


# ============================================================================
# RF Chamber Certificate Templates
# ============================================================================

class RFChamberCertificateTemplates:
    """
    Pre-defined certificate templates for RF chamber measurements.
    """
    
    @staticmethod
    def antenna_pattern_certificate(
        generator: CertificateGenerator,
        equipment: EquipmentInfo,
        pattern_results: Dict[str, Any],
        chamber_info: Dict[str, Any],
        technician: Optional[TechnicianInfo] = None
    ) -> CalibrationCertificate:
        """
        Create certificate for antenna pattern measurement.
        
        Args:
            generator: Certificate generator
            equipment: Antenna under test
            pattern_results: Pattern measurement results
            chamber_info: Chamber configuration info
            technician: Technician who performed measurement
        
        Returns:
            Calibration certificate
        """
        # Create measurement results
        results = []
        
        # Gain measurement
        if 'peak_gain_dbi' in pattern_results:
            results.append(MeasurementResult(
                parameter="Peak Gain",
                nominal_value=pattern_results.get('specified_gain_dbi', pattern_results['peak_gain_dbi']),
                measured_value=pattern_results['peak_gain_dbi'],
                unit="dBi",
                tolerance_low=pattern_results.get('gain_tolerance_low'),
                tolerance_high=pattern_results.get('gain_tolerance_high'),
                uncertainty=pattern_results.get('gain_uncertainty', 0.5),
                test_point_id="GAIN_001"
            ))
        
        # 3dB beamwidth
        if 'beamwidth_3db' in pattern_results:
            for plane, bw in pattern_results['beamwidth_3db'].items():
                results.append(MeasurementResult(
                    parameter=f"3dB Beamwidth ({plane})",
                    nominal_value=pattern_results.get('specified_beamwidth', {}).get(plane, bw),
                    measured_value=bw,
                    unit="degrees",
                    tolerance_low=pattern_results.get('beamwidth_tolerance_low'),
                    tolerance_high=pattern_results.get('beamwidth_tolerance_high'),
                    uncertainty=pattern_results.get('beamwidth_uncertainty', 1.0),
                    test_point_id=f"BW_{plane}_001"
                ))
        
        # Front-to-back ratio
        if 'front_to_back_db' in pattern_results:
            results.append(MeasurementResult(
                parameter="Front-to-Back Ratio",
                nominal_value=pattern_results.get('specified_f2b_db', pattern_results['front_to_back_db']),
                measured_value=pattern_results['front_to_back_db'],
                unit="dB",
                tolerance_low=pattern_results.get('f2b_tolerance_low'),
                uncertainty=pattern_results.get('f2b_uncertainty', 0.5),
                test_point_id="F2B_001"
            ))
        
        # Sidelobe level
        if 'max_sidelobe_db' in pattern_results:
            results.append(MeasurementResult(
                parameter="Maximum Sidelobe Level",
                nominal_value=pattern_results.get('specified_sll_db', pattern_results['max_sidelobe_db']),
                measured_value=pattern_results['max_sidelobe_db'],
                unit="dB",
                tolerance_high=pattern_results.get('sll_tolerance_high'),
                uncertainty=pattern_results.get('sll_uncertainty', 0.5),
                test_point_id="SLL_001"
            ))
        
        # Create procedure reference
        procedure = CalibrationProcedure(
            procedure_number="RFA-ANT-001",
            title="Antenna Pattern Measurement Procedure",
            revision="A",
            effective_date=datetime.now() - timedelta(days=90),
            scope="Antenna radiation pattern characterization",
            method=chamber_info.get('measurement_method', 'Far-field'),
            standards_referenced=["IEEE 149-2021", "CTIA OTA Test Plan"]
        )
        
        # Create environmental conditions
        env_conditions = EnvironmentalConditions(
            temperature_c=chamber_info.get('temperature_c', 23.0),
            humidity_percent=chamber_info.get('humidity_percent', 50.0),
            notes=f"Chamber: {chamber_info.get('chamber_name', 'Anechoic Chamber')}"
        )
        
        # Build certificate
        remarks = f"""
Antenna pattern measurement performed in {chamber_info.get('chamber_name', 'RF anechoic chamber')}.
Test frequency: {pattern_results.get('frequency_mhz', 'N/A')} MHz
Polarization: {pattern_results.get('polarization', 'N/A')}
Angular resolution: {pattern_results.get('angular_resolution_deg', 1.0)}°
"""
        
        return generator.create_certificate(
            equipment=equipment,
            measurement_results=results,
            certificate_type=CertificateType.FULL_CALIBRATION,
            procedures=[procedure],
            environmental_conditions=env_conditions,
            technician=technician,
            calibration_interval_months=12,
            remarks=remarks.strip()
        )
    
    @staticmethod
    def power_calibration_certificate(
        generator: CertificateGenerator,
        equipment: EquipmentInfo,
        power_results: List[Dict[str, Any]],
        reference_standard: ReferenceStandard,
        technician: Optional[TechnicianInfo] = None
    ) -> CalibrationCertificate:
        """
        Create certificate for power meter/sensor calibration.
        
        Args:
            generator: Certificate generator
            equipment: Power meter/sensor under test
            power_results: Power measurement results at various levels
            reference_standard: Reference power standard
            technician: Technician who performed calibration
        
        Returns:
            Calibration certificate
        """
        results = []
        
        for i, pr in enumerate(power_results):
            # Create uncertainty budget for each point
            uncertainty_components = [
                UncertaintyCalculator.create_type_b_component(
                    name="Reference standard",
                    value=reference_standard.uncertainty,
                    distribution="normal",
                    source=f"Certificate {reference_standard.certificate_number}"
                ),
                UncertaintyCalculator.create_type_b_component(
                    name="Mismatch",
                    value=pr.get('mismatch_uncertainty', 0.02),
                    distribution="u-shaped",
                    source="Mismatch loss calculation"
                ),
                UncertaintyCalculator.create_type_b_component(
                    name="Connector repeatability",
                    value=pr.get('connector_uncertainty', 0.01),
                    distribution="rectangular",
                    source="Type N connector spec"
                ),
                UncertaintyCalculator.create_type_b_component(
                    name="Resolution",
                    value=pr.get('resolution', 0.01),
                    distribution="rectangular",
                    source="DUT resolution"
                )
            ]
            
            if pr.get('repeated_measurements'):
                uncertainty_components.append(
                    UncertaintyCalculator.create_type_a_component(
                        name="Repeatability",
                        measurements=pr['repeated_measurements']
                    )
                )
            
            budget = UncertaintyCalculator.create_uncertainty_budget(
                parameter="Power",
                components=uncertainty_components,
                unit="dB"
            )
            
            results.append(MeasurementResult(
                parameter=f"Power @ {pr.get('frequency_mhz', 'N/A')} MHz, {pr.get('nominal_dbm', 'N/A')} dBm",
                nominal_value=pr['nominal_dbm'],
                measured_value=pr['measured_dbm'],
                unit="dBm",
                tolerance_low=pr.get('tolerance_low'),
                tolerance_high=pr.get('tolerance_high'),
                uncertainty=budget.expanded_uncertainty(),
                uncertainty_budget=budget,
                as_found=pr.get('as_found_dbm'),
                as_left=pr.get('as_left_dbm'),
                adjustment_made=pr.get('adjusted', False),
                test_point_id=f"PWR_{i+1:03d}"
            ))
        
        procedure = CalibrationProcedure(
            procedure_number="RFA-PWR-001",
            title="RF Power Measurement Calibration",
            revision="B",
            effective_date=datetime.now() - timedelta(days=60),
            scope="Power meter and sensor calibration",
            method="Direct comparison",
            standards_referenced=["IEEE 1451.2", "IEC 62659"]
        )
        
        return generator.create_certificate(
            equipment=equipment,
            measurement_results=results,
            certificate_type=CertificateType.AS_FOUND_AS_LEFT,
            procedures=[procedure],
            reference_standards=[reference_standard],
            technician=technician,
            calibration_interval_months=12
        )
    
    @staticmethod
    def chamber_field_uniformity_certificate(
        generator: CertificateGenerator,
        chamber_equipment: EquipmentInfo,
        uniformity_results: Dict[str, Any],
        test_configuration: Dict[str, Any],
        technician: Optional[TechnicianInfo] = None
    ) -> CalibrationCertificate:
        """
        Create certificate for chamber field uniformity validation.
        
        Args:
            generator: Certificate generator
            chamber_equipment: Chamber under test
            uniformity_results: Field uniformity measurement results
            test_configuration: Test configuration details
            technician: Technician who performed validation
        
        Returns:
            Calibration certificate
        """
        results = []
        
        # Field uniformity by frequency
        for freq_data in uniformity_results.get('frequency_data', []):
            freq_mhz = freq_data['frequency_mhz']
            
            results.append(MeasurementResult(
                parameter=f"Field Uniformity @ {freq_mhz} MHz",
                nominal_value=0.0,  # Ideal is 0 dB deviation
                measured_value=freq_data.get('max_deviation_db', 0),
                unit="dB",
                tolerance_high=test_configuration.get('uniformity_tolerance_db', 6.0),
                uncertainty=0.5,
                test_point_id=f"UNIF_{freq_mhz}"
            ))
        
        # Site VSWR (for semi-anechoic)
        if 'site_vswr' in uniformity_results:
            for freq, vswr in uniformity_results['site_vswr'].items():
                results.append(MeasurementResult(
                    parameter=f"Site VSWR @ {freq} MHz",
                    nominal_value=1.0,  # Ideal VSWR
                    measured_value=vswr,
                    unit="",
                    tolerance_high=test_configuration.get('vswr_tolerance', 6.0),
                    uncertainty=0.2,
                    test_point_id=f"VSWR_{freq}"
                ))
        
        # Reference site comparison (NSA)
        if 'nsa_deviation' in uniformity_results:
            for freq, deviation in uniformity_results['nsa_deviation'].items():
                results.append(MeasurementResult(
                    parameter=f"NSA Deviation @ {freq} MHz",
                    nominal_value=0.0,
                    measured_value=deviation,
                    unit="dB",
                    tolerance_low=-4.0,
                    tolerance_high=4.0,
                    uncertainty=0.5,
                    test_point_id=f"NSA_{freq}"
                ))
        
        procedure = CalibrationProcedure(
            procedure_number="RFA-CHM-001",
            title="RF Chamber Validation Procedure",
            revision="A",
            effective_date=datetime.now() - timedelta(days=180),
            scope="Chamber field uniformity and site validation",
            method="Per CISPR 16-1-4 / IEC 61000-4-3",
            standards_referenced=["CISPR 16-1-4", "IEC 61000-4-3", "IEC 61000-4-21"]
        )
        
        env_conditions = EnvironmentalConditions(
            temperature_c=test_configuration.get('temperature_c', 23.0),
            humidity_percent=test_configuration.get('humidity_percent', 50.0),
            notes="Chamber validation environmental conditions"
        )
        
        remarks = f"""
Chamber validation performed per {test_configuration.get('standard', 'CISPR 16-1-4')}.
Test volume: {test_configuration.get('test_volume', 'N/A')}
Frequency range: {test_configuration.get('freq_range', 'N/A')}
Polarizations tested: {test_configuration.get('polarizations', 'H/V')}
"""
        
        return generator.create_certificate(
            equipment=chamber_equipment,
            measurement_results=results,
            certificate_type=CertificateType.VERIFICATION,
            procedures=[procedure],
            environmental_conditions=env_conditions,
            technician=technician,
            calibration_interval_months=24,  # Chamber validation typically 2 years
            remarks=remarks.strip()
        )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Enumerations
    'CertificateType',
    'CertificateStatus',
    'AccreditationBody',
    'UncertaintyType',
    'SignatureAlgorithm',
    
    # Data classes
    'LaboratoryInfo',
    'EquipmentInfo',
    'ReferenceStandard',
    'UncertaintyComponent',
    'UncertaintyBudget',
    'MeasurementResult',
    'EnvironmentalConditions',
    'CalibrationProcedure',
    'TechnicianInfo',
    'CalibrationCertificate',
    
    # Main classes
    'CertificateGenerator',
    'UncertaintyCalculator',
    'TraceabilityChainManager',
    'CertificateDatabase',
    'RFChamberCertificateTemplates'
]
