"""
Automated Test Sequences and Reporting for RF Chamber Testing.

This module provides comprehensive automated test sequence execution,
result management, and calibration certificate generation for RF
chamber testing operations.

Features:
- Configurable test sequence definitions
- Automated test execution with progress tracking
- Real-time result collection and analysis
- Comprehensive reporting and documentation
- ISO 17025 compliant calibration certificates
- Traceability chain management

Standards:
- ISO/IEC 17025: Testing and calibration laboratory requirements
- ANSI/NCSL Z540.3: Requirements for calibration
- ILAC-G8: Guidelines on decision rules and statements of conformity

Author: RF Arsenal Development Team
License: Proprietary - Calibration Sensitive
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from . import (
    ChamberType,
    MeasurementType,
    CalibrationStatus,
    ChamberTestResult,
    ChamberSpecification,
    MeasurementPoint,
    CalibrationPoint,
    ChamberTestSequenceResult,
    UncertaintyBudget,
    ChamberInterface,
    SimulatedChamber,
    AntennaPatternMeasurement,
    PowerCalibration,
    FrequencyCalibration,
    CalibrationValidator,
    UncertaintyAnalyzer,
)

logger = logging.getLogger(__name__)


class TestSequenceType(Enum):
    """Types of test sequences."""
    
    FULL_CALIBRATION = "full_calibration"
    POWER_CALIBRATION = "power_calibration"
    FREQUENCY_CALIBRATION = "frequency_calibration"
    ANTENNA_PATTERN = "antenna_pattern"
    CHAMBER_VALIDATION = "chamber_validation"
    QUICK_CHECK = "quick_check"
    CUSTOM = "custom"


class TestPriority(Enum):
    """Test execution priority."""
    
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class SequenceStatus(Enum):
    """Test sequence status."""
    
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class TestStep:
    """Single test step definition."""
    
    step_id: str
    name: str
    description: str
    test_type: MeasurementType
    parameters: Dict[str, Any]
    expected_result: Optional[Any]
    tolerance: Optional[float]
    priority: TestPriority
    timeout_s: float
    retry_count: int = 0
    skip_on_failure: bool = False


@dataclass
class TestStepResult:
    """Result of a test step execution."""
    
    step_id: str
    step_name: str
    start_time: datetime
    end_time: datetime
    result: TestResult
    measured_value: Any
    expected_value: Any
    deviation: Optional[float]
    within_tolerance: bool
    error_message: Optional[str]
    retry_attempts: int
    measurements: List[MeasurementPoint]


@dataclass
class TestSequenceDefinition:
    """Complete test sequence definition."""
    
    sequence_id: str
    sequence_type: TestSequenceType
    name: str
    description: str
    version: str
    steps: List[TestStep]
    frequency_range_hz: Tuple[float, float]
    chamber_type: ChamberType
    estimated_duration_minutes: float
    created_date: datetime
    created_by: str
    approved_by: Optional[str]
    approval_date: Optional[datetime]


@dataclass
class CalibrationCertificate:
    """Calibration certificate data."""
    
    certificate_number: str
    issue_date: datetime
    expiration_date: datetime
    device_id: str
    device_description: str
    serial_number: str
    calibration_date: datetime
    calibration_location: str
    chamber_id: str
    procedure_id: str
    environmental_conditions: Dict[str, Any]
    calibration_results: List[Dict[str, Any]]
    measurement_uncertainty: List[UncertaintyBudget]
    traceability_chain: List[str]
    calibration_status: CalibrationStatus
    next_calibration_due: datetime
    performed_by: str
    approved_by: str
    notes: str


class TestSequenceBuilder:
    """
    Builder for creating test sequences.
    
    Provides fluent interface for defining test sequences
    with validation.
    """
    
    def __init__(self):
        self._sequence_id = secrets.token_hex(8)
        self._sequence_type = TestSequenceType.CUSTOM
        self._name = ""
        self._description = ""
        self._version = "1.0"
        self._steps: List[TestStep] = []
        self._frequency_range = (1e6, 6e9)
        self._chamber_type = ChamberType.ANECHOIC
        self._created_by = "system"
    
    def with_type(self, sequence_type: TestSequenceType) -> "TestSequenceBuilder":
        """Set sequence type."""
        self._sequence_type = sequence_type
        return self
    
    def with_name(self, name: str) -> "TestSequenceBuilder":
        """Set sequence name."""
        self._name = name
        return self
    
    def with_description(self, description: str) -> "TestSequenceBuilder":
        """Set sequence description."""
        self._description = description
        return self
    
    def with_frequency_range(
        self,
        start_hz: float,
        stop_hz: float
    ) -> "TestSequenceBuilder":
        """Set frequency range."""
        self._frequency_range = (start_hz, stop_hz)
        return self
    
    def with_chamber_type(self, chamber_type: ChamberType) -> "TestSequenceBuilder":
        """Set chamber type."""
        self._chamber_type = chamber_type
        return self
    
    def add_step(
        self,
        name: str,
        test_type: MeasurementType,
        parameters: Dict[str, Any],
        expected_result: Optional[Any] = None,
        tolerance: Optional[float] = None,
        priority: TestPriority = TestPriority.NORMAL,
        timeout_s: float = 60.0,
        description: str = ""
    ) -> "TestSequenceBuilder":
        """Add a test step."""
        step = TestStep(
            step_id=f"step_{len(self._steps) + 1:03d}",
            name=name,
            description=description or name,
            test_type=test_type,
            parameters=parameters,
            expected_result=expected_result,
            tolerance=tolerance,
            priority=priority,
            timeout_s=timeout_s
        )
        self._steps.append(step)
        return self
    
    def build(self) -> TestSequenceDefinition:
        """Build the test sequence definition."""
        # Estimate duration
        duration = sum(step.timeout_s for step in self._steps) / 60 * 1.2  # 20% margin
        
        return TestSequenceDefinition(
            sequence_id=self._sequence_id,
            sequence_type=self._sequence_type,
            name=self._name,
            description=self._description,
            version=self._version,
            steps=self._steps,
            frequency_range_hz=self._frequency_range,
            chamber_type=self._chamber_type,
            estimated_duration_minutes=duration,
            created_date=datetime.utcnow(),
            created_by=self._created_by,
            approved_by=None,
            approval_date=None
        )


class StandardTestSequences:
    """
    Factory for standard test sequences.
    
    Provides pre-defined test sequences for common
    calibration scenarios.
    """
    
    @staticmethod
    def create_full_calibration_sequence(
        frequency_points_hz: List[float],
        power_levels_dbm: List[float]
    ) -> TestSequenceDefinition:
        """Create full system calibration sequence."""
        builder = TestSequenceBuilder()
        builder.with_type(TestSequenceType.FULL_CALIBRATION)
        builder.with_name("Full System Calibration")
        builder.with_description(
            "Complete calibration including power, frequency, "
            "and antenna pattern validation"
        )
        
        # Power calibration steps
        for freq in frequency_points_hz:
            for power in power_levels_dbm:
                builder.add_step(
                    name=f"Power Cal {freq/1e9:.2f}GHz @ {power}dBm",
                    test_type=MeasurementType.POWER,
                    parameters={
                        "frequency_hz": freq,
                        "power_level_dbm": power
                    },
                    tolerance=0.5,
                    priority=TestPriority.HIGH
                )
        
        # Frequency calibration steps
        for freq in frequency_points_hz:
            builder.add_step(
                name=f"Frequency Cal {freq/1e9:.2f}GHz",
                test_type=MeasurementType.FREQUENCY,
                parameters={"frequency_hz": freq},
                tolerance=1.0,  # ppm
                priority=TestPriority.HIGH
            )
        
        # Linearity test
        builder.add_step(
            name="Power Linearity Test",
            test_type=MeasurementType.POWER,
            parameters={
                "frequency_hz": frequency_points_hz[len(frequency_points_hz)//2],
                "power_range_dbm": (min(power_levels_dbm), max(power_levels_dbm)),
                "test_type": "linearity"
            },
            tolerance=0.3,
            priority=TestPriority.NORMAL
        )
        
        return builder.build()
    
    @staticmethod
    def create_antenna_pattern_sequence(
        frequency_hz: float,
        polarizations: List[str] = ["vertical", "horizontal"]
    ) -> TestSequenceDefinition:
        """Create antenna pattern measurement sequence."""
        builder = TestSequenceBuilder()
        builder.with_type(TestSequenceType.ANTENNA_PATTERN)
        builder.with_name(f"Antenna Pattern @ {frequency_hz/1e9:.2f}GHz")
        builder.with_description("Complete antenna radiation pattern measurement")
        
        for pol in polarizations:
            # E-plane cut
            builder.add_step(
                name=f"E-Plane Cut ({pol})",
                test_type=MeasurementType.ANTENNA_PATTERN,
                parameters={
                    "frequency_hz": frequency_hz,
                    "polarization": pol,
                    "cut_plane": "E",
                    "theta_range": (0, 180, 2)
                },
                priority=TestPriority.HIGH,
                timeout_s=300
            )
            
            # H-plane cut
            builder.add_step(
                name=f"H-Plane Cut ({pol})",
                test_type=MeasurementType.ANTENNA_PATTERN,
                parameters={
                    "frequency_hz": frequency_hz,
                    "polarization": pol,
                    "cut_plane": "H",
                    "phi_range": (0, 360, 2)
                },
                priority=TestPriority.HIGH,
                timeout_s=300
            )
        
        # 3D pattern
        builder.add_step(
            name="Full 3D Pattern",
            test_type=MeasurementType.ANTENNA_PATTERN,
            parameters={
                "frequency_hz": frequency_hz,
                "theta_range": (0, 180, 5),
                "phi_range": (0, 360, 5),
                "mode": "3d"
            },
            priority=TestPriority.NORMAL,
            timeout_s=1800
        )
        
        return builder.build()
    
    @staticmethod
    def create_quick_check_sequence(
        frequencies_hz: List[float]
    ) -> TestSequenceDefinition:
        """Create quick verification sequence."""
        builder = TestSequenceBuilder()
        builder.with_type(TestSequenceType.QUICK_CHECK)
        builder.with_name("Quick Calibration Check")
        builder.with_description("Fast verification of calibration status")
        
        for freq in frequencies_hz:
            # Single power check
            builder.add_step(
                name=f"Power Check {freq/1e9:.2f}GHz",
                test_type=MeasurementType.POWER,
                parameters={"frequency_hz": freq, "power_level_dbm": 0},
                tolerance=1.0,
                priority=TestPriority.NORMAL,
                timeout_s=10
            )
            
            # Single frequency check
            builder.add_step(
                name=f"Frequency Check {freq/1e9:.2f}GHz",
                test_type=MeasurementType.FREQUENCY,
                parameters={"frequency_hz": freq},
                tolerance=2.0,
                priority=TestPriority.NORMAL,
                timeout_s=10
            )
        
        return builder.build()


class TestSequenceExecutor:
    """
    Test sequence execution engine.
    
    Executes test sequences with progress tracking,
    error handling, and result collection.
    """
    
    def __init__(
        self,
        chamber: ChamberInterface,
        specification: ChamberSpecification
    ):
        self.chamber = chamber
        self.specification = specification
        
        self._current_sequence: Optional[TestSequenceDefinition] = None
        self._status = SequenceStatus.PENDING
        self._step_results: List[TestStepResult] = []
        self._current_step_index = 0
        
        self._progress_callbacks: List[Callable[[float, str], None]] = []
        self._result_callbacks: List[Callable[[TestStepResult], None]] = []
        
        self._abort_requested = False
        self._pause_requested = False
        self._lock = threading.Lock()
    
    def register_progress_callback(
        self,
        callback: Callable[[float, str], None]
    ) -> None:
        """Register callback for progress updates."""
        self._progress_callbacks.append(callback)
    
    def register_result_callback(
        self,
        callback: Callable[[TestStepResult], None]
    ) -> None:
        """Register callback for step results."""
        self._result_callbacks.append(callback)
    
    def execute_sequence(
        self,
        sequence: TestSequenceDefinition
    ) -> ChamberTestSequenceResult:
        """
        Execute a test sequence.
        
        Args:
            sequence: Test sequence to execute
            
        Returns:
            Complete test sequence result
        """
        self._current_sequence = sequence
        self._step_results = []
        self._current_step_index = 0
        self._abort_requested = False
        self._pause_requested = False
        
        with self._lock:
            self._status = SequenceStatus.RUNNING
        
        start_time = datetime.utcnow()
        all_measurements: List[MeasurementPoint] = []
        all_calibration_points: List[CalibrationPoint] = []
        
        logger.info(f"Starting test sequence: {sequence.name}")
        self._notify_progress(0, f"Starting sequence: {sequence.name}")
        
        try:
            # Connect to chamber if not connected
            if not self.chamber.connect():
                raise Exception("Failed to connect to chamber")
            
            for idx, step in enumerate(sequence.steps):
                # Check for abort/pause
                if self._abort_requested:
                    with self._lock:
                        self._status = SequenceStatus.ABORTED
                    break
                
                while self._pause_requested:
                    with self._lock:
                        self._status = SequenceStatus.PAUSED
                    time.sleep(0.1)
                
                with self._lock:
                    self._status = SequenceStatus.RUNNING
                
                self._current_step_index = idx
                progress = (idx / len(sequence.steps)) * 100
                self._notify_progress(progress, f"Executing: {step.name}")
                
                # Execute step
                step_result = self._execute_step(step)
                self._step_results.append(step_result)
                
                # Collect measurements
                all_measurements.extend(step_result.measurements)
                
                # Notify callbacks
                for callback in self._result_callbacks:
                    try:
                        callback(step_result)
                    except Exception as e:
                        logger.error(f"Result callback error: {e}")
                
                # Check for critical failure
                if (step.priority == TestPriority.CRITICAL and 
                    step_result.result == ChamberTestResult.FAIL and
                    not step.skip_on_failure):
                    logger.error(f"Critical step failed: {step.name}")
                    break
            
            # Sequence complete
            end_time = datetime.utcnow()
            
            with self._lock:
                if self._status != SequenceStatus.ABORTED:
                    self._status = SequenceStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Sequence execution error: {e}")
            end_time = datetime.utcnow()
            with self._lock:
                self._status = SequenceStatus.FAILED
        
        finally:
            self.chamber.disconnect()
        
        # Calculate overall result
        passed = sum(1 for r in self._step_results if r.result == ChamberTestResult.PASS)
        failed = sum(1 for r in self._step_results if r.result == ChamberTestResult.FAIL)
        skipped = sum(1 for r in self._step_results if r.result == ChamberTestResult.SKIPPED)
        
        overall_result = (
            ChamberTestResult.PASS if failed == 0 and self._status == SequenceStatus.COMPLETED
            else ChamberTestResult.FAIL
        )
        
        self._notify_progress(100, "Sequence complete")
        
        return ChamberTestSequenceResult(
            sequence_id=sequence.sequence_id,
            sequence_name=sequence.name,
            start_time=start_time,
            end_time=end_time,
            overall_result=overall_result,
            tests_passed=passed,
            tests_failed=failed,
            tests_skipped=skipped,
            measurements=all_measurements,
            calibration_points=all_calibration_points,
            report_path=None,
            certificate_id=None
        )
    
    def _execute_step(self, step: TestStep) -> TestStepResult:
        """Execute a single test step."""
        start_time = datetime.utcnow()
        measurements = []
        measured_value = None
        error_message = None
        retry_attempts = 0
        
        try:
            # Set frequency if specified
            if "frequency_hz" in step.parameters:
                self.chamber.set_frequency(step.parameters["frequency_hz"])
            
            # Perform measurement based on type
            if step.test_type == MeasurementType.POWER:
                measurement = self.chamber.measure(MeasurementType.POWER)
                measurements.append(measurement)
                measured_value = measurement.value
                
            elif step.test_type == MeasurementType.FREQUENCY:
                measurement = self.chamber.measure(MeasurementType.FREQUENCY)
                measurements.append(measurement)
                measured_value = measurement.value
                
            elif step.test_type == MeasurementType.ANTENNA_PATTERN:
                # Pattern measurement may involve multiple points
                if step.parameters.get("mode") == "3d":
                    # 3D pattern - simplified
                    for theta in range(0, 181, step.parameters.get("theta_range", (0, 180, 5))[2]):
                        self.chamber.set_position(theta, 0)
                        measurement = self.chamber.measure(MeasurementType.ANTENNA_PATTERN)
                        measurements.append(measurement)
                else:
                    measurement = self.chamber.measure(MeasurementType.ANTENNA_PATTERN)
                    measurements.append(measurement)
                    measured_value = measurement.value
            
            else:
                measurement = self.chamber.measure(step.test_type)
                measurements.append(measurement)
                measured_value = measurement.value
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Step execution error: {e}")
        
        end_time = datetime.utcnow()
        
        # Determine result
        if error_message:
            result = ChamberTestResult.ERROR
            within_tolerance = False
            deviation = None
        elif step.expected_result is not None and step.tolerance is not None:
            deviation = abs(measured_value - step.expected_result) if measured_value else None
            within_tolerance = deviation <= step.tolerance if deviation is not None else False
            result = ChamberTestResult.PASS if within_tolerance else ChamberTestResult.FAIL
        else:
            # No expected value - just pass if measurement succeeded
            result = ChamberTestResult.PASS if measured_value is not None else ChamberTestResult.FAIL
            within_tolerance = True
            deviation = None
        
        return TestStepResult(
            step_id=step.step_id,
            step_name=step.name,
            start_time=start_time,
            end_time=end_time,
            result=result,
            measured_value=measured_value,
            expected_value=step.expected_result,
            deviation=deviation,
            within_tolerance=within_tolerance,
            error_message=error_message,
            retry_attempts=retry_attempts,
            measurements=measurements
        )
    
    def _notify_progress(self, progress: float, message: str) -> None:
        """Notify progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(progress, message)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def request_abort(self) -> None:
        """Request sequence abort."""
        self._abort_requested = True
    
    def request_pause(self) -> None:
        """Request sequence pause."""
        self._pause_requested = True
    
    def request_resume(self) -> None:
        """Request sequence resume."""
        self._pause_requested = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        with self._lock:
            return {
                "status": self._status.value,
                "current_step": self._current_step_index,
                "total_steps": len(self._current_sequence.steps) if self._current_sequence else 0,
                "steps_completed": len(self._step_results),
                "passed": sum(1 for r in self._step_results if r.result == ChamberTestResult.PASS),
                "failed": sum(1 for r in self._step_results if r.result == ChamberTestResult.FAIL)
            }


class CalibrationCertificateGenerator:
    """
    Calibration certificate generator.
    
    Generates ISO 17025 compliant calibration certificates
    with full traceability documentation.
    """
    
    def __init__(
        self,
        laboratory_name: str = "RF Arsenal Calibration Laboratory",
        laboratory_id: str = "RFAL-001",
        accreditation_number: Optional[str] = None
    ):
        self.laboratory_name = laboratory_name
        self.laboratory_id = laboratory_id
        self.accreditation_number = accreditation_number
        
        # Certificate number counter
        self._certificate_counter = 0
    
    def generate_certificate(
        self,
        sequence_result: ChamberTestSequenceResult,
        device_info: Dict[str, str],
        chamber_info: ChamberSpecification,
        environmental_conditions: Dict[str, Any],
        uncertainty_budgets: List[UncertaintyBudget],
        traceability: List[str],
        performed_by: str,
        approved_by: str,
        validity_months: int = 12,
        notes: str = ""
    ) -> CalibrationCertificate:
        """
        Generate a calibration certificate.
        
        Args:
            sequence_result: Test sequence results
            device_info: Device identification info
            chamber_info: Chamber specification
            environmental_conditions: Environmental data
            uncertainty_budgets: Measurement uncertainties
            traceability: Traceability chain
            performed_by: Technician name
            approved_by: Approver name
            validity_months: Certificate validity
            notes: Additional notes
            
        Returns:
            Complete calibration certificate
        """
        self._certificate_counter += 1
        
        # Generate certificate number
        now = datetime.utcnow()
        certificate_number = (
            f"{self.laboratory_id}-"
            f"{now.strftime('%Y%m%d')}-"
            f"{self._certificate_counter:04d}"
        )
        
        # Determine calibration status
        if sequence_result.overall_result == ChamberTestResult.PASS:
            status = CalibrationStatus.VALID
        else:
            status = CalibrationStatus.INVALID
        
        # Format calibration results
        calibration_results = []
        for step_result in getattr(sequence_result, '_step_results', []):
            if hasattr(step_result, 'measured_value'):
                calibration_results.append({
                    "parameter": step_result.step_name,
                    "measured_value": step_result.measured_value,
                    "expected_value": step_result.expected_value,
                    "deviation": step_result.deviation,
                    "within_tolerance": step_result.within_tolerance,
                    "result": step_result.result.value if hasattr(step_result.result, 'value') else str(step_result.result)
                })
        
        return CalibrationCertificate(
            certificate_number=certificate_number,
            issue_date=now,
            expiration_date=now + timedelta(days=validity_months * 30),
            device_id=device_info.get("device_id", ""),
            device_description=device_info.get("description", ""),
            serial_number=device_info.get("serial_number", ""),
            calibration_date=sequence_result.start_time,
            calibration_location=self.laboratory_name,
            chamber_id=chamber_info.chamber_id,
            procedure_id=sequence_result.sequence_id,
            environmental_conditions=environmental_conditions,
            calibration_results=calibration_results,
            measurement_uncertainty=uncertainty_budgets,
            traceability_chain=traceability,
            calibration_status=status,
            next_calibration_due=now + timedelta(days=validity_months * 30),
            performed_by=performed_by,
            approved_by=approved_by,
            notes=notes
        )
    
    def export_certificate_json(
        self,
        certificate: CalibrationCertificate
    ) -> str:
        """Export certificate as JSON."""
        data = {
            "certificate_number": certificate.certificate_number,
            "laboratory": {
                "name": self.laboratory_name,
                "id": self.laboratory_id,
                "accreditation": self.accreditation_number
            },
            "issue_date": certificate.issue_date.isoformat(),
            "expiration_date": certificate.expiration_date.isoformat(),
            "device": {
                "id": certificate.device_id,
                "description": certificate.device_description,
                "serial_number": certificate.serial_number
            },
            "calibration": {
                "date": certificate.calibration_date.isoformat(),
                "location": certificate.calibration_location,
                "chamber_id": certificate.chamber_id,
                "procedure_id": certificate.procedure_id
            },
            "environmental_conditions": certificate.environmental_conditions,
            "results": certificate.calibration_results,
            "uncertainty": [
                {
                    "measurement_type": u.measurement_type.value,
                    "frequency_hz": u.frequency_hz,
                    "expanded_uncertainty": u.expanded_uncertainty,
                    "coverage_factor": u.coverage_factor
                }
                for u in certificate.measurement_uncertainty
            ],
            "traceability": certificate.traceability_chain,
            "status": certificate.calibration_status.value,
            "next_due": certificate.next_calibration_due.isoformat(),
            "signatures": {
                "performed_by": certificate.performed_by,
                "approved_by": certificate.approved_by
            },
            "notes": certificate.notes
        }
        
        return json.dumps(data, indent=2)
    
    def export_certificate_html(
        self,
        certificate: CalibrationCertificate
    ) -> str:
        """Export certificate as HTML."""
        # Generate HTML certificate
        results_html = ""
        for result in certificate.calibration_results:
            status_class = "pass" if result.get("within_tolerance", False) else "fail"
            results_html += f"""
            <tr class="{status_class}">
                <td>{result.get('parameter', '')}</td>
                <td>{result.get('measured_value', '')}</td>
                <td>{result.get('expected_value', '')}</td>
                <td>{result.get('deviation', '')}</td>
                <td>{result.get('result', '')}</td>
            </tr>
            """
        
        uncertainty_html = ""
        for u in certificate.measurement_uncertainty:
            uncertainty_html += f"""
            <tr>
                <td>{u.measurement_type.value}</td>
                <td>{u.frequency_hz / 1e9:.3f} GHz</td>
                <td>±{u.expanded_uncertainty:.3f}</td>
                <td>k={u.coverage_factor}</td>
            </tr>
            """
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Calibration Certificate {certificate.certificate_number}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
        .certificate-number {{ font-size: 24px; font-weight: bold; }}
        .section {{ margin: 20px 0; }}
        .section-title {{ font-size: 16px; font-weight: bold; color: #333; border-bottom: 1px solid #ccc; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        .pass {{ background-color: #e6ffe6; }}
        .fail {{ background-color: #ffe6e6; }}
        .status-valid {{ color: green; font-weight: bold; }}
        .status-invalid {{ color: red; font-weight: bold; }}
        .signatures {{ margin-top: 40px; }}
        .signature-line {{ border-top: 1px solid #333; width: 200px; display: inline-block; margin: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.laboratory_name}</h1>
        <p class="certificate-number">Certificate No: {certificate.certificate_number}</p>
        {f'<p>Accreditation: {self.accreditation_number}</p>' if self.accreditation_number else ''}
    </div>
    
    <div class="section">
        <h2 class="section-title">Device Information</h2>
        <table>
            <tr><th>Device ID</th><td>{certificate.device_id}</td></tr>
            <tr><th>Description</th><td>{certificate.device_description}</td></tr>
            <tr><th>Serial Number</th><td>{certificate.serial_number}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2 class="section-title">Calibration Details</h2>
        <table>
            <tr><th>Calibration Date</th><td>{certificate.calibration_date.strftime('%Y-%m-%d')}</td></tr>
            <tr><th>Location</th><td>{certificate.calibration_location}</td></tr>
            <tr><th>Chamber ID</th><td>{certificate.chamber_id}</td></tr>
            <tr><th>Procedure ID</th><td>{certificate.procedure_id}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2 class="section-title">Environmental Conditions</h2>
        <table>
            <tr><th>Temperature</th><td>{certificate.environmental_conditions.get('temperature_c', 'N/A')} °C</td></tr>
            <tr><th>Humidity</th><td>{certificate.environmental_conditions.get('humidity_percent', 'N/A')} %</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2 class="section-title">Calibration Results</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Measured</th>
                <th>Expected</th>
                <th>Deviation</th>
                <th>Result</th>
            </tr>
            {results_html}
        </table>
    </div>
    
    <div class="section">
        <h2 class="section-title">Measurement Uncertainty</h2>
        <table>
            <tr>
                <th>Measurement Type</th>
                <th>Frequency</th>
                <th>Expanded Uncertainty</th>
                <th>Coverage Factor</th>
            </tr>
            {uncertainty_html}
        </table>
    </div>
    
    <div class="section">
        <h2 class="section-title">Calibration Status</h2>
        <p class="status-{certificate.calibration_status.value}">
            Status: {certificate.calibration_status.value.upper()}
        </p>
        <p>Next Calibration Due: {certificate.next_calibration_due.strftime('%Y-%m-%d')}</p>
    </div>
    
    <div class="section">
        <h2 class="section-title">Traceability</h2>
        <ul>
            {''.join(f'<li>{t}</li>' for t in certificate.traceability_chain)}
        </ul>
    </div>
    
    {f'<div class="section"><h2 class="section-title">Notes</h2><p>{certificate.notes}</p></div>' if certificate.notes else ''}
    
    <div class="signatures">
        <div style="display: inline-block; margin-right: 100px;">
            <p class="signature-line"></p>
            <p>Performed by: {certificate.performed_by}</p>
        </div>
        <div style="display: inline-block;">
            <p class="signature-line"></p>
            <p>Approved by: {certificate.approved_by}</p>
        </div>
    </div>
    
    <div class="footer" style="margin-top: 40px; font-size: 10px; color: #666;">
        <p>Issue Date: {certificate.issue_date.strftime('%Y-%m-%d')}</p>
        <p>Expiration Date: {certificate.expiration_date.strftime('%Y-%m-%d')}</p>
    </div>
</body>
</html>
        """
        
        return html


class ReportGenerator:
    """
    Test report generator.
    
    Generates comprehensive test reports in various formats.
    """
    
    def __init__(self):
        self._report_counter = 0
    
    def generate_summary_report(
        self,
        sequence_result: ChamberTestSequenceResult,
        step_results: List[TestStepResult]
    ) -> Dict[str, Any]:
        """Generate summary report."""
        duration = (sequence_result.end_time - sequence_result.start_time).total_seconds()
        
        # Group by result
        by_result = {}
        for result in step_results:
            key = result.result.value
            if key not in by_result:
                by_result[key] = []
            by_result[key].append(result.step_name)
        
        return {
            "report_type": "summary",
            "generated_at": datetime.utcnow().isoformat(),
            "sequence_id": sequence_result.sequence_id,
            "sequence_name": sequence_result.sequence_name,
            "duration_seconds": duration,
            "overall_result": sequence_result.overall_result.value,
            "statistics": {
                "total_tests": len(step_results),
                "passed": sequence_result.tests_passed,
                "failed": sequence_result.tests_failed,
                "skipped": sequence_result.tests_skipped,
                "pass_rate_percent": (
                    sequence_result.tests_passed / len(step_results) * 100
                    if step_results else 0
                )
            },
            "results_by_status": by_result,
            "start_time": sequence_result.start_time.isoformat(),
            "end_time": sequence_result.end_time.isoformat()
        }
    
    def generate_detailed_report(
        self,
        sequence_result: ChamberTestSequenceResult,
        step_results: List[TestStepResult],
        measurements: List[MeasurementPoint]
    ) -> Dict[str, Any]:
        """Generate detailed report with all measurements."""
        summary = self.generate_summary_report(sequence_result, step_results)
        
        # Add detailed step results
        detailed_steps = []
        for result in step_results:
            detailed_steps.append({
                "step_id": result.step_id,
                "step_name": result.step_name,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "duration_ms": (result.end_time - result.start_time).total_seconds() * 1000,
                "result": result.result.value,
                "measured_value": result.measured_value,
                "expected_value": result.expected_value,
                "deviation": result.deviation,
                "within_tolerance": result.within_tolerance,
                "error_message": result.error_message,
                "retry_attempts": result.retry_attempts,
                "measurements_count": len(result.measurements)
            })
        
        summary["report_type"] = "detailed"
        summary["step_details"] = detailed_steps
        summary["total_measurements"] = len(measurements)
        
        return summary


# Export public API
__all__ = [
    # Enums
    "TestSequenceType",
    "TestPriority",
    "SequenceStatus",
    
    # Data classes
    "TestStep",
    "TestStepResult",
    "TestSequenceDefinition",
    "CalibrationCertificate",
    
    # Builders
    "TestSequenceBuilder",
    "StandardTestSequences",
    
    # Executors
    "TestSequenceExecutor",
    
    # Generators
    "CalibrationCertificateGenerator",
    "ReportGenerator",
]
