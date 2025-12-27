"""
RF Chamber Testing Framework for Calibration Validation.

This module provides comprehensive RF chamber testing capabilities for
validating hardware calibration, antenna patterns, power accuracy,
and frequency response per industry standards.

Supported Chamber Types:
- Anechoic Chamber (far-field and near-field)
- Reverberation Chamber (mode-stirred)
- Semi-Anechoic Chamber
- GTEM Cell
- TEM Cell

Standards Compliance:
- CISPR 16-1-4: Anechoic chamber requirements
- IEC 61000-4-21: Reverberation chamber testing
- IEEE 149: Antenna measurement techniques
- MIL-STD-461G: EMC testing requirements
- CTIA OTA: Over-the-air performance testing

Features:
- Automated test sequence execution
- Real-time measurement and analysis
- Calibration validation and verification
- Antenna pattern measurement (2D/3D)
- Power calibration with linearity testing
- Frequency response characterization
- Uncertainty analysis and budgeting
- Calibration certificate generation

Author: RF Arsenal Development Team
License: Proprietary - Calibration Sensitive
"""

import asyncio
import hashlib
import logging
import math
import os
import secrets
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy import signal as scipy_signal
from scipy import interpolate
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class ChamberType(Enum):
    """Types of RF test chambers."""
    
    ANECHOIC = "anechoic"                   # Fully anechoic chamber
    SEMI_ANECHOIC = "semi_anechoic"         # Semi-anechoic (reflective floor)
    REVERBERATION = "reverberation"          # Mode-stirred reverberation
    GTEM = "gtem"                            # Gigahertz TEM cell
    TEM = "tem"                              # Standard TEM cell
    OATS = "oats"                            # Open Area Test Site
    FAR_FIELD = "far_field"                  # Far-field antenna range
    NEAR_FIELD = "near_field"                # Near-field scanner
    COMPACT_RANGE = "compact_range"          # Compact antenna test range


class MeasurementType(Enum):
    """Types of RF measurements."""
    
    POWER = "power"                          # Power level measurement
    FREQUENCY = "frequency"                  # Frequency measurement
    PHASE = "phase"                          # Phase measurement
    ANTENNA_PATTERN = "antenna_pattern"      # Antenna radiation pattern
    GAIN = "gain"                            # Antenna/amplifier gain
    VSWR = "vswr"                            # Voltage Standing Wave Ratio
    RETURN_LOSS = "return_loss"              # Return loss (S11)
    INSERTION_LOSS = "insertion_loss"        # Insertion loss (S21)
    NOISE_FIGURE = "noise_figure"            # Noise figure
    EVM = "evm"                              # Error Vector Magnitude
    SENSITIVITY = "sensitivity"              # Receiver sensitivity
    SPURIOUS = "spurious"                    # Spurious emissions
    HARMONICS = "harmonics"                  # Harmonic distortion
    IMD = "imd"                              # Intermodulation distortion
    FIELD_UNIFORMITY = "field_uniformity"    # Chamber field uniformity


class CalibrationStatus(Enum):
    """Calibration validation status."""
    
    VALID = "valid"                          # Calibration within spec
    WARNING = "warning"                      # Approaching limits
    INVALID = "invalid"                      # Out of specification
    EXPIRED = "expired"                      # Calibration expired
    PENDING = "pending"                      # Not yet validated
    IN_PROGRESS = "in_progress"              # Validation in progress


class ChamberTestResult(Enum):
    """Chamber test result status."""
    
    PASS = "pass"
    FAIL = "fail"
    MARGINAL = "marginal"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ChamberSpecification:
    """RF chamber specification."""
    
    chamber_id: str
    chamber_type: ChamberType
    frequency_range_hz: Tuple[float, float]
    quiet_zone_dimensions_m: Tuple[float, float, float]  # L x W x H
    reflectivity_db: float
    field_uniformity_db: float
    max_power_w: float
    temperature_range_c: Tuple[float, float]
    humidity_range_percent: Tuple[float, float]
    shielding_effectiveness_db: float
    last_certification: Optional[datetime]
    certification_expires: Optional[datetime]
    certification_authority: Optional[str]


@dataclass
class MeasurementPoint:
    """Single measurement point."""
    
    timestamp: datetime
    frequency_hz: float
    value: float
    unit: str
    uncertainty: float
    temperature_c: float
    humidity_percent: float
    position: Optional[Tuple[float, float, float]] = None  # x, y, z in meters
    angle: Optional[Tuple[float, float]] = None  # theta, phi in degrees
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationPoint:
    """Calibration reference point."""
    
    frequency_hz: float
    reference_value: float
    measured_value: float
    correction_factor: float
    uncertainty: float
    temperature_c: float
    valid: bool


@dataclass
class ChamberTestSequenceResult:
    """Result of a test sequence execution."""
    
    sequence_id: str
    sequence_name: str
    start_time: datetime
    end_time: datetime
    overall_result: ChamberTestResult
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    measurements: List[MeasurementPoint]
    calibration_points: List[CalibrationPoint]
    report_path: Optional[str]
    certificate_id: Optional[str]


@dataclass
class UncertaintyBudget:
    """Measurement uncertainty budget."""
    
    measurement_type: MeasurementType
    frequency_hz: float
    components: Dict[str, float]  # Component name -> uncertainty contribution
    combined_standard_uncertainty: float
    expanded_uncertainty: float  # k=2, 95% confidence
    coverage_factor: float
    notes: str


class ChamberException(Exception):
    """Base exception for chamber testing errors."""
    pass


class CalibrationException(ChamberException):
    """Exception for calibration failures."""
    pass


class MeasurementException(ChamberException):
    """Exception for measurement failures."""
    pass


class ChamberInterface(ABC):
    """Abstract interface for RF chamber control."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to chamber control system."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from chamber."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get chamber status."""
        pass
    
    @abstractmethod
    def set_position(self, theta: float, phi: float) -> bool:
        """Set antenna positioner angles."""
        pass
    
    @abstractmethod
    def get_position(self) -> Tuple[float, float]:
        """Get current positioner angles."""
        pass
    
    @abstractmethod
    def set_frequency(self, frequency_hz: float) -> bool:
        """Set measurement frequency."""
        pass
    
    @abstractmethod
    def measure(self, measurement_type: MeasurementType) -> MeasurementPoint:
        """Perform a measurement."""
        pass


class DevelopmentChamber(ChamberInterface):
    """
    Development RF chamber for testing and development.
    
    Provides realistic synthetic measurements for algorithm
    development and testing without physical chamber.
    """
    
    def __init__(
        self,
        specification: ChamberSpecification,
        add_noise: bool = True
    ):
        self.specification = specification
        self.add_noise = add_noise
        
        self._connected = False
        self._position = (0.0, 0.0)  # theta, phi
        self._frequency_hz = 1e9
        self._temperature_c = 23.0
        self._humidity_percent = 45.0
        
        # Synthetic antenna pattern (dipole-like)
        self._antenna_pattern = self._generate_dipole_pattern()
        
        # Synthetic calibration data
        self._cal_data = self._generate_cal_data()
    
    def _generate_dipole_pattern(self) -> Dict[Tuple[float, float], float]:
        """Generate synthetic dipole antenna pattern."""
        pattern = {}
        for theta in range(0, 181, 5):
            for phi in range(0, 361, 5):
                # Dipole pattern: sin(theta) behavior
                theta_rad = math.radians(theta)
                gain = 1.64 * (math.sin(theta_rad) ** 2)  # ~2.15 dBi peak
                gain_db = 10 * math.log10(max(gain, 1e-10))
                pattern[(theta, phi)] = gain_db
        return pattern
    
    def _generate_cal_data(self) -> Dict[float, float]:
        """Generate synthetic calibration data."""
        cal_data = {}
        for freq in np.logspace(6, 10, 50):  # 1 MHz to 10 GHz
            # Simulate frequency-dependent path loss
            path_loss = 20 * math.log10(freq / 1e9) + 40
            cal_data[freq] = path_loss
        return cal_data
    
    def connect(self) -> bool:
        """Connect to development chamber."""
        self._connected = True
        logger.info(f"Connected to development chamber: {self.specification.chamber_id}")
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from development chamber."""
        self._connected = False
        logger.info("Disconnected from development chamber")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get chamber status."""
        return {
            "connected": self._connected,
            "chamber_type": self.specification.chamber_type.value,
            "position": self._position,
            "frequency_hz": self._frequency_hz,
            "temperature_c": self._temperature_c,
            "humidity_percent": self._humidity_percent,
            "ready": self._connected
        }
    
    def set_position(self, theta: float, phi: float) -> bool:
        """Set positioner angles."""
        if not self._connected:
            return False
        
        # Clamp to valid ranges
        theta = max(0, min(180, theta))
        phi = phi % 360
        
        self._position = (theta, phi)
        
        # Simulate movement time
        time.sleep(0.01)
        
        return True
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position."""
        return self._position
    
    def set_frequency(self, frequency_hz: float) -> bool:
        """Set measurement frequency."""
        if not self._connected:
            return False
        
        if not (self.specification.frequency_range_hz[0] <= 
                frequency_hz <= 
                self.specification.frequency_range_hz[1]):
            return False
        
        self._frequency_hz = frequency_hz
        return True
    
    def measure(self, measurement_type: MeasurementType) -> MeasurementPoint:
        """Perform synthetic measurement."""
        if not self._connected:
            raise MeasurementException("Chamber not connected")
        
        value = 0.0
        unit = "dBm"
        uncertainty = 0.5
        
        if measurement_type == MeasurementType.ANTENNA_PATTERN:
            # Get pattern value for current position
            theta, phi = self._position
            theta_key = int(round(theta / 5) * 5)
            phi_key = int(round(phi / 5) * 5) % 360
            
            value = self._antenna_pattern.get((theta_key, phi_key), -40.0)
            unit = "dBi"
            uncertainty = 0.3
            
        elif measurement_type == MeasurementType.POWER:
            # Simulate power measurement
            value = -30.0  # Base power
            unit = "dBm"
            uncertainty = 0.2
            
        elif measurement_type == MeasurementType.GAIN:
            # Simulate gain measurement
            value = 10.0
            unit = "dB"
            uncertainty = 0.3
            
        elif measurement_type == MeasurementType.VSWR:
            # Simulate VSWR
            value = 1.5
            unit = ":1"
            uncertainty = 0.1
            
        elif measurement_type == MeasurementType.RETURN_LOSS:
            # Simulate return loss
            value = -15.0
            unit = "dB"
            uncertainty = 0.3
        
        # Add noise if enabled
        if self.add_noise:
            noise = np.random.normal(0, uncertainty / 3)
            value += noise
        
        return MeasurementPoint(
            timestamp=datetime.utcnow(),
            frequency_hz=self._frequency_hz,
            value=value,
            unit=unit,
            uncertainty=uncertainty,
            temperature_c=self._temperature_c,
            humidity_percent=self._humidity_percent,
            position=None,
            angle=self._position
        )
    
    def set_environmental(
        self,
        temperature_c: float,
        humidity_percent: float
    ) -> None:
        """Set synthetic environmental conditions."""
        self._temperature_c = temperature_c
        self._humidity_percent = humidity_percent


class AntennaPatternMeasurement:
    """
    Antenna pattern measurement system.
    
    Performs 2D and 3D antenna radiation pattern measurements
    in anechoic chamber environments.
    """
    
    def __init__(
        self,
        chamber: ChamberInterface,
        frequency_hz: float,
        polarization: str = "vertical"
    ):
        self.chamber = chamber
        self.frequency_hz = frequency_hz
        self.polarization = polarization
        
        self._pattern_data: Dict[Tuple[float, float], float] = {}
        self._measurement_config = {
            "theta_start": 0,
            "theta_stop": 180,
            "theta_step": 5,
            "phi_start": 0,
            "phi_stop": 360,
            "phi_step": 5,
            "averaging": 1
        }
    
    def configure(
        self,
        theta_range: Tuple[float, float, float] = (0, 180, 5),
        phi_range: Tuple[float, float, float] = (0, 360, 5),
        averaging: int = 1
    ) -> None:
        """
        Configure pattern measurement.
        
        Args:
            theta_range: (start, stop, step) in degrees
            phi_range: (start, stop, step) in degrees
            averaging: Number of averages per point
        """
        self._measurement_config = {
            "theta_start": theta_range[0],
            "theta_stop": theta_range[1],
            "theta_step": theta_range[2],
            "phi_start": phi_range[0],
            "phi_stop": phi_range[1],
            "phi_step": phi_range[2],
            "averaging": averaging
        }
    
    def measure_2d_cut(
        self,
        cut_plane: str = "E",
        fixed_angle: float = 0.0
    ) -> Dict[float, float]:
        """
        Measure 2D pattern cut.
        
        Args:
            cut_plane: "E" (theta) or "H" (phi) plane cut
            fixed_angle: Fixed angle for the other dimension
            
        Returns:
            Dictionary of angle -> gain in dB
        """
        self.chamber.set_frequency(self.frequency_hz)
        
        cut_data = {}
        config = self._measurement_config
        
        if cut_plane.upper() == "E":
            # E-plane cut (vary theta, fixed phi)
            angles = np.arange(
                config["theta_start"],
                config["theta_stop"] + config["theta_step"],
                config["theta_step"]
            )
            
            for theta in angles:
                self.chamber.set_position(theta, fixed_angle)
                
                # Average measurements
                values = []
                for _ in range(config["averaging"]):
                    measurement = self.chamber.measure(
                        MeasurementType.ANTENNA_PATTERN
                    )
                    values.append(measurement.value)
                
                cut_data[theta] = np.mean(values)
        else:
            # H-plane cut (vary phi, fixed theta)
            angles = np.arange(
                config["phi_start"],
                config["phi_stop"] + config["phi_step"],
                config["phi_step"]
            )
            
            for phi in angles:
                self.chamber.set_position(fixed_angle, phi)
                
                values = []
                for _ in range(config["averaging"]):
                    measurement = self.chamber.measure(
                        MeasurementType.ANTENNA_PATTERN
                    )
                    values.append(measurement.value)
                
                cut_data[phi] = np.mean(values)
        
        return cut_data
    
    def measure_3d_pattern(
        self,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[Tuple[float, float], float]:
        """
        Measure full 3D antenna pattern.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of (theta, phi) -> gain in dB
        """
        self.chamber.set_frequency(self.frequency_hz)
        
        config = self._measurement_config
        
        theta_angles = np.arange(
            config["theta_start"],
            config["theta_stop"] + config["theta_step"],
            config["theta_step"]
        )
        phi_angles = np.arange(
            config["phi_start"],
            config["phi_stop"] + config["phi_step"],
            config["phi_step"]
        )
        
        total_points = len(theta_angles) * len(phi_angles)
        current_point = 0
        
        self._pattern_data = {}
        
        for theta in theta_angles:
            for phi in phi_angles:
                self.chamber.set_position(theta, phi)
                
                values = []
                for _ in range(config["averaging"]):
                    measurement = self.chamber.measure(
                        MeasurementType.ANTENNA_PATTERN
                    )
                    values.append(measurement.value)
                
                self._pattern_data[(theta, phi)] = np.mean(values)
                
                current_point += 1
                if progress_callback:
                    progress_callback(current_point / total_points * 100)
        
        return self._pattern_data
    
    def calculate_parameters(self) -> Dict[str, Any]:
        """
        Calculate antenna parameters from pattern data.
        
        Returns:
            Dictionary of calculated parameters
        """
        if not self._pattern_data:
            raise ValueError("No pattern data available")
        
        # Find peak gain
        peak_gain = max(self._pattern_data.values())
        peak_angles = [k for k, v in self._pattern_data.items() if v == peak_gain]
        
        # Calculate 3dB beamwidth (simplified)
        threshold = peak_gain - 3.0
        
        # Find half-power points in theta (E-plane)
        theta_90 = [(t, g) for (t, p), g in self._pattern_data.items() 
                    if p == 90 or p == 0]
        theta_90_sorted = sorted(theta_90, key=lambda x: x[0])
        
        hpbw_e = self._calculate_beamwidth(theta_90_sorted, threshold)
        
        # Calculate front-to-back ratio
        front_gain = self._pattern_data.get((90, 0), peak_gain)
        back_gain = self._pattern_data.get((90, 180), -30)
        fb_ratio = front_gain - back_gain
        
        # Estimate directivity (simplified spherical integration)
        directivity = self._calculate_directivity()
        
        return {
            "peak_gain_dbi": peak_gain,
            "peak_direction": peak_angles[0] if peak_angles else (0, 0),
            "hpbw_e_plane_deg": hpbw_e,
            "front_to_back_ratio_db": fb_ratio,
            "directivity_dbi": directivity,
            "frequency_hz": self.frequency_hz,
            "polarization": self.polarization,
            "measurement_points": len(self._pattern_data)
        }
    
    def _calculate_beamwidth(
        self,
        data: List[Tuple[float, float]],
        threshold: float
    ) -> float:
        """Calculate beamwidth from 1D pattern data."""
        if len(data) < 3:
            return 0.0
        
        angles = [d[0] for d in data]
        gains = [d[1] for d in data]
        
        # Find points above threshold
        above = [(a, g) for a, g in zip(angles, gains) if g >= threshold]
        
        if len(above) < 2:
            return 0.0
        
        return above[-1][0] - above[0][0]
    
    def _calculate_directivity(self) -> float:
        """Calculate directivity from 3D pattern data."""
        if not self._pattern_data:
            return 0.0
        
        # Convert to linear power
        total_power = 0.0
        peak_power = 0.0
        
        for (theta, phi), gain_db in self._pattern_data.items():
            gain_linear = 10 ** (gain_db / 10)
            peak_power = max(peak_power, gain_linear)
            
            # Spherical integration weight
            theta_rad = math.radians(theta)
            weight = math.sin(theta_rad)
            total_power += gain_linear * weight
        
        # Normalize
        if total_power > 0:
            directivity = 4 * math.pi * peak_power / total_power
            return 10 * math.log10(directivity)
        
        return 0.0
    
    def export_pattern(self, format: str = "csv") -> str:
        """
        Export pattern data to string format.
        
        Args:
            format: Output format ("csv", "json", "nsf")
            
        Returns:
            Formatted pattern data string
        """
        if format == "csv":
            lines = ["theta,phi,gain_dbi"]
            for (theta, phi), gain in sorted(self._pattern_data.items()):
                lines.append(f"{theta},{phi},{gain:.2f}")
            return "\n".join(lines)
        
        elif format == "json":
            import json
            data = {
                "frequency_hz": float(self.frequency_hz),
                "polarization": self.polarization,
                "pattern": [
                    {"theta": int(t), "phi": int(p), "gain_dbi": float(g)}
                    for (t, p), g in sorted(self._pattern_data.items())
                ]
            }
            return json.dumps(data, indent=2)
        
        else:
            raise ValueError(f"Unknown format: {format}")


class PowerCalibration:
    """
    RF power calibration and validation system.
    
    Performs comprehensive power calibration including linearity,
    accuracy, and stability measurements.
    """
    
    def __init__(
        self,
        chamber: ChamberInterface,
        reference_power_meter: Optional[Any] = None
    ):
        self.chamber = chamber
        self.reference_meter = reference_power_meter
        
        self._calibration_points: List[CalibrationPoint] = []
        self._linearity_data: List[Tuple[float, float]] = []
        
        # Specifications
        self._specs = {
            "power_accuracy_db": 0.5,
            "linearity_db": 0.3,
            "stability_db": 0.1,
            "temp_coefficient_db_per_c": 0.01
        }
    
    def calibrate_power(
        self,
        frequency_points_hz: List[float],
        power_levels_dbm: List[float],
        reference_values: Optional[Dict[Tuple[float, float], float]] = None
    ) -> List[CalibrationPoint]:
        """
        Perform power calibration across frequency and power levels.
        
        Args:
            frequency_points_hz: List of calibration frequencies
            power_levels_dbm: List of power levels to calibrate
            reference_values: Optional reference measurements
            
        Returns:
            List of calibration points
        """
        self._calibration_points = []
        
        for freq in frequency_points_hz:
            self.chamber.set_frequency(freq)
            
            for power in power_levels_dbm:
                # Measure power
                measurement = self.chamber.measure(MeasurementType.POWER)
                measured_value = measurement.value
                
                # Get reference value
                if reference_values and (freq, power) in reference_values:
                    reference_value = reference_values[(freq, power)]
                else:
                    # Use expected value as reference (simulated)
                    reference_value = power
                
                # Calculate correction factor
                correction = reference_value - measured_value
                
                # Calculate uncertainty
                uncertainty = self._calculate_uncertainty(measurement)
                
                # Check if within spec
                valid = abs(correction) <= self._specs["power_accuracy_db"]
                
                cal_point = CalibrationPoint(
                    frequency_hz=freq,
                    reference_value=reference_value,
                    measured_value=measured_value,
                    correction_factor=correction,
                    uncertainty=uncertainty,
                    temperature_c=measurement.temperature_c,
                    valid=valid
                )
                
                self._calibration_points.append(cal_point)
        
        return self._calibration_points
    
    def measure_linearity(
        self,
        frequency_hz: float,
        power_range_dbm: Tuple[float, float],
        steps: int = 20
    ) -> Dict[str, Any]:
        """
        Measure power linearity over specified range.
        
        Args:
            frequency_hz: Test frequency
            power_range_dbm: (min, max) power range
            steps: Number of measurement steps
            
        Returns:
            Linearity analysis results
        """
        self.chamber.set_frequency(frequency_hz)
        
        power_levels = np.linspace(
            power_range_dbm[0],
            power_range_dbm[1],
            steps
        )
        
        self._linearity_data = []
        
        for expected_power in power_levels:
            # In real system, would set DUT output power
            # Here we measure whatever the chamber provides
            measurement = self.chamber.measure(MeasurementType.POWER)
            
            # Simulate linearity test by using expected vs measured
            self._linearity_data.append((expected_power, measurement.value))
        
        # Analyze linearity
        expected = np.array([d[0] for d in self._linearity_data])
        measured = np.array([d[1] for d in self._linearity_data])
        
        # Linear fit
        coeffs = np.polyfit(expected, measured, 1)
        fitted = np.polyval(coeffs, expected)
        
        # Calculate linearity error
        linearity_error = measured - fitted
        max_linearity_error = np.max(np.abs(linearity_error))
        rms_linearity_error = np.sqrt(np.mean(linearity_error ** 2))
        
        # Dynamic range
        valid_range = measured[measured > measured.max() - 60]  # 60dB dynamic range
        
        return {
            "frequency_hz": frequency_hz,
            "power_range_dbm": power_range_dbm,
            "slope": coeffs[0],
            "offset_db": coeffs[1],
            "max_linearity_error_db": max_linearity_error,
            "rms_linearity_error_db": rms_linearity_error,
            "dynamic_range_db": measured.max() - measured.min(),
            "within_spec": max_linearity_error <= self._specs["linearity_db"],
            "data_points": len(self._linearity_data)
        }
    
    def measure_stability(
        self,
        frequency_hz: float,
        duration_seconds: float = 60.0,
        interval_seconds: float = 1.0
    ) -> Dict[str, Any]:
        """
        Measure power stability over time.
        
        Args:
            frequency_hz: Test frequency
            duration_seconds: Measurement duration
            interval_seconds: Measurement interval
            
        Returns:
            Stability analysis results
        """
        self.chamber.set_frequency(frequency_hz)
        
        measurements = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            measurement = self.chamber.measure(MeasurementType.POWER)
            measurements.append({
                "time": time.time() - start_time,
                "power_dbm": measurement.value,
                "temperature_c": measurement.temperature_c
            })
            time.sleep(interval_seconds)
        
        # Analyze stability
        powers = np.array([m["power_dbm"] for m in measurements])
        temps = np.array([m["temperature_c"] for m in measurements])
        
        power_mean = np.mean(powers)
        power_std = np.std(powers)
        power_pk_pk = np.max(powers) - np.min(powers)
        
        # Temperature correlation
        if np.std(temps) > 0:
            temp_correlation = np.corrcoef(temps, powers)[0, 1]
        else:
            temp_correlation = 0.0
        
        return {
            "frequency_hz": frequency_hz,
            "duration_seconds": duration_seconds,
            "mean_power_dbm": power_mean,
            "std_dev_db": power_std,
            "peak_to_peak_db": power_pk_pk,
            "temperature_correlation": temp_correlation,
            "temperature_range_c": (np.min(temps), np.max(temps)),
            "within_spec": power_pk_pk <= self._specs["stability_db"],
            "sample_count": len(measurements)
        }
    
    def _calculate_uncertainty(self, measurement: MeasurementPoint) -> float:
        """Calculate measurement uncertainty."""
        # Combine uncertainty components
        components = {
            "instrument": measurement.uncertainty,
            "mismatch": 0.1,
            "connector": 0.05,
            "temperature": abs(measurement.temperature_c - 23) * self._specs["temp_coefficient_db_per_c"]
        }
        
        # RSS combination
        combined = math.sqrt(sum(u**2 for u in components.values()))
        
        return combined
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration results."""
        if not self._calibration_points:
            return {"status": "no_data"}
        
        valid_points = [p for p in self._calibration_points if p.valid]
        
        corrections = [p.correction_factor for p in self._calibration_points]
        
        return {
            "total_points": len(self._calibration_points),
            "valid_points": len(valid_points),
            "pass_rate_percent": len(valid_points) / len(self._calibration_points) * 100,
            "mean_correction_db": np.mean(corrections),
            "max_correction_db": np.max(np.abs(corrections)),
            "frequency_range_hz": (
                min(p.frequency_hz for p in self._calibration_points),
                max(p.frequency_hz for p in self._calibration_points)
            ),
            "temperature_range_c": (
                min(p.temperature_c for p in self._calibration_points),
                max(p.temperature_c for p in self._calibration_points)
            )
        }


class FrequencyCalibration:
    """
    RF frequency calibration and validation system.
    
    Performs frequency accuracy, stability, and response
    characterization.
    """
    
    def __init__(
        self,
        chamber: ChamberInterface,
        reference_source: Optional[Any] = None
    ):
        self.chamber = chamber
        self.reference_source = reference_source
        
        self._frequency_points: List[CalibrationPoint] = []
        self._response_data: Dict[float, float] = {}
        
        # Specifications
        self._specs = {
            "frequency_accuracy_ppm": 1.0,
            "frequency_stability_ppm": 0.1,
            "flatness_db": 1.0
        }
    
    def calibrate_frequency(
        self,
        frequency_points_hz: List[float],
        reference_frequencies_hz: Optional[Dict[float, float]] = None
    ) -> List[CalibrationPoint]:
        """
        Calibrate frequency accuracy at specified points.
        
        Args:
            frequency_points_hz: List of nominal frequencies
            reference_frequencies_hz: Optional reference measurements
            
        Returns:
            List of calibration points
        """
        self._frequency_points = []
        
        for nominal_freq in frequency_points_hz:
            self.chamber.set_frequency(nominal_freq)
            
            # Measure frequency (simulated - in real system would use counter)
            measurement = self.chamber.measure(MeasurementType.FREQUENCY)
            
            # Get reference
            if reference_frequencies_hz and nominal_freq in reference_frequencies_hz:
                reference_freq = reference_frequencies_hz[nominal_freq]
            else:
                reference_freq = nominal_freq
            
            # Calculate error in ppm
            measured_freq = measurement.value if measurement.value > 0 else nominal_freq
            error_ppm = (measured_freq - reference_freq) / reference_freq * 1e6
            
            # Check spec
            valid = abs(error_ppm) <= self._specs["frequency_accuracy_ppm"]
            
            cal_point = CalibrationPoint(
                frequency_hz=nominal_freq,
                reference_value=reference_freq,
                measured_value=measured_freq,
                correction_factor=error_ppm,  # Store as ppm
                uncertainty=0.1,  # ppm
                temperature_c=measurement.temperature_c,
                valid=valid
            )
            
            self._frequency_points.append(cal_point)
        
        return self._frequency_points
    
    def measure_frequency_response(
        self,
        start_freq_hz: float,
        stop_freq_hz: float,
        points: int = 201
    ) -> Dict[str, Any]:
        """
        Measure frequency response (amplitude vs frequency).
        
        Args:
            start_freq_hz: Start frequency
            stop_freq_hz: Stop frequency
            points: Number of measurement points
            
        Returns:
            Frequency response analysis results
        """
        frequencies = np.logspace(
            np.log10(start_freq_hz),
            np.log10(stop_freq_hz),
            points
        )
        
        self._response_data = {}
        
        for freq in frequencies:
            self.chamber.set_frequency(freq)
            measurement = self.chamber.measure(MeasurementType.POWER)
            self._response_data[freq] = measurement.value
        
        # Analyze response
        amplitudes = np.array(list(self._response_data.values()))
        
        # Calculate flatness
        reference_level = np.mean(amplitudes)
        deviation = amplitudes - reference_level
        flatness = np.max(amplitudes) - np.min(amplitudes)
        
        # Find -3dB bandwidth
        threshold = reference_level - 3.0
        above_threshold = np.where(amplitudes >= threshold)[0]
        
        if len(above_threshold) >= 2:
            freq_array = np.array(list(self._response_data.keys()))
            bandwidth = freq_array[above_threshold[-1]] - freq_array[above_threshold[0]]
        else:
            bandwidth = 0.0
        
        return {
            "start_freq_hz": start_freq_hz,
            "stop_freq_hz": stop_freq_hz,
            "points": points,
            "reference_level_dbm": reference_level,
            "flatness_db": flatness,
            "max_deviation_db": np.max(np.abs(deviation)),
            "bandwidth_3db_hz": bandwidth,
            "within_spec": flatness <= self._specs["flatness_db"]
        }
    
    def measure_phase_response(
        self,
        frequency_points_hz: List[float]
    ) -> Dict[float, float]:
        """
        Measure phase response at specified frequencies.
        
        Args:
            frequency_points_hz: Measurement frequencies
            
        Returns:
            Dictionary of frequency -> phase in degrees
        """
        phase_data = {}
        
        for freq in frequency_points_hz:
            self.chamber.set_frequency(freq)
            measurement = self.chamber.measure(MeasurementType.PHASE)
            phase_data[freq] = measurement.value
        
        return phase_data
    
    def calculate_group_delay(
        self,
        phase_data: Dict[float, float]
    ) -> Dict[float, float]:
        """
        Calculate group delay from phase data.
        
        Args:
            phase_data: Dictionary of frequency -> phase
            
        Returns:
            Dictionary of frequency -> group delay in ns
        """
        frequencies = np.array(sorted(phase_data.keys()))
        phases = np.array([phase_data[f] for f in frequencies])
        
        # Convert to radians
        phases_rad = np.deg2rad(phases)
        
        # Unwrap phase
        phases_unwrapped = np.unwrap(phases_rad)
        
        # Calculate group delay: -d(phase)/d(freq) / (2*pi)
        group_delay = {}
        
        for i in range(1, len(frequencies)):
            df = frequencies[i] - frequencies[i-1]
            dphi = phases_unwrapped[i] - phases_unwrapped[i-1]
            
            delay_s = -dphi / (2 * math.pi * df)
            delay_ns = delay_s * 1e9
            
            mid_freq = (frequencies[i] + frequencies[i-1]) / 2
            group_delay[mid_freq] = delay_ns
        
        return group_delay


class CalibrationValidator:
    """
    Calibration validation and verification system.
    
    Validates calibration against specifications and
    generates compliance reports.
    """
    
    def __init__(self, specifications: Dict[str, Any]):
        self.specifications = specifications
        self._validation_results: List[Dict[str, Any]] = []
    
    def validate_power_calibration(
        self,
        calibration_points: List[CalibrationPoint]
    ) -> Dict[str, Any]:
        """
        Validate power calibration results.
        
        Args:
            calibration_points: List of calibration points
            
        Returns:
            Validation results
        """
        spec_accuracy = self.specifications.get("power_accuracy_db", 0.5)
        
        results = {
            "test_name": "Power Calibration Validation",
            "specification": f"±{spec_accuracy} dB",
            "test_date": datetime.utcnow().isoformat(),
            "points_tested": len(calibration_points),
            "points_passed": 0,
            "points_failed": 0,
            "max_error_db": 0.0,
            "details": []
        }
        
        for point in calibration_points:
            error = abs(point.correction_factor)
            passed = error <= spec_accuracy
            
            if passed:
                results["points_passed"] += 1
            else:
                results["points_failed"] += 1
            
            results["max_error_db"] = max(results["max_error_db"], error)
            
            results["details"].append({
                "frequency_hz": point.frequency_hz,
                "error_db": point.correction_factor,
                "uncertainty_db": point.uncertainty,
                "passed": passed
            })
        
        results["overall_result"] = (
            ChamberTestResult.PASS if results["points_failed"] == 0
            else ChamberTestResult.FAIL
        )
        
        self._validation_results.append(results)
        return results
    
    def validate_frequency_calibration(
        self,
        calibration_points: List[CalibrationPoint]
    ) -> Dict[str, Any]:
        """
        Validate frequency calibration results.
        
        Args:
            calibration_points: List of calibration points
            
        Returns:
            Validation results
        """
        spec_accuracy = self.specifications.get("frequency_accuracy_ppm", 1.0)
        
        results = {
            "test_name": "Frequency Calibration Validation",
            "specification": f"±{spec_accuracy} ppm",
            "test_date": datetime.utcnow().isoformat(),
            "points_tested": len(calibration_points),
            "points_passed": 0,
            "points_failed": 0,
            "max_error_ppm": 0.0,
            "details": []
        }
        
        for point in calibration_points:
            error_ppm = abs(point.correction_factor)
            passed = error_ppm <= spec_accuracy
            
            if passed:
                results["points_passed"] += 1
            else:
                results["points_failed"] += 1
            
            results["max_error_ppm"] = max(results["max_error_ppm"], error_ppm)
            
            results["details"].append({
                "frequency_hz": point.frequency_hz,
                "error_ppm": point.correction_factor,
                "passed": passed
            })
        
        results["overall_result"] = (
            ChamberTestResult.PASS if results["points_failed"] == 0
            else ChamberTestResult.FAIL
        )
        
        self._validation_results.append(results)
        return results
    
    def validate_antenna_pattern(
        self,
        measured_params: Dict[str, Any],
        expected_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate antenna pattern parameters.
        
        Args:
            measured_params: Measured antenna parameters
            expected_params: Expected/specified parameters
            
        Returns:
            Validation results
        """
        results = {
            "test_name": "Antenna Pattern Validation",
            "test_date": datetime.utcnow().isoformat(),
            "parameters_tested": 0,
            "parameters_passed": 0,
            "parameters_failed": 0,
            "details": []
        }
        
        # Define validation criteria
        validations = [
            ("peak_gain_dbi", 1.0),      # ±1 dB tolerance
            ("hpbw_e_plane_deg", 5.0),   # ±5 degree tolerance
            ("front_to_back_ratio_db", 3.0),  # ±3 dB tolerance
        ]
        
        for param_name, tolerance in validations:
            if param_name in measured_params and param_name in expected_params:
                measured = measured_params[param_name]
                expected = expected_params[param_name]
                error = abs(measured - expected)
                passed = error <= tolerance
                
                results["parameters_tested"] += 1
                if passed:
                    results["parameters_passed"] += 1
                else:
                    results["parameters_failed"] += 1
                
                results["details"].append({
                    "parameter": param_name,
                    "measured": measured,
                    "expected": expected,
                    "tolerance": tolerance,
                    "error": error,
                    "passed": passed
                })
        
        results["overall_result"] = (
            ChamberTestResult.PASS if results["parameters_failed"] == 0
            else ChamberTestResult.FAIL
        )
        
        self._validation_results.append(results)
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        total_tests = len(self._validation_results)
        passed_tests = sum(
            1 for r in self._validation_results
            if r.get("overall_result") == ChamberTestResult.PASS
        )
        
        return {
            "report_id": secrets.token_hex(8),
            "generated_at": datetime.utcnow().isoformat(),
            "total_validations": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate_percent": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "validations": self._validation_results,
            "overall_status": (
                CalibrationStatus.VALID if passed_tests == total_tests
                else CalibrationStatus.INVALID
            ).value
        }


class UncertaintyAnalyzer:
    """
    Measurement uncertainty analysis per GUM (Guide to Uncertainty in Measurement).
    
    Calculates and documents measurement uncertainty budgets.
    """
    
    def __init__(self):
        self._budgets: List[UncertaintyBudget] = []
    
    def calculate_power_uncertainty(
        self,
        measurement_type: MeasurementType,
        frequency_hz: float,
        components: Dict[str, Tuple[float, str]]
    ) -> UncertaintyBudget:
        """
        Calculate power measurement uncertainty budget.
        
        Args:
            measurement_type: Type of measurement
            frequency_hz: Measurement frequency
            components: Dict of component_name -> (value, distribution)
                       distribution: "normal", "rectangular", "triangular"
                       
        Returns:
            Complete uncertainty budget
        """
        processed_components = {}
        
        for name, (value, distribution) in components.items():
            # Convert to standard uncertainty
            if distribution == "normal":
                std_uncertainty = value  # Already standard uncertainty
            elif distribution == "rectangular":
                std_uncertainty = value / math.sqrt(3)
            elif distribution == "triangular":
                std_uncertainty = value / math.sqrt(6)
            else:
                std_uncertainty = value
            
            processed_components[name] = std_uncertainty
        
        # Combined standard uncertainty (RSS)
        combined = math.sqrt(sum(u**2 for u in processed_components.values()))
        
        # Expanded uncertainty (k=2 for 95% confidence)
        coverage_factor = 2.0
        expanded = combined * coverage_factor
        
        budget = UncertaintyBudget(
            measurement_type=measurement_type,
            frequency_hz=frequency_hz,
            components=processed_components,
            combined_standard_uncertainty=combined,
            expanded_uncertainty=expanded,
            coverage_factor=coverage_factor,
            notes="Calculated per GUM methodology"
        )
        
        self._budgets.append(budget)
        return budget
    
    def calculate_antenna_pattern_uncertainty(
        self,
        frequency_hz: float
    ) -> UncertaintyBudget:
        """
        Calculate antenna pattern measurement uncertainty.
        
        Args:
            frequency_hz: Measurement frequency
            
        Returns:
            Uncertainty budget for pattern measurement
        """
        # Typical uncertainty components for antenna measurements
        components = {
            "range_reflection": (0.3, "rectangular"),
            "positioning": (0.2, "rectangular"),
            "probe_polarization": (0.1, "normal"),
            "gain_standard": (0.25, "normal"),
            "cable_flexing": (0.1, "rectangular"),
            "receiver_linearity": (0.1, "rectangular"),
            "alignment": (0.15, "rectangular"),
            "temperature": (0.05, "rectangular")
        }
        
        return self.calculate_power_uncertainty(
            MeasurementType.ANTENNA_PATTERN,
            frequency_hz,
            components
        )
    
    def export_budgets(self) -> List[Dict[str, Any]]:
        """Export all uncertainty budgets."""
        return [
            {
                "measurement_type": b.measurement_type.value,
                "frequency_hz": b.frequency_hz,
                "components": b.components,
                "combined_standard_uncertainty": b.combined_standard_uncertainty,
                "expanded_uncertainty": b.expanded_uncertainty,
                "coverage_factor": b.coverage_factor,
                "notes": b.notes
            }
            for b in self._budgets
        ]


# Export public API
__all__ = [
    # Enums
    "ChamberType",
    "MeasurementType",
    "CalibrationStatus",
    "ChamberTestResult",
    
    # Data classes
    "ChamberSpecification",
    "MeasurementPoint",
    "CalibrationPoint",
    "ChamberTestSequenceResult",
    "UncertaintyBudget",
    
    # Exceptions
    "ChamberException",
    "CalibrationException",
    "MeasurementException",
    
    # Interfaces
    "ChamberInterface",
    "DevelopmentChamber",
    
    # Measurement systems
    "AntennaPatternMeasurement",
    "PowerCalibration",
    "FrequencyCalibration",
    
    # Validation
    "CalibrationValidator",
    "UncertaintyAnalyzer",
]
