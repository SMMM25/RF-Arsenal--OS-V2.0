"""
Reverberation Chamber Testing Module for RF Arsenal OS.

This module provides comprehensive reverberation (mode-stirred) chamber
testing capabilities for EMC testing and over-the-air performance
evaluation per IEC 61000-4-21.

Features:
- Mode-stirred chamber measurements
- Field uniformity verification
- Stirrer effectiveness characterization
- Loading effect compensation
- OTA performance testing
- Statistical analysis of measurements

Standards:
- IEC 61000-4-21: Reverberation chamber methods
- IEC 61000-4-3: Radiated immunity testing
- CTIA OTA: Over-the-air performance testing

Author: RF Arsenal Development Team
License: Proprietary - Calibration Sensitive
"""

import asyncio
import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy import stats
from scipy import signal as scipy_signal

from . import (
    ChamberType,
    MeasurementType,
    CalibrationStatus,
    TestResult,
    ChamberSpecification,
    MeasurementPoint,
    ChamberInterface,
    ChamberException,
    MeasurementException,
)

logger = logging.getLogger(__name__)


class StirrerType(Enum):
    """Types of mode stirrers."""
    
    MECHANICAL = "mechanical"       # Rotating paddle stirrer
    PLATFORM = "platform"           # Rotating platform
    FREQUENCY = "frequency"         # Frequency stirring
    SOURCE = "source"               # Source position stirring
    COMBINED = "combined"           # Combined stirring methods


class LoadingCondition(Enum):
    """Chamber loading conditions."""
    
    UNLOADED = "unloaded"           # Empty chamber
    REFERENCE = "reference"         # With reference absorber
    DUT_LOADED = "dut_loaded"       # With device under test
    HEAVILY_LOADED = "heavily_loaded"  # High loading condition


@dataclass
class StirrerConfig:
    """Stirrer configuration."""
    
    stirrer_type: StirrerType
    num_positions: int              # Number of stirrer positions
    rotation_speed_rpm: float       # For mechanical stirrer
    step_mode: bool                 # Step vs continuous mode
    dwell_time_s: float            # Dwell time at each position


@dataclass
class FieldUniformityData:
    """Field uniformity measurement data."""
    
    frequency_hz: float
    positions: List[Tuple[float, float, float]]
    amplitudes_dbm: np.ndarray      # [positions, stirrer_positions]
    mean_amplitude_dbm: float
    std_dev_db: float
    max_deviation_db: float
    uniformity_pass: bool


@dataclass
class ChamberCalibrationData:
    """Reverberation chamber calibration data."""
    
    frequency_hz: float
    chamber_factor_db: float        # Chamber insertion gain
    loading_factor_db: float        # Loading correction
    stirrer_efficiency: float       # 0-1, stirring efficiency
    q_factor: float                 # Chamber Q-factor
    coherence_bandwidth_hz: float
    decay_time_s: float            # Chamber decay time constant


@dataclass
class OTAMeasurementResult:
    """Over-the-air measurement result."""
    
    frequency_hz: float
    trp_dbm: float                  # Total Radiated Power
    tis_dbm: float                  # Total Isotropic Sensitivity
    eirp_dbm: float                 # Effective Isotropic Radiated Power
    uncertainty_db: float
    measurement_time_s: float
    stirrer_positions: int


class ReverberationChamberController:
    """
    Reverberation chamber control and measurement system.
    
    Controls stirrer, RF equipment, and coordinates
    mode-stirred measurements.
    """
    
    def __init__(
        self,
        chamber: ChamberInterface,
        specification: ChamberSpecification,
        stirrer_config: StirrerConfig
    ):
        self.chamber = chamber
        self.specification = specification
        self.stirrer_config = stirrer_config
        
        self._connected = False
        self._stirrer_position = 0
        self._current_frequency = 1e9
        self._loading = LoadingCondition.UNLOADED
        
        # Calibration data
        self._calibration: Dict[float, ChamberCalibrationData] = {}
        
        # Measurement data buffer
        self._measurement_buffer: List[MeasurementPoint] = []
    
    def connect(self) -> bool:
        """Connect to chamber control system."""
        try:
            self._connected = self.chamber.connect()
            
            if self._connected:
                logger.info(
                    f"Connected to reverberation chamber: "
                    f"{self.specification.chamber_id}"
                )
            
            return self._connected
            
        except Exception as e:
            logger.error(f"Chamber connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from chamber."""
        self._connected = False
        self.chamber.disconnect()
        return True
    
    def set_frequency(self, frequency_hz: float) -> bool:
        """Set measurement frequency."""
        if not self._connected:
            return False
        
        self._current_frequency = frequency_hz
        return self.chamber.set_frequency(frequency_hz)
    
    def move_stirrer(self, position: int, wait: bool = True) -> bool:
        """
        Move stirrer to specified position.
        
        Args:
            position: Stirrer position (0 to num_positions-1)
            wait: Wait for movement and settling
            
        Returns:
            True if successful
        """
        if not self._connected:
            return False
        
        if position < 0 or position >= self.stirrer_config.num_positions:
            return False
        
        self._stirrer_position = position
        
        if wait:
            time.sleep(self.stirrer_config.dwell_time_s)
        
        return True
    
    def rotate_stirrer_continuous(
        self,
        duration_s: float,
        rpm: Optional[float] = None
    ) -> bool:
        """
        Rotate stirrer continuously for specified duration.
        
        Args:
            duration_s: Rotation duration
            rpm: Override rotation speed (optional)
            
        Returns:
            True if successful
        """
        if not self._connected:
            return False
        
        # In real system, would control motor
        time.sleep(duration_s)
        
        return True
    
    def measure_at_stirrer_positions(
        self,
        measurement_type: MeasurementType = MeasurementType.POWER,
        num_positions: Optional[int] = None
    ) -> List[MeasurementPoint]:
        """
        Measure at all stirrer positions.
        
        Args:
            measurement_type: Type of measurement
            num_positions: Override number of positions
            
        Returns:
            List of measurements at each position
        """
        if not self._connected:
            raise MeasurementException("Chamber not connected")
        
        positions = num_positions or self.stirrer_config.num_positions
        measurements = []
        
        for pos in range(positions):
            self.move_stirrer(pos)
            measurement = self.chamber.measure(measurement_type)
            measurement.metadata["stirrer_position"] = pos
            measurements.append(measurement)
        
        self._measurement_buffer = measurements
        return measurements
    
    def get_statistics(
        self,
        measurements: Optional[List[MeasurementPoint]] = None
    ) -> Dict[str, float]:
        """
        Calculate statistics from measurements.
        
        Args:
            measurements: Measurements to analyze (uses buffer if None)
            
        Returns:
            Statistical summary
        """
        meas = measurements or self._measurement_buffer
        
        if not meas:
            return {}
        
        values = np.array([m.value for m in meas])
        
        # Convert from dBm to linear for proper averaging
        linear_values = 10 ** (values / 10)
        
        return {
            "count": len(values),
            "mean_dbm": 10 * np.log10(np.mean(linear_values)),
            "std_dev_db": np.std(values),
            "max_dbm": np.max(values),
            "min_dbm": np.min(values),
            "peak_to_peak_db": np.max(values) - np.min(values),
            "median_dbm": np.median(values)
        }
    
    def set_loading(self, loading: LoadingCondition) -> None:
        """Set chamber loading condition."""
        self._loading = loading
    
    def get_status(self) -> Dict[str, Any]:
        """Get chamber status."""
        return {
            "connected": self._connected,
            "chamber_id": self.specification.chamber_id,
            "stirrer_position": self._stirrer_position,
            "stirrer_config": {
                "type": self.stirrer_config.stirrer_type.value,
                "positions": self.stirrer_config.num_positions
            },
            "current_frequency_hz": self._current_frequency,
            "loading": self._loading.value,
            "calibrated_frequencies": len(self._calibration)
        }


class FieldUniformityTest:
    """
    Field uniformity verification test.
    
    Verifies chamber field uniformity per IEC 61000-4-21
    requirements.
    """
    
    def __init__(
        self,
        controller: ReverberationChamberController,
        working_volume_m: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ):
        self.controller = controller
        self.working_volume_m = working_volume_m
        
        self._uniformity_data: List[FieldUniformityData] = []
        
        # IEC 61000-4-21 requirements
        self._requirements = {
            "min_positions": 8,          # Minimum measurement positions
            "max_std_dev_db": 3.0,       # Maximum standard deviation
            "min_stirrer_positions": 12,  # Minimum stirrer positions
            "loading_factor_tolerance_db": 3.0
        }
    
    def define_measurement_positions(
        self,
        num_positions: int = 8
    ) -> List[Tuple[float, float, float]]:
        """
        Define measurement positions within working volume.
        
        Args:
            num_positions: Number of positions (8 for corners)
            
        Returns:
            List of (x, y, z) positions in meters
        """
        wx, wy, wz = self.working_volume_m
        
        if num_positions == 8:
            # 8 corners of working volume (standard configuration)
            positions = [
                (-wx/2, -wy/2, -wz/2),
                (-wx/2, -wy/2, +wz/2),
                (-wx/2, +wy/2, -wz/2),
                (-wx/2, +wy/2, +wz/2),
                (+wx/2, -wy/2, -wz/2),
                (+wx/2, -wy/2, +wz/2),
                (+wx/2, +wy/2, -wz/2),
                (+wx/2, +wy/2, +wz/2),
            ]
        else:
            # Generate grid of positions
            positions = []
            nx = int(np.ceil(num_positions ** (1/3)))
            ny = nx
            nz = int(np.ceil(num_positions / (nx * ny)))
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if len(positions) < num_positions:
                            x = -wx/2 + (i + 0.5) * wx / nx
                            y = -wy/2 + (j + 0.5) * wy / ny
                            z = -wz/2 + (k + 0.5) * wz / nz
                            positions.append((x, y, z))
        
        return positions
    
    def measure_uniformity(
        self,
        frequency_hz: float,
        positions: Optional[List[Tuple[float, float, float]]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> FieldUniformityData:
        """
        Measure field uniformity at specified frequency.
        
        Args:
            frequency_hz: Test frequency
            positions: Measurement positions (default: 8 corners)
            progress_callback: Progress callback
            
        Returns:
            Field uniformity data
        """
        self.controller.set_frequency(frequency_hz)
        
        meas_positions = positions or self.define_measurement_positions()
        num_positions = len(meas_positions)
        num_stirrer = self.controller.stirrer_config.num_positions
        
        # Initialize data array
        amplitudes = np.zeros((num_positions, num_stirrer))
        
        total_measurements = num_positions * num_stirrer
        current = 0
        
        for pos_idx, position in enumerate(meas_positions):
            # In real system, would move field probe to position
            
            for stir_idx in range(num_stirrer):
                self.controller.move_stirrer(stir_idx)
                
                measurement = self.controller.chamber.measure(MeasurementType.POWER)
                amplitudes[pos_idx, stir_idx] = measurement.value
                
                current += 1
                if progress_callback:
                    progress_callback(current / total_measurements * 100)
        
        # Calculate statistics
        # Average over stirrer positions for each location
        position_means = np.mean(amplitudes, axis=1)
        
        overall_mean = np.mean(position_means)
        std_dev = np.std(position_means)
        max_deviation = np.max(np.abs(position_means - overall_mean))
        
        # Check against requirements
        uniformity_pass = (
            std_dev <= self._requirements["max_std_dev_db"] and
            num_positions >= self._requirements["min_positions"] and
            num_stirrer >= self._requirements["min_stirrer_positions"]
        )
        
        data = FieldUniformityData(
            frequency_hz=frequency_hz,
            positions=meas_positions,
            amplitudes_dbm=amplitudes,
            mean_amplitude_dbm=overall_mean,
            std_dev_db=std_dev,
            max_deviation_db=max_deviation,
            uniformity_pass=uniformity_pass
        )
        
        self._uniformity_data.append(data)
        return data
    
    def measure_uniformity_vs_frequency(
        self,
        frequencies_hz: List[float],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[FieldUniformityData]:
        """
        Measure field uniformity across frequency range.
        
        Args:
            frequencies_hz: Test frequencies
            progress_callback: Progress callback
            
        Returns:
            List of uniformity data per frequency
        """
        results = []
        
        for idx, freq in enumerate(frequencies_hz):
            data = self.measure_uniformity(
                freq,
                progress_callback=lambda p: progress_callback(
                    (idx + p/100) / len(frequencies_hz) * 100
                ) if progress_callback else None
            )
            results.append(data)
        
        return results
    
    def generate_uniformity_report(self) -> Dict[str, Any]:
        """Generate field uniformity report."""
        if not self._uniformity_data:
            return {"status": "no_data"}
        
        all_pass = all(d.uniformity_pass for d in self._uniformity_data)
        
        return {
            "report_type": "field_uniformity",
            "generated_at": datetime.utcnow().isoformat(),
            "working_volume_m": self.working_volume_m,
            "requirements": self._requirements,
            "frequencies_tested": len(self._uniformity_data),
            "frequencies_passed": sum(1 for d in self._uniformity_data if d.uniformity_pass),
            "overall_pass": all_pass,
            "frequency_results": [
                {
                    "frequency_hz": d.frequency_hz,
                    "mean_dbm": d.mean_amplitude_dbm,
                    "std_dev_db": d.std_dev_db,
                    "max_deviation_db": d.max_deviation_db,
                    "pass": d.uniformity_pass
                }
                for d in self._uniformity_data
            ]
        }


class StirrerEfficiencyTest:
    """
    Stirrer efficiency characterization.
    
    Evaluates mode stirrer effectiveness for achieving
    statistical uniformity.
    """
    
    def __init__(self, controller: ReverberationChamberController):
        self.controller = controller
        self._efficiency_data: Dict[float, Dict[str, Any]] = {}
    
    def measure_stirrer_efficiency(
        self,
        frequency_hz: float,
        num_samples_per_position: int = 10
    ) -> Dict[str, Any]:
        """
        Measure stirrer efficiency at specified frequency.
        
        Args:
            frequency_hz: Test frequency
            num_samples_per_position: Samples per stirrer position
            
        Returns:
            Stirrer efficiency analysis
        """
        self.controller.set_frequency(frequency_hz)
        
        num_positions = self.controller.stirrer_config.num_positions
        
        # Collect data at each stirrer position
        all_samples = []
        position_samples = []
        
        for pos in range(num_positions):
            self.controller.move_stirrer(pos)
            
            samples = []
            for _ in range(num_samples_per_position):
                measurement = self.controller.chamber.measure(MeasurementType.POWER)
                samples.append(measurement.value)
            
            position_samples.append(samples)
            all_samples.extend(samples)
        
        all_samples = np.array(all_samples)
        position_samples = np.array(position_samples)
        
        # Calculate efficiency metrics
        
        # 1. Stirrer correlation - lower is better
        # Correlation between adjacent positions
        correlations = []
        for i in range(num_positions - 1):
            if len(position_samples[i]) == len(position_samples[i+1]):
                corr = np.corrcoef(position_samples[i], position_samples[i+1])[0, 1]
                correlations.append(corr)
        
        mean_correlation = np.mean(correlations) if correlations else 0
        
        # 2. Statistical independence test (chi-squared)
        # Test if distribution matches expected Rayleigh distribution
        
        # 3. Number of independent samples estimation
        # Based on autocorrelation
        autocorr = np.correlate(all_samples - np.mean(all_samples), 
                                all_samples - np.mean(all_samples), 
                                mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find first zero crossing for correlation length
        zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
        correlation_length = zero_crossings[0] if len(zero_crossings) > 0 else len(autocorr)
        
        # Effective independent samples
        n_independent = len(all_samples) / max(correlation_length, 1)
        
        # 4. Efficiency score (0-1)
        efficiency = min(1.0, n_independent / num_positions) * (1 - abs(mean_correlation))
        
        result = {
            "frequency_hz": frequency_hz,
            "stirrer_positions": num_positions,
            "samples_per_position": num_samples_per_position,
            "total_samples": len(all_samples),
            "mean_dbm": float(np.mean(all_samples)),
            "std_dev_db": float(np.std(all_samples)),
            "mean_correlation": float(mean_correlation),
            "correlation_length": int(correlation_length),
            "independent_samples": float(n_independent),
            "efficiency_score": float(efficiency),
            "efficiency_rating": (
                "excellent" if efficiency > 0.9 else
                "good" if efficiency > 0.7 else
                "acceptable" if efficiency > 0.5 else
                "poor"
            )
        }
        
        self._efficiency_data[frequency_hz] = result
        return result
    
    def optimize_stirrer_positions(
        self,
        frequency_hz: float,
        target_uncertainty_db: float = 1.0
    ) -> Dict[str, Any]:
        """
        Determine optimal number of stirrer positions for target uncertainty.
        
        Args:
            frequency_hz: Test frequency
            target_uncertainty_db: Target measurement uncertainty
            
        Returns:
            Optimization results
        """
        self.controller.set_frequency(frequency_hz)
        
        # Measure at increasing number of positions
        results = []
        
        for num_pos in [12, 24, 50, 100, 200]:
            if num_pos > self.controller.stirrer_config.num_positions:
                break
            
            # Collect samples
            samples = []
            step = self.controller.stirrer_config.num_positions // num_pos
            
            for i in range(0, self.controller.stirrer_config.num_positions, max(step, 1)):
                if len(samples) >= num_pos:
                    break
                self.controller.move_stirrer(i)
                measurement = self.controller.chamber.measure(MeasurementType.POWER)
                samples.append(measurement.value)
            
            if len(samples) < 3:
                continue
            
            samples = np.array(samples)
            
            # Calculate uncertainty (std error of mean)
            uncertainty_db = np.std(samples) / np.sqrt(len(samples))
            
            results.append({
                "positions": len(samples),
                "uncertainty_db": float(uncertainty_db),
                "meets_target": uncertainty_db <= target_uncertainty_db
            })
        
        # Find minimum positions meeting target
        meeting_target = [r for r in results if r["meets_target"]]
        recommended = min(meeting_target, key=lambda x: x["positions"]) if meeting_target else results[-1]
        
        return {
            "frequency_hz": frequency_hz,
            "target_uncertainty_db": target_uncertainty_db,
            "analysis": results,
            "recommended_positions": recommended["positions"],
            "achieved_uncertainty_db": recommended["uncertainty_db"]
        }


class ChamberCalibration:
    """
    Reverberation chamber calibration system.
    
    Performs chamber characterization and generates
    calibration factors.
    """
    
    def __init__(self, controller: ReverberationChamberController):
        self.controller = controller
        self._calibration_data: Dict[float, ChamberCalibrationData] = {}
    
    def calibrate_chamber_factor(
        self,
        frequencies_hz: List[float],
        reference_power_dbm: float = 0.0
    ) -> Dict[float, float]:
        """
        Calibrate chamber insertion factor.
        
        Args:
            frequencies_hz: Calibration frequencies
            reference_power_dbm: Reference transmit power
            
        Returns:
            Dictionary of frequency -> chamber factor
        """
        chamber_factors = {}
        
        for freq in frequencies_hz:
            self.controller.set_frequency(freq)
            
            # Measure at all stirrer positions
            measurements = self.controller.measure_at_stirrer_positions()
            stats = self.controller.get_statistics(measurements)
            
            # Chamber factor = Reference - Measured mean
            chamber_factor = reference_power_dbm - stats["mean_dbm"]
            chamber_factors[freq] = chamber_factor
            
            # Store in calibration data
            if freq not in self._calibration_data:
                self._calibration_data[freq] = ChamberCalibrationData(
                    frequency_hz=freq,
                    chamber_factor_db=chamber_factor,
                    loading_factor_db=0.0,
                    stirrer_efficiency=0.0,
                    q_factor=0.0,
                    coherence_bandwidth_hz=0.0,
                    decay_time_s=0.0
                )
            else:
                self._calibration_data[freq].chamber_factor_db = chamber_factor
        
        return chamber_factors
    
    def measure_loading_factor(
        self,
        frequency_hz: float,
        reference_absorber_area_m2: float = 0.1
    ) -> float:
        """
        Measure chamber loading factor.
        
        Args:
            frequency_hz: Test frequency
            reference_absorber_area_m2: Area of reference absorber
            
        Returns:
            Loading factor in dB
        """
        self.controller.set_frequency(frequency_hz)
        
        # Measure unloaded
        self.controller.set_loading(LoadingCondition.UNLOADED)
        unloaded_measurements = self.controller.measure_at_stirrer_positions()
        unloaded_stats = self.controller.get_statistics(unloaded_measurements)
        
        # Measure with reference load
        self.controller.set_loading(LoadingCondition.REFERENCE)
        loaded_measurements = self.controller.measure_at_stirrer_positions()
        loaded_stats = self.controller.get_statistics(loaded_measurements)
        
        # Loading factor
        loading_factor = unloaded_stats["mean_dbm"] - loaded_stats["mean_dbm"]
        
        # Normalize by absorber area
        loading_per_m2 = loading_factor / reference_absorber_area_m2
        
        if frequency_hz in self._calibration_data:
            self._calibration_data[frequency_hz].loading_factor_db = loading_factor
        
        return loading_factor
    
    def measure_q_factor(
        self,
        frequency_hz: float
    ) -> float:
        """
        Measure chamber Q-factor.
        
        Args:
            frequency_hz: Test frequency
            
        Returns:
            Chamber Q-factor
        """
        self.controller.set_frequency(frequency_hz)
        
        # Q-factor from power balance method
        # Q = 16 * pi^2 * V * <|E|^2> / (lambda^3 * P_in)
        
        # Estimate Q from measurement variance across stirrer positions
        measurements = self.controller.measure_at_stirrer_positions()
        values = np.array([m.value for m in measurements])
        
        # Convert to linear power
        linear_power = 10 ** (values / 10)
        mean_power = np.mean(linear_power)
        var_power = np.var(linear_power)
        
        # Q estimate (simplified)
        wavelength = 3e8 / frequency_hz
        chamber_volume = 50.0  # m^3, typical chamber
        
        # Rayleigh distribution parameter
        if var_power > 0:
            k_ratio = mean_power / np.sqrt(var_power)
        else:
            k_ratio = 1.0
        
        q_factor = 16 * np.pi**2 * chamber_volume * k_ratio / (wavelength**3)
        
        if frequency_hz in self._calibration_data:
            self._calibration_data[frequency_hz].q_factor = q_factor
        
        return q_factor
    
    def measure_coherence_bandwidth(
        self,
        center_frequency_hz: float,
        span_hz: float = 100e6
    ) -> float:
        """
        Measure chamber coherence bandwidth.
        
        Args:
            center_frequency_hz: Center frequency
            span_hz: Frequency span for measurement
            
        Returns:
            Coherence bandwidth in Hz
        """
        # Measure transfer function across frequency span
        num_points = 201
        frequencies = np.linspace(
            center_frequency_hz - span_hz/2,
            center_frequency_hz + span_hz/2,
            num_points
        )
        
        # Collect one stirrer position
        transfer_function = []
        
        for freq in frequencies:
            self.controller.set_frequency(freq)
            measurement = self.controller.chamber.measure(MeasurementType.POWER)
            # Include phase (simulated)
            phase = np.random.uniform(-180, 180)
            amplitude = 10 ** (measurement.value / 20)
            transfer_function.append(amplitude * np.exp(1j * np.deg2rad(phase)))
        
        transfer_function = np.array(transfer_function)
        
        # Calculate frequency correlation
        autocorr = np.correlate(
            transfer_function - np.mean(transfer_function),
            transfer_function - np.mean(transfer_function),
            mode='full'
        )
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = np.abs(autocorr) / np.abs(autocorr[0])
        
        # Find coherence bandwidth (correlation drops to 1/e)
        threshold = 1 / np.e
        below_threshold = np.where(autocorr < threshold)[0]
        
        if len(below_threshold) > 0:
            coherence_samples = below_threshold[0]
            freq_step = span_hz / (num_points - 1)
            coherence_bandwidth = coherence_samples * freq_step
        else:
            coherence_bandwidth = span_hz
        
        if center_frequency_hz in self._calibration_data:
            self._calibration_data[center_frequency_hz].coherence_bandwidth_hz = coherence_bandwidth
        
        return coherence_bandwidth
    
    def get_calibration_data(self) -> Dict[float, ChamberCalibrationData]:
        """Get all calibration data."""
        return self._calibration_data.copy()
    
    def export_calibration(self) -> Dict[str, Any]:
        """Export calibration data for storage."""
        return {
            "chamber_id": self.controller.specification.chamber_id,
            "calibration_date": datetime.utcnow().isoformat(),
            "frequencies": {
                str(freq): {
                    "chamber_factor_db": data.chamber_factor_db,
                    "loading_factor_db": data.loading_factor_db,
                    "q_factor": data.q_factor,
                    "coherence_bandwidth_hz": data.coherence_bandwidth_hz,
                    "stirrer_efficiency": data.stirrer_efficiency
                }
                for freq, data in self._calibration_data.items()
            }
        }


class OTAMeasurement:
    """
    Over-the-Air (OTA) performance measurement.
    
    Measures TRP, TIS, and other OTA metrics using
    reverberation chamber method.
    """
    
    def __init__(
        self,
        controller: ReverberationChamberController,
        calibration: ChamberCalibration
    ):
        self.controller = controller
        self.calibration = calibration
        self._results: List[OTAMeasurementResult] = []
    
    def measure_trp(
        self,
        frequency_hz: float,
        dut_max_power_dbm: float = 23.0
    ) -> OTAMeasurementResult:
        """
        Measure Total Radiated Power (TRP).
        
        Args:
            frequency_hz: Test frequency
            dut_max_power_dbm: DUT maximum transmit power
            
        Returns:
            TRP measurement result
        """
        start_time = time.time()
        
        self.controller.set_frequency(frequency_hz)
        self.controller.set_loading(LoadingCondition.DUT_LOADED)
        
        # Measure at all stirrer positions
        measurements = self.controller.measure_at_stirrer_positions()
        stats = self.controller.get_statistics(measurements)
        
        # Get calibration factors
        cal_data = self.calibration.get_calibration_data().get(frequency_hz)
        
        if cal_data:
            chamber_factor = cal_data.chamber_factor_db
            loading_factor = cal_data.loading_factor_db
        else:
            chamber_factor = 0.0
            loading_factor = 0.0
        
        # Calculate TRP
        # TRP = Measured + Chamber Factor + Loading Factor
        trp = stats["mean_dbm"] + chamber_factor + loading_factor
        
        # Calculate uncertainty
        # Based on number of independent samples and std deviation
        n_samples = len(measurements)
        uncertainty = stats["std_dev_db"] / np.sqrt(n_samples) * 2  # 95% confidence
        
        # EIRP (assuming isotropic, so EIRP = TRP)
        eirp = trp
        
        result = OTAMeasurementResult(
            frequency_hz=frequency_hz,
            trp_dbm=trp,
            tis_dbm=float('nan'),  # Not measured
            eirp_dbm=eirp,
            uncertainty_db=uncertainty,
            measurement_time_s=time.time() - start_time,
            stirrer_positions=len(measurements)
        )
        
        self._results.append(result)
        return result
    
    def measure_tis(
        self,
        frequency_hz: float,
        sensitivity_threshold_dbm: float = -100.0
    ) -> OTAMeasurementResult:
        """
        Measure Total Isotropic Sensitivity (TIS).
        
        Args:
            frequency_hz: Test frequency
            sensitivity_threshold_dbm: DUT sensitivity threshold
            
        Returns:
            TIS measurement result
        """
        start_time = time.time()
        
        self.controller.set_frequency(frequency_hz)
        self.controller.set_loading(LoadingCondition.DUT_LOADED)
        
        # Measure received power distribution
        measurements = self.controller.measure_at_stirrer_positions()
        stats = self.controller.get_statistics(measurements)
        
        # Get calibration factors
        cal_data = self.calibration.get_calibration_data().get(frequency_hz)
        
        if cal_data:
            chamber_factor = cal_data.chamber_factor_db
        else:
            chamber_factor = 0.0
        
        # Calculate TIS
        # TIS = Sensitivity + Chamber Factor - Isotropic correction
        # Simplified calculation
        tis = sensitivity_threshold_dbm + chamber_factor
        
        uncertainty = stats["std_dev_db"] / np.sqrt(len(measurements)) * 2
        
        result = OTAMeasurementResult(
            frequency_hz=frequency_hz,
            trp_dbm=float('nan'),  # Not measured
            tis_dbm=tis,
            eirp_dbm=float('nan'),
            uncertainty_db=uncertainty,
            measurement_time_s=time.time() - start_time,
            stirrer_positions=len(measurements)
        )
        
        self._results.append(result)
        return result
    
    def get_results(self) -> List[OTAMeasurementResult]:
        """Get all OTA measurement results."""
        return self._results.copy()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate OTA measurement report."""
        trp_results = [r for r in self._results if not np.isnan(r.trp_dbm)]
        tis_results = [r for r in self._results if not np.isnan(r.tis_dbm)]
        
        return {
            "report_type": "ota_measurement",
            "generated_at": datetime.utcnow().isoformat(),
            "chamber_id": self.controller.specification.chamber_id,
            "trp_measurements": [
                {
                    "frequency_hz": r.frequency_hz,
                    "trp_dbm": r.trp_dbm,
                    "eirp_dbm": r.eirp_dbm,
                    "uncertainty_db": r.uncertainty_db,
                    "measurement_time_s": r.measurement_time_s
                }
                for r in trp_results
            ],
            "tis_measurements": [
                {
                    "frequency_hz": r.frequency_hz,
                    "tis_dbm": r.tis_dbm,
                    "uncertainty_db": r.uncertainty_db,
                    "measurement_time_s": r.measurement_time_s
                }
                for r in tis_results
            ]
        }


# Export public API
__all__ = [
    # Enums
    "StirrerType",
    "LoadingCondition",
    
    # Data classes
    "StirrerConfig",
    "FieldUniformityData",
    "ChamberCalibrationData",
    "OTAMeasurementResult",
    
    # Controllers
    "ReverberationChamberController",
    
    # Test systems
    "FieldUniformityTest",
    "StirrerEfficiencyTest",
    "ChamberCalibration",
    "OTAMeasurement",
]
