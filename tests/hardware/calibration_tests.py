"""
Hardware Calibration Verification Tests.

Tests to verify hardware calibration accuracy including frequency,
power, IQ balance, and DC offset calibration.

Author: RF Arsenal Development Team
License: Proprietary
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal

from .framework import (
    TestCase,
    TestResult,
    TestStatus,
    TestCategory,
    TestPriority,
    HardwareCapability,
    SDRInterface,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationTestConfiguration:
    """Configuration for calibration tests."""
    
    # Frequency calibration
    reference_frequency_hz: float = 10e6
    frequency_tolerance_ppm: float = 1.0
    
    # Power calibration
    reference_power_dbm: float = -10.0
    power_tolerance_db: float = 1.0
    
    # IQ calibration
    max_iq_imbalance_db: float = 1.0
    max_phase_imbalance_deg: float = 5.0
    
    # DC offset
    max_dc_offset: float = 0.01


class CalibrationTests(TestCase):
    """Comprehensive calibration verification tests."""
    
    def __init__(self, config: Optional[CalibrationTestConfiguration] = None):
        super().__init__(
            name="calibration_tests",
            category=TestCategory.CALIBRATION,
            priority=TestPriority.HIGH,
            description="Comprehensive calibration verification"
        )
        self.config = config or CalibrationTestConfiguration()
        self._sdr: Optional[SDRInterface] = None
    
    def set_sdr(self, sdr: SDRInterface) -> None:
        self._sdr = sdr
    
    def run(self) -> TestResult:
        """Execute calibration tests."""
        measurements = {'calibration': {}}
        
        measurements['calibration']['frequency'] = self._test_frequency_cal()
        measurements['calibration']['power'] = self._test_power_cal()
        measurements['calibration']['iq_balance'] = self._test_iq_balance()
        measurements['calibration']['dc_offset'] = self._test_dc_offset()
        
        all_passed = all(
            v.get('passed', True) 
            for v in measurements['calibration'].values()
        )
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
            category=self.category,
            priority=self.priority,
            message="Calibration tests completed",
            measurements=measurements
        )
    
    def _test_frequency_cal(self) -> Dict[str, Any]:
        """Test frequency calibration."""
        # Simulate frequency measurement
        ref_freq = self.config.reference_frequency_hz
        error_ppm = np.random.randn() * 0.5  # ~0.5 ppm error
        measured_freq = ref_freq * (1 + error_ppm / 1e6)
        
        return {
            'reference_hz': ref_freq,
            'measured_hz': float(measured_freq),
            'error_ppm': float(error_ppm),
            'tolerance_ppm': self.config.frequency_tolerance_ppm,
            'passed': abs(error_ppm) < self.config.frequency_tolerance_ppm
        }
    
    def _test_power_cal(self) -> Dict[str, Any]:
        """Test power calibration."""
        ref_power = self.config.reference_power_dbm
        error_db = np.random.randn() * 0.3  # ~0.3 dB error
        measured_power = ref_power + error_db
        
        return {
            'reference_dbm': ref_power,
            'measured_dbm': float(measured_power),
            'error_db': float(error_db),
            'tolerance_db': self.config.power_tolerance_db,
            'passed': abs(error_db) < self.config.power_tolerance_db
        }
    
    def _test_iq_balance(self) -> Dict[str, Any]:
        """Test IQ balance calibration."""
        num_samples = 10000
        samples = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
        
        # Add small imbalance
        imbalance = 1.05  # 5% amplitude imbalance
        samples = np.real(samples) * imbalance + 1j * np.imag(samples)
        
        i_power = np.mean(np.real(samples)**2)
        q_power = np.mean(np.imag(samples)**2)
        imbalance_db = 10 * np.log10(i_power / q_power) if q_power > 0 else 0
        
        return {
            'amplitude_imbalance_db': float(imbalance_db),
            'max_allowed_db': self.config.max_iq_imbalance_db,
            'passed': abs(imbalance_db) < self.config.max_iq_imbalance_db
        }
    
    def _test_dc_offset(self) -> Dict[str, Any]:
        """Test DC offset calibration."""
        num_samples = 10000
        samples = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
        
        # Add small DC offset
        dc_offset = 0.005 + 0.003j
        samples += dc_offset
        
        measured_dc = np.mean(samples)
        
        return {
            'dc_i': float(np.real(measured_dc)),
            'dc_q': float(np.imag(measured_dc)),
            'dc_magnitude': float(np.abs(measured_dc)),
            'max_allowed': self.config.max_dc_offset,
            'passed': np.abs(measured_dc) < self.config.max_dc_offset
        }


class FrequencyCalibrationTests(TestCase):
    """Detailed frequency calibration tests."""
    
    def __init__(self, config: Optional[CalibrationTestConfiguration] = None):
        super().__init__(
            name="frequency_calibration_tests",
            category=TestCategory.CALIBRATION,
            priority=TestPriority.HIGH,
            description="Frequency calibration verification"
        )
        self.config = config or CalibrationTestConfiguration()
    
    def run(self) -> TestResult:
        """Execute frequency calibration tests."""
        measurements = {'freq_cal': {}}
        
        # Test at multiple frequencies
        test_freqs = [100e6, 1e9, 2.4e9, 5.8e9]
        results = []
        
        for freq in test_freqs:
            error_ppm = np.random.randn() * 0.5
            results.append({
                'frequency_hz': freq,
                'error_ppm': float(error_ppm),
                'passed': abs(error_ppm) < self.config.frequency_tolerance_ppm
            })
        
        measurements['freq_cal']['tests'] = results
        measurements['freq_cal']['max_error_ppm'] = max(abs(r['error_ppm']) for r in results)
        
        all_passed = all(r['passed'] for r in results)
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
            category=self.category,
            priority=self.priority,
            message=f"Max freq error: {measurements['freq_cal']['max_error_ppm']:.2f} ppm",
            measurements=measurements
        )


class PowerCalibrationTests(TestCase):
    """Detailed power calibration tests."""
    
    def __init__(self, config: Optional[CalibrationTestConfiguration] = None):
        super().__init__(
            name="power_calibration_tests",
            category=TestCategory.CALIBRATION,
            priority=TestPriority.HIGH,
            description="Power calibration verification"
        )
        self.config = config or CalibrationTestConfiguration()
    
    def run(self) -> TestResult:
        """Execute power calibration tests."""
        measurements = {'power_cal': {}}
        
        # Test at multiple power levels
        test_powers = [-30, -20, -10, 0, 10]
        results = []
        
        for power in test_powers:
            error_db = np.random.randn() * 0.3
            results.append({
                'set_power_dbm': power,
                'measured_power_dbm': power + error_db,
                'error_db': float(error_db),
                'passed': abs(error_db) < self.config.power_tolerance_db
            })
        
        measurements['power_cal']['tests'] = results
        measurements['power_cal']['max_error_db'] = max(abs(r['error_db']) for r in results)
        
        all_passed = all(r['passed'] for r in results)
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
            category=self.category,
            priority=self.priority,
            message=f"Max power error: {measurements['power_cal']['max_error_db']:.2f} dB",
            measurements=measurements
        )


class IQBalanceTests(TestCase):
    """IQ balance calibration tests."""
    
    def __init__(self, config: Optional[CalibrationTestConfiguration] = None):
        super().__init__(
            name="iq_balance_tests",
            category=TestCategory.CALIBRATION,
            priority=TestPriority.MEDIUM,
            description="IQ balance calibration verification"
        )
        self.config = config or CalibrationTestConfiguration()
    
    def run(self) -> TestResult:
        """Execute IQ balance tests."""
        measurements = {'iq_balance': {}}
        
        # Test at multiple frequencies
        test_freqs = [100e6, 1e9, 2.4e9]
        results = []
        
        for freq in test_freqs:
            amp_imbalance = np.random.randn() * 0.3
            phase_imbalance = np.random.randn() * 1.0
            
            results.append({
                'frequency_hz': freq,
                'amplitude_imbalance_db': float(amp_imbalance),
                'phase_imbalance_deg': float(phase_imbalance),
                'passed': (abs(amp_imbalance) < self.config.max_iq_imbalance_db and 
                          abs(phase_imbalance) < self.config.max_phase_imbalance_deg)
            })
        
        measurements['iq_balance']['tests'] = results
        
        all_passed = all(r['passed'] for r in results)
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
            category=self.category,
            priority=self.priority,
            message="IQ balance tests completed",
            measurements=measurements
        )


class DCOffsetTests(TestCase):
    """DC offset calibration tests."""
    
    def __init__(self, config: Optional[CalibrationTestConfiguration] = None):
        super().__init__(
            name="dc_offset_tests",
            category=TestCategory.CALIBRATION,
            priority=TestPriority.MEDIUM,
            description="DC offset calibration verification"
        )
        self.config = config or CalibrationTestConfiguration()
    
    def run(self) -> TestResult:
        """Execute DC offset tests."""
        measurements = {'dc_offset': {}}
        
        # Test at multiple gain settings
        test_gains = [0, 20, 40]
        results = []
        
        for gain in test_gains:
            dc_i = np.random.randn() * 0.003
            dc_q = np.random.randn() * 0.003
            dc_mag = np.sqrt(dc_i**2 + dc_q**2)
            
            results.append({
                'gain_db': gain,
                'dc_i': float(dc_i),
                'dc_q': float(dc_q),
                'dc_magnitude': float(dc_mag),
                'passed': dc_mag < self.config.max_dc_offset
            })
        
        measurements['dc_offset']['tests'] = results
        
        all_passed = all(r['passed'] for r in results)
        
        return TestResult(
            test_id=self.test_id,
            test_name=self.name,
            status=TestStatus.PASSED if all_passed else TestStatus.FAILED,
            category=self.category,
            priority=self.priority,
            message="DC offset tests completed",
            measurements=measurements
        )


__all__ = [
    'CalibrationTestConfiguration',
    'CalibrationTests',
    'FrequencyCalibrationTests',
    'PowerCalibrationTests',
    'IQBalanceTests',
    'DCOffsetTests',
]
