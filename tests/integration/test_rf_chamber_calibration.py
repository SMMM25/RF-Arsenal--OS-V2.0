"""
RF Arsenal OS - RF Chamber Calibration Integration Tests

Tests for RF chamber calibration and validation functionality.
Tests the simulated chamber environment for algorithm verification.
"""

import unittest
import sys
import os
import math
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.calibration.chamber import (
    ChamberType,
    MeasurementType,
    CalibrationStatus,
    ChamberTestResult,
    ChamberSpecification,
    MeasurementPoint,
    CalibrationPoint,
    ChamberTestSequenceResult,
    UncertaintyBudget,
    ChamberException,
    CalibrationException,
    MeasurementException,
    ChamberInterface,
    SimulatedChamber,
    AntennaPatternMeasurement,
    PowerCalibration,
)


class TestChamberTypes(unittest.TestCase):
    """Test chamber type enumeration."""
    
    def test_chamber_types_exist(self):
        """Test that expected chamber types are defined."""
        expected = ['ANECHOIC', 'SEMI_ANECHOIC', 'REVERBERATION', 'GTEM', 'TEM']
        for chamber_type in expected:
            self.assertTrue(hasattr(ChamberType, chamber_type))
    
    def test_chamber_type_values(self):
        """Test chamber type values are strings."""
        for chamber_type in ChamberType:
            self.assertIsInstance(chamber_type.value, str)


class TestMeasurementTypes(unittest.TestCase):
    """Test measurement type enumeration."""
    
    def test_measurement_types_exist(self):
        """Test that expected measurement types are defined."""
        expected = ['POWER', 'FREQUENCY', 'ANTENNA_PATTERN', 'GAIN', 'VSWR']
        for meas_type in expected:
            self.assertTrue(hasattr(MeasurementType, meas_type))


class TestCalibrationStatus(unittest.TestCase):
    """Test calibration status enumeration."""
    
    def test_status_types_exist(self):
        """Test that expected status types are defined."""
        expected = ['VALID', 'WARNING', 'INVALID', 'EXPIRED', 'PENDING']
        for status in expected:
            self.assertTrue(hasattr(CalibrationStatus, status))


class TestChamberSpecification(unittest.TestCase):
    """Test chamber specification dataclass."""
    
    def test_create_specification(self):
        """Test creating chamber specification."""
        spec = ChamberSpecification(
            chamber_id="TEST_001",
            chamber_type=ChamberType.ANECHOIC,
            frequency_range_hz=(1e6, 6e9),
            quiet_zone_dimensions_m=(1.0, 1.0, 1.0),
            reflectivity_db=-40.0,
            field_uniformity_db=1.0,
            max_power_w=100.0,
            temperature_range_c=(18.0, 28.0),
            humidity_range_percent=(30.0, 70.0),
            shielding_effectiveness_db=100.0,
            last_certification=datetime.now(),
            certification_expires=datetime.now() + timedelta(days=365),
            certification_authority="Test Lab"
        )
        
        self.assertEqual(spec.chamber_id, "TEST_001")
        self.assertEqual(spec.chamber_type, ChamberType.ANECHOIC)


class TestMeasurementPoint(unittest.TestCase):
    """Test measurement point dataclass."""
    
    def test_create_measurement(self):
        """Test creating measurement point."""
        measurement = MeasurementPoint(
            timestamp=datetime.now(),
            frequency_hz=1e9,
            value=-30.0,
            unit="dBm",
            uncertainty=0.5,
            temperature_c=23.0,
            humidity_percent=45.0
        )
        
        self.assertEqual(measurement.frequency_hz, 1e9)
        self.assertEqual(measurement.value, -30.0)
        self.assertEqual(measurement.unit, "dBm")


class TestSimulatedChamber(unittest.TestCase):
    """Test simulated chamber implementation."""
    
    def setUp(self):
        """Set up test chamber."""
        self.spec = ChamberSpecification(
            chamber_id="SIM_001",
            chamber_type=ChamberType.ANECHOIC,
            frequency_range_hz=(1e6, 6e9),
            quiet_zone_dimensions_m=(1.0, 1.0, 1.0),
            reflectivity_db=-40.0,
            field_uniformity_db=1.0,
            max_power_w=100.0,
            temperature_range_c=(18.0, 28.0),
            humidity_range_percent=(30.0, 70.0),
            shielding_effectiveness_db=100.0,
            last_certification=None,
            certification_expires=None,
            certification_authority=None
        )
        self.chamber = SimulatedChamber(self.spec)
    
    def test_connect(self):
        """Test chamber connection."""
        result = self.chamber.connect()
        self.assertTrue(result)
    
    def test_disconnect(self):
        """Test chamber disconnection."""
        self.chamber.connect()
        result = self.chamber.disconnect()
        self.assertTrue(result)
    
    def test_get_status(self):
        """Test getting chamber status."""
        self.chamber.connect()
        status = self.chamber.get_status()
        
        self.assertIn("connected", status)
        self.assertIn("chamber_type", status)
        self.assertIn("position", status)
        self.assertTrue(status["connected"])
    
    def test_set_position(self):
        """Test setting positioner position."""
        self.chamber.connect()
        result = self.chamber.set_position(45.0, 90.0)
        self.assertTrue(result)
        
        position = self.chamber.get_position()
        self.assertEqual(position, (45.0, 90.0))
    
    def test_set_frequency(self):
        """Test setting measurement frequency."""
        self.chamber.connect()
        result = self.chamber.set_frequency(1e9)
        self.assertTrue(result)
    
    def test_set_frequency_out_of_range(self):
        """Test setting frequency outside valid range."""
        self.chamber.connect()
        result = self.chamber.set_frequency(100e9)  # Way outside range
        self.assertFalse(result)
    
    def test_measure_power(self):
        """Test power measurement."""
        self.chamber.connect()
        self.chamber.set_frequency(1e9)
        
        measurement = self.chamber.measure(MeasurementType.POWER)
        
        self.assertIsInstance(measurement, MeasurementPoint)
        self.assertEqual(measurement.unit, "dBm")
        self.assertIsInstance(measurement.value, float)
    
    def test_measure_antenna_pattern(self):
        """Test antenna pattern measurement."""
        self.chamber.connect()
        self.chamber.set_frequency(1e9)
        self.chamber.set_position(90.0, 0.0)
        
        measurement = self.chamber.measure(MeasurementType.ANTENNA_PATTERN)
        
        self.assertIsInstance(measurement, MeasurementPoint)
        self.assertEqual(measurement.unit, "dBi")
    
    def test_measure_without_connection(self):
        """Test that measurement fails without connection."""
        with self.assertRaises(MeasurementException):
            self.chamber.measure(MeasurementType.POWER)


class TestAntennaPatternMeasurement(unittest.TestCase):
    """Test antenna pattern measurement system."""
    
    def setUp(self):
        """Set up test environment."""
        spec = ChamberSpecification(
            chamber_id="PAT_001",
            chamber_type=ChamberType.ANECHOIC,
            frequency_range_hz=(1e6, 6e9),
            quiet_zone_dimensions_m=(1.0, 1.0, 1.0),
            reflectivity_db=-40.0,
            field_uniformity_db=1.0,
            max_power_w=100.0,
            temperature_range_c=(18.0, 28.0),
            humidity_range_percent=(30.0, 70.0),
            shielding_effectiveness_db=100.0,
            last_certification=None,
            certification_expires=None,
            certification_authority=None
        )
        self.chamber = SimulatedChamber(spec)
        self.chamber.connect()
        self.pattern_meas = AntennaPatternMeasurement(
            self.chamber,
            frequency_hz=2.4e9
        )
    
    def tearDown(self):
        """Clean up."""
        self.chamber.disconnect()
    
    def test_configure_measurement(self):
        """Test configuring pattern measurement."""
        self.pattern_meas.configure(
            theta_range=(0, 180, 10),
            phi_range=(0, 360, 10),
            averaging=2
        )
        # Just verify no exception
        self.assertIsNotNone(self.pattern_meas)
    
    def test_measure_2d_e_plane_cut(self):
        """Test 2D E-plane pattern cut measurement."""
        self.pattern_meas.configure(
            theta_range=(0, 180, 30),
            phi_range=(0, 360, 30),
            averaging=1
        )
        
        cut_data = self.pattern_meas.measure_2d_cut(cut_plane="E", fixed_angle=0.0)
        
        self.assertIsInstance(cut_data, dict)
        self.assertGreater(len(cut_data), 0)
    
    def test_measure_2d_h_plane_cut(self):
        """Test 2D H-plane pattern cut measurement."""
        self.pattern_meas.configure(
            theta_range=(0, 180, 30),
            phi_range=(0, 360, 30),
            averaging=1
        )
        
        cut_data = self.pattern_meas.measure_2d_cut(cut_plane="H", fixed_angle=90.0)
        
        self.assertIsInstance(cut_data, dict)
        self.assertGreater(len(cut_data), 0)
    
    def test_measure_3d_pattern(self):
        """Test 3D pattern measurement (small grid)."""
        self.pattern_meas.configure(
            theta_range=(0, 180, 45),  # Coarse grid for speed
            phi_range=(0, 360, 90),
            averaging=1
        )
        
        pattern_data = self.pattern_meas.measure_3d_pattern()
        
        self.assertIsInstance(pattern_data, dict)
        self.assertGreater(len(pattern_data), 0)
    
    def test_calculate_parameters(self):
        """Test antenna parameter calculation."""
        self.pattern_meas.configure(
            theta_range=(0, 180, 45),
            phi_range=(0, 360, 90),
            averaging=1
        )
        
        self.pattern_meas.measure_3d_pattern()
        params = self.pattern_meas.calculate_parameters()
        
        self.assertIn("peak_gain_dbi", params)
        self.assertIn("peak_direction", params)
        self.assertIn("frequency_hz", params)
    
    def test_export_pattern_csv(self):
        """Test exporting pattern to CSV."""
        self.pattern_meas.configure(
            theta_range=(0, 180, 90),
            phi_range=(0, 360, 180),
            averaging=1
        )
        
        self.pattern_meas.measure_3d_pattern()
        csv_output = self.pattern_meas.export_pattern(format="csv")
        
        self.assertIsInstance(csv_output, str)
        self.assertIn("theta,phi,gain_dbi", csv_output)
    
    def test_export_pattern_json(self):
        """Test exporting pattern to JSON."""
        self.pattern_meas.configure(
            theta_range=(0, 180, 90),
            phi_range=(0, 360, 180),
            averaging=1
        )
        
        self.pattern_meas.measure_3d_pattern()
        json_output = self.pattern_meas.export_pattern(format="json")
        
        self.assertIsInstance(json_output, str)
        self.assertIn("frequency_hz", json_output)
        self.assertIn("pattern", json_output)


class TestPowerCalibration(unittest.TestCase):
    """Test power calibration system."""
    
    def setUp(self):
        """Set up test environment."""
        spec = ChamberSpecification(
            chamber_id="PWR_001",
            chamber_type=ChamberType.ANECHOIC,
            frequency_range_hz=(1e6, 6e9),
            quiet_zone_dimensions_m=(1.0, 1.0, 1.0),
            reflectivity_db=-40.0,
            field_uniformity_db=1.0,
            max_power_w=100.0,
            temperature_range_c=(18.0, 28.0),
            humidity_range_percent=(30.0, 70.0),
            shielding_effectiveness_db=100.0,
            last_certification=None,
            certification_expires=None,
            certification_authority=None
        )
        self.chamber = SimulatedChamber(spec)
        self.chamber.connect()
        self.power_cal = PowerCalibration(self.chamber)
    
    def tearDown(self):
        """Clean up."""
        self.chamber.disconnect()
    
    def test_calibrate_power_single_frequency(self):
        """Test power calibration at single frequency."""
        cal_points = self.power_cal.calibrate_power(
            frequency_points_hz=[1e9],
            power_levels_dbm=[-30, -20, -10]
        )
        
        self.assertIsInstance(cal_points, list)
        self.assertEqual(len(cal_points), 3)
    
    def test_calibrate_power_multiple_frequencies(self):
        """Test power calibration across multiple frequencies."""
        cal_points = self.power_cal.calibrate_power(
            frequency_points_hz=[1e9, 2e9],
            power_levels_dbm=[-30, -20]
        )
        
        self.assertIsInstance(cal_points, list)
        self.assertEqual(len(cal_points), 4)


class TestExceptions(unittest.TestCase):
    """Test exception classes."""
    
    def test_chamber_exception(self):
        """Test ChamberException."""
        with self.assertRaises(ChamberException):
            raise ChamberException("Test error")
    
    def test_calibration_exception(self):
        """Test CalibrationException."""
        with self.assertRaises(CalibrationException):
            raise CalibrationException("Calibration failed")
        
        # CalibrationException should be a subclass of ChamberException
        with self.assertRaises(ChamberException):
            raise CalibrationException("Also a chamber exception")
    
    def test_measurement_exception(self):
        """Test MeasurementException."""
        with self.assertRaises(MeasurementException):
            raise MeasurementException("Measurement failed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
