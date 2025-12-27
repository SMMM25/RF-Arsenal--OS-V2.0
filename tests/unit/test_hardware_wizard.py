#!/usr/bin/env python3
"""
Unit Tests for Hardware Setup Wizard
Tests SDR detection, calibration, and setup functionality
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from install.hardware_wizard import (
    HardwareSetupWizard,
    HardwareDetector,
    HardwareCalibrator,
    SDRType,
    SDRCapabilities,
    DetectedSDR,
    CalibrationStatus,
    DriverStatus,
    AntennaRecommendation,
    SDR_CAPABILITIES,
    ANTENNA_GUIDE,
    get_hardware_wizard
)


class TestSDRCapabilities(unittest.TestCase):
    """Test SDR capability definitions"""
    
    def test_bladerf_xa9_capabilities(self):
        """Test BladeRF xA9 capabilities are defined correctly"""
        cap = SDR_CAPABILITIES[SDRType.BLADERF_XA9]
        
        self.assertEqual(cap.frequency_min_hz, 47_000_000)
        self.assertEqual(cap.frequency_max_hz, 6_000_000_000)
        self.assertTrue(cap.tx_capable)
        self.assertTrue(cap.rx_capable)
        self.assertTrue(cap.full_duplex)
        self.assertEqual(cap.num_tx_channels, 2)
        self.assertEqual(cap.num_rx_channels, 2)
    
    def test_hackrf_capabilities(self):
        """Test HackRF capabilities are defined correctly"""
        cap = SDR_CAPABILITIES[SDRType.HACKRF]
        
        self.assertTrue(cap.tx_capable)
        self.assertTrue(cap.rx_capable)
        self.assertFalse(cap.full_duplex)  # Half-duplex
        self.assertEqual(cap.num_tx_channels, 1)
    
    def test_rtlsdr_rx_only(self):
        """Test RTL-SDR is receive only"""
        cap = SDR_CAPABILITIES[SDRType.RTL_SDR]
        
        self.assertFalse(cap.tx_capable)
        self.assertTrue(cap.rx_capable)
        self.assertEqual(cap.num_tx_channels, 0)
    
    def test_all_sdrs_have_capabilities(self):
        """Test that major SDR types have capability definitions"""
        expected_types = [
            SDRType.BLADERF_XA9,
            SDRType.BLADERF_XA4,
            SDRType.HACKRF,
            SDRType.LIMESDR_MINI,
            SDRType.LIMESDR_USB,
            SDRType.RTL_SDR,
            SDRType.AIRSPY,
        ]
        for sdr_type in expected_types:
            self.assertIn(sdr_type, SDR_CAPABILITIES, f"Missing capabilities for {sdr_type}")


class TestAntennaGuide(unittest.TestCase):
    """Test antenna recommendation guide"""
    
    def test_antenna_guide_not_empty(self):
        """Test that antenna guide has entries"""
        self.assertGreater(len(ANTENNA_GUIDE), 0)
    
    def test_antenna_recommendation_structure(self):
        """Test antenna recommendation structure"""
        for antenna in ANTENNA_GUIDE:
            self.assertIsInstance(antenna, AntennaRecommendation)
            self.assertTrue(antenna.name)
            self.assertGreaterEqual(antenna.freq_max_mhz, antenna.freq_min_mhz, 
                                    f"Antenna {antenna.name}: freq_max ({antenna.freq_max_mhz}) must be >= freq_min ({antenna.freq_min_mhz})")
    
    def test_433mhz_antenna_exists(self):
        """Test that 433 MHz antenna recommendation exists"""
        found = False
        for antenna in ANTENNA_GUIDE:
            if antenna.freq_min_mhz <= 433 <= antenna.freq_max_mhz:
                found = True
                break
        self.assertTrue(found, "No antenna covers 433 MHz")
    
    def test_wifi_antenna_exists(self):
        """Test that WiFi antenna recommendation exists"""
        found = False
        for antenna in ANTENNA_GUIDE:
            if antenna.freq_min_mhz <= 2400 <= antenna.freq_max_mhz:
                found = True
                break
        self.assertTrue(found, "No antenna covers 2.4 GHz WiFi")


class TestDetectedSDR(unittest.TestCase):
    """Test DetectedSDR dataclass"""
    
    def test_create_detected_sdr(self):
        """Test creating DetectedSDR instance"""
        detected = DetectedSDR(
            sdr_type=SDRType.HACKRF,
            serial_number="1234567890",
            firmware_version="2021.03.1"
        )
        
        self.assertEqual(detected.sdr_type, SDRType.HACKRF)
        self.assertEqual(detected.serial_number, "1234567890")
        self.assertEqual(detected.driver_status, DriverStatus.NOT_INSTALLED)
    
    def test_detected_sdr_to_dict(self):
        """Test converting DetectedSDR to dictionary"""
        detected = DetectedSDR(
            sdr_type=SDRType.BLADERF_XA9,
            serial_number="ABC123"
        )
        
        data = detected.to_dict()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data['type'], "BladeRF 2.0 micro xA9")
        self.assertEqual(data['serial'], "ABC123")


class TestHardwareDetector(unittest.TestCase):
    """Test HardwareDetector class"""
    
    def setUp(self):
        self.detector = HardwareDetector()
    
    def test_usb_ids_defined(self):
        """Test that USB IDs are defined for detection"""
        self.assertGreater(len(self.detector.USB_IDS), 0)
    
    def test_bladerf_usb_id(self):
        """Test BladeRF USB ID is correct"""
        # BladeRF 2.0 vendor:product
        self.assertIn((0x2cf0, 0x5246), self.detector.USB_IDS)
    
    def test_hackrf_usb_id(self):
        """Test HackRF USB ID is correct"""
        self.assertIn((0x1d50, 0x6089), self.detector.USB_IDS)
    
    def test_detect_all_returns_list(self):
        """Test that detect_all returns a list"""
        # This may find real devices or return empty
        result = self.detector.detect_all()
        self.assertIsInstance(result, list)


class TestHardwareCalibrator(unittest.TestCase):
    """Test HardwareCalibrator class"""
    
    def setUp(self):
        self.calibrator = HardwareCalibrator()
    
    def test_calibrate_dc_offset(self):
        """Test DC offset calibration returns result"""
        result = self.calibrator.calibrate_dc_offset()
        
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
    
    def test_calibrate_iq_balance(self):
        """Test IQ balance calibration returns result"""
        result = self.calibrator.calibrate_iq_balance()
        
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
    
    def test_full_calibration(self):
        """Test full calibration routine"""
        result = self.calibrator.run_full_calibration()
        
        self.assertIsInstance(result, dict)
        self.assertIn('dc_offset', result)
        self.assertIn('iq_balance', result)
        self.assertIn('gain', result)
        self.assertIn('overall_status', result)


class TestHardwareSetupWizard(unittest.TestCase):
    """Test HardwareSetupWizard class"""
    
    def setUp(self):
        self.wizard = HardwareSetupWizard()
    
    def test_wizard_initialization(self):
        """Test wizard initializes correctly"""
        self.assertIsNotNone(self.wizard.detector)
        self.assertIsNotNone(self.wizard.calibrator)
        self.assertEqual(self.wizard.wizard_state, 'start')
        self.assertFalse(self.wizard.setup_complete)
    
    def test_get_antenna_recommendations_all(self):
        """Test getting all antenna recommendations"""
        recommendations = self.wizard.get_antenna_recommendations()
        
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), len(ANTENNA_GUIDE))
    
    def test_get_antenna_recommendations_filtered(self):
        """Test getting filtered antenna recommendations"""
        recommendations = self.wizard.get_antenna_recommendations(frequency_mhz=433)
        
        self.assertIsInstance(recommendations, list)
        # All returned antennas should cover 433 MHz
        for antenna in recommendations:
            self.assertLessEqual(antenna.freq_min_mhz, 433)
            self.assertGreaterEqual(antenna.freq_max_mhz, 433)
    
    def test_check_driver_status(self):
        """Test driver status check"""
        status = self.wizard.check_driver_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('bladerf', status)
        self.assertIn('hackrf', status)
        self.assertIn('limesdr', status)
        self.assertIn('rtlsdr', status)
    
    def test_get_troubleshooting_tips(self):
        """Test getting troubleshooting tips"""
        tips = self.wizard.get_troubleshooting_tips()
        
        self.assertIsInstance(tips, list)
        self.assertGreater(len(tips), 0)
    
    def test_get_troubleshooting_tips_for_sdr(self):
        """Test getting SDR-specific troubleshooting tips"""
        tips = self.wizard.get_troubleshooting_tips(SDRType.BLADERF_XA9)
        
        self.assertIsInstance(tips, list)
        # Should have more tips for specific SDR
        generic_tips = self.wizard.get_troubleshooting_tips()
        self.assertGreaterEqual(len(tips), len(generic_tips))
    
    def test_get_wizard_state(self):
        """Test getting wizard state"""
        state = self.wizard.get_wizard_state()
        
        self.assertIsInstance(state, dict)
        self.assertIn('state', state)
        self.assertIn('detected_devices', state)
        self.assertIn('setup_complete', state)
    
    def test_get_status_display(self):
        """Test getting status display"""
        display = self.wizard.get_status_display()
        
        self.assertIsInstance(display, str)
        self.assertIn("HARDWARE SETUP WIZARD", display)
    
    def test_select_device_invalid_index(self):
        """Test selecting device with invalid index"""
        result = self.wizard.select_device(999)
        self.assertFalse(result)
    
    def test_run_detection(self):
        """Test running hardware detection"""
        devices = self.wizard.run_detection()
        
        self.assertIsInstance(devices, list)
        # Should be same as detected_devices
        self.assertEqual(devices, self.wizard.detected_devices)


class TestGlobalInstance(unittest.TestCase):
    """Test global singleton pattern"""
    
    def test_singleton(self):
        """Test that get_hardware_wizard returns same instance"""
        wizard1 = get_hardware_wizard()
        wizard2 = get_hardware_wizard()
        self.assertIs(wizard1, wizard2)


class TestSDRTypeEnum(unittest.TestCase):
    """Test SDRType enum"""
    
    def test_all_sdr_types_exist(self):
        """Test that expected SDR types exist"""
        expected = [
            'BLADERF_XA9', 'BLADERF_XA4', 'HACKRF',
            'LIMESDR_MINI', 'LIMESDR_USB', 'RTL_SDR',
            'AIRSPY', 'UNKNOWN'
        ]
        for sdr in expected:
            self.assertTrue(hasattr(SDRType, sdr))


if __name__ == '__main__':
    unittest.main(verbosity=2)
