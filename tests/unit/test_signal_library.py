#!/usr/bin/env python3
"""
Unit Tests for Signal Replay Library
Tests signal capture, storage, analysis, and replay functionality
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.replay.signal_library import (
    SignalLibrary,
    SignalMetadata,
    SignalAnalyzer,
    CaptureSettings,
    ModulationType,
    SignalCategory,
    EncodingType,
    get_signal_library
)


class TestSignalMetadata(unittest.TestCase):
    """Test SignalMetadata dataclass"""
    
    def test_create_metadata(self):
        """Test creating signal metadata"""
        metadata = SignalMetadata(
            signal_id="SIG_TEST_001",
            name="Test Signal",
            frequency=433_920_000,
            sample_rate=2_000_000,
            category=SignalCategory.KEYFOB
        )
        self.assertEqual(metadata.signal_id, "SIG_TEST_001")
        self.assertEqual(metadata.name, "Test Signal")
        self.assertEqual(metadata.frequency, 433_920_000)
        self.assertEqual(metadata.category, SignalCategory.KEYFOB)
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary"""
        metadata = SignalMetadata(
            signal_id="SIG_TEST_002",
            name="Test",
            modulation=ModulationType.OOK,
            encoding=EncodingType.PRINCETON
        )
        data = metadata.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['signal_id'], "SIG_TEST_002")
        self.assertEqual(data['modulation'], "OOK")
        self.assertEqual(data['encoding'], "princeton")
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary"""
        data = {
            'signal_id': 'SIG_TEST_003',
            'name': 'From Dict',
            'description': 'Test description',
            'category': 'keyfob',
            'frequency': 315_000_000,
            'sample_rate': 1_000_000,
            'bandwidth': 500_000,
            'modulation': 'ASK',
            'encoding': 'manchester',
            'bit_rate': 1000,
            'symbol_duration_us': 1000,
            'preamble_length': 0,
            'capture_time': '',
            'capture_location': '',
            'capture_device': '',
            'signal_strength_dbm': -70,
            'decoded_data': 'AABBCC',
            'rolling_code': False,
            'replay_safe': True,
            'raw_file': '',
            'demod_file': '',
            'file_size_bytes': 0,
            'duration_ms': 1000,
            'tags': ['test'],
            'notes': '',
            'checksum': ''
        }
        metadata = SignalMetadata.from_dict(data)
        self.assertEqual(metadata.signal_id, 'SIG_TEST_003')
        self.assertEqual(metadata.category, SignalCategory.KEYFOB)
        self.assertEqual(metadata.modulation, ModulationType.ASK)


class TestCaptureSettings(unittest.TestCase):
    """Test CaptureSettings dataclass"""
    
    def test_default_settings(self):
        """Test default capture settings"""
        settings = CaptureSettings()
        self.assertEqual(settings.frequency, 433_920_000)
        self.assertEqual(settings.sample_rate, 2_000_000)
        self.assertEqual(settings.duration_ms, 5000)
    
    def test_custom_settings(self):
        """Test custom capture settings"""
        settings = CaptureSettings(
            frequency=315_000_000,
            sample_rate=1_000_000,
            bandwidth=500_000,
            duration_ms=10000
        )
        self.assertEqual(settings.frequency, 315_000_000)
        self.assertEqual(settings.duration_ms, 10000)


class TestSignalAnalyzer(unittest.TestCase):
    """Test SignalAnalyzer class"""
    
    def setUp(self):
        self.analyzer = SignalAnalyzer()
    
    def test_common_frequencies(self):
        """Test that common frequencies are defined"""
        self.assertIn(433_920_000, self.analyzer.COMMON_FREQUENCIES)
        self.assertIn(315_000_000, self.analyzer.COMMON_FREQUENCIES)
        self.assertIn(915_000_000, self.analyzer.COMMON_FREQUENCIES)
    
    def test_analyze_returns_dict(self):
        """Test that analyze_signal returns a dictionary"""
        import numpy as np
        # Create dummy signal
        iq_data = np.random.randn(1000) + 1j * np.random.randn(1000)
        result = self.analyzer.analyze_signal(iq_data, 2_000_000, 433_920_000)
        
        self.assertIsInstance(result, dict)
        self.assertIn('frequency', result)
        self.assertIn('sample_rate', result)
        self.assertIn('modulation', result)


class TestSignalLibrary(unittest.TestCase):
    """Test SignalLibrary class"""
    
    def setUp(self):
        """Create temporary library directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.library = SignalLibrary(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_library_initialization(self):
        """Test library initializes correctly"""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue((Path(self.temp_dir) / 'signals').exists())
        self.assertTrue((Path(self.temp_dir) / 'raw').exists())
    
    def test_generate_signal_id(self):
        """Test signal ID generation"""
        id1 = self.library._generate_signal_id()
        id2 = self.library._generate_signal_id()
        
        self.assertTrue(id1.startswith('SIG_'))
        self.assertNotEqual(id1, id2)  # Should be unique
    
    def test_capture_signal_no_hardware(self):
        """Test capturing signal without hardware (simulated)"""
        settings = CaptureSettings(
            frequency=433_920_000,
            duration_ms=100
        )
        
        metadata = self.library.capture_signal(
            None,  # No hardware
            settings,
            name="Test Capture",
            category=SignalCategory.KEYFOB
        )
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.name, "Test Capture")
        self.assertEqual(metadata.category, SignalCategory.KEYFOB)
        self.assertTrue(Path(metadata.raw_file).exists())
    
    def test_list_signals(self):
        """Test listing signals"""
        # Capture a signal first
        settings = CaptureSettings(duration_ms=100)
        self.library.capture_signal(None, settings, name="Test1")
        self.library.capture_signal(None, settings, name="Test2")
        
        signals = self.library.list_signals()
        self.assertEqual(len(signals), 2)
    
    def test_get_signal(self):
        """Test getting signal by ID"""
        settings = CaptureSettings(duration_ms=100)
        metadata = self.library.capture_signal(None, settings, name="Get Test")
        
        retrieved = self.library.get_signal(metadata.signal_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Get Test")
    
    def test_delete_signal(self):
        """Test deleting signal"""
        settings = CaptureSettings(duration_ms=100)
        metadata = self.library.capture_signal(None, settings, name="Delete Test")
        
        # Verify it exists
        self.assertIn(metadata.signal_id, self.library.catalog)
        
        # Delete it
        result = self.library.delete_signal(metadata.signal_id)
        self.assertTrue(result)
        
        # Verify it's gone
        self.assertNotIn(metadata.signal_id, self.library.catalog)
    
    def test_search_by_category(self):
        """Test searching signals by category"""
        settings = CaptureSettings(duration_ms=100)
        self.library.capture_signal(None, settings, name="Keyfob1", category=SignalCategory.KEYFOB)
        self.library.capture_signal(None, settings, name="Garage1", category=SignalCategory.GARAGE_DOOR)
        self.library.capture_signal(None, settings, name="Keyfob2", category=SignalCategory.KEYFOB)
        
        keyfobs = self.library.search_signals(category=SignalCategory.KEYFOB)
        self.assertEqual(len(keyfobs), 2)
        
        garages = self.library.search_signals(category=SignalCategory.GARAGE_DOOR)
        self.assertEqual(len(garages), 1)
    
    def test_search_by_name(self):
        """Test searching signals by name"""
        settings = CaptureSettings(duration_ms=100)
        self.library.capture_signal(None, settings, name="Front Door Sensor")
        self.library.capture_signal(None, settings, name="Back Door Sensor")
        self.library.capture_signal(None, settings, name="Garage Remote")
        
        results = self.library.search_signals(name_contains="Door")
        self.assertEqual(len(results), 2)
    
    def test_get_statistics(self):
        """Test getting library statistics"""
        settings = CaptureSettings(duration_ms=100)
        self.library.capture_signal(None, settings, name="Test1", category=SignalCategory.KEYFOB)
        self.library.capture_signal(None, settings, name="Test2", category=SignalCategory.KEYFOB)
        
        stats = self.library.get_statistics()
        
        self.assertEqual(stats['total_signals'], 2)
        self.assertIn('by_category', stats)
        self.assertIn('total_size_mb', stats)
    
    def test_checksum_calculation(self):
        """Test checksum calculation"""
        data = b"test data for checksum"
        checksum = self.library._calculate_checksum(data)
        
        self.assertEqual(len(checksum), 64)  # SHA-256 hex is 64 chars
        # Same data should produce same checksum
        self.assertEqual(checksum, self.library._calculate_checksum(data))
    
    def test_catalog_persistence(self):
        """Test that catalog is saved and loaded"""
        settings = CaptureSettings(duration_ms=100)
        metadata = self.library.capture_signal(None, settings, name="Persist Test")
        signal_id = metadata.signal_id
        
        # Create new library instance pointing to same directory
        library2 = SignalLibrary(self.temp_dir)
        
        # Should have the signal
        self.assertIn(signal_id, library2.catalog)
        self.assertEqual(library2.catalog[signal_id].name, "Persist Test")


class TestSignalCategories(unittest.TestCase):
    """Test signal category enums"""
    
    def test_all_categories_exist(self):
        """Test that expected categories exist"""
        expected = [
            'KEYFOB', 'GARAGE_DOOR', 'CAR_KEY', 'WIRELESS_SENSOR',
            'DOORBELL', 'REMOTE_CONTROL', 'TPMS', 'WEATHER_STATION',
            'SMART_HOME', 'INDUSTRIAL', 'CUSTOM', 'UNKNOWN'
        ]
        for cat in expected:
            self.assertTrue(hasattr(SignalCategory, cat))


class TestModulationTypes(unittest.TestCase):
    """Test modulation type enums"""
    
    def test_all_modulation_types_exist(self):
        """Test that expected modulation types exist"""
        expected = ['OOK', 'ASK', 'FSK', 'GFSK', 'PSK', 'QPSK', 'UNKNOWN']
        for mod in expected:
            self.assertTrue(hasattr(ModulationType, mod))


class TestEncodingTypes(unittest.TestCase):
    """Test encoding type enums"""
    
    def test_all_encoding_types_exist(self):
        """Test that expected encoding types exist"""
        expected = [
            'RAW', 'MANCHESTER', 'BIPHASE', 'PWM', 'PPM', 'NRZ',
            'PRINCETON', 'HOLTEK', 'KEELOQ', 'UNKNOWN'
        ]
        for enc in expected:
            self.assertTrue(hasattr(EncodingType, enc))


if __name__ == '__main__':
    unittest.main(verbosity=2)
