#!/usr/bin/env python3
"""
RF Arsenal OS - Tests for New Attack Modules
Tests YateBTS, NFC/RFID, ADS-B, TEMPEST, and Power Analysis modules
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock


class TestYateBTS:
    """Test YateBTS GSM/LTE BTS module"""
    
    def test_import(self):
        """Test module imports correctly"""
        from modules.cellular.yatebts import (
            YateBTSController,
            BTSConfig,
            BTSMode,
            CellularBand,
            CapturedDevice
        )
        assert YateBTSController is not None
        
    def test_config_creation(self):
        """Test BTS configuration"""
        from modules.cellular.yatebts import BTSConfig, BTSMode, CellularBand
        
        config = BTSConfig(
            mcc="001",
            mnc="01",
            shortname="Test-BTS",
            mode=BTSMode.IMSI_CATCHER
        )
        assert config.mcc == "001"
        assert config.shortname == "Test-BTS"
        
    def test_controller_creation(self):
        """Test controller instantiation"""
        from modules.cellular.yatebts import YateBTSController
        
        controller = YateBTSController()
        assert controller is not None
        assert controller.running == False
        
    def test_captured_device_dataclass(self):
        """Test CapturedDevice dataclass"""
        from modules.cellular.yatebts import CapturedDevice
        
        device = CapturedDevice(
            imsi="001010123456789",
            imei="123456789012345"
        )
        data = device.to_dict()
        assert data['imsi'] == "001010123456789"
        assert data['imei'] == "123456789012345"
        
    def test_status(self):
        """Test status retrieval"""
        from modules.cellular.yatebts import YateBTSController
        
        controller = YateBTSController()
        status = controller.get_status()
        assert 'running' in status
        assert 'mode' in status
        assert 'statistics' in status
        
    def test_config_generation(self):
        """Test YateBTS config file generation"""
        from modules.cellular.yatebts import YateBTSController
        
        controller = YateBTSController()
        config = controller.generate_config()
        assert 'Identity.MCC' in config
        assert 'Identity.MNC' in config


class TestNFCProxmark3:
    """Test NFC/RFID Proxmark3 module"""
    
    def test_import(self):
        """Test module imports correctly"""
        from modules.nfc import (
            Proxmark3Controller,
            RFIDCard,
            AttackResult,
            CardType,
            AttackType
        )
        assert Proxmark3Controller is not None
        
    def test_card_types(self):
        """Test card type enumeration"""
        from modules.nfc import CardType
        
        assert CardType.MIFARE_CLASSIC_1K.value == "mf_classic_1k"
        assert CardType.EM4100.value == "em4100"
        assert CardType.HID_PROX.value == "hid_prox"
        
    def test_attack_types(self):
        """Test attack type enumeration"""
        from modules.nfc import AttackType
        
        assert AttackType.DARKSIDE.value == "darkside"
        assert AttackType.NESTED.value == "nested"
        assert AttackType.HARDNESTED.value == "hardnested"
        
    def test_controller_creation(self):
        """Test controller instantiation"""
        from modules.nfc import Proxmark3Controller
        
        controller = Proxmark3Controller()
        assert controller is not None
        assert controller.connected == False
        
    def test_rfid_card_dataclass(self):
        """Test RFIDCard dataclass"""
        from modules.nfc import RFIDCard, CardType
        
        card = RFIDCard(
            uid="DEADBEEF",
            card_type=CardType.MIFARE_CLASSIC_1K,
            sak="08"
        )
        data = card.to_dict()
        assert data['uid'] == "DEADBEEF"
        assert data['card_type'] == "mf_classic_1k"
        
    def test_status(self):
        """Test status retrieval"""
        from modules.nfc import Proxmark3Controller
        
        controller = Proxmark3Controller()
        status = controller.get_status()
        assert 'connected' in status
        assert 'cards_found' in status
        
    def test_default_keys(self):
        """Test default key list exists"""
        from modules.nfc import Proxmark3Controller
        
        controller = Proxmark3Controller()
        assert len(controller.DEFAULT_KEYS) > 5
        assert "FFFFFFFFFFFF" in controller.DEFAULT_KEYS


class TestADSB:
    """Test ADS-B aircraft tracking module"""
    
    def test_import(self):
        """Test module imports correctly"""
        from modules.adsb import (
            ADSBController,
            Aircraft,
            ADSBMessage,
            ADSBMessageType
        )
        assert ADSBController is not None
        
    def test_message_types(self):
        """Test message type enumeration"""
        from modules.adsb import ADSBMessageType
        
        assert ADSBMessageType.IDENTIFICATION.value == 1
        assert ADSBMessageType.AIRBORNE_POSITION.value == 3
        assert ADSBMessageType.AIRBORNE_VELOCITY.value == 4
        
    def test_aircraft_dataclass(self):
        """Test Aircraft dataclass"""
        from modules.adsb import Aircraft
        
        aircraft = Aircraft(
            icao="ABC123",
            callsign="TEST123",
            altitude=35000,
            ground_speed=450.0
        )
        data = aircraft.to_dict()
        assert data['icao'] == "ABC123"
        assert data['callsign'] == "TEST123"
        assert data['altitude'] == 35000
        
    def test_controller_creation(self):
        """Test controller instantiation"""
        from modules.adsb import ADSBController
        
        controller = ADSBController()
        assert controller is not None
        assert controller.running == False
        
    def test_status(self):
        """Test status retrieval"""
        from modules.adsb import ADSBController
        
        controller = ADSBController()
        status = controller.get_status()
        assert 'running' in status
        assert 'aircraft_tracked' in status
        assert status['frequency'] == 1090_000_000
        
    def test_crc_calculation(self):
        """Test CRC-24 calculation"""
        from modules.adsb import ADSBController
        
        controller = ADSBController()
        # Test with known data
        crc = controller._calculate_crc(b'\x8d\x4b\x17\x57\x58\xc3\x82\xd6\x90\xc8\xac')
        assert isinstance(crc, int)


class TestTEMPEST:
    """Test TEMPEST/Van Eck EM surveillance module"""
    
    def test_import(self):
        """Test module imports correctly"""
        from modules.tempest import (
            TEMPESTController,
            TEMPESTMode,
            DisplayType,
            EMSource,
            ReconstructedFrame,
            KeystrokeCapture
        )
        assert TEMPESTController is not None
        
    def test_display_types(self):
        """Test display type enumeration"""
        from modules.tempest import DisplayType
        
        assert DisplayType.VGA_1024x768.value['width'] == 1024
        assert DisplayType.VGA_1024x768.value['height'] == 768
        
    def test_tempest_modes(self):
        """Test TEMPEST mode enumeration"""
        from modules.tempest import TEMPESTMode
        
        assert TEMPESTMode.VIDEO_RECONSTRUCT.value == "video"
        assert TEMPESTMode.KEYBOARD_CAPTURE.value == "keyboard"
        
    def test_controller_creation(self):
        """Test controller instantiation"""
        from modules.tempest import TEMPESTController
        
        controller = TEMPESTController()
        assert controller is not None
        assert controller.running == False
        
    def test_em_source_dataclass(self):
        """Test EMSource dataclass"""
        from modules.tempest import EMSource
        
        source = EMSource(
            frequency=65e6,
            power=-45.0,
            bandwidth=100e3,
            source_type="VGA pixel clock"
        )
        data = source.to_dict()
        assert data['frequency'] == 65e6
        assert data['source_type'] == "VGA pixel clock"
        
    def test_status(self):
        """Test status retrieval"""
        from modules.tempest import TEMPESTController
        
        controller = TEMPESTController()
        status = controller.get_status()
        assert 'running' in status
        assert 'mode' in status
        
    def test_video_harmonics(self):
        """Test video harmonic frequencies"""
        from modules.tempest import TEMPESTController
        
        controller = TEMPESTController()
        assert 'vga' in controller.VIDEO_HARMONICS
        assert 65e6 in controller.VIDEO_HARMONICS['vga']


class TestPowerAnalysis:
    """Test Power Analysis side-channel module"""
    
    def test_import(self):
        """Test module imports correctly"""
        from modules.power_analysis import (
            PowerAnalysisController,
            PowerTrace,
            KeyHypothesis,
            AttackResult,
            AttackMode,
            TargetAlgorithm
        )
        assert PowerAnalysisController is not None
        
    def test_attack_modes(self):
        """Test attack mode enumeration"""
        from modules.power_analysis import AttackMode
        
        assert AttackMode.SPA.value == "spa"
        assert AttackMode.DPA.value == "dpa"
        assert AttackMode.CPA.value == "cpa"
        
    def test_target_algorithms(self):
        """Test target algorithm enumeration"""
        from modules.power_analysis import TargetAlgorithm
        
        assert TargetAlgorithm.AES_128.value == "aes128"
        assert TargetAlgorithm.RSA.value == "rsa"
        
    def test_controller_creation(self):
        """Test controller instantiation"""
        from modules.power_analysis import PowerAnalysisController
        
        controller = PowerAnalysisController()
        assert controller is not None
        assert len(controller.traces) == 0
        
    def test_power_trace_dataclass(self):
        """Test PowerTrace dataclass"""
        from modules.power_analysis import PowerTrace
        
        samples = np.random.randn(1000).astype(np.float32)
        trace = PowerTrace(
            samples=samples,
            plaintext=b'\x00' * 16
        )
        assert len(trace) == 1000
        
    def test_sbox(self):
        """Test AES S-box is correct"""
        from modules.power_analysis import PowerAnalysisController
        
        controller = PowerAnalysisController()
        # Known S-box values
        assert controller.AES_SBOX[0] == 0x63
        assert controller.AES_SBOX[1] == 0x7c
        assert controller.AES_SBOX[255] == 0x16
        
    def test_status(self):
        """Test status retrieval"""
        from modules.power_analysis import PowerAnalysisController
        
        controller = PowerAnalysisController()
        status = controller.get_status()
        assert 'traces_collected' in status
        assert 'attacks_performed' in status
        
    def test_trace_simulation(self):
        """Test trace simulation"""
        from modules.power_analysis import PowerAnalysisController
        
        controller = PowerAnalysisController()
        trace = controller._simulate_trace(b'\x00' * 16, 10000)
        assert len(trace) == 10000
        assert trace.dtype == np.float32


class TestAICommandCenterIntegration:
    """Test AI Command Center integration with new modules"""
    
    def test_new_categories_exist(self):
        """Test new CommandCategories exist"""
        from core.ai_command_center import CommandCategory
        
        assert hasattr(CommandCategory, 'YATEBTS')
        assert hasattr(CommandCategory, 'NFC')
        assert hasattr(CommandCategory, 'ADSB')
        assert hasattr(CommandCategory, 'TEMPEST')
        assert hasattr(CommandCategory, 'POWERANALYSIS')
        
    def test_help_topics_exist(self):
        """Test help topics for new modules"""
        from core.ai_command_center import AICommandCenter
        
        ai = AICommandCenter()
        assert 'yatebts' in ai.HELP_TOPICS
        assert 'nfc' in ai.HELP_TOPICS
        assert 'adsb' in ai.HELP_TOPICS
        assert 'tempest' in ai.HELP_TOPICS
        assert 'poweranalysis' in ai.HELP_TOPICS
        
    def test_yatebts_command_parsing(self):
        """Test YateBTS command parsing"""
        from core.ai_command_center import AICommandCenter, CommandCategory
        
        ai = AICommandCenter()
        
        # Test YateBTS specific command - use 'yatebts' keyword
        context = ai._parse_command("start yatebts gsm bts")
        assert context.category == CommandCategory.YATEBTS
        
        # Test status command
        context = ai._parse_command("yatebts status")
        assert context.category == CommandCategory.YATEBTS
        
    def test_nfc_command_parsing(self):
        """Test NFC command parsing"""
        from core.ai_command_center import AICommandCenter, CommandCategory
        
        ai = AICommandCenter()
        
        # Test Proxmark3 command
        context = ai._parse_command("proxmark3 scan hf")
        assert context.category == CommandCategory.NFC
        
        # Test attack command
        context = ai._parse_command("run mifare darkside attack")
        assert context.category == CommandCategory.NFC
        assert context.intent == 'nfc_darkside'
        
    def test_adsb_command_parsing(self):
        """Test ADS-B command parsing"""
        from core.ai_command_center import AICommandCenter, CommandCategory
        
        ai = AICommandCenter()
        
        # Test with explicit ADS-B keyword
        context = ai._parse_command("adsb receiver start")
        assert context.category == CommandCategory.ADSB
        
        # Test track aircraft with ICAO
        context = ai._parse_command("track aircraft icao ABC123")
        assert context.category == CommandCategory.ADSB
        
    def test_tempest_command_parsing(self):
        """Test TEMPEST command parsing"""
        from core.ai_command_center import AICommandCenter, CommandCategory
        
        ai = AICommandCenter()
        
        # Test with explicit TEMPEST keyword
        context = ai._parse_command("tempest scan em sources")
        assert context.category == CommandCategory.TEMPEST
        
        # Test van eck keyword
        context = ai._parse_command("van eck phreaking start")
        assert context.category == CommandCategory.TEMPEST
        
    def test_power_analysis_command_parsing(self):
        """Test power analysis command parsing"""
        from core.ai_command_center import AICommandCenter, CommandCategory
        
        ai = AICommandCenter()
        
        # Test with explicit power analysis keyword
        context = ai._parse_command("power analysis capture 100 traces")
        assert context.category == CommandCategory.POWERANALYSIS
        
        # Test CPA attack
        context = ai._parse_command("correlation power analysis cpa attack")
        assert context.category == CommandCategory.POWERANALYSIS
        assert context.intent == 'power_cpa'


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
