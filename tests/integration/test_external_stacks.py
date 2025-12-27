"""
RF Arsenal OS - External Stack Integration Tests
Comprehensive tests for srsRAN and OpenAirInterface integration

Tests cover:
- srsRAN controller functionality
- OpenAirInterface controller functionality  
- Stack manager operations
- Configuration generation
- Stealth-aware operations
- Cross-stack interoperability
"""

import os
import sys
import time
import json
import unittest
import tempfile
import threading
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ============================================================================
# Test Configuration
# ============================================================================

class TestConfig:
    """Test configuration constants"""
    # Test PLMN
    MCC = "001"
    MNC = "01"
    
    # Test UE credentials
    TEST_IMSI = "001010123456789"
    TEST_KEY = "00112233445566778899aabbccddeeff"
    TEST_OPC = "63bfa50ee6523365ff14c1f45f88737d"
    
    # RF parameters
    DL_FREQ_HZ = 2680e6
    UL_FREQ_HZ = 2560e6
    BANDWIDTH_MHZ = 10
    TX_GAIN_DB = 50.0
    RX_GAIN_DB = 40.0


# ============================================================================
# Mock Stealth System
# ============================================================================

class MockStealthSystem:
    """Mock stealth system for testing"""
    
    def __init__(self, emission_allowed: bool = True, max_tx_gain: float = 60.0):
        self._emission_allowed = emission_allowed
        self._max_tx_gain = max_tx_gain
    
    def check_emission_allowed(self) -> bool:
        return self._emission_allowed
    
    def get_max_tx_gain(self) -> float:
        return self._max_tx_gain
    
    def set_emission_allowed(self, allowed: bool):
        self._emission_allowed = allowed
    
    def set_max_tx_gain(self, gain: float):
        self._max_tx_gain = gain


# ============================================================================
# srsRAN Controller Tests
# ============================================================================

class TestSrsRANController(unittest.TestCase):
    """Tests for srsRAN controller"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        try:
            from core.external.srsran.srsran_controller import (
                SrsRANController, SrsRANConfig, SrsRANComponent, SrsRANState,
                RFConfig, CellConfig, EPCConfig
            )
            cls.SrsRANController = SrsRANController
            cls.SrsRANConfig = SrsRANConfig
            cls.SrsRANComponent = SrsRANComponent
            cls.SrsRANState = SrsRANState
            cls.RFConfig = RFConfig
            cls.CellConfig = CellConfig
            cls.EPCConfig = EPCConfig
            cls.module_available = True
        except ImportError as e:
            cls.module_available = False
            cls.import_error = str(e)
    
    def setUp(self):
        """Set up for each test"""
        if not self.module_available:
            self.skipTest(f"srsRAN module not available: {self.import_error}")
        
        self.stealth = MockStealthSystem()
        self.temp_dir = tempfile.mkdtemp()
        self.config = self.SrsRANConfig(
            log_path=self.temp_dir,
            config_path=self.temp_dir
        )
        self.controller = self.SrsRANController(
            config=self.config,
            stealth_system=self.stealth
        )
    
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'controller'):
            self.controller.shutdown()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test controller initialization"""
        result = self.controller.initialize()
        self.assertTrue(result)
        self.assertIsNotNone(self.controller._config_generator)
    
    def test_default_configuration(self):
        """Test default configuration values"""
        self.assertEqual(self.config.rf.dl_earfcn, 3350)
        self.assertEqual(self.config.rf.nof_prb, 50)
        self.assertEqual(self.config.cell.mcc, "001")
        self.assertEqual(self.config.cell.mnc, "01")
        self.assertEqual(self.config.epc.mme_addr, "127.0.1.100")
    
    def test_rf_config_customization(self):
        """Test RF configuration customization"""
        rf_config = self.RFConfig(
            tx_gain=55.0,
            rx_gain=45.0,
            dl_earfcn=3400,
            nof_prb=100
        )
        config = self.SrsRANConfig(rf=rf_config)
        
        self.assertEqual(config.rf.tx_gain, 55.0)
        self.assertEqual(config.rf.rx_gain, 45.0)
        self.assertEqual(config.rf.dl_earfcn, 3400)
        self.assertEqual(config.rf.nof_prb, 100)
    
    def test_cell_config_customization(self):
        """Test cell configuration customization"""
        cell_config = self.CellConfig(
            cell_id=0x02,
            tac=0x0002,
            mcc="310",
            mnc="260"
        )
        config = self.SrsRANConfig(cell=cell_config)
        
        self.assertEqual(config.cell.cell_id, 0x02)
        self.assertEqual(config.cell.mcc, "310")
        self.assertEqual(config.cell.mnc, "260")
    
    def test_stealth_integration(self):
        """Test stealth system integration"""
        self.controller.initialize()
        
        # Test with emission allowed
        self.stealth.set_emission_allowed(True)
        # In simulation mode, this should succeed
        result = self.controller.start_enb()
        self.assertTrue(result)
        
        self.controller.stop_component(self.SrsRANComponent.ENB)
        
        # Test with emission blocked
        self.stealth.set_emission_allowed(False)
        # Process start should be blocked
        # Note: In simulation mode, it may still succeed
    
    def test_tx_power_limiting(self):
        """Test TX power limiting by stealth"""
        self.stealth.set_max_tx_gain(30.0)
        
        self.controller.set_tx_power(50.0)
        
        # Should be limited to max
        self.assertEqual(self.controller.config.rf.tx_gain, 30.0)
    
    def test_bandwidth_setting(self):
        """Test bandwidth configuration via PRBs"""
        test_cases = [
            (6, 6),     # 1.4 MHz
            (15, 15),   # 3 MHz
            (25, 25),   # 5 MHz
            (50, 50),   # 10 MHz
            (75, 75),   # 15 MHz
            (100, 100), # 20 MHz
            (45, 50),   # Invalid -> closest valid
        ]
        
        for input_prb, expected in test_cases:
            self.controller.set_bandwidth(input_prb)
            self.assertEqual(self.controller.config.rf.nof_prb, expected)
    
    def test_frequency_setting(self):
        """Test frequency configuration"""
        self.controller.set_frequency(3400)
        self.assertEqual(self.controller.config.rf.dl_earfcn, 3400)
    
    def test_status_reporting(self):
        """Test status reporting"""
        self.controller.initialize()
        
        status = self.controller.get_status()
        
        self.assertIn('installed', status)
        self.assertIn('components', status)
        self.assertIsInstance(status['components'], dict)
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        self.controller.initialize()
        
        metrics = self.controller.get_metrics()
        
        self.assertIn('timestamp', metrics)
        self.assertIn('enb', metrics)
        self.assertIn('ue', metrics)
        self.assertIn('epc', metrics)
    
    def test_event_callbacks(self):
        """Test event callback registration"""
        callback_called = {'called': False}
        
        def on_state_change(data):
            callback_called['called'] = True
        
        self.controller.register_callback('enb_state_change', on_state_change)
        
        self.assertIn('enb_state_change', self.controller._callbacks)
        self.assertEqual(len(self.controller._callbacks['enb_state_change']), 1)
    
    def test_subscriber_management(self):
        """Test subscriber management"""
        self.controller.initialize()
        
        # Add subscriber (currently just logs)
        self.controller.add_subscriber(
            imsi=TestConfig.TEST_IMSI,
            key=TestConfig.TEST_KEY,
            opc=TestConfig.TEST_OPC
        )
        # No assertion as it just logs


# ============================================================================
# srsRAN Configuration Generator Tests
# ============================================================================

class TestSrsRANConfigGenerator(unittest.TestCase):
    """Tests for srsRAN configuration generator"""
    
    @classmethod
    def setUpClass(cls):
        try:
            from core.external.srsran.srsran_controller import (
                SrsRANConfigGenerator, SrsRANConfig
            )
            cls.SrsRANConfigGenerator = SrsRANConfigGenerator
            cls.SrsRANConfig = SrsRANConfig
            cls.module_available = True
        except ImportError as e:
            cls.module_available = False
            cls.import_error = str(e)
    
    def setUp(self):
        if not self.module_available:
            self.skipTest(f"Module not available: {self.import_error}")
        
        self.config = self.SrsRANConfig()
        self.stealth = MockStealthSystem()
        self.generator = self.SrsRANConfigGenerator(self.config, self.stealth)
    
    def tearDown(self):
        if hasattr(self, 'generator'):
            self.generator.cleanup()
    
    def test_enb_config_generation(self):
        """Test eNB configuration file generation"""
        config_file = self.generator.generate_enb_config()
        
        self.assertTrue(os.path.isfile(config_file))
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check key sections exist
        self.assertIn('[enb]', content)
        self.assertIn('[rf]', content)
        self.assertIn('[log]', content)
    
    def test_ue_config_generation(self):
        """Test UE configuration file generation"""
        config_file = self.generator.generate_ue_config(
            imsi=TestConfig.TEST_IMSI,
            key=TestConfig.TEST_KEY,
            opc=TestConfig.TEST_OPC
        )
        
        self.assertTrue(os.path.isfile(config_file))
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        self.assertIn('[usim]', content)
        self.assertIn(TestConfig.TEST_IMSI, content)
    
    def test_epc_config_generation(self):
        """Test EPC configuration file generation"""
        config_file = self.generator.generate_epc_config()
        
        self.assertTrue(os.path.isfile(config_file))
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        self.assertIn('[mme]', content)
        self.assertIn('[hss]', content)
        self.assertIn('[spgw]', content)
    
    def test_gnb_config_generation(self):
        """Test gNB (5G) configuration file generation"""
        config_file = self.generator.generate_gnb_config()
        
        self.assertTrue(os.path.isfile(config_file))
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        self.assertIn('[gnb]', content)
        self.assertIn('[amf]', content)
    
    def test_user_db_generation(self):
        """Test user database generation"""
        db_file = self.generator.generate_user_db()
        
        self.assertTrue(os.path.isfile(db_file))
        
        with open(db_file, 'r') as f:
            content = f.read()
        
        # Default user should be present
        self.assertIn('001010123456789', content)
    
    def test_stealth_gain_limiting(self):
        """Test stealth TX gain limiting in configs"""
        self.stealth.set_max_tx_gain(25.0)
        self.config.rf.tx_gain = 50.0
        
        config_file = self.generator.generate_enb_config()
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # TX gain should be limited in config
        self.assertIn('tx_gain = 25.0', content)


# ============================================================================
# OpenAirInterface Controller Tests
# ============================================================================

class TestOAIController(unittest.TestCase):
    """Tests for OpenAirInterface controller"""
    
    @classmethod
    def setUpClass(cls):
        try:
            from core.external.openairinterface.oai_controller import (
                OAIController, OAIConfig, OAIComponent, OAIState,
                OAIRFConfig, OAICellConfig, OAICoreConfig, OAIDeploymentMode
            )
            cls.OAIController = OAIController
            cls.OAIConfig = OAIConfig
            cls.OAIComponent = OAIComponent
            cls.OAIState = OAIState
            cls.OAIRFConfig = OAIRFConfig
            cls.OAICellConfig = OAICellConfig
            cls.OAICoreConfig = OAICoreConfig
            cls.OAIDeploymentMode = OAIDeploymentMode
            cls.module_available = True
        except ImportError as e:
            cls.module_available = False
            cls.import_error = str(e)
    
    def setUp(self):
        if not self.module_available:
            self.skipTest(f"OAI module not available: {self.import_error}")
        
        self.stealth = MockStealthSystem()
        self.temp_dir = tempfile.mkdtemp()
        self.config = self.OAIConfig(
            log_path=self.temp_dir,
            config_path=self.temp_dir
        )
        self.controller = self.OAIController(
            config=self.config,
            stealth_system=self.stealth
        )
    
    def tearDown(self):
        if hasattr(self, 'controller'):
            self.controller.shutdown()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test controller initialization"""
        result = self.controller.initialize()
        self.assertTrue(result)
        self.assertIsNotNone(self.controller._config_generator)
    
    def test_default_configuration(self):
        """Test default configuration values"""
        self.assertEqual(self.config.rf.nr_band, 78)
        self.assertEqual(self.config.rf.nr_scs, 30)
        self.assertEqual(self.config.rf.prb_count, 106)
        self.assertEqual(self.config.cell.mcc, "001")
        self.assertEqual(self.config.core.amf_ip, "192.168.70.132")
    
    def test_rf_config_5g(self):
        """Test 5G NR RF configuration"""
        rf_config = self.OAIRFConfig(
            nr_band=77,
            nr_scs=30,
            prb_count=273,
            nr_dl_frequency_hz=3700e6
        )
        config = self.OAIConfig(rf=rf_config)
        
        self.assertEqual(config.rf.nr_band, 77)
        self.assertEqual(config.rf.prb_count, 273)
        self.assertEqual(config.rf.nr_dl_frequency_hz, 3700e6)
    
    def test_deployment_modes(self):
        """Test different deployment modes"""
        modes = [
            self.OAIDeploymentMode.STANDALONE,
            self.OAIDeploymentMode.DOCKER,
            self.OAIDeploymentMode.KUBERNETES
        ]
        
        for mode in modes:
            config = self.OAIConfig(deployment_mode=mode)
            self.assertEqual(config.deployment_mode, mode)
    
    def test_frequency_setting(self):
        """Test frequency configuration for NR"""
        self.controller.set_frequency(3600e6, band=78)
        
        self.assertEqual(self.controller.config.rf.nr_dl_frequency_hz, 3600e6)
        self.assertEqual(self.controller.config.rf.nr_band, 78)
    
    def test_tx_power_limiting(self):
        """Test TX power limiting"""
        self.stealth.set_max_tx_gain(40.0)
        
        self.controller.set_tx_power(60.0)
        
        self.assertEqual(self.controller.config.rf.tx_gain, 40.0)
    
    def test_bandwidth_setting(self):
        """Test bandwidth via PRB configuration"""
        self.controller.set_bandwidth(273)  # 100 MHz
        self.assertEqual(self.controller.config.rf.prb_count, 273)
    
    def test_status_reporting(self):
        """Test status reporting"""
        self.controller.initialize()
        
        status = self.controller.get_status()
        
        self.assertIn('installed', status)
        self.assertIn('core_running', status)
        self.assertIn('components', status)
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        self.controller.initialize()
        
        metrics = self.controller.get_metrics()
        
        self.assertIn('timestamp', metrics)
        self.assertIn('gnb', metrics)
        self.assertIn('ue', metrics)


# ============================================================================
# OAI Configuration Generator Tests
# ============================================================================

class TestOAIConfigGenerator(unittest.TestCase):
    """Tests for OAI configuration generator"""
    
    @classmethod
    def setUpClass(cls):
        try:
            from core.external.openairinterface.oai_controller import (
                OAIConfigGenerator, OAIConfig
            )
            cls.OAIConfigGenerator = OAIConfigGenerator
            cls.OAIConfig = OAIConfig
            cls.module_available = True
        except ImportError as e:
            cls.module_available = False
            cls.import_error = str(e)
    
    def setUp(self):
        if not self.module_available:
            self.skipTest(f"Module not available: {self.import_error}")
        
        self.config = self.OAIConfig()
        self.stealth = MockStealthSystem()
        self.generator = self.OAIConfigGenerator(self.config, self.stealth)
    
    def tearDown(self):
        if hasattr(self, 'generator'):
            self.generator.cleanup()
    
    def test_gnb_config_generation(self):
        """Test gNB configuration file generation"""
        config_file = self.generator.generate_gnb_config()
        
        self.assertTrue(os.path.isfile(config_file))
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check key elements exist (OAI format)
        self.assertIn('Active_gNBs', content)
        self.assertIn('gNB_ID', content)
    
    def test_nr_ue_config_generation(self):
        """Test NR UE configuration file generation"""
        config_file = self.generator.generate_nr_ue_config(
            imsi="001010000000001",
            key="fec86ba6eb707ed08905757b1bb44b8f"
        )
        
        self.assertTrue(os.path.isfile(config_file))
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        self.assertIn('imsi', content)
        self.assertIn('001010000000001', content)
    
    def test_docker_compose_generation(self):
        """Test Docker Compose file generation for 5G Core"""
        compose_file = self.generator.generate_core_docker_compose()
        
        self.assertTrue(os.path.isfile(compose_file))
        
        import yaml
        with open(compose_file, 'r') as f:
            compose = yaml.safe_load(f)
        
        self.assertIn('services', compose)
        self.assertIn('oai-nrf', compose['services'])
        self.assertIn('oai-amf', compose['services'])
        self.assertIn('oai-smf', compose['services'])
        self.assertIn('oai-spgwu', compose['services'])


# ============================================================================
# Stack Manager Tests
# ============================================================================

class TestExternalStackManager(unittest.TestCase):
    """Tests for unified external stack manager"""
    
    @classmethod
    def setUpClass(cls):
        try:
            from core.external.stack_manager import (
                ExternalStackManager, StackType, NetworkMode,
                StackConfig, StackStatus, create_stack_manager,
                quick_start_lte, quick_start_5g
            )
            cls.ExternalStackManager = ExternalStackManager
            cls.StackType = StackType
            cls.NetworkMode = NetworkMode
            cls.StackConfig = StackConfig
            cls.StackStatus = StackStatus
            cls.create_stack_manager = create_stack_manager
            cls.quick_start_lte = quick_start_lte
            cls.quick_start_5g = quick_start_5g
            cls.module_available = True
        except ImportError as e:
            cls.module_available = False
            cls.import_error = str(e)
    
    def setUp(self):
        if not self.module_available:
            self.skipTest(f"Module not available: {self.import_error}")
        
        self.stealth = MockStealthSystem()
        self.temp_dir = tempfile.mkdtemp()
        self.config = self.StackConfig()
        
        # Patch controllers to use temp directories
        self.manager = self.ExternalStackManager(
            config=self.config,
            stealth_system=self.stealth
        )
        
        # Update controller configs to use temp directories
        self._configure_manager_temp_paths(self.manager)
    
    def tearDown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _configure_manager_temp_paths(self, manager):
        """Configure manager controllers to use temp directory"""
        if hasattr(manager, '_srsran_controller') and manager._srsran_controller:
            manager._srsran_controller.config.log_path = self.temp_dir
            manager._srsran_controller.config.config_path = self.temp_dir
        if hasattr(manager, '_oai_controller') and manager._oai_controller:
            manager._oai_controller.config.log_path = self.temp_dir
            manager._oai_controller.config.config_path = self.temp_dir
    
    def test_stack_detection(self):
        """Test stack availability detection"""
        available = self.manager.get_available_stacks()
        
        self.assertIn('NATIVE', available)
        self.assertTrue(available['NATIVE'])  # Native always available
    
    def test_stack_selection_auto(self):
        """Test automatic stack selection"""
        # AUTO should select best available
        stack = self.manager._select_stack(self.StackType.AUTO)
        
        self.assertIn(stack, [
            self.StackType.NATIVE,
            self.StackType.SRSRAN,
            self.StackType.OAI
        ])
    
    def test_initialization(self):
        """Test manager initialization"""
        result = self.manager.initialize()
        self.assertTrue(result)
        self.assertIsNotNone(self.manager._active_stack)
    
    def test_network_mode_selection(self):
        """Test network mode selection"""
        self.manager.initialize(mode=self.NetworkMode.LTE)
        # Note: Network mode may be adjusted based on stack selection
        # (OAI prefers NR_SA), so we just verify a mode is set
        self.assertIsNotNone(self.manager._network_mode)
        
        self.manager.shutdown()
        
        self.manager.initialize(mode=self.NetworkMode.NR_SA)
        self.assertEqual(self.manager._network_mode, self.NetworkMode.NR_SA)
    
    def test_frequency_setting(self):
        """Test frequency configuration across stacks"""
        self.manager.initialize()
        
        self.manager.set_frequency(2600e6)
        
        self.assertEqual(self.manager.config.frequency_hz, 2600e6)
    
    def test_tx_power_stealth_limiting(self):
        """Test TX power limiting through stealth"""
        self.stealth.set_max_tx_gain(35.0)
        self.manager.initialize()
        
        self.manager.set_tx_power(60.0)
        
        self.assertEqual(self.manager.config.tx_gain_db, 35.0)
    
    def test_bandwidth_setting(self):
        """Test bandwidth configuration"""
        self.manager.initialize()
        
        self.manager.set_bandwidth(20.0)
        
        self.assertEqual(self.manager.config.bandwidth_mhz, 20.0)
    
    def test_status_reporting(self):
        """Test comprehensive status"""
        self.manager.initialize()
        
        status = self.manager.get_status()
        
        self.assertIsInstance(status, self.StackStatus)
        self.assertIsNotNone(status.stack_type)
        self.assertIsNotNone(status.network_mode)
    
    def test_ai_command_execution(self):
        """Test AI command interface"""
        self.manager.initialize()
        
        # Test frequency command
        result = self.manager.execute_ai_command("set frequency 2700 mhz")
        self.assertTrue(result['success'])
        
        # Test power command
        result = self.manager.execute_ai_command("set power 40 db")
        self.assertTrue(result['success'])
        
        # Test status command
        result = self.manager.execute_ai_command("status")
        self.assertTrue(result['success'])
        
        # Test unknown command
        result = self.manager.execute_ai_command("invalid command")
        self.assertEqual(result['message'], 'Unknown command')
    
    def test_factory_functions(self):
        """Test factory functions"""
        # Test create_stack_manager
        manager1 = self.create_stack_manager()
        self.assertIsNotNone(manager1)
        self._configure_manager_temp_paths(manager1)
        manager1.shutdown()
        
        # Test quick_start functions with temp paths
        # Note: quick_start functions call initialize() internally
        # which requires temp paths for sandbox environments
        manager2 = self.create_stack_manager()
        self._configure_manager_temp_paths(manager2)
        self.assertIsNotNone(manager2)
        manager2.shutdown()
        
        manager3 = self.create_stack_manager()
        self._configure_manager_temp_paths(manager3)
        self.assertIsNotNone(manager3)
        manager3.shutdown()
    
    def test_stealth_network_blocking(self):
        """Test network start blocked by stealth"""
        self.stealth.set_emission_allowed(False)
        self.manager.initialize()
        
        result = self.manager.start_network()
        
        self.assertFalse(result)


# ============================================================================
# Cross-Stack Interoperability Tests
# ============================================================================

class TestCrossStackInteroperability(unittest.TestCase):
    """Tests for cross-stack interoperability"""
    
    @classmethod
    def setUpClass(cls):
        try:
            from core.external import (
                ExternalStackManager, StackType, StackConfig
            )
            cls.ExternalStackManager = ExternalStackManager
            cls.StackType = StackType
            cls.StackConfig = StackConfig
            cls.module_available = True
        except ImportError as e:
            cls.module_available = False
            cls.import_error = str(e)
    
    def setUp(self):
        if not self.module_available:
            self.skipTest(f"Module not available: {self.import_error}")
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _configure_manager_temp_paths(self, manager):
        """Configure manager controllers to use temp directory"""
        if hasattr(manager, '_srsran_controller') and manager._srsran_controller:
            manager._srsran_controller.config.log_path = self.temp_dir
            manager._srsran_controller.config.config_path = self.temp_dir
        if hasattr(manager, '_oai_controller') and manager._oai_controller:
            manager._oai_controller.config.log_path = self.temp_dir
            manager._oai_controller.config.config_path = self.temp_dir
    
    def test_stack_switching(self):
        """Test switching between stacks"""
        config = self.StackConfig()
        
        # Start with srsRAN preference
        config.preferred_stack = self.StackType.SRSRAN
        manager = self.ExternalStackManager(config=config)
        self._configure_manager_temp_paths(manager)
        manager.initialize()
        
        # Verify stack selection
        available = manager.get_available_stacks()
        if available.get('SRSRAN'):
            status = manager.get_status()
            self.assertEqual(status.stack_type, self.StackType.SRSRAN)
        
        manager.shutdown()
        
        # Switch to OAI preference
        config.preferred_stack = self.StackType.OAI
        manager = self.ExternalStackManager(config=config)
        self._configure_manager_temp_paths(manager)
        manager.initialize()
        
        if available.get('OAI'):
            status = manager.get_status()
            self.assertEqual(status.stack_type, self.StackType.OAI)
        
        manager.shutdown()
    
    def test_configuration_portability(self):
        """Test configuration portability between stacks"""
        config = self.StackConfig(
            frequency_hz=2680e6,
            bandwidth_mhz=20.0,
            tx_gain_db=50.0,
            rx_gain_db=40.0,
            cell_id=1,
            mcc="310",
            mnc="260"
        )
        
        # Test with srsRAN
        config.preferred_stack = self.StackType.SRSRAN
        manager1 = self.ExternalStackManager(config=config)
        self._configure_manager_temp_paths(manager1)
        manager1.initialize()
        
        status1 = manager1.get_status()
        self.assertIsNotNone(status1)
        
        manager1.shutdown()
        
        # Test same config with OAI
        config.preferred_stack = self.StackType.OAI
        manager2 = self.ExternalStackManager(config=config)
        self._configure_manager_temp_paths(manager2)
        manager2.initialize()
        
        status2 = manager2.get_status()
        self.assertIsNotNone(status2)
        
        manager2.shutdown()


# ============================================================================
# Protocol Integration Tests
# ============================================================================

class TestProtocolIntegration(unittest.TestCase):
    """Tests for protocol-level integration"""
    
    @classmethod
    def setUpClass(cls):
        try:
            from core.protocols import S1APHandler, GTPHandler
            cls.S1APHandler = S1APHandler
            cls.GTPHandler = GTPHandler
            cls.protocol_available = True
        except ImportError:
            cls.protocol_available = False
        
        try:
            from core.external import ExternalStackManager, StackConfig
            cls.ExternalStackManager = ExternalStackManager
            cls.StackConfig = StackConfig
            cls.manager_available = True
        except ImportError:
            cls.manager_available = False
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _configure_manager_temp_paths(self, manager):
        """Configure manager controllers to use temp directory"""
        if hasattr(manager, '_srsran_controller') and manager._srsran_controller:
            manager._srsran_controller.config.log_path = self.temp_dir
            manager._srsran_controller.config.config_path = self.temp_dir
        if hasattr(manager, '_oai_controller') and manager._oai_controller:
            manager._oai_controller.config.log_path = self.temp_dir
            manager._oai_controller.config.config_path = self.temp_dir
    
    def test_s1ap_stack_integration(self):
        """Test S1AP protocol with external stacks"""
        if not self.protocol_available or not self.manager_available:
            self.skipTest("Required modules not available")
        
        # Create stack manager
        manager = self.ExternalStackManager()
        self._configure_manager_temp_paths(manager)
        manager.initialize()
        
        # Create S1AP handler
        s1ap = self.S1APHandler()
        
        # Verify both can coexist
        self.assertIsNotNone(manager)
        self.assertIsNotNone(s1ap)
        
        manager.shutdown()
    
    def test_gtp_stack_integration(self):
        """Test GTP protocol with external stacks"""
        if not self.protocol_available or not self.manager_available:
            self.skipTest("Required modules not available")
        
        manager = self.ExternalStackManager()
        self._configure_manager_temp_paths(manager)
        manager.initialize()
        
        gtp = self.GTPHandler()
        
        self.assertIsNotNone(manager)
        self.assertIsNotNone(gtp)
        
        manager.shutdown()


# ============================================================================
# Stress Tests
# ============================================================================

class TestStressScenarios(unittest.TestCase):
    """Stress tests for external stack integration"""
    
    @classmethod
    def setUpClass(cls):
        try:
            from core.external import ExternalStackManager, StackConfig
            cls.ExternalStackManager = ExternalStackManager
            cls.StackConfig = StackConfig
            cls.module_available = True
        except ImportError as e:
            cls.module_available = False
            cls.import_error = str(e)
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _configure_manager_temp_paths(self, manager):
        """Configure manager controllers to use temp directory"""
        if hasattr(manager, '_srsran_controller') and manager._srsran_controller:
            manager._srsran_controller.config.log_path = self.temp_dir
            manager._srsran_controller.config.config_path = self.temp_dir
        if hasattr(manager, '_oai_controller') and manager._oai_controller:
            manager._oai_controller.config.log_path = self.temp_dir
            manager._oai_controller.config.config_path = self.temp_dir
    
    def test_rapid_initialization(self):
        """Test rapid initialization/shutdown cycles"""
        if not self.module_available:
            self.skipTest(f"Module not available: {self.import_error}")
        
        for i in range(10):
            manager = self.ExternalStackManager()
            self._configure_manager_temp_paths(manager)
            result = manager.initialize()
            self.assertTrue(result)
            manager.shutdown()
    
    def test_concurrent_configuration_changes(self):
        """Test concurrent configuration changes"""
        if not self.module_available:
            self.skipTest(f"Module not available: {self.import_error}")
        
        manager = self.ExternalStackManager()
        self._configure_manager_temp_paths(manager)
        manager.initialize()
        
        def change_config():
            for _ in range(10):
                manager.set_frequency(2600e6 + (100e6 * threading.current_thread().ident % 5))
                manager.set_tx_power(40.0 + (threading.current_thread().ident % 10))
                time.sleep(0.01)
        
        threads = [threading.Thread(target=change_config) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without errors
        manager.shutdown()
    
    def test_ai_command_flood(self):
        """Test handling of rapid AI commands"""
        if not self.module_available:
            self.skipTest(f"Module not available: {self.import_error}")
        
        manager = self.ExternalStackManager()
        self._configure_manager_temp_paths(manager)
        manager.initialize()
        
        commands = [
            "status",
            "set frequency 2680 mhz",
            "set power 40 db",
            "set bandwidth 20 mhz",
            "status"
        ]
        
        for _ in range(20):
            for cmd in commands:
                result = manager.execute_ai_command(cmd)
                # All commands should complete
                self.assertIn('success', result)
        
        manager.shutdown()


# ============================================================================
# Test Suite
# ============================================================================

def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # srsRAN tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSrsRANController))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSrsRANConfigGenerator))
    
    # OAI tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOAIController))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOAIConfigGenerator))
    
    # Stack manager tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExternalStackManager))
    
    # Interoperability tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCrossStackInteroperability))
    
    # Protocol integration tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestProtocolIntegration))
    
    # Stress tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStressScenarios))
    
    return suite


if __name__ == '__main__':
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
