"""
Comprehensive Hardware Integration Test Framework.

This package provides extensive testing capabilities for real RF hardware
including SDR devices, signal generators, spectrum analyzers, and
complete RF signal chains.

Supported Hardware:
- HackRF One/PortaPack
- BladeRF (x40, x115, xA4, xA5, xA9)
- USRP (B200, B210, X300, X310, N200, N210)
- RTL-SDR (various dongles)
- LimeSDR (Mini, USB)
- PlutoSDR
- Airspy (R2, Mini, HF+)

Test Categories:
- Hardware Detection and Initialization
- RF Signal Generation and Reception
- Performance Benchmarking
- Calibration Verification
- Fault Detection and Diagnostics
- Stress Testing and Reliability
- End-to-End Communication Tests
- Protocol Compliance Testing

Standards and Specifications:
- USB 2.0/3.0 interface compliance
- SMA/MCX connector specifications
- SDR hardware specifications per manufacturer
- RF measurement accuracy standards
- IEC 61508 (Functional Safety)
- MIL-HDBK-217F (Reliability Prediction)
- IEC 60068-2 (Environmental Testing)

Author: RF Arsenal Development Team
License: Proprietary
"""

from .framework import (
    # Core Framework
    HardwareTestFramework,
    TestCategory,
    TestPriority,
    TestResult,
    TestStatus,
    HardwareCapability,
    
    # Hardware Interfaces
    HardwareInterface,
    SDRInterface,
    SignalGeneratorInterface,
    SpectrumAnalyzerInterface,
    
    # Test Infrastructure
    TestRunner,
    TestSuite,
    TestCase,
    TestFixture,
    
    # Hardware Discovery
    HardwareDiscovery,
    DeviceInfo,
    DeviceCapabilities,
    
    # Configuration
    HardwareTestConfig,
    TestEnvironment,
)

from .sdr_tests import (
    # SDR Test Suites
    SDRTestSuite,
    HackRFTestSuite,
    BladeRFTestSuite,
    USRPTestSuite,
    RTLSDRTestSuite,
    LimeSDRTestSuite,
    PlutoSDRTestSuite,
    AirspyTestSuite,
    
    # Common SDR Tests
    SDRInitializationTests,
    SDRFrequencyTests,
    SDRGainTests,
    SDRBandwidthTests,
    SDRStreamingTests,
)

from .signal_tests import (
    # Signal Generation Tests
    SignalGenerationTests,
    ToneGenerationTests,
    ModulationTests,
    WaveformTests,
    
    # Signal Analysis Tests
    SignalAnalysisTests,
    SpectrumAnalysisTests,
    PowerMeasurementTests,
    SNRMeasurementTests,
    
    # Signal Quality Tests
    SignalQualityTests,
    EVMTests,
    ACPRTests,
    SpuriousTests,
)

from .performance_tests import (
    # Performance Benchmarks
    PerformanceBenchmarks,
    ThroughputTests,
    LatencyTests,
    CPUUtilizationTests,
    MemoryUsageTests,
    
    # Stress Tests
    StressTests,
    ContinuousOperationTests,
    ThermalTests,
    PowerCyclingTests,
)

from .calibration_tests import (
    # Calibration Verification
    CalibrationTests,
    FrequencyCalibrationTests,
    PowerCalibrationTests,
    IQBalanceTests,
    DCOffsetTests,
)

from .diagnostics import (
    # Hardware Diagnostics
    HardwareDiagnostics,
    USBDiagnostics,
    RFPathDiagnostics,
    FPGADiagnostics,
    FirmwareDiagnostics,
)

from .e2e_tests import (
    # End-to-End Tests
    EndToEndTests,
    TransmitReceiveLoopback,
    MultiDeviceTests,
    ProtocolTests,
    CommunicationChainTests,
)

from .reporting import (
    # Test Reporting
    TestReporter,
    HTMLReporter,
    JSONReporter,
    JUnitReporter,
    CSVReporter,
    ConsolidatedReporter,
    TestMetrics,
    TestCoverage,
)

from .stress_reliability import (
    # Stress Test Configuration
    StressLevel,
    StressTestConfiguration,
    StressTestMetrics,
    
    # Core Stress Tests
    ContinuousOperationStressTest,
    HighLoadStressTest,
    MemoryLeakDetectionTest,
    PowerCycleReliabilityTest,
    RecoveryAndFaultToleranceTest,
    EnduranceTest,
    
    # Reliability Suite
    ReliabilityValidationSuite,
)

__all__ = [
    # Framework
    'HardwareTestFramework',
    'TestCategory',
    'TestPriority', 
    'TestResult',
    'TestStatus',
    'HardwareCapability',
    
    # Interfaces
    'HardwareInterface',
    'SDRInterface',
    'SignalGeneratorInterface',
    'SpectrumAnalyzerInterface',
    
    # Infrastructure
    'TestRunner',
    'TestSuite',
    'TestCase',
    'TestFixture',
    
    # Discovery
    'HardwareDiscovery',
    'DeviceInfo',
    'DeviceCapabilities',
    
    # Configuration
    'HardwareTestConfig',
    'TestEnvironment',
    
    # SDR Tests
    'SDRTestSuite',
    'HackRFTestSuite',
    'BladeRFTestSuite',
    'USRPTestSuite',
    'RTLSDRTestSuite',
    'LimeSDRTestSuite',
    'PlutoSDRTestSuite',
    'AirspyTestSuite',
    'SDRInitializationTests',
    'SDRFrequencyTests',
    'SDRGainTests',
    'SDRBandwidthTests',
    'SDRStreamingTests',
    
    # Signal Tests
    'SignalGenerationTests',
    'ToneGenerationTests',
    'ModulationTests',
    'WaveformTests',
    'SignalAnalysisTests',
    'SpectrumAnalysisTests',
    'PowerMeasurementTests',
    'SNRMeasurementTests',
    'SignalQualityTests',
    'EVMTests',
    'ACPRTests',
    'SpuriousTests',
    
    # Performance Tests
    'PerformanceBenchmarks',
    'ThroughputTests',
    'LatencyTests',
    'CPUUtilizationTests',
    'MemoryUsageTests',
    'StressTests',
    'ContinuousOperationTests',
    'ThermalTests',
    'PowerCyclingTests',
    
    # Calibration
    'CalibrationTests',
    'FrequencyCalibrationTests',
    'PowerCalibrationTests',
    'IQBalanceTests',
    'DCOffsetTests',
    
    # Diagnostics
    'HardwareDiagnostics',
    'USBDiagnostics',
    'RFPathDiagnostics',
    'FPGADiagnostics',
    'FirmwareDiagnostics',
    
    # E2E Tests
    'EndToEndTests',
    'TransmitReceiveLoopback',
    'MultiDeviceTests',
    'ProtocolTests',
    'CommunicationChainTests',
    
    # Reporting
    'TestReporter',
    'HTMLReporter',
    'JSONReporter',
    'JUnitReporter',
    'CSVReporter',
    'ConsolidatedReporter',
    'TestMetrics',
    'TestCoverage',
    
    # Stress & Reliability
    'StressLevel',
    'StressTestConfiguration',
    'StressTestMetrics',
    'ContinuousOperationStressTest',
    'HighLoadStressTest',
    'MemoryLeakDetectionTest',
    'PowerCycleReliabilityTest',
    'RecoveryAndFaultToleranceTest',
    'EnduranceTest',
    'ReliabilityValidationSuite',
]
