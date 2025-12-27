#!/usr/bin/env python3
"""
Unit tests for RF Arsenal OS SoapySDR Hardware Abstraction Layer

Tests the universal SDR backend supporting BladeRF, HackRF, RTL-SDR,
USRP, LimeSDR, Airspy, and PlutoSDR.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.sdr import (
    SDRType,
    StreamFormat,
    StreamDirection,
    GainMode,
    SDRDeviceInfo,
    StreamConfig,
    IQCapture,
    DSPProcessor,
    SoapySDRDevice,
    SDRManager,
    get_sdr_manager,
    SDR_SPECIFICATIONS,
    SOAPY_AVAILABLE,
)


class TestSDREnums:
    """Test SDR enumeration types"""
    
    def test_sdr_type_values(self):
        """Test SDRType enum values"""
        assert SDRType.BLADERF.value == "bladerf"
        assert SDRType.BLADERF2.value == "bladerf2"
        assert SDRType.HACKRF.value == "hackrf"
        assert SDRType.RTLSDR.value == "rtlsdr"
        assert SDRType.USRP.value == "uhd"
        assert SDRType.LIMESDR.value == "lime"
        assert SDRType.AIRSPY.value == "airspy"
        assert SDRType.PLUTOSDR.value == "plutosdr"
    
    def test_stream_format_values(self):
        """Test StreamFormat enum values"""
        assert StreamFormat.CF32.value == "CF32"
        assert StreamFormat.CS16.value == "CS16"
        assert StreamFormat.CS8.value == "CS8"
        assert StreamFormat.CU8.value == "CU8"
    
    def test_stream_direction(self):
        """Test StreamDirection enum"""
        assert StreamDirection.RX is not None
        assert StreamDirection.TX is not None
    
    def test_gain_mode(self):
        """Test GainMode enum"""
        assert GainMode.MANUAL is not None
        assert GainMode.AGC_SLOW is not None
        assert GainMode.AGC_FAST is not None
        assert GainMode.AGC_HYBRID is not None


class TestSDRDeviceInfo:
    """Test SDRDeviceInfo data class"""
    
    def test_device_info_creation(self):
        """Test creating SDRDeviceInfo"""
        info = SDRDeviceInfo(
            driver="hackrf",
            label="HackRF One",
            serial="000000001",
            hardware="HackRF One",
            sdr_type=SDRType.HACKRF,
            index=0
        )
        
        assert info.driver == "hackrf"
        assert info.label == "HackRF One"
        assert info.serial == "000000001"
        assert info.sdr_type == SDRType.HACKRF
        assert info.index == 0
    
    def test_device_info_defaults(self):
        """Test SDRDeviceInfo default values"""
        info = SDRDeviceInfo(
            driver="test",
            label="Test",
            serial="123",
            hardware="Test",
            sdr_type=SDRType.UNKNOWN,
            index=0
        )
        
        assert info.freq_range == (0, 6e9)
        assert info.sample_rate_range == (0, 61.44e6)
        assert info.tx_capable == False
        assert info.full_duplex == False
        assert info.mimo_capable == False
        assert info.num_channels == 1
    
    def test_device_info_str(self):
        """Test SDRDeviceInfo string representation"""
        info = SDRDeviceInfo(
            driver="bladerf2",
            label="BladeRF 2.0 micro",
            serial="ABC123",
            hardware="BladeRF 2.0",
            sdr_type=SDRType.BLADERF2,
            index=0
        )
        
        str_repr = str(info)
        assert "BladeRF 2.0 micro" in str_repr
        assert "bladerf2" in str_repr
        assert "ABC123" in str_repr


class TestStreamConfig:
    """Test StreamConfig data class"""
    
    def test_stream_config_defaults(self):
        """Test StreamConfig default values"""
        config = StreamConfig()
        
        assert config.frequency == 100e6
        assert config.sample_rate == 2e6
        assert config.bandwidth == 0
        assert config.gain == 30
        assert config.gain_mode == GainMode.MANUAL
        assert config.channel == 0
        assert config.buffer_size == 16384
        assert config.format == StreamFormat.CF32
    
    def test_stream_config_custom(self):
        """Test StreamConfig with custom values"""
        config = StreamConfig(
            frequency=433.92e6,
            sample_rate=10e6,
            bandwidth=8e6,
            gain=45,
            gain_mode=GainMode.AGC_FAST,
            channel=1
        )
        
        assert config.frequency == 433.92e6
        assert config.sample_rate == 10e6
        assert config.bandwidth == 8e6
        assert config.gain == 45
        assert config.gain_mode == GainMode.AGC_FAST
        assert config.channel == 1


class TestIQCapture:
    """Test IQCapture data class"""
    
    def test_iq_capture_creation(self):
        """Test creating IQCapture"""
        samples = np.random.randn(1000) + 1j * np.random.randn(1000)
        samples = samples.astype(np.complex64)
        
        capture = IQCapture(
            samples=samples,
            frequency=100e6,
            sample_rate=2e6,
            bandwidth=2e6,
            timestamp=1234567890.0,
            duration=0.5,
            device="test"
        )
        
        assert len(capture.samples) == 1000
        assert capture.frequency == 100e6
        assert capture.sample_rate == 2e6
        assert capture.duration == 0.5
        assert capture.device == "test"
    
    def test_iq_capture_save_load_npy(self, tmp_path):
        """Test saving and loading IQ capture as NPY"""
        samples = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)
        
        capture = IQCapture(
            samples=samples,
            frequency=100e6,
            sample_rate=2e6,
            bandwidth=2e6,
            timestamp=0,
            duration=0.5,
            device="test"
        )
        
        filepath = str(tmp_path / "test.npy")
        capture.save(filepath, format="npy")
        
        loaded = IQCapture.load(filepath, frequency=100e6, sample_rate=2e6)
        np.testing.assert_array_almost_equal(capture.samples, loaded.samples)
    
    def test_iq_capture_save_cf32(self, tmp_path):
        """Test saving IQ capture as CF32"""
        samples = (np.random.randn(100) + 1j * np.random.randn(100)).astype(np.complex64)
        
        capture = IQCapture(
            samples=samples,
            frequency=100e6,
            sample_rate=2e6,
            bandwidth=2e6,
            timestamp=0,
            duration=0.05,
            device="test"
        )
        
        filepath = str(tmp_path / "test.cf32")
        capture.save(filepath, format="cf32")
        
        # Verify file exists
        assert os.path.exists(filepath)


class TestDSPProcessor:
    """Test DSP processing capabilities"""
    
    @pytest.fixture
    def dsp(self):
        return DSPProcessor()
    
    def test_compute_fft(self, dsp):
        """Test FFT computation"""
        # Generate test signal
        t = np.arange(1024) / 1000
        signal = np.exp(2j * np.pi * 100 * t).astype(np.complex64)
        
        fft_result = dsp.compute_fft(signal, fft_size=1024)
        
        assert len(fft_result) == 1024
        assert fft_result.dtype == np.float64  # dB values
    
    def test_compute_fft_short_signal(self, dsp):
        """Test FFT with short signal (zero padding)"""
        signal = np.ones(100, dtype=np.complex64)
        fft_result = dsp.compute_fft(signal, fft_size=256)
        
        assert len(fft_result) == 256
    
    def test_measure_power(self, dsp):
        """Test power measurement"""
        # Unit power signal
        signal = np.ones(1000, dtype=np.complex64)
        power = dsp.measure_power(signal)
        
        assert np.isclose(power, 0, atol=0.1)  # Should be ~0 dB
        
        # Half power signal
        signal = np.ones(1000, dtype=np.complex64) * 0.5
        power = dsp.measure_power(signal)
        
        assert power < 0  # Should be negative dB
    
    def test_detect_signal(self, dsp):
        """Test signal detection"""
        # Strong signal
        signal = np.ones(1000, dtype=np.complex64) * 10
        assert dsp.detect_signal(signal, threshold_db=-30) == True
        
        # Weak signal
        signal = np.ones(1000, dtype=np.complex64) * 0.0001
        assert dsp.detect_signal(signal, threshold_db=-30) == False
    
    def test_estimate_frequency(self, dsp):
        """Test frequency estimation"""
        sample_rate = 1000
        test_freq = 100
        
        t = np.arange(1024) / sample_rate
        signal = np.exp(2j * np.pi * test_freq * t).astype(np.complex64)
        
        estimated = dsp.estimate_frequency(signal, sample_rate)
        
        # Should be close to 100 Hz
        assert np.isclose(estimated, test_freq, atol=5)
    
    def test_demodulate_am(self, dsp):
        """Test AM demodulation"""
        # AM signal: carrier with amplitude modulation
        t = np.arange(1000) / 1000
        carrier = np.exp(2j * np.pi * 100 * t)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 10 * t)
        signal = (carrier * modulation).astype(np.complex64)
        
        demod = dsp.demodulate_am(signal)
        
        assert len(demod) == len(signal)
        assert demod.dtype == np.float32
    
    def test_lowpass_filter(self, dsp):
        """Test lowpass filter"""
        signal = np.random.randn(1000).astype(np.complex64)
        filtered = dsp.lowpass_filter(signal, cutoff=100, sample_rate=1000)
        
        assert len(filtered) == len(signal)
    
    def test_compute_spectrogram(self, dsp):
        """Test spectrogram computation"""
        signal = np.random.randn(10000) + 1j * np.random.randn(10000)
        signal = signal.astype(np.complex64)
        
        spectrogram = dsp.compute_spectrogram(signal, fft_size=256, overlap=128)
        
        assert spectrogram.ndim == 2
        assert spectrogram.shape[1] == 256


class TestSoapySDRDevice:
    """Test SoapySDR device wrapper"""
    
    def test_device_creation(self):
        """Test creating SoapySDRDevice"""
        device = SoapySDRDevice()
        
        assert device._device is None
        assert device._streaming == False
        assert device.dsp is not None
    
    def test_device_with_info(self):
        """Test creating device with info"""
        info = SDRDeviceInfo(
            driver="hackrf",
            label="HackRF One",
            serial="123",
            hardware="HackRF",
            sdr_type=SDRType.HACKRF,
            index=0
        )
        
        device = SoapySDRDevice(info)
        assert device.device_info == info
    
    def test_enumerate_devices(self):
        """Test device enumeration (simulation mode)"""
        devices = SoapySDRDevice.enumerate_devices()
        
        # In simulation mode or without hardware, may return empty
        assert isinstance(devices, list)
    
    def test_configure_rx_simulation(self):
        """Test RX configuration in simulation mode"""
        device = SoapySDRDevice()
        
        config = StreamConfig(
            frequency=433.92e6,
            sample_rate=2e6,
            gain=40
        )
        
        result = device.configure_rx(config)
        assert result == True
        assert device._rx_config.frequency == 433.92e6
    
    def test_configure_tx_simulation(self):
        """Test TX configuration in simulation mode"""
        device = SoapySDRDevice()
        
        config = StreamConfig(
            frequency=433.92e6,
            sample_rate=2e6,
            gain=20
        )
        
        result = device.configure_tx(config)
        assert result == True
        assert device._tx_config.frequency == 433.92e6
    
    def test_set_frequency(self):
        """Test setting frequency"""
        device = SoapySDRDevice()
        device.set_frequency(915e6)
        
        assert device._rx_config.frequency == 915e6
    
    def test_set_sample_rate(self):
        """Test setting sample rate"""
        device = SoapySDRDevice()
        device.set_sample_rate(10e6)
        
        assert device._rx_config.sample_rate == 10e6
    
    def test_set_gain(self):
        """Test setting gain"""
        device = SoapySDRDevice()
        device.set_gain(45)
        
        assert device._rx_config.gain == 45
    
    def test_get_info(self):
        """Test getting device info"""
        device = SoapySDRDevice()
        info = device.get_info()
        
        assert 'driver' in info
        assert 'soapy_available' in info
        assert 'streaming' in info


class TestSDRManager:
    """Test SDR manager"""
    
    def test_manager_creation(self):
        """Test creating SDR manager"""
        manager = SDRManager()
        
        assert manager._devices == []
        assert manager._active_device is None
    
    def test_scan_devices(self):
        """Test scanning for devices"""
        manager = SDRManager()
        devices = manager.scan_devices()
        
        assert isinstance(devices, list)
    
    def test_list_devices(self):
        """Test listing devices"""
        manager = SDRManager()
        devices = manager.list_devices()
        
        assert isinstance(devices, list)
    
    def test_get_status(self):
        """Test getting manager status"""
        manager = SDRManager()
        status = manager.get_status()
        
        assert 'soapy_available' in status
        assert 'devices_found' in status
        assert 'devices' in status
        assert 'active_device' in status
    
    def test_close_all(self):
        """Test closing all devices"""
        manager = SDRManager()
        manager.close_all()
        
        assert manager._device_cache == {}
        assert manager._active_device is None


class TestSDRManagerSingleton:
    """Test SDR manager singleton"""
    
    def test_get_sdr_manager(self):
        """Test getting singleton instance"""
        manager1 = get_sdr_manager()
        manager2 = get_sdr_manager()
        
        assert manager1 is manager2
    
    def test_singleton_type(self):
        """Test singleton is correct type"""
        manager = get_sdr_manager()
        assert isinstance(manager, SDRManager)


class TestSDRSpecifications:
    """Test SDR specifications database"""
    
    def test_specifications_exist(self):
        """Test that specifications exist for common SDRs"""
        assert SDRType.BLADERF in SDR_SPECIFICATIONS
        assert SDRType.HACKRF in SDR_SPECIFICATIONS
        assert SDRType.RTLSDR in SDR_SPECIFICATIONS
        assert SDRType.USRP in SDR_SPECIFICATIONS
        assert SDRType.LIMESDR in SDR_SPECIFICATIONS
    
    def test_bladerf2_specs(self):
        """Test BladeRF 2.0 specifications"""
        specs = SDR_SPECIFICATIONS[SDRType.BLADERF2]
        
        assert specs['freq_range'] == (47e6, 6e9)
        assert specs['sample_rate_max'] == 61.44e6
        assert specs['bits'] == 12
        assert specs['full_duplex'] == True
        assert specs['tx_capable'] == True
        assert specs['mimo'] == True
    
    def test_hackrf_specs(self):
        """Test HackRF specifications"""
        specs = SDR_SPECIFICATIONS[SDRType.HACKRF]
        
        assert specs['freq_range'] == (1e6, 6e9)
        assert specs['sample_rate_max'] == 20e6
        assert specs['bits'] == 8
        assert specs['full_duplex'] == False
        assert specs['tx_capable'] == True
    
    def test_rtlsdr_specs(self):
        """Test RTL-SDR specifications"""
        specs = SDR_SPECIFICATIONS[SDRType.RTLSDR]
        
        assert specs['freq_range'] == (24e6, 1.766e9)
        assert specs['sample_rate_max'] == 3.2e6
        assert specs['bits'] == 8
        assert specs['tx_capable'] == False


class TestAICommandCenterIntegration:
    """Test AI Command Center SDR integration"""
    
    def test_sdr_command_category_exists(self):
        """Test SDR command category exists"""
        from core.ai_command_center import CommandCategory
        
        assert hasattr(CommandCategory, 'SDR')
        assert CommandCategory.SDR.value == "sdr"
    
    def test_sdr_commands_processed(self):
        """Test SDR commands are processed"""
        from core.ai_command_center import AICommandCenter
        
        ai = AICommandCenter()
        
        # Test various SDR commands
        commands = [
            "scan sdr",
            "list sdr devices",
            "sdr status",
            "sdr info",
        ]
        
        for cmd in commands:
            result = ai.process_command(cmd)
            assert hasattr(result, 'success'), f"Command '{cmd}' failed to return proper result"
    
    def test_sdr_help_topic_exists(self):
        """Test SDR help topic exists"""
        from core.ai_command_center import AICommandCenter
        
        ai = AICommandCenter()
        assert 'sdr' in ai.HELP_TOPICS
        assert 'scan sdr' in ai.HELP_TOPICS['sdr']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
