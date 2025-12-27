#!/usr/bin/env python3
"""
RF Arsenal OS - Radar Systems Module
Hardware: BladeRF 2.0 micro xA9
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RadarConfig:
    """Radar Configuration"""
    frequency: int = 10_000_000_000  # 10 GHz (X-band)
    sample_rate: int = 40_000_000    # 40 MSPS
    tx_power: int = 30               # dBm
    pulse_width: float = 1e-6        # 1 microsecond
    prf: int = 1000                  # Pulse Repetition Frequency (Hz)
    radar_type: str = "fmcw"         # fmcw, pulse, cw

@dataclass
class Target:
    """Detected Target"""
    range_m: float
    velocity_mps: float
    azimuth_deg: float
    rcs: float  # Radar Cross Section (dBsm)
    timestamp: datetime
    snr: float

class RadarSystems:
    """Multi-mode Radar System (FMCW, Pulse, CW)"""
    
    # Speed of light
    C = 299_792_458  # m/s
    
    def __init__(self, hardware_controller):
        """
        Initialize radar system
        
        Args:
            hardware_controller: BladeRF hardware controller instance
        """
        self.hw = hardware_controller
        self.config = RadarConfig()
        self.is_running = False
        self.detected_targets: List[Target] = []
        
    def configure(self, config: RadarConfig) -> bool:
        """Configure radar parameters"""
        try:
            self.config = config
            
            # Configure BladeRF
            if not self.hw.configure_hardware({
                'frequency': config.frequency,
                'sample_rate': config.sample_rate,
                'bandwidth': config.sample_rate,
                'tx_gain': config.tx_power,
                'rx_gain': 40
            }):
                logger.error("Failed to configure hardware")
                return False
                
            logger.info(f"Radar configured: {config.frequency/1e9:.1f} GHz, "
                       f"Type: {config.radar_type}, PRF: {config.prf} Hz")
            return True
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return False
    
    def fmcw_radar(self, duration: float = 1.0) -> List[Target]:
        """
        Frequency Modulated Continuous Wave (FMCW) Radar
        Used for range and velocity measurement
        
        Args:
            duration: Measurement duration in seconds
            
        Returns:
            List of detected targets
        """
        try:
            logger.info("FMCW radar active...")
            
            # FMCW parameters
            sweep_time = 1e-3  # 1ms sweep
            bandwidth = 200e6  # 200 MHz sweep
            
            targets = []
            num_sweeps = int(duration / sweep_time)
            
            for sweep_idx in range(num_sweeps):
                # Generate FMCW chirp
                chirp = self._generate_fmcw_chirp(sweep_time, bandwidth)
                
                # Transmit chirp
                self.hw.transmit_burst(chirp)
                
                # Receive echo
                echo = self.hw.receive_samples(len(chirp))
                
                if echo is None:
                    continue
                
                # Process echo
                targets_in_sweep = self._process_fmcw_echo(chirp, echo, bandwidth)
                targets.extend(targets_in_sweep)
            
            # Remove duplicates and average
            targets = self._merge_targets(targets)
            
            self.detected_targets.extend(targets)
            logger.info(f"FMCW: Detected {len(targets)} target(s)")
            return targets
            
        except Exception as e:
            logger.error(f"FMCW radar error: {e}")
            return []
    
    def _generate_fmcw_chirp(self, sweep_time: float, bandwidth: float) -> np.ndarray:
        """Generate FMCW chirp signal"""
        num_samples = int(self.config.sample_rate * sweep_time)
        t = np.linspace(0, sweep_time, num_samples, endpoint=False)
        
        # Linear frequency sweep (chirp)
        # f(t) = f0 + (bandwidth / sweep_time) * t
        chirp_rate = bandwidth / sweep_time
        instantaneous_freq = chirp_rate * t
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.config.sample_rate
        
        chirp = np.exp(1j * phase).astype(np.complex64)
        chirp *= 0.5  # Power
        
        return chirp
    
    def _process_fmcw_echo(self, tx_chirp: np.ndarray, rx_echo: np.ndarray,
                          bandwidth: float) -> List[Target]:
        """Process FMCW echo to detect targets"""
        targets = []
        
        # Mix transmitted and received signals
        mixed = tx_chirp * np.conj(rx_echo)
        
        # FFT to get range profile
        fft_result = np.fft.fft(mixed)
        range_profile = np.abs(fft_result)
        
        # Detect peaks (targets)
        threshold = np.mean(range_profile) + 3 * np.std(range_profile)
        peaks = np.where(range_profile > threshold)[0]
        
        for peak_idx in peaks[:10]:  # Limit to 10 targets
            # Calculate range
            beat_freq = peak_idx * self.config.sample_rate / len(mixed)
            target_range = (self.C * beat_freq) / (2 * bandwidth / 1e-3)  # sweep_time = 1ms
            
            # Calculate velocity (from Doppler shift, simplified)
            doppler_shift = self._estimate_doppler(rx_echo, peak_idx)
            velocity = (doppler_shift * self.C) / (2 * self.config.frequency)
            
            # Estimate RCS
            signal_power = range_profile[peak_idx]
            rcs = self._estimate_rcs(signal_power, target_range)
            
            # Calculate SNR
            noise_floor = np.median(range_profile)
            snr = 10 * np.log10(signal_power / noise_floor)
            
            target = Target(
                range_m=target_range,
                velocity_mps=velocity,
                azimuth_deg=0.0,  # Would need antenna array for azimuth
                rcs=rcs,
                timestamp=datetime.now(),
                snr=snr
            )
            
            targets.append(target)
        
        return targets
    
    def _estimate_doppler(self, signal: np.ndarray, peak_idx: int) -> float:
        """Estimate Doppler shift"""
        # Simple phase difference method
        if len(signal) < 2:
            return 0.0
        
        phase = np.angle(signal)
        phase_diff = np.diff(phase)
        
        # Unwrap and average
        phase_diff = np.unwrap(phase_diff)
        doppler = np.mean(phase_diff) * self.config.sample_rate / (2 * np.pi)
        
        return doppler
    
    def _estimate_rcs(self, signal_power: float, target_range: float) -> float:
        """Estimate Radar Cross Section (RCS)"""
        # Simplified radar equation
        # RCS (dBsm) = Signal_power + 40*log10(range) - Pt - Gt - Gr - wavelength
        
        if target_range <= 0:
            return -100.0
        
        # Simplified calculation
        range_loss = 40 * np.log10(target_range)
        rcs_db = 10 * np.log10(signal_power) + range_loss - self.config.tx_power
        
        return rcs_db
    
    def pulse_radar(self, duration: float = 1.0) -> List[Target]:
        """
        Pulse Radar (traditional ranging radar)
        
        Args:
            duration: Measurement duration in seconds
            
        Returns:
            List of detected targets
        """
        try:
            logger.info("Pulse radar active...")
            
            targets = []
            num_pulses = int(duration * self.config.prf)
            
            for pulse_idx in range(num_pulses):
                # Generate pulse
                pulse = self._generate_pulse()
                
                # Transmit pulse
                self.hw.transmit_burst(pulse)
                
                # Wait for echoes (listen period)
                listen_time = (1.0 / self.config.prf) - self.config.pulse_width
                echo = self.hw.receive_samples(
                    int(self.config.sample_rate * listen_time)
                )
                
                if echo is None:
                    continue
                
                # Process echo
                targets_in_pulse = self._process_pulse_echo(echo, pulse_idx)
                targets.extend(targets_in_pulse)
            
            # Merge targets
            targets = self._merge_targets(targets)
            
            self.detected_targets.extend(targets)
            logger.info(f"Pulse: Detected {len(targets)} target(s)")
            return targets
            
        except Exception as e:
            logger.error(f"Pulse radar error: {e}")
            return []
    
    def _generate_pulse(self) -> np.ndarray:
        """Generate radar pulse"""
        num_samples = int(self.config.sample_rate * self.config.pulse_width)
        
        # Rectangular pulse with carrier
        pulse = np.ones(num_samples, dtype=np.complex64)
        pulse *= 0.7  # Power
        
        return pulse
    
    def _process_pulse_echo(self, echo: np.ndarray, pulse_idx: int) -> List[Target]:
        """Process pulse echo to detect targets"""
        targets = []
        
        # Matched filter (correlation)
        pulse = self._generate_pulse()
        correlation = np.correlate(echo, pulse, mode='valid')
        power = np.abs(correlation) ** 2
        
        # Detect peaks
        threshold = np.mean(power) + 5 * np.std(power)
        peaks = np.where(power > threshold)[0]
        
        for peak_idx in peaks[:10]:
            # Calculate range from time delay
            time_delay = peak_idx / self.config.sample_rate
            target_range = (self.C * time_delay) / 2
            
            # Estimate RCS
            signal_power = power[peak_idx]
            rcs = self._estimate_rcs(signal_power, target_range)
            
            # Calculate SNR
            noise_floor = np.median(power)
            snr = 10 * np.log10(signal_power / noise_floor)
            
            target = Target(
                range_m=target_range,
                velocity_mps=0.0,  # Pulse radar doesn't measure velocity directly
                azimuth_deg=0.0,
                rcs=rcs,
                timestamp=datetime.now(),
                snr=snr
            )
            
            targets.append(target)
        
        return targets
    
    def passive_radar(self, duration: float = 5.0) -> List[Target]:
        """
        Passive Bistatic Radar (uses external illuminators like TV/FM)
        
        Args:
            duration: Measurement duration in seconds
            
        Returns:
            List of detected targets
        """
        try:
            logger.info("Passive bistatic radar active...")
            
            # Receive reference signal (direct path)
            ref_samples = self.hw.receive_samples(
                int(self.config.sample_rate * 0.1)
            )
            
            if ref_samples is None:
                return []
            
            targets = []
            num_measurements = int(duration / 0.1)
            
            for i in range(num_measurements):
                # Receive surveillance signal (reflected path)
                surv_samples = self.hw.receive_samples(
                    int(self.config.sample_rate * 0.1)
                )
                
                if surv_samples is None:
                    continue
                
                # Cross-correlation to detect targets
                targets_detected = self._process_passive_radar(
                    ref_samples, surv_samples
                )
                targets.extend(targets_detected)
            
            # Merge targets
            targets = self._merge_targets(targets)
            
            self.detected_targets.extend(targets)
            logger.info(f"Passive: Detected {len(targets)} target(s)")
            return targets
            
        except Exception as e:
            logger.error(f"Passive radar error: {e}")
            return []
    
    def _process_passive_radar(self, reference: np.ndarray,
                              surveillance: np.ndarray) -> List[Target]:
        """Process passive radar signals"""
        targets = []
        
        # Cross-correlation
        correlation = np.correlate(surveillance, reference, mode='valid')
        power = np.abs(correlation) ** 2
        
        # Detect peaks
        threshold = np.mean(power) + 6 * np.std(power)
        peaks = np.where(power > threshold)[0]
        
        for peak_idx in peaks[:5]:
            # Calculate bistatic range
            time_delay = peak_idx / self.config.sample_rate
            bistatic_range = self.C * time_delay
            
            # Estimate monostatic range (simplified)
            target_range = bistatic_range / 2
            
            # Doppler estimation
            doppler = self._estimate_doppler(surveillance, peak_idx)
            velocity = (doppler * self.C) / (2 * self.config.frequency)
            
            # Calculate SNR
            noise_floor = np.median(power)
            signal_power = power[peak_idx]
            snr = 10 * np.log10(signal_power / noise_floor)
            
            target = Target(
                range_m=target_range,
                velocity_mps=velocity,
                azimuth_deg=0.0,
                rcs=0.0,  # Cannot estimate RCS in passive mode
                timestamp=datetime.now(),
                snr=snr
            )
            
            targets.append(target)
        
        return targets
    
    def _merge_targets(self, targets: List[Target]) -> List[Target]:
        """Merge duplicate targets"""
        if len(targets) <= 1:
            return targets
        
        merged = []
        used = set()
        
        for i, target1 in enumerate(targets):
            if i in used:
                continue
            
            # Find similar targets
            similar = [target1]
            for j, target2 in enumerate(targets[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check if targets are similar (within 10m range, 5 m/s velocity)
                if (abs(target1.range_m - target2.range_m) < 10 and
                    abs(target1.velocity_mps - target2.velocity_mps) < 5):
                    similar.append(target2)
                    used.add(j)
            
            # Average similar targets
            avg_target = Target(
                range_m=np.mean([t.range_m for t in similar]),
                velocity_mps=np.mean([t.velocity_mps for t in similar]),
                azimuth_deg=np.mean([t.azimuth_deg for t in similar]),
                rcs=np.mean([t.rcs for t in similar]),
                timestamp=similar[0].timestamp,
                snr=np.mean([t.snr for t in similar])
            )
            
            merged.append(avg_target)
            used.add(i)
        
        return merged
    
    def track_target(self, target: Target, duration: float = 10.0) -> List[Target]:
        """
        Track specific target over time
        
        Args:
            target: Initial target to track
            duration: Tracking duration in seconds
            
        Returns:
            List of target positions over time
        """
        try:
            logger.info(f"Tracking target at {target.range_m:.1f}m...")
            
            track = [target]
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < duration:
                # Get new measurement
                if self.config.radar_type == "fmcw":
                    new_targets = self.fmcw_radar(duration=0.1)
                else:
                    new_targets = self.pulse_radar(duration=0.1)
                
                # Find closest target to previous position
                if new_targets:
                    closest = min(
                        new_targets,
                        key=lambda t: abs(t.range_m - track[-1].range_m)
                    )
                    track.append(closest)
            
            logger.info(f"Tracked target for {len(track)} measurements")
            return track
            
        except Exception as e:
            logger.error(f"Target tracking error: {e}")
            return [target]
    
    def get_detected_targets(self) -> List[Target]:
        """Get all detected targets"""
        return self.detected_targets
    
    def stop(self):
        """Stop radar operations"""
        self.is_running = False
        self.hw.stop_transmission()
        logger.info("Radar systems stopped")

def main():
    """Test radar systems"""
    from core.hardware import HardwareController
    
    # Initialize hardware
    hw = HardwareController()
    if not hw.connect():
        print("Failed to connect to BladeRF")
        return
    
    # Create radar system
    radar = RadarSystems(hw)
    
    # Configure for X-band (10 GHz) - Note: BladeRF may not reach this frequency
    # In practice, use lower frequencies within BladeRF range
    config = RadarConfig(
        frequency=2_400_000_000,  # 2.4 GHz (achievable with BladeRF)
        radar_type="fmcw"
    )
    
    if not radar.configure(config):
        print("Configuration failed")
        return
    
    print("RF Arsenal OS - Radar Systems")
    print("=" * 50)
    
    # FMCW Radar demo
    print("\nRunning FMCW radar (1s)...")
    targets = radar.fmcw_radar(duration=1.0)
    
    print(f"\nDetected {len(targets)} target(s):")
    for i, target in enumerate(targets, 1):
        print(f"{i}. Range: {target.range_m:.1f}m, "
              f"Velocity: {target.velocity_mps:.1f} m/s, "
              f"RCS: {target.rcs:.1f} dBsm, "
              f"SNR: {target.snr:.1f} dB")
    
    radar.stop()
    hw.disconnect()

if __name__ == "__main__":
    main()
