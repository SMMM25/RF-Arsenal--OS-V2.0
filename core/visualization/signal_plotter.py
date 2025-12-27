"""
RF Arsenal OS - Signal Plotter and Timeseries Display
Real-time signal visualization for time-domain analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import deque


class PlotMode(Enum):
    """Signal plot modes"""
    REALTIME = "realtime"
    TRIGGERED = "triggered"
    SINGLE_SHOT = "single_shot"
    ROLL = "roll"


class TriggerEdge(Enum):
    """Trigger edge types"""
    RISING = "rising"
    FALLING = "falling"
    EITHER = "either"


@dataclass
class TriggerSettings:
    """Oscilloscope-style trigger settings"""
    enabled: bool = False
    level: float = 0.0
    edge: TriggerEdge = TriggerEdge.RISING
    holdoff_samples: int = 100
    pre_trigger_samples: int = 500
    channel: str = "I"  # I, Q, or magnitude


@dataclass
class Marker:
    """Measurement marker on signal plot"""
    marker_id: str
    x_position: float  # Time or sample index
    y_value: float     # Signal value at marker
    label: str = ""
    color: str = "#FF0000"


class SignalPlotter:
    """
    Production-grade signal plotter for time-domain analysis.
    
    Features:
    - Real-time IQ signal plotting
    - Oscilloscope-style triggering
    - Multiple measurement channels
    - Markers and measurements
    - Amplitude and time measurements
    - Export capabilities
    """
    
    def __init__(self,
                 sample_rate: float = 1e6,
                 buffer_size: int = 10000,
                 plot_mode: PlotMode = PlotMode.REALTIME):
        """
        Initialize signal plotter.
        
        Args:
            sample_rate: Sample rate in Hz
            buffer_size: Buffer size in samples
            plot_mode: Plot update mode
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.plot_mode = plot_mode
        
        # Signal buffers
        self._i_buffer = deque(maxlen=buffer_size)
        self._q_buffer = deque(maxlen=buffer_size)
        self._timestamp_buffer = deque(maxlen=buffer_size)
        
        # Trigger
        self.trigger = TriggerSettings()
        self._triggered = False
        self._trigger_index = 0
        self._waiting_for_trigger = False
        
        # Display settings
        self.time_scale = 1e-3  # Seconds per division
        self.amplitude_scale = 1.0  # Volts per division
        self.time_offset = 0.0
        self.amplitude_offset = 0.0
        
        # Markers
        self._markers: Dict[str, Marker] = {}
        
        # Measurements
        self._measurements: Dict[str, float] = {}
        
        # Callbacks
        self._update_callbacks: List[Callable] = []
        
        # Lock
        self._lock = threading.Lock()
        
        # Statistics
        self._sample_count = 0
        self._start_time = time.time()
        
    def add_samples(self, iq_samples: np.ndarray) -> None:
        """
        Add IQ samples to the plotter.
        
        Args:
            iq_samples: Complex IQ samples
        """
        timestamp = time.time()
        
        with self._lock:
            for sample in iq_samples:
                self._i_buffer.append(float(np.real(sample)))
                self._q_buffer.append(float(np.imag(sample)))
                self._timestamp_buffer.append(timestamp)
                self._sample_count += 1
                
            # Check for trigger
            if self.trigger.enabled and not self._triggered:
                self._check_trigger()
                
        # Update measurements
        self._update_measurements()
        
        # Notify callbacks
        for callback in self._update_callbacks:
            callback(self.get_plot_data())
            
    def _check_trigger(self) -> None:
        """Check for trigger condition"""
        if len(self._i_buffer) < self.trigger.pre_trigger_samples + 10:
            return
            
        # Select trigger channel
        if self.trigger.channel == "I":
            data = list(self._i_buffer)
        elif self.trigger.channel == "Q":
            data = list(self._q_buffer)
        else:  # magnitude
            data = [np.sqrt(i**2 + q**2) 
                    for i, q in zip(self._i_buffer, self._q_buffer)]
            
        # Look for trigger edge
        for i in range(len(data) - 1):
            if self.trigger.edge == TriggerEdge.RISING:
                if data[i] < self.trigger.level <= data[i + 1]:
                    self._triggered = True
                    self._trigger_index = i
                    return
            elif self.trigger.edge == TriggerEdge.FALLING:
                if data[i] > self.trigger.level >= data[i + 1]:
                    self._triggered = True
                    self._trigger_index = i
                    return
            else:  # Either edge
                if ((data[i] < self.trigger.level <= data[i + 1]) or
                    (data[i] > self.trigger.level >= data[i + 1])):
                    self._triggered = True
                    self._trigger_index = i
                    return
                    
    def _update_measurements(self) -> None:
        """Update signal measurements"""
        with self._lock:
            if len(self._i_buffer) < 100:
                return
                
            i_data = np.array(list(self._i_buffer))
            q_data = np.array(list(self._q_buffer))
            magnitude = np.sqrt(i_data**2 + q_data**2)
            
            # Amplitude measurements
            self._measurements["i_peak_peak"] = float(np.max(i_data) - np.min(i_data))
            self._measurements["q_peak_peak"] = float(np.max(q_data) - np.min(q_data))
            self._measurements["i_rms"] = float(np.sqrt(np.mean(i_data**2)))
            self._measurements["q_rms"] = float(np.sqrt(np.mean(q_data**2)))
            self._measurements["magnitude_peak"] = float(np.max(magnitude))
            self._measurements["magnitude_rms"] = float(np.sqrt(np.mean(magnitude**2)))
            self._measurements["magnitude_avg"] = float(np.mean(magnitude))
            
            # DC offset
            self._measurements["i_dc"] = float(np.mean(i_data))
            self._measurements["q_dc"] = float(np.mean(q_data))
            
            # Frequency estimation (zero-crossing)
            zero_crossings = np.where(np.diff(np.signbit(i_data - np.mean(i_data))))[0]
            if len(zero_crossings) >= 2:
                periods = np.diff(zero_crossings)
                avg_period_samples = np.mean(periods) * 2  # Full period
                if avg_period_samples > 0:
                    self._measurements["frequency_hz"] = float(
                        self.sample_rate / avg_period_samples
                    )
                    
            # Phase (average phase angle)
            phases = np.arctan2(q_data, i_data)
            self._measurements["phase_deg"] = float(np.mean(phases) * 180 / np.pi)
            
    def add_marker(self, marker_id: str, x_position: float, label: str = "") -> None:
        """Add a measurement marker"""
        with self._lock:
            # Interpolate y value at x position
            sample_index = int(x_position * self.sample_rate)
            sample_index = np.clip(sample_index, 0, len(self._i_buffer) - 1)
            
            if self._i_buffer:
                y_value = list(self._i_buffer)[sample_index]
            else:
                y_value = 0.0
                
            self._markers[marker_id] = Marker(
                marker_id=marker_id,
                x_position=x_position,
                y_value=y_value,
                label=label
            )
            
    def remove_marker(self, marker_id: str) -> None:
        """Remove a marker"""
        if marker_id in self._markers:
            del self._markers[marker_id]
            
    def measure_between_markers(self, marker1_id: str, marker2_id: str) -> Dict[str, float]:
        """Measure delta between two markers"""
        if marker1_id not in self._markers or marker2_id not in self._markers:
            return {"error": "Markers not found"}
            
        m1 = self._markers[marker1_id]
        m2 = self._markers[marker2_id]
        
        return {
            "delta_time": abs(m2.x_position - m1.x_position),
            "delta_amplitude": abs(m2.y_value - m1.y_value),
            "frequency_hz": 1.0 / abs(m2.x_position - m1.x_position) if abs(m2.x_position - m1.x_position) > 0 else 0
        }
        
    def get_plot_data(self) -> Dict[str, Any]:
        """Get current plot data for visualization"""
        with self._lock:
            i_data = list(self._i_buffer)
            q_data = list(self._q_buffer)
            
            # Create time axis
            num_samples = len(i_data)
            time_axis = [i / self.sample_rate for i in range(num_samples)]
            
            return {
                "i_data": i_data,
                "q_data": q_data,
                "magnitude": [np.sqrt(i**2 + q**2) for i, q in zip(i_data, q_data)],
                "phase_deg": [np.arctan2(q, i) * 180 / np.pi for i, q in zip(i_data, q_data)],
                "time_axis": time_axis,
                "sample_rate": self.sample_rate,
                "measurements": self._measurements.copy(),
                "markers": {k: {"x": v.x_position, "y": v.y_value, "label": v.label}
                           for k, v in self._markers.items()},
                "trigger": {
                    "enabled": self.trigger.enabled,
                    "level": self.trigger.level,
                    "triggered": self._triggered,
                    "edge": self.trigger.edge.value
                },
                "plot_mode": self.plot_mode.value,
                "sample_count": self._sample_count
            }
            
    def get_measurements(self) -> Dict[str, float]:
        """Get current measurements"""
        return self._measurements.copy()
        
    def set_trigger(self, enabled: bool = True, level: float = 0.0,
                   edge: TriggerEdge = TriggerEdge.RISING,
                   channel: str = "I") -> None:
        """Configure trigger settings"""
        self.trigger.enabled = enabled
        self.trigger.level = level
        self.trigger.edge = edge
        self.trigger.channel = channel
        self._triggered = False
        
    def force_trigger(self) -> None:
        """Force immediate trigger"""
        self._triggered = True
        self._trigger_index = len(self._i_buffer) // 2
        
    def run_single(self) -> None:
        """Run single acquisition"""
        self.plot_mode = PlotMode.SINGLE_SHOT
        self._triggered = False
        self._waiting_for_trigger = True
        
    def clear(self) -> None:
        """Clear all buffers"""
        with self._lock:
            self._i_buffer.clear()
            self._q_buffer.clear()
            self._timestamp_buffer.clear()
            self._triggered = False
            self._sample_count = 0
            
    def register_callback(self, callback: Callable) -> None:
        """Register callback for plot updates"""
        self._update_callbacks.append(callback)
        
    def export_csv(self, filepath: str) -> bool:
        """Export data to CSV"""
        try:
            with self._lock:
                with open(filepath, 'w') as f:
                    f.write("time_s,i_value,q_value,magnitude,phase_deg\n")
                    for i, (iv, qv) in enumerate(zip(self._i_buffer, self._q_buffer)):
                        t = i / self.sample_rate
                        mag = np.sqrt(iv**2 + qv**2)
                        phase = np.arctan2(qv, iv) * 180 / np.pi
                        f.write(f"{t},{iv},{qv},{mag},{phase}\n")
            return True
        except Exception:
            return False


class TimeseriesDisplay:
    """
    Multi-channel timeseries display for signal comparison.
    """
    
    def __init__(self, num_channels: int = 4, buffer_size: int = 10000):
        """
        Initialize timeseries display.
        
        Args:
            num_channels: Number of signal channels
            buffer_size: Buffer size per channel
        """
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        
        # Channel buffers
        self._channels: Dict[str, deque] = {}
        self._channel_settings: Dict[str, Dict] = {}
        
        # Initialize default channels
        for i in range(num_channels):
            channel_name = f"CH{i + 1}"
            self._channels[channel_name] = deque(maxlen=buffer_size)
            self._channel_settings[channel_name] = {
                "enabled": True,
                "color": self._default_colors()[i % len(self._default_colors())],
                "scale": 1.0,
                "offset": 0.0,
                "label": channel_name
            }
            
        # Sample rate
        self.sample_rate = 1e6
        
        # Lock
        self._lock = threading.Lock()
        
    def _default_colors(self) -> List[str]:
        """Default channel colors"""
        return ["#FFD700", "#00BFFF", "#FF69B4", "#32CD32", "#FF6347", "#9370DB"]
        
    def add_samples(self, channel: str, samples: np.ndarray) -> None:
        """Add samples to a channel"""
        if channel not in self._channels:
            return
            
        with self._lock:
            for sample in samples:
                self._channels[channel].append(float(sample))
                
    def add_channel(self, name: str, color: str = "#FFFFFF") -> None:
        """Add a new channel"""
        if name in self._channels:
            return
            
        self._channels[name] = deque(maxlen=self.buffer_size)
        self._channel_settings[name] = {
            "enabled": True,
            "color": color,
            "scale": 1.0,
            "offset": 0.0,
            "label": name
        }
        
    def set_channel_settings(self, channel: str, **settings) -> None:
        """Update channel settings"""
        if channel in self._channel_settings:
            self._channel_settings[channel].update(settings)
            
    def get_display_data(self) -> Dict[str, Any]:
        """Get all channel data for display"""
        with self._lock:
            channels_data = {}
            
            for name, buffer in self._channels.items():
                settings = self._channel_settings[name]
                data = list(buffer)
                
                channels_data[name] = {
                    "data": data,
                    "time_axis": [i / self.sample_rate for i in range(len(data))],
                    "settings": settings
                }
                
            return {
                "channels": channels_data,
                "sample_rate": self.sample_rate,
                "num_channels": len(self._channels)
            }
            
    def clear_channel(self, channel: str) -> None:
        """Clear a specific channel"""
        if channel in self._channels:
            self._channels[channel].clear()
            
    def clear_all(self) -> None:
        """Clear all channels"""
        for channel in self._channels.values():
            channel.clear()
