#!/usr/bin/env python3
"""
RF Arsenal OS - Main GUI
Professional PyQt5 interface for RF security operations
Integrates all 17 modules with visual controls
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QGroupBox, QGridLayout, QCheckBox, QLineEdit, QProgressBar,
    QTableWidget, QTableWidgetItem, QMessageBox, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPalette
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import json
from datetime import datetime

# Import RF Arsenal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.hardware import BladeRFController, FrequencyBand
from core.stealth import StealthController
from core.emergency import EmergencyProtocol


class SpectrumCanvas(FigureCanvas):
    """Real-time spectrum display widget"""
    
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Configure plot
        self.axes.set_xlabel('Frequency (MHz)')
        self.axes.set_ylabel('Power (dBm)')
        self.axes.set_title('Real-time Spectrum')
        self.axes.grid(True, alpha=0.3)
        self.fig.tight_layout()
        
        # Initialize empty plot
        self.line, = self.axes.plot([], [], 'b-', linewidth=1)
        
    def update_spectrum(self, frequencies, powers):
        """Update spectrum display"""
        self.line.set_data(frequencies / 1e6, powers)
        self.axes.relim()
        self.axes.autoscale_view()
        self.draw()


class WaterfallCanvas(FigureCanvas):
    """Waterfall display widget"""
    
    def __init__(self, parent=None, width=8, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Configure plot
        self.axes.set_xlabel('Frequency (MHz)')
        self.axes.set_ylabel('Time')
        self.axes.set_title('Waterfall Display')
        self.fig.tight_layout()
        
        # Initialize waterfall data
        self.waterfall_data = []
        self.max_history = 100
        
    def update_waterfall(self, frequencies, powers):
        """Add new spectrum line to waterfall"""
        self.waterfall_data.append(powers)
        if len(self.waterfall_data) > self.max_history:
            self.waterfall_data.pop(0)
        
        self.axes.clear()
        if self.waterfall_data:
            data = np.array(self.waterfall_data)
            self.axes.imshow(data, aspect='auto', cmap='viridis',
                           extent=[frequencies[0]/1e6, frequencies[-1]/1e6, 
                                 0, len(self.waterfall_data)])
        self.axes.set_xlabel('Frequency (MHz)')
        self.axes.set_ylabel('Time')
        self.draw()


class HardwareControlPanel(QWidget):
    """Hardware control and status panel"""
    
    def __init__(self):
        super().__init__()
        self.hardware = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Hardware connection group
        hw_group = QGroupBox("BladeRF Hardware")
        hw_layout = QGridLayout()
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_hardware)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_hardware)
        self.disconnect_btn.setEnabled(False)
        
        self.status_label = QLabel("Status: Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        hw_layout.addWidget(QLabel("Device:"), 0, 0)
        hw_layout.addWidget(self.connect_btn, 0, 1)
        hw_layout.addWidget(self.disconnect_btn, 0, 2)
        hw_layout.addWidget(self.status_label, 1, 0, 1, 3)
        
        hw_group.setLayout(hw_layout)
        layout.addWidget(hw_group)
        
        # Frequency control
        freq_group = QGroupBox("Frequency Control")
        freq_layout = QGridLayout()
        
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(70, 6000)
        self.freq_spin.setValue(2400)
        self.freq_spin.setSuffix(" MHz")
        self.freq_spin.setDecimals(3)
        
        self.bandwidth_combo = QComboBox()
        self.bandwidth_combo.addItems(["1.5 MHz", "5 MHz", "10 MHz", "20 MHz", "28 MHz", "40 MHz", "56 MHz"])
        self.bandwidth_combo.setCurrentText("20 MHz")
        
        freq_layout.addWidget(QLabel("Center Frequency:"), 0, 0)
        freq_layout.addWidget(self.freq_spin, 0, 1)
        freq_layout.addWidget(QLabel("Bandwidth:"), 1, 0)
        freq_layout.addWidget(self.bandwidth_combo, 1, 1)
        
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)
        
        # TX/RX control
        txrx_group = QGroupBox("TX/RX Control")
        txrx_layout = QGridLayout()
        
        self.tx_power_spin = QSpinBox()
        self.tx_power_spin.setRange(-20, 30)
        self.tx_power_spin.setValue(10)
        self.tx_power_spin.setSuffix(" dBm")
        
        self.rx_gain_spin = QSpinBox()
        self.rx_gain_spin.setRange(0, 60)
        self.rx_gain_spin.setValue(40)
        self.rx_gain_spin.setSuffix(" dB")
        
        self.tx_enable = QCheckBox("TX Enable")
        self.rx_enable = QCheckBox("RX Enable")
        self.rx_enable.setChecked(True)
        
        txrx_layout.addWidget(QLabel("TX Power:"), 0, 0)
        txrx_layout.addWidget(self.tx_power_spin, 0, 1)
        txrx_layout.addWidget(self.tx_enable, 0, 2)
        txrx_layout.addWidget(QLabel("RX Gain:"), 1, 0)
        txrx_layout.addWidget(self.rx_gain_spin, 1, 1)
        txrx_layout.addWidget(self.rx_enable, 1, 2)
        
        txrx_group.setLayout(txrx_layout)
        layout.addWidget(txrx_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def connect_hardware(self):
        """Connect to BladeRF hardware"""
        try:
            self.hardware = BladeRFController()
            self.status_label.setText("Status: Connected")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            QMessageBox.information(self, "Success", "BladeRF connected successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect: {str(e)}")
            
    def disconnect_hardware(self):
        """Disconnect from hardware"""
        if self.hardware:
            self.hardware = None
        self.status_label.setText("Status: Disconnected")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)


class CellularPanel(QWidget):
    """Cellular operations (2G/3G/4G/5G)"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Generation selector
        gen_group = QGroupBox("Cellular Generation")
        gen_layout = QHBoxLayout()
        
        self.gen_combo = QComboBox()
        self.gen_combo.addItems(["2G/GSM", "3G/UMTS", "4G/LTE", "5G/NR"])
        self.gen_combo.currentTextChanged.connect(self.update_generation)
        
        gen_layout.addWidget(QLabel("Generation:"))
        gen_layout.addWidget(self.gen_combo)
        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)
        
        # Configuration
        config_group = QGroupBox("Base Station Configuration")
        config_layout = QGridLayout()
        
        self.mcc_edit = QLineEdit("310")  # USA
        self.mnc_edit = QLineEdit("260")  # T-Mobile
        self.band_combo = QComboBox()
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 3000)
        
        config_layout.addWidget(QLabel("MCC:"), 0, 0)
        config_layout.addWidget(self.mcc_edit, 0, 1)
        config_layout.addWidget(QLabel("MNC:"), 0, 2)
        config_layout.addWidget(self.mnc_edit, 0, 3)
        config_layout.addWidget(QLabel("Band:"), 1, 0)
        config_layout.addWidget(self.band_combo, 1, 1)
        config_layout.addWidget(QLabel("Channel:"), 1, 2)
        config_layout.addWidget(self.channel_spin, 1, 3)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Features
        features_group = QGroupBox("Features")
        features_layout = QVBoxLayout()
        
        self.imsi_catcher_cb = QCheckBox("IMSI Catcher")
        self.silent_sms_cb = QCheckBox("Silent SMS")
        self.location_track_cb = QCheckBox("Location Tracking")
        self.call_intercept_cb = QCheckBox("Call Interception")
        
        features_layout.addWidget(self.imsi_catcher_cb)
        features_layout.addWidget(self.silent_sms_cb)
        features_layout.addWidget(self.location_track_cb)
        features_layout.addWidget(self.call_intercept_cb)
        
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Base Station")
        self.start_btn.clicked.connect(self.start_basestation)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_basestation)
        self.stop_btn.setEnabled(False)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        
        # Device table
        self.device_table = QTableWidget()
        self.device_table.setColumnCount(5)
        self.device_table.setHorizontalHeaderLabels(["IMSI", "IMEI", "Signal", "Status", "Time"])
        layout.addWidget(QLabel("Connected Devices:"))
        layout.addWidget(self.device_table)
        
        self.setLayout(layout)
        self.update_generation("2G/GSM")
        
    def update_generation(self, gen):
        """Update UI based on selected generation"""
        self.band_combo.clear()
        if gen == "2G/GSM":
            self.band_combo.addItems(["GSM-900", "GSM-1800", "GSM-850", "GSM-1900"])
        elif gen == "3G/UMTS":
            self.band_combo.addItems(["Band 1 (2100)", "Band 2 (1900)", "Band 5 (850)"])
        elif gen == "4G/LTE":
            self.band_combo.addItems(["Band 1 (2100)", "Band 3 (1800)", "Band 7 (2600)", "Band 20 (800)"])
        elif gen == "5G/NR":
            self.band_combo.addItems(["n78 (3.5 GHz)", "n77 (3.7 GHz)", "n79 (4.7 GHz)"])
            
    def start_basestation(self):
        """Start cellular base station"""
        gen = self.gen_combo.currentText()
        QMessageBox.information(self, "Starting", f"Starting {gen} base station...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
    def stop_basestation(self):
        """Stop base station"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


class AttackPanel(QWidget):
    """Attack modules (WiFi, GPS, Drone, Jamming)"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Attack type selector
        attack_group = QGroupBox("Attack Module")
        attack_layout = QHBoxLayout()
        
        self.attack_combo = QComboBox()
        self.attack_combo.addItems([
            "WiFi Attacks",
            "GPS Spoofing", 
            "Drone Warfare",
            "Electronic Warfare Jamming"
        ])
        self.attack_combo.currentTextChanged.connect(self.update_attack_panel)
        
        attack_layout.addWidget(QLabel("Module:"))
        attack_layout.addWidget(self.attack_combo)
        attack_group.setLayout(attack_layout)
        layout.addWidget(attack_group)
        
        # Stacked widget for different attack panels
        self.attack_stack = QWidget()
        self.attack_stack_layout = QVBoxLayout()
        self.attack_stack.setLayout(self.attack_stack_layout)
        layout.addWidget(self.attack_stack)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.start_attack_btn = QPushButton("Start Attack")
        self.start_attack_btn.clicked.connect(self.start_attack)
        self.stop_attack_btn = QPushButton("Stop Attack")
        self.stop_attack_btn.clicked.connect(self.stop_attack)
        self.stop_attack_btn.setEnabled(False)
        
        btn_layout.addWidget(self.start_attack_btn)
        btn_layout.addWidget(self.stop_attack_btn)
        layout.addLayout(btn_layout)
        
        # Results area
        layout.addWidget(QLabel("Attack Results:"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        self.setLayout(layout)
        self.update_attack_panel("WiFi Attacks")
        
    def update_attack_panel(self, attack_type):
        """Update panel based on selected attack"""
        # Clear current panel
        for i in reversed(range(self.attack_stack_layout.count())): 
            self.attack_stack_layout.itemAt(i).widget().setParent(None)
        
        if attack_type == "WiFi Attacks":
            self.create_wifi_panel()
        elif attack_type == "GPS Spoofing":
            self.create_gps_panel()
        elif attack_type == "Drone Warfare":
            self.create_drone_panel()
        elif attack_type == "Electronic Warfare Jamming":
            self.create_jamming_panel()
            
    def create_wifi_panel(self):
        """WiFi attack configuration"""
        group = QGroupBox("WiFi Attack Configuration")
        layout = QGridLayout()
        
        self.wifi_attack_combo = QComboBox()
        self.wifi_attack_combo.addItems([
            "Deauthentication",
            "Evil Twin",
            "WPS Bruteforce",
            "Beacon Flood"
        ])
        
        self.wifi_channel_spin = QSpinBox()
        self.wifi_channel_spin.setRange(1, 14)
        self.wifi_channel_spin.setValue(6)
        
        self.target_ssid_edit = QLineEdit()
        self.target_mac_edit = QLineEdit()
        
        layout.addWidget(QLabel("Attack Type:"), 0, 0)
        layout.addWidget(self.wifi_attack_combo, 0, 1)
        layout.addWidget(QLabel("Channel:"), 1, 0)
        layout.addWidget(self.wifi_channel_spin, 1, 1)
        layout.addWidget(QLabel("Target SSID:"), 2, 0)
        layout.addWidget(self.target_ssid_edit, 2, 1)
        layout.addWidget(QLabel("Target MAC:"), 3, 0)
        layout.addWidget(self.target_mac_edit, 3, 1)
        
        group.setLayout(layout)
        self.attack_stack_layout.addWidget(group)
        
    def create_gps_panel(self):
        """GPS spoofing configuration"""
        group = QGroupBox("GPS Spoofing Configuration")
        layout = QGridLayout()
        
        self.gps_lat_spin = QDoubleSpinBox()
        self.gps_lat_spin.setRange(-90, 90)
        self.gps_lat_spin.setValue(37.7749)
        self.gps_lat_spin.setDecimals(6)
        
        self.gps_lon_spin = QDoubleSpinBox()
        self.gps_lon_spin.setRange(-180, 180)
        self.gps_lon_spin.setValue(-122.4194)
        self.gps_lon_spin.setDecimals(6)
        
        self.gps_alt_spin = QDoubleSpinBox()
        self.gps_alt_spin.setRange(0, 10000)
        self.gps_alt_spin.setValue(100)
        self.gps_alt_spin.setSuffix(" m")
        
        self.gps_satellites_spin = QSpinBox()
        self.gps_satellites_spin.setRange(4, 32)
        self.gps_satellites_spin.setValue(8)
        
        layout.addWidget(QLabel("Latitude:"), 0, 0)
        layout.addWidget(self.gps_lat_spin, 0, 1)
        layout.addWidget(QLabel("Longitude:"), 1, 0)
        layout.addWidget(self.gps_lon_spin, 1, 1)
        layout.addWidget(QLabel("Altitude:"), 2, 0)
        layout.addWidget(self.gps_alt_spin, 2, 1)
        layout.addWidget(QLabel("Satellites:"), 3, 0)
        layout.addWidget(self.gps_satellites_spin, 3, 1)
        
        group.setLayout(layout)
        self.attack_stack_layout.addWidget(group)
        
    def create_drone_panel(self):
        """Drone warfare configuration"""
        group = QGroupBox("Drone Warfare Configuration")
        layout = QGridLayout()
        
        self.drone_mode_combo = QComboBox()
        self.drone_mode_combo.addItems([
            "Detection Only",
            "Jamming",
            "GPS Spoofing",
            "Hijacking",
            "Force Landing"
        ])
        
        self.drone_freq_combo = QComboBox()
        self.drone_freq_combo.addItems(["2.4 GHz", "5.8 GHz", "Both"])
        
        self.drone_protocol_combo = QComboBox()
        self.drone_protocol_combo.addItems(["DJI OcuSync", "MAVLink", "Parrot", "Auto-detect"])
        
        layout.addWidget(QLabel("Mode:"), 0, 0)
        layout.addWidget(self.drone_mode_combo, 0, 1)
        layout.addWidget(QLabel("Frequency:"), 1, 0)
        layout.addWidget(self.drone_freq_combo, 1, 1)
        layout.addWidget(QLabel("Protocol:"), 2, 0)
        layout.addWidget(self.drone_protocol_combo, 2, 1)
        
        group.setLayout(layout)
        self.attack_stack_layout.addWidget(group)
        
    def create_jamming_panel(self):
        """Electronic warfare jamming configuration"""
        group = QGroupBox("Jamming Configuration")
        layout = QGridLayout()
        
        self.jam_mode_combo = QComboBox()
        self.jam_mode_combo.addItems([
            "Noise Jamming",
            "Tone Jamming",
            "Sweep Jamming",
            "Pulse Jamming",
            "Barrage Jamming"
        ])
        
        self.jam_start_freq = QDoubleSpinBox()
        self.jam_start_freq.setRange(70, 6000)
        self.jam_start_freq.setValue(2400)
        self.jam_start_freq.setSuffix(" MHz")
        
        self.jam_end_freq = QDoubleSpinBox()
        self.jam_end_freq.setRange(70, 6000)
        self.jam_end_freq.setValue(2500)
        self.jam_end_freq.setSuffix(" MHz")
        
        self.jam_power_spin = QSpinBox()
        self.jam_power_spin.setRange(0, 30)
        self.jam_power_spin.setValue(20)
        self.jam_power_spin.setSuffix(" dBm")
        
        layout.addWidget(QLabel("Mode:"), 0, 0)
        layout.addWidget(self.jam_mode_combo, 0, 1)
        layout.addWidget(QLabel("Start Freq:"), 1, 0)
        layout.addWidget(self.jam_start_freq, 1, 1)
        layout.addWidget(QLabel("End Freq:"), 2, 0)
        layout.addWidget(self.jam_end_freq, 2, 1)
        layout.addWidget(QLabel("Power:"), 3, 0)
        layout.addWidget(self.jam_power_spin, 3, 1)
        
        group.setLayout(layout)
        self.attack_stack_layout.addWidget(group)
        
    def start_attack(self):
        """Start selected attack"""
        attack_type = self.attack_combo.currentText()
        self.results_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {attack_type}...")
        self.start_attack_btn.setEnabled(False)
        self.stop_attack_btn.setEnabled(True)
        
    def stop_attack(self):
        """Stop attack"""
        self.results_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Attack stopped")
        self.start_attack_btn.setEnabled(True)
        self.stop_attack_btn.setEnabled(False)


class AnalysisPanel(QWidget):
    """Analysis modules (Spectrum, SIGINT, Radar)"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Module selector
        module_group = QGroupBox("Analysis Module")
        module_layout = QHBoxLayout()
        
        self.module_combo = QComboBox()
        self.module_combo.addItems([
            "Spectrum Analyzer",
            "SIGINT Engine",
            "Radar Systems"
        ])
        
        module_layout.addWidget(QLabel("Module:"))
        module_layout.addWidget(self.module_combo)
        module_group.setLayout(module_layout)
        layout.addWidget(module_group)
        
        # Spectrum display
        self.spectrum_canvas = SpectrumCanvas(self, width=10, height=3)
        layout.addWidget(self.spectrum_canvas)
        
        # Waterfall display
        self.waterfall_canvas = WaterfallCanvas(self, width=10, height=2)
        layout.addWidget(self.waterfall_canvas)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.start_analysis_btn = QPushButton("Start Analysis")
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        self.stop_analysis_btn = QPushButton("Stop")
        self.stop_analysis_btn.clicked.connect(self.stop_analysis)
        self.stop_analysis_btn.setEnabled(False)
        
        btn_layout.addWidget(self.start_analysis_btn)
        btn_layout.addWidget(self.stop_analysis_btn)
        layout.addLayout(btn_layout)
        
        # Signal table
        self.signal_table = QTableWidget()
        self.signal_table.setColumnCount(6)
        self.signal_table.setHorizontalHeaderLabels([
            "Frequency", "Power", "Bandwidth", "Modulation", "Type", "Time"
        ])
        layout.addWidget(QLabel("Detected Signals:"))
        layout.addWidget(self.signal_table)
        
        self.setLayout(layout)
        
        # Update timer for displays
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        
    def start_analysis(self):
        """Start analysis"""
        self.start_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.setEnabled(True)
        self.update_timer.start(100)  # Update every 100ms
        
    def stop_analysis(self):
        """Stop analysis"""
        self.start_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setEnabled(False)
        self.update_timer.stop()
        
    def update_displays(self):
        """Update spectrum and waterfall displays with simulated data"""
        # Generate simulated spectrum data
        freqs = np.linspace(2400e6, 2500e6, 1000)
        powers = -80 + 20 * np.random.randn(1000)
        
        # Add some peaks
        peaks = [2.42e9, 2.45e9, 2.48e9]
        for peak in peaks:
            idx = np.argmin(np.abs(freqs - peak))
            powers[idx-10:idx+10] += 30
        
        self.spectrum_canvas.update_spectrum(freqs, powers)
        self.waterfall_canvas.update_waterfall(freqs, powers)


class AIControlPanel(QWidget):
    """AI natural language control interface"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Voice control
        voice_group = QGroupBox("Voice Control")
        voice_layout = QHBoxLayout()
        
        self.voice_btn = QPushButton("🎤 Hold to Speak")
        self.voice_btn.setMinimumHeight(50)
        self.voice_enabled = QCheckBox("Enable Voice")
        
        voice_layout.addWidget(self.voice_btn)
        voice_layout.addWidget(self.voice_enabled)
        voice_group.setLayout(voice_layout)
        layout.addWidget(voice_group)
        
        # Text command
        cmd_group = QGroupBox("Text Command")
        cmd_layout = QVBoxLayout()
        
        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText("Enter command (e.g., 'start 5g base station with imsi catcher')")
        self.cmd_input.returnPressed.connect(self.execute_command)
        
        self.execute_btn = QPushButton("Execute Command")
        self.execute_btn.clicked.connect(self.execute_command)
        
        cmd_layout.addWidget(self.cmd_input)
        cmd_layout.addWidget(self.execute_btn)
        cmd_group.setLayout(cmd_layout)
        layout.addWidget(cmd_group)
        
        # Command history
        layout.addWidget(QLabel("Command History:"))
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        layout.addWidget(self.history_text)
        
        # Example commands
        examples_group = QGroupBox("Example Commands")
        examples_layout = QVBoxLayout()
        
        examples = [
            "start 5g base station with imsi catcher",
            "scan wifi networks on channel 6",
            "spoof gps to coordinates 37.7749, -122.4194",
            "jam drone frequencies",
            "analyze spectrum from 2.4 to 2.5 GHz"
        ]
        
        for example in examples:
            btn = QPushButton(example)
            btn.clicked.connect(lambda checked, cmd=example: self.set_command(cmd))
            examples_layout.addWidget(btn)
        
        examples_group.setLayout(examples_layout)
        layout.addWidget(examples_group)
        
        self.setLayout(layout)
        
    def set_command(self, command):
        """Set command in input field"""
        self.cmd_input.setText(command)
        
    def execute_command(self):
        """Execute natural language command"""
        command = self.cmd_input.text()
        if not command:
            return
            
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.history_text.append(f"[{timestamp}] > {command}")
        self.history_text.append(f"[{timestamp}] Parsing command...")
        
        # Simulate AI processing
        QTimer.singleShot(500, lambda: self.show_result(command))
        self.cmd_input.clear()
        
    def show_result(self, command):
        """Show command execution result"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.history_text.append(f"[{timestamp}] ✓ Command executed successfully\n")


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("RF Arsenal OS - Professional RF Security Platform")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Hardware control
        self.hw_panel = HardwareControlPanel()
        main_layout.addWidget(self.hw_panel, 1)
        
        # Right panel - Main tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(CellularPanel(), "📱 Cellular")
        self.tabs.addTab(AttackPanel(), "⚔️ Attacks")
        self.tabs.addTab(AnalysisPanel(), "📊 Analysis")
        self.tabs.addTab(AIControlPanel(), "🤖 AI Control")
        
        main_layout.addWidget(self.tabs, 4)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Apply dark theme
        self.apply_dark_theme()
        
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Save Configuration")
        file_menu.addAction("Load Configuration")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("System Monitor")
        tools_menu.addAction("Log Viewer")
        tools_menu.addAction("Export Data")
        
        # Security menu
        security_menu = menubar.addMenu("Security")
        security_menu.addAction("Enable Stealth Mode")
        security_menu.addAction("Emergency Shutdown")
        security_menu.addAction("Clear Evidence")
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("Documentation")
        help_menu.addAction("About")
        
    def apply_dark_theme(self):
        """Apply dark theme to application"""
        dark_stylesheet = """
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        QGroupBox {
            border: 2px solid #555555;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #0d47a1;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1565c0;
        }
        QPushButton:pressed {
            background-color: #0a3d91;
        }
        QPushButton:disabled {
            background-color: #555555;
            color: #888888;
        }
        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #3d3d3d;
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 5px;
            color: #ffffff;
        }
        QTableWidget {
            background-color: #2d2d2d;
            alternate-background-color: #3d3d3d;
            gridline-color: #555555;
        }
        QHeaderView::section {
            background-color: #1e1e1e;
            color: #ffffff;
            padding: 5px;
            border: 1px solid #555555;
        }
        QTabWidget::pane {
            border: 1px solid #555555;
        }
        QTabBar::tab {
            background-color: #2d2d2d;
            color: #ffffff;
            padding: 10px;
            border: 1px solid #555555;
        }
        QTabBar::tab:selected {
            background-color: #0d47a1;
        }
        """
        self.setStyleSheet(dark_stylesheet)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("RF Arsenal OS")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
