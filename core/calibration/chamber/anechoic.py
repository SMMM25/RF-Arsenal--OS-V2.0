"""
Anechoic Chamber Testing Module for RF Arsenal OS.

This module provides comprehensive anechoic chamber testing capabilities
including far-field, near-field, and compact range measurements per
IEEE 149 and CISPR 16-1-4 standards.

Features:
- Far-field antenna pattern measurements
- Near-field planar/cylindrical/spherical scanning
- Near-field to far-field transformation
- Quiet zone characterization
- Absorber performance verification
- Site VSWR measurements
- Cross-polarization discrimination

Standards:
- IEEE 149-1979: Antenna measurement techniques
- CISPR 16-1-4: Anechoic chamber requirements
- IEC 61000-4-3: Radiated immunity testing

Author: RF Arsenal Development Team
License: Proprietary - Calibration Sensitive
"""

import asyncio
import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy import signal as scipy_signal
from scipy import fft as scipy_fft
from scipy import interpolate
from scipy.ndimage import gaussian_filter

from . import (
    ChamberType,
    MeasurementType,
    CalibrationStatus,
    TestResult,
    ChamberSpecification,
    MeasurementPoint,
    ChamberInterface,
    ChamberException,
    MeasurementException,
)

logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Near-field scan types."""
    
    PLANAR = "planar"               # Planar near-field scanning
    CYLINDRICAL = "cylindrical"     # Cylindrical near-field scanning
    SPHERICAL = "spherical"         # Spherical near-field scanning


class PolarizationType(Enum):
    """Antenna polarization types."""
    
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    RHCP = "rhcp"                   # Right-hand circular
    LHCP = "lhcp"                   # Left-hand circular
    SLANT_45 = "slant_45"
    SLANT_135 = "slant_135"


class AbsorberType(Enum):
    """RF absorber types."""
    
    PYRAMID = "pyramid"             # Pyramidal foam absorber
    WEDGE = "wedge"                 # Wedge absorber
    FERRITE = "ferrite"             # Ferrite tile
    HYBRID = "hybrid"               # Hybrid pyramid + ferrite
    WALK_ON = "walk_on"             # Walk-on floor absorber


@dataclass
class QuietZoneSpec:
    """Quiet zone specification."""
    
    dimensions_m: Tuple[float, float, float]  # L x W x H
    center_position_m: Tuple[float, float, float]
    amplitude_taper_db: float
    phase_taper_deg: float
    reflectivity_db: float
    frequency_range_hz: Tuple[float, float]


@dataclass
class AbsorberSpec:
    """Absorber specification."""
    
    absorber_type: AbsorberType
    manufacturer: str
    model: str
    height_m: float
    reflectivity_db: Dict[float, float]  # Frequency -> reflectivity
    frequency_range_hz: Tuple[float, float]
    fire_rating: str


@dataclass
class NearFieldData:
    """Near-field measurement data."""
    
    scan_type: ScanType
    frequency_hz: float
    polarization: PolarizationType
    positions: np.ndarray          # Scan positions
    amplitude: np.ndarray          # Complex amplitude data
    phase: np.ndarray              # Phase data in degrees
    grid_spacing_m: float
    scan_area_m: Tuple[float, float]
    aut_position_m: Tuple[float, float, float]
    probe_correction: Optional[np.ndarray] = None


@dataclass
class FarFieldData:
    """Far-field pattern data."""
    
    frequency_hz: float
    polarization: PolarizationType
    theta: np.ndarray              # Theta angles in degrees
    phi: np.ndarray                # Phi angles in degrees
    amplitude_db: np.ndarray       # Amplitude in dB
    phase_deg: np.ndarray          # Phase in degrees
    co_pol: np.ndarray             # Co-polarization component
    cross_pol: np.ndarray          # Cross-polarization component


class AnechoicChamberController:
    """
    Anechoic chamber control and measurement system.
    
    Controls chamber positioner, RF equipment, and
    coordinates measurement sequences.
    """
    
    def __init__(
        self,
        chamber: ChamberInterface,
        specification: ChamberSpecification
    ):
        self.chamber = chamber
        self.specification = specification
        
        self._connected = False
        self._positioner_position = (0.0, 0.0)  # theta, phi
        self._current_frequency = 1e9
        self._polarization = PolarizationType.VERTICAL
        
        # Equipment states
        self._equipment_status = {
            "signal_generator": False,
            "receiver": False,
            "positioner": False,
            "probe": False
        }
        
        # Calibration data
        self._path_loss_cal: Dict[float, float] = {}
        self._probe_cal: Dict[float, complex] = {}
    
    def connect(self) -> bool:
        """Connect to chamber control system."""
        try:
            self._connected = self.chamber.connect()
            
            if self._connected:
                # Initialize equipment
                self._equipment_status["signal_generator"] = True
                self._equipment_status["receiver"] = True
                self._equipment_status["positioner"] = True
                
                logger.info(f"Connected to anechoic chamber: {self.specification.chamber_id}")
            
            return self._connected
            
        except Exception as e:
            logger.error(f"Chamber connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from chamber."""
        self._connected = False
        self.chamber.disconnect()
        return True
    
    def set_frequency(self, frequency_hz: float) -> bool:
        """Set measurement frequency."""
        if not self._connected:
            return False
        
        freq_range = self.specification.frequency_range_hz
        if not (freq_range[0] <= frequency_hz <= freq_range[1]):
            logger.error(f"Frequency {frequency_hz} Hz out of range")
            return False
        
        self._current_frequency = frequency_hz
        return self.chamber.set_frequency(frequency_hz)
    
    def set_polarization(self, polarization: PolarizationType) -> bool:
        """Set measurement polarization."""
        self._polarization = polarization
        return True
    
    def move_positioner(
        self,
        theta: float,
        phi: float,
        wait: bool = True
    ) -> bool:
        """
        Move antenna positioner to specified angles.
        
        Args:
            theta: Elevation angle in degrees (0-180)
            phi: Azimuth angle in degrees (0-360)
            wait: Wait for movement to complete
            
        Returns:
            True if successful
        """
        if not self._connected:
            return False
        
        success = self.chamber.set_position(theta, phi)
        
        if success and wait:
            # Wait for settling
            time.sleep(0.05)
            self._positioner_position = (theta, phi)
        
        return success
    
    def measure_pattern_point(self) -> MeasurementPoint:
        """Measure a single pattern point."""
        if not self._connected:
            raise MeasurementException("Chamber not connected")
        
        return self.chamber.measure(MeasurementType.ANTENNA_PATTERN)
    
    def calibrate_path_loss(
        self,
        frequencies_hz: List[float],
        reference_antenna_gain_db: float = 0.0
    ) -> Dict[float, float]:
        """
        Calibrate system path loss at specified frequencies.
        
        Args:
            frequencies_hz: Calibration frequencies
            reference_antenna_gain_db: Known gain of reference antenna
            
        Returns:
            Dictionary of frequency -> path loss
        """
        self._path_loss_cal = {}
        
        # Move to boresight
        self.move_positioner(90, 0)
        
        for freq in frequencies_hz:
            self.set_frequency(freq)
            
            # Measure received power
            measurement = self.chamber.measure(MeasurementType.POWER)
            
            # Calculate path loss (assuming known transmit power)
            tx_power_dbm = 0.0  # Reference level
            path_loss = tx_power_dbm - measurement.value - reference_antenna_gain_db
            
            self._path_loss_cal[freq] = path_loss
        
        logger.info(f"Path loss calibration complete for {len(frequencies_hz)} frequencies")
        return self._path_loss_cal
    
    def get_path_loss(self, frequency_hz: float) -> float:
        """
        Get interpolated path loss for frequency.
        
        Args:
            frequency_hz: Target frequency
            
        Returns:
            Path loss in dB
        """
        if not self._path_loss_cal:
            return 0.0
        
        # Find nearest calibration points
        cal_freqs = sorted(self._path_loss_cal.keys())
        
        if frequency_hz <= cal_freqs[0]:
            return self._path_loss_cal[cal_freqs[0]]
        if frequency_hz >= cal_freqs[-1]:
            return self._path_loss_cal[cal_freqs[-1]]
        
        # Linear interpolation
        for i in range(len(cal_freqs) - 1):
            if cal_freqs[i] <= frequency_hz <= cal_freqs[i + 1]:
                f1, f2 = cal_freqs[i], cal_freqs[i + 1]
                pl1, pl2 = self._path_loss_cal[f1], self._path_loss_cal[f2]
                
                # Log-frequency interpolation
                factor = (np.log10(frequency_hz) - np.log10(f1)) / (np.log10(f2) - np.log10(f1))
                return pl1 + factor * (pl2 - pl1)
        
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get chamber system status."""
        return {
            "connected": self._connected,
            "chamber_id": self.specification.chamber_id,
            "chamber_type": self.specification.chamber_type.value,
            "positioner_position": self._positioner_position,
            "current_frequency_hz": self._current_frequency,
            "polarization": self._polarization.value,
            "equipment_status": self._equipment_status,
            "path_loss_calibrated": len(self._path_loss_cal) > 0
        }


class QuietZoneCharacterization:
    """
    Quiet zone characterization system.
    
    Measures and validates quiet zone performance including
    amplitude taper, phase taper, and reflectivity.
    """
    
    def __init__(
        self,
        controller: AnechoicChamberController,
        quiet_zone_spec: QuietZoneSpec
    ):
        self.controller = controller
        self.spec = quiet_zone_spec
        
        self._amplitude_data: Dict[Tuple[float, float, float], float] = {}
        self._phase_data: Dict[Tuple[float, float, float], float] = {}
        self._measurement_points: List[MeasurementPoint] = []
    
    def measure_field_uniformity(
        self,
        frequency_hz: float,
        grid_points: int = 9,
        height_points: int = 3
    ) -> Dict[str, Any]:
        """
        Measure field uniformity within quiet zone.
        
        Args:
            frequency_hz: Test frequency
            grid_points: Points per side of measurement grid
            height_points: Number of height levels
            
        Returns:
            Field uniformity analysis results
        """
        self.controller.set_frequency(frequency_hz)
        
        # Define measurement grid within quiet zone
        qz = self.spec
        x_points = np.linspace(
            qz.center_position_m[0] - qz.dimensions_m[0]/2,
            qz.center_position_m[0] + qz.dimensions_m[0]/2,
            grid_points
        )
        y_points = np.linspace(
            qz.center_position_m[1] - qz.dimensions_m[1]/2,
            qz.center_position_m[1] + qz.dimensions_m[1]/2,
            grid_points
        )
        z_points = np.linspace(
            qz.center_position_m[2] - qz.dimensions_m[2]/2,
            qz.center_position_m[2] + qz.dimensions_m[2]/2,
            height_points
        )
        
        self._amplitude_data = {}
        self._phase_data = {}
        
        for z in z_points:
            for y in y_points:
                for x in x_points:
                    # In real system, would move probe to position
                    # For simulation, measure at current position
                    measurement = self.controller.chamber.measure(
                        MeasurementType.POWER
                    )
                    
                    self._amplitude_data[(x, y, z)] = measurement.value
                    
                    # Simulate phase measurement
                    phase_meas = self.controller.chamber.measure(
                        MeasurementType.PHASE
                    )
                    self._phase_data[(x, y, z)] = phase_meas.value
        
        # Analyze uniformity
        amplitudes = np.array(list(self._amplitude_data.values()))
        phases = np.array(list(self._phase_data.values()))
        
        # Reference is center point
        center_key = (
            qz.center_position_m[0],
            qz.center_position_m[1],
            qz.center_position_m[2]
        )
        
        # Find closest point to center
        closest_key = min(
            self._amplitude_data.keys(),
            key=lambda k: sum((a-b)**2 for a, b in zip(k, center_key))
        )
        
        ref_amplitude = self._amplitude_data[closest_key]
        ref_phase = self._phase_data[closest_key]
        
        amplitude_taper = np.max(np.abs(amplitudes - ref_amplitude))
        phase_taper = np.max(np.abs(phases - ref_phase))
        
        return {
            "frequency_hz": frequency_hz,
            "measurement_points": len(self._amplitude_data),
            "reference_amplitude_dbm": ref_amplitude,
            "amplitude_taper_db": amplitude_taper,
            "amplitude_taper_spec_db": qz.amplitude_taper_db,
            "amplitude_pass": amplitude_taper <= qz.amplitude_taper_db,
            "phase_taper_deg": phase_taper,
            "phase_taper_spec_deg": qz.phase_taper_deg,
            "phase_pass": phase_taper <= qz.phase_taper_deg,
            "overall_pass": (
                amplitude_taper <= qz.amplitude_taper_db and
                phase_taper <= qz.phase_taper_deg
            )
        }
    
    def measure_reflectivity(
        self,
        frequency_hz: float,
        wall_positions: List[str] = ["front", "back", "left", "right", "ceiling", "floor"]
    ) -> Dict[str, Any]:
        """
        Measure chamber wall reflectivity.
        
        Args:
            frequency_hz: Test frequency
            wall_positions: Walls to measure
            
        Returns:
            Reflectivity measurements
        """
        self.controller.set_frequency(frequency_hz)
        
        reflectivity_results = {}
        
        # Position mapping to angles (simplified)
        wall_angles = {
            "front": (90, 0),
            "back": (90, 180),
            "left": (90, 270),
            "right": (90, 90),
            "ceiling": (0, 0),
            "floor": (180, 0)
        }
        
        for wall in wall_positions:
            if wall in wall_angles:
                theta, phi = wall_angles[wall]
                
                # Move to face wall
                self.controller.move_positioner(theta, phi)
                
                # Measure reflected signal
                measurement = self.controller.chamber.measure(MeasurementType.POWER)
                
                # Calculate reflectivity (simplified)
                # In real system, would compare to free-space reference
                reflectivity_db = measurement.value + 60  # Approximate
                
                reflectivity_results[wall] = {
                    "reflectivity_db": reflectivity_db,
                    "spec_db": self.spec.reflectivity_db,
                    "pass": reflectivity_db <= self.spec.reflectivity_db
                }
        
        # Overall result
        all_pass = all(r["pass"] for r in reflectivity_results.values())
        max_reflectivity = max(r["reflectivity_db"] for r in reflectivity_results.values())
        
        return {
            "frequency_hz": frequency_hz,
            "walls_tested": len(reflectivity_results),
            "wall_results": reflectivity_results,
            "max_reflectivity_db": max_reflectivity,
            "spec_db": self.spec.reflectivity_db,
            "overall_pass": all_pass
        }
    
    def generate_quiet_zone_report(self) -> Dict[str, Any]:
        """Generate comprehensive quiet zone report."""
        return {
            "quiet_zone_spec": {
                "dimensions_m": self.spec.dimensions_m,
                "center_position_m": self.spec.center_position_m,
                "amplitude_taper_spec_db": self.spec.amplitude_taper_db,
                "phase_taper_spec_deg": self.spec.phase_taper_deg,
                "reflectivity_spec_db": self.spec.reflectivity_db,
                "frequency_range_hz": self.spec.frequency_range_hz
            },
            "measurement_data": {
                "amplitude_points": len(self._amplitude_data),
                "phase_points": len(self._phase_data)
            },
            "generated_at": datetime.utcnow().isoformat()
        }


class NearFieldScanner:
    """
    Near-field antenna measurement scanner.
    
    Supports planar, cylindrical, and spherical near-field
    scanning with near-field to far-field transformation.
    """
    
    def __init__(
        self,
        controller: AnechoicChamberController,
        scan_type: ScanType = ScanType.PLANAR
    ):
        self.controller = controller
        self.scan_type = scan_type
        
        self._near_field_data: Optional[NearFieldData] = None
        self._far_field_data: Optional[FarFieldData] = None
        
        # Scan configuration
        self._scan_config = {
            "grid_spacing_lambda": 0.5,
            "scan_area_lambda": (10, 10),
            "probe_correction": True,
            "averaging": 1
        }
    
    def configure_scan(
        self,
        grid_spacing_lambda: float = 0.5,
        scan_area_lambda: Tuple[float, float] = (10, 10),
        probe_correction: bool = True,
        averaging: int = 1
    ) -> None:
        """
        Configure near-field scan parameters.
        
        Args:
            grid_spacing_lambda: Grid spacing in wavelengths
            scan_area_lambda: Scan area in wavelengths (width, height)
            probe_correction: Enable probe correction
            averaging: Number of averages per point
        """
        self._scan_config = {
            "grid_spacing_lambda": grid_spacing_lambda,
            "scan_area_lambda": scan_area_lambda,
            "probe_correction": probe_correction,
            "averaging": averaging
        }
    
    def perform_planar_scan(
        self,
        frequency_hz: float,
        polarization: PolarizationType = PolarizationType.VERTICAL,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> NearFieldData:
        """
        Perform planar near-field scan.
        
        Args:
            frequency_hz: Measurement frequency
            polarization: Measurement polarization
            progress_callback: Optional progress callback
            
        Returns:
            Near-field measurement data
        """
        self.controller.set_frequency(frequency_hz)
        self.controller.set_polarization(polarization)
        
        # Calculate scan parameters
        wavelength = 3e8 / frequency_hz
        config = self._scan_config
        
        grid_spacing_m = config["grid_spacing_lambda"] * wavelength
        scan_width_m = config["scan_area_lambda"][0] * wavelength
        scan_height_m = config["scan_area_lambda"][1] * wavelength
        
        # Generate scan grid
        x_points = np.arange(
            -scan_width_m/2,
            scan_width_m/2 + grid_spacing_m,
            grid_spacing_m
        )
        y_points = np.arange(
            -scan_height_m/2,
            scan_height_m/2 + grid_spacing_m,
            grid_spacing_m
        )
        
        nx, ny = len(x_points), len(y_points)
        total_points = nx * ny
        
        # Initialize data arrays
        amplitude = np.zeros((ny, nx), dtype=complex)
        phase = np.zeros((ny, nx))
        positions = np.zeros((ny, nx, 2))
        
        current_point = 0
        
        for j, y in enumerate(y_points):
            for i, x in enumerate(x_points):
                # In real system, move scanner to position
                # Measure amplitude and phase
                measurement = self.controller.chamber.measure(
                    MeasurementType.ANTENNA_PATTERN
                )
                
                # Store data
                amp_linear = 10 ** (measurement.value / 20)
                phase_deg = np.random.uniform(-180, 180)  # Simulated phase
                
                amplitude[j, i] = amp_linear * np.exp(1j * np.deg2rad(phase_deg))
                phase[j, i] = phase_deg
                positions[j, i] = [x, y]
                
                current_point += 1
                if progress_callback:
                    progress_callback(current_point / total_points * 100)
        
        self._near_field_data = NearFieldData(
            scan_type=ScanType.PLANAR,
            frequency_hz=frequency_hz,
            polarization=polarization,
            positions=positions,
            amplitude=amplitude,
            phase=phase,
            grid_spacing_m=grid_spacing_m,
            scan_area_m=(scan_width_m, scan_height_m),
            aut_position_m=(0, 0, wavelength * 2)  # 2 wavelengths behind scan plane
        )
        
        return self._near_field_data
    
    def transform_to_far_field(
        self,
        theta_range: Tuple[float, float] = (-90, 90),
        phi_range: Tuple[float, float] = (-90, 90),
        angular_resolution: float = 1.0
    ) -> FarFieldData:
        """
        Transform near-field data to far-field pattern.
        
        Args:
            theta_range: Theta angle range in degrees
            phi_range: Phi angle range in degrees
            angular_resolution: Angular resolution in degrees
            
        Returns:
            Far-field pattern data
        """
        if self._near_field_data is None:
            raise ValueError("No near-field data available")
        
        nf_data = self._near_field_data
        
        # Generate far-field angles
        theta = np.arange(theta_range[0], theta_range[1] + angular_resolution, angular_resolution)
        phi = np.arange(phi_range[0], phi_range[1] + angular_resolution, angular_resolution)
        
        n_theta, n_phi = len(theta), len(phi)
        
        # Initialize far-field arrays
        ff_amplitude = np.zeros((n_theta, n_phi), dtype=complex)
        
        wavelength = 3e8 / nf_data.frequency_hz
        k = 2 * np.pi / wavelength
        
        # Near-field to far-field transformation (simplified 2D FFT approach)
        # In production, would use proper plane-wave spectrum method
        
        # Zero-pad for better angular resolution
        pad_factor = 4
        nf_padded = np.zeros(
            (nf_data.amplitude.shape[0] * pad_factor,
             nf_data.amplitude.shape[1] * pad_factor),
            dtype=complex
        )
        
        # Center the data in padded array
        start_y = (nf_padded.shape[0] - nf_data.amplitude.shape[0]) // 2
        start_x = (nf_padded.shape[1] - nf_data.amplitude.shape[1]) // 2
        nf_padded[
            start_y:start_y + nf_data.amplitude.shape[0],
            start_x:start_x + nf_data.amplitude.shape[1]
        ] = nf_data.amplitude
        
        # 2D FFT
        ff_spectrum = scipy_fft.fftshift(scipy_fft.fft2(nf_padded))
        
        # Map to angular coordinates
        dx = nf_data.grid_spacing_m
        du = 1 / (nf_padded.shape[1] * dx)
        dv = 1 / (nf_padded.shape[0] * dx)
        
        u = (np.arange(nf_padded.shape[1]) - nf_padded.shape[1]//2) * du * wavelength
        v = (np.arange(nf_padded.shape[0]) - nf_padded.shape[0]//2) * dv * wavelength
        
        # Interpolate to requested angles
        for i, th in enumerate(theta):
            for j, ph in enumerate(phi):
                # Convert angles to direction cosines
                sin_th = np.sin(np.deg2rad(th))
                cos_th = np.cos(np.deg2rad(th))
                sin_ph = np.sin(np.deg2rad(ph))
                cos_ph = np.cos(np.deg2rad(ph))
                
                u_val = sin_th * cos_ph
                v_val = sin_th * sin_ph
                
                # Check valid region
                if u_val**2 + v_val**2 <= 1:
                    # Interpolate
                    u_idx = np.argmin(np.abs(u - u_val))
                    v_idx = np.argmin(np.abs(v - v_val))
                    
                    if 0 <= u_idx < ff_spectrum.shape[1] and 0 <= v_idx < ff_spectrum.shape[0]:
                        ff_amplitude[i, j] = ff_spectrum[v_idx, u_idx]
        
        # Convert to dB
        ff_amplitude_db = 20 * np.log10(np.abs(ff_amplitude) + 1e-10)
        ff_amplitude_db -= np.max(ff_amplitude_db)  # Normalize to peak
        
        ff_phase = np.angle(ff_amplitude, deg=True)
        
        self._far_field_data = FarFieldData(
            frequency_hz=nf_data.frequency_hz,
            polarization=nf_data.polarization,
            theta=theta,
            phi=phi,
            amplitude_db=ff_amplitude_db,
            phase_deg=ff_phase,
            co_pol=ff_amplitude_db,  # Simplified - same as total
            cross_pol=ff_amplitude_db - 30  # Simplified cross-pol estimate
        )
        
        return self._far_field_data
    
    def calculate_antenna_parameters(self) -> Dict[str, Any]:
        """Calculate antenna parameters from far-field data."""
        if self._far_field_data is None:
            raise ValueError("No far-field data available")
        
        ff = self._far_field_data
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(ff.amplitude_db), ff.amplitude_db.shape)
        peak_gain = ff.amplitude_db[peak_idx]
        peak_theta = ff.theta[peak_idx[0]]
        peak_phi = ff.phi[peak_idx[1]]
        
        # Calculate beamwidths (E-plane and H-plane)
        # E-plane: phi = 0
        phi_0_idx = np.argmin(np.abs(ff.phi))
        e_plane_pattern = ff.amplitude_db[:, phi_0_idx]
        hpbw_e = self._calculate_hpbw(ff.theta, e_plane_pattern)
        
        # H-plane: theta = 90
        theta_90_idx = np.argmin(np.abs(ff.theta - 90))
        h_plane_pattern = ff.amplitude_db[theta_90_idx, :]
        hpbw_h = self._calculate_hpbw(ff.phi, h_plane_pattern)
        
        # Sidelobe level
        sll = self._calculate_sidelobe_level(e_plane_pattern)
        
        # Cross-pol discrimination
        xpd = np.max(ff.co_pol) - np.max(ff.cross_pol)
        
        return {
            "frequency_hz": ff.frequency_hz,
            "polarization": ff.polarization.value,
            "peak_gain_db": float(peak_gain),
            "peak_direction_deg": (float(peak_theta), float(peak_phi)),
            "hpbw_e_plane_deg": hpbw_e,
            "hpbw_h_plane_deg": hpbw_h,
            "first_sidelobe_level_db": sll,
            "cross_pol_discrimination_db": xpd,
            "theta_range_deg": (float(ff.theta[0]), float(ff.theta[-1])),
            "phi_range_deg": (float(ff.phi[0]), float(ff.phi[-1]))
        }
    
    def _calculate_hpbw(self, angles: np.ndarray, pattern_db: np.ndarray) -> float:
        """Calculate half-power beamwidth."""
        peak_val = np.max(pattern_db)
        threshold = peak_val - 3.0
        
        above_threshold = np.where(pattern_db >= threshold)[0]
        
        if len(above_threshold) >= 2:
            return float(angles[above_threshold[-1]] - angles[above_threshold[0]])
        
        return 0.0
    
    def _calculate_sidelobe_level(self, pattern_db: np.ndarray) -> float:
        """Calculate first sidelobe level."""
        # Find main lobe peak
        peak_idx = np.argmax(pattern_db)
        peak_val = pattern_db[peak_idx]
        
        # Find first null after peak
        for i in range(peak_idx + 1, len(pattern_db) - 1):
            if pattern_db[i] < pattern_db[i-1] and pattern_db[i] < pattern_db[i+1]:
                # Found null, look for sidelobe
                for j in range(i + 1, len(pattern_db) - 1):
                    if pattern_db[j] > pattern_db[j-1] and pattern_db[j] > pattern_db[j+1]:
                        return float(pattern_db[j] - peak_val)
                break
        
        return -20.0  # Default if not found


class SiteVSWRMeasurement:
    """
    Site VSWR (Voltage Standing Wave Ratio) measurement.
    
    Validates chamber performance by measuring site VSWR
    per CISPR 16-1-4 requirements.
    """
    
    def __init__(self, controller: AnechoicChamberController):
        self.controller = controller
        self._vswr_data: Dict[float, Dict[str, float]] = {}
    
    def measure_site_vswr(
        self,
        frequencies_hz: List[float],
        heights_m: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    ) -> Dict[str, Any]:
        """
        Measure site VSWR at specified frequencies and heights.
        
        Args:
            frequencies_hz: Test frequencies
            heights_m: Antenna heights to measure
            
        Returns:
            Site VSWR results
        """
        self._vswr_data = {}
        
        for freq in frequencies_hz:
            self.controller.set_frequency(freq)
            
            height_measurements = []
            
            for height in heights_m:
                # In real system, would adjust antenna height
                measurement = self.controller.chamber.measure(MeasurementType.POWER)
                height_measurements.append(measurement.value)
            
            # Calculate site VSWR
            e_max = max(height_measurements)
            e_min = min(height_measurements)
            
            # Convert dB difference to VSWR
            delta_db = e_max - e_min
            
            # VSWR = (10^(delta/20) + 1) / (10^(delta/20) - 1) approximately
            ratio = 10 ** (delta_db / 20)
            if ratio > 1:
                vswr = (ratio + 1) / (ratio - 1) if ratio != 1 else float('inf')
            else:
                vswr = 1.0
            
            self._vswr_data[freq] = {
                "e_max_dbm": e_max,
                "e_min_dbm": e_min,
                "delta_db": delta_db,
                "vswr": vswr,
                "vswr_db": delta_db
            }
        
        # CISPR 16-1-4 limits (simplified)
        # Site VSWR should be < 6 dB for frequencies > 1 GHz
        pass_count = sum(
            1 for data in self._vswr_data.values()
            if data["vswr_db"] < 6.0
        )
        
        return {
            "frequencies_tested": len(frequencies_hz),
            "heights_tested": len(heights_m),
            "frequency_results": self._vswr_data,
            "max_vswr_db": max(d["vswr_db"] for d in self._vswr_data.values()),
            "pass_count": pass_count,
            "fail_count": len(frequencies_hz) - pass_count,
            "overall_pass": pass_count == len(frequencies_hz)
        }


# Export public API
__all__ = [
    # Enums
    "ScanType",
    "PolarizationType",
    "AbsorberType",
    
    # Data classes
    "QuietZoneSpec",
    "AbsorberSpec",
    "NearFieldData",
    "FarFieldData",
    
    # Controllers
    "AnechoicChamberController",
    
    # Measurement systems
    "QuietZoneCharacterization",
    "NearFieldScanner",
    "SiteVSWRMeasurement",
]
