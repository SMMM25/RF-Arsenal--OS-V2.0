#!/usr/bin/env python3
"""
RF Arsenal OS - Vehicle Penetration Testing Module Tests

Comprehensive tests for vehicle security testing components:
- CAN Bus analysis
- UDS diagnostics
- Key Fob attacks
- TPMS spoofing
- GPS spoofing
- Bluetooth/BLE attacks
- V2X communication attacks

Author: RF Arsenal Team
"""

import pytest
import time
import struct
import numpy as np
from unittest.mock import Mock, MagicMock, patch


# ============================================================================
# CAN Bus Tests
# ============================================================================

class TestCANBusController:
    """Tests for CAN Bus Controller"""
    
    def test_can_frame_creation(self):
        """Test CAN frame creation"""
        from core.vehicle.can_bus import CANFrame, CANProtocol
        
        frame = CANFrame(
            arbitration_id=0x123,
            data=bytes([0x01, 0x02, 0x03, 0x04])
        )
        
        assert frame.arbitration_id == 0x123
        assert frame.data == bytes([0x01, 0x02, 0x03, 0x04])
    
    def test_can_controller_initialization(self):
        """Test CAN controller initialization"""
        from core.vehicle.can_bus import CANBusController, CANInterface, CANSpeed
        
        controller = CANBusController()
        assert controller is not None
        # Check internal connected state
        assert hasattr(controller, '_connected')
    
    def test_can_protocols(self):
        """Test CAN protocol enumeration"""
        from core.vehicle.can_bus import CANProtocol
        
        assert CANProtocol.CAN_2_0A.value == "can_2.0a"
        assert CANProtocol.CAN_2_0B.value == "can_2.0b"
        assert CANProtocol.CAN_FD.value == "can_fd"
    
    def test_can_speeds(self):
        """Test CAN speed configurations"""
        from core.vehicle.can_bus import CANSpeed
        
        assert CANSpeed.CAN_125KBPS.value == 125000
        assert CANSpeed.CAN_250KBPS.value == 250000
        assert CANSpeed.CAN_500KBPS.value == 500000
        assert CANSpeed.CAN_1MBPS.value == 1000000
    
    def test_can_filter(self):
        """Test CAN filter creation"""
        from core.vehicle.can_bus import CANFilter
        
        filter = CANFilter(
            can_id=0x100,
            can_mask=0x7FF
        )
        
        assert filter.can_mask == 0x7FF
        assert filter.can_id == 0x100
        assert filter.matches(0x100)
    
    def test_can_frame_extended_id(self):
        """Test extended CAN ID frame"""
        from core.vehicle.can_bus import CANFrame
        
        frame = CANFrame(
            arbitration_id=0x18FEF100,
            data=bytes([0xFF] * 8),
            is_extended=True
        )
        
        assert frame.is_extended
        assert frame.arbitration_id == 0x18FEF100


class TestCANOperations:
    """Tests for CAN Bus Operations"""
    
    def test_can_interface_types(self):
        """Test CAN interface enumeration"""
        from core.vehicle.can_bus import CANInterface
        
        assert CANInterface.SOCKETCAN.value == "socketcan"
        assert CANInterface.SLCAN.value == "slcan"
        assert CANInterface.ELM327.value == "elm327"
    
    def test_can_error_exception(self):
        """Test CAN error exception"""
        from core.vehicle.can_bus import CANError
        
        with pytest.raises(CANError):
            raise CANError("Test CAN error")


# ============================================================================
# UDS Tests
# ============================================================================

class TestUDSProtocol:
    """Tests for UDS Protocol"""
    
    def test_uds_service_ids(self):
        """Test UDS service ID enumeration"""
        from core.vehicle.uds import UDSService
        
        assert UDSService.DIAGNOSTIC_SESSION_CONTROL == 0x10
        assert UDSService.ECU_RESET == 0x11
        assert UDSService.SECURITY_ACCESS == 0x27
        assert UDSService.READ_DATA_BY_ID == 0x22
        assert UDSService.WRITE_DATA_BY_ID == 0x2E
    
    def test_uds_session_types(self):
        """Test UDS session types"""
        from core.vehicle.uds import UDSSession
        
        assert UDSSession.DEFAULT == 0x01
        assert UDSSession.PROGRAMMING == 0x02
        assert UDSSession.EXTENDED_DIAGNOSTIC == 0x03
    
    def test_uds_negative_responses(self):
        """Test UDS negative response codes"""
        from core.vehicle.uds import UDSNegativeResponse
        
        assert UDSNegativeResponse.SECURITY_ACCESS_DENIED == 0x33
        assert UDSNegativeResponse.INVALID_KEY == 0x35
        assert UDSNegativeResponse.REQUEST_OUT_OF_RANGE == 0x31
    
    def test_dtc_formatting(self):
        """Test DTC code formatting"""
        from core.vehicle.uds import DiagnosticTroubleCode
        
        # P0123 style code
        dtc = DiagnosticTroubleCode(code=0x0123, status=0x09)
        
        code_str = dtc.code_string
        assert code_str.startswith('P')
        assert dtc.is_active or dtc.is_confirmed
    
    def test_uds_client_creation(self):
        """Test UDS client initialization"""
        from core.vehicle.uds import UDSClient
        from core.vehicle.can_bus import CANBusController
        
        can = CANBusController()
        client = UDSClient(can, tx_id=0x7E0, rx_id=0x7E8)
        
        assert client.tx_id == 0x7E0
        assert client.rx_id == 0x7E8
        assert client.current_session.value == 0x01  # Default
    
    def test_uds_response(self):
        """Test UDS response container"""
        from core.vehicle.uds import UDSResponse
        
        response = UDSResponse(
            service_id=0x22,
            data=bytes([0x01, 0x02, 0x03]),
            is_positive=True
        )
        
        assert response.is_positive
        assert '+' in str(response)


# ============================================================================
# Key Fob Tests
# ============================================================================

class TestKeyFobAttack:
    """Tests for Key Fob Attack Module"""
    
    def test_key_fob_frequencies(self):
        """Test key fob frequency constants"""
        from core.vehicle.key_fob import KeyFobFrequency
        
        assert KeyFobFrequency.FREQ_315MHZ.value == 315_000_000
        assert KeyFobFrequency.FREQ_433MHZ.value == 433_920_000
        assert KeyFobFrequency.FREQ_868MHZ.value == 868_000_000
    
    def test_key_fob_protocols(self):
        """Test key fob protocol types"""
        from core.vehicle.key_fob import KeyFobProtocol
        
        assert KeyFobProtocol.KEELOQ.value == "keeloq"
        assert KeyFobProtocol.HITAG2.value == "hitag2"
        assert KeyFobProtocol.FIXED_CODE.value == "fixed_code"
    
    def test_key_fob_capture_creation(self):
        """Test key fob capture dataclass"""
        from core.vehicle.key_fob import KeyFobCapture, KeyFobProtocol
        
        capture = KeyFobCapture(
            timestamp=time.time(),
            frequency=433.92e6,
            raw_iq=np.zeros(1000),
            protocol=KeyFobProtocol.KEELOQ,
            serial_number=0x12345678,
            rolling_code=0x87654321
        )
        
        assert capture.serial_number == 0x12345678
        assert 'keeloq' in str(capture).lower()
    
    def test_key_fob_attack_initialization(self):
        """Test key fob attack controller"""
        from core.vehicle.key_fob import KeyFobAttack
        
        attack = KeyFobAttack(frequency=315e6)
        
        assert attack.frequency == 315e6
        assert len(attack.get_captures()) == 0
    
    def test_rolling_code_analyzer(self):
        """Test rolling code analyzer"""
        from core.vehicle.key_fob import (
            RollingCodeAnalyzer, KeyFobCapture, KeyFobProtocol
        )
        
        analyzer = RollingCodeAnalyzer()
        
        # Add captures with sequential rolling codes
        for i in range(5):
            capture = KeyFobCapture(
                timestamp=time.time(),
                frequency=433.92e6,
                raw_iq=np.zeros(100),
                serial_number=0x12345678,
                rolling_code=1000 + i
            )
            analyzer.add_capture(capture)
        
        state = analyzer.get_state(0x12345678)
        assert state is not None
        assert len(state.code_history) == 5
    
    def test_keeloq_decrypt(self):
        """Test KeeLoq decryption function exists"""
        from core.vehicle.key_fob import RollingCodeAnalyzer
        
        analyzer = RollingCodeAnalyzer()
        
        # Test decryption function exists
        assert hasattr(analyzer, 'keeloq_decrypt')
        assert callable(analyzer.keeloq_decrypt)
    
    def test_rolljam_attack(self):
        """Test RollJam attack initialization"""
        from core.vehicle.key_fob import RollJamAttack
        
        attack = RollJamAttack(frequency=433.92e6)
        
        assert attack.frequency == 433.92e6
        assert attack.get_saved_code() is None


# ============================================================================
# TPMS Tests
# ============================================================================

class TestTPMSSpoofer:
    """Tests for TPMS Spoofer Module"""
    
    def test_tpms_protocols(self):
        """Test TPMS protocol enumeration"""
        from core.vehicle.tpms import TPMSProtocol
        
        assert TPMSProtocol.SCHRADER.value == "schrader"
        assert TPMSProtocol.HUF_BERU.value == "huf_beru"
        assert TPMSProtocol.CONTINENTAL.value == "continental"
    
    def test_tpms_sensor_creation(self):
        """Test TPMS sensor dataclass"""
        from core.vehicle.tpms import TPMSSensor, TPMSManufacturer
        
        sensor = TPMSSensor(
            sensor_id=0x12345678,
            tire_position="FL",
            pressure_psi=32.5,
            temperature_f=75.0
        )
        
        assert sensor.sensor_id == 0x12345678
        assert sensor.tire_position == "FL"
        assert 'FL' in str(sensor)
    
    def test_tpms_packet_pressure_conversion(self):
        """Test TPMS packet pressure conversion"""
        from core.vehicle.tpms import TPMSPacket, TPMSProtocol
        
        packet = TPMSPacket(
            sensor_id=0x12345678,
            pressure_raw=200,  # ~50 kPa = ~7.25 PSI
            temperature_raw=70,  # 20°C = 68°F
            flags=0,
            checksum=0,
            raw_data=b''
        )
        
        assert packet.pressure_psi > 0
        assert packet.temperature_f > 0
    
    def test_tpms_spoofer_initialization(self):
        """Test TPMS spoofer initialization"""
        from core.vehicle.tpms import TPMSSpoofer
        
        spoofer = TPMSSpoofer(frequency=433.92e6)
        
        assert spoofer.frequency == 433.92e6
        assert len(spoofer.get_discovered_sensors()) == 0
    
    def test_tpms_encoder(self):
        """Test TPMS packet encoder"""
        from core.vehicle.tpms import TPMSEncoder, TPMSProtocol
        
        encoder = TPMSEncoder()
        
        packet = encoder.encode_packet(
            sensor_id=0x12345678,
            pressure_psi=32.0,
            temperature_f=70.0,
            protocol=TPMSProtocol.GENERIC
        )
        
        assert isinstance(packet, bytes)
        assert len(packet) >= 8
    
    def test_tpms_decoder(self):
        """Test TPMS packet decoder"""
        from core.vehicle.tpms import TPMSDecoder, TPMSEncoder, TPMSProtocol
        
        encoder = TPMSEncoder()
        decoder = TPMSDecoder()
        
        # Encode a packet
        packet_bytes = encoder.encode_packet(
            sensor_id=0x12345678,
            pressure_psi=32.0,
            temperature_f=70.0
        )
        
        # Decode it back
        packet = decoder.decode_packet(packet_bytes)
        
        assert packet is not None
        assert packet.sensor_id == 0x12345678


# ============================================================================
# GPS Spoofing Tests
# ============================================================================

class TestGPSSpoofer:
    """Tests for GPS Spoofing Module"""
    
    def test_gps_coordinate_creation(self):
        """Test GPS coordinate creation"""
        from core.vehicle.gps_spoof import GPSCoordinate
        
        coord = GPSCoordinate(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=10.0
        )
        
        assert coord.latitude == 37.7749
        assert coord.longitude == -122.4194
        assert '37.77' in str(coord)
    
    def test_gps_coordinate_ecef_conversion(self):
        """Test GPS coordinate ECEF conversion"""
        from core.vehicle.gps_spoof import GPSCoordinate
        
        coord = GPSCoordinate(
            latitude=0.0,
            longitude=0.0,
            altitude=0.0
        )
        
        ecef = coord.to_ecef()
        
        assert len(ecef) == 3
        assert ecef[0] > 6000000  # ~6371km Earth radius
    
    def test_gps_coordinate_distance(self):
        """Test GPS distance calculation"""
        from core.vehicle.gps_spoof import GPSCoordinate
        
        sf = GPSCoordinate(latitude=37.7749, longitude=-122.4194)
        la = GPSCoordinate(latitude=34.0522, longitude=-118.2437)
        
        distance = sf.distance_to(la)
        
        # SF to LA is roughly 560 km
        assert 500000 < distance < 600000
    
    def test_gps_trajectory_creation(self):
        """Test GPS trajectory creation"""
        from core.vehicle.gps_spoof import GPSTrajectory, GPSCoordinate
        
        trajectory = GPSTrajectory(name="Test Route")
        
        trajectory.add_waypoint(GPSCoordinate(37.7749, -122.4194))
        trajectory.add_waypoint(GPSCoordinate(37.7849, -122.4094))
        
        assert len(trajectory.waypoints) == 2
        assert trajectory.total_distance() > 0
    
    def test_gps_trajectory_interpolation(self):
        """Test GPS trajectory position interpolation"""
        from core.vehicle.gps_spoof import GPSTrajectory, GPSCoordinate
        
        trajectory = GPSTrajectory()
        trajectory.add_waypoint(GPSCoordinate(0.0, 0.0))
        trajectory.add_waypoint(GPSCoordinate(0.001, 0.0))  # ~111m apart
        
        # Get position at start
        pos = trajectory.get_position_at_time(0, speed_mps=10)
        assert pos.latitude == 0.0
        
        # Get position partway
        pos = trajectory.get_position_at_time(5, speed_mps=10)  # 50m traveled
        assert 0 < pos.latitude < 0.001
    
    def test_gps_satellite_creation(self):
        """Test GPS satellite creation"""
        from core.vehicle.gps_spoof import GPSSatellite
        
        sat = GPSSatellite(
            prn=1,
            elevation=45.0,
            azimuth=90.0
        )
        
        assert sat.prn == 1
        assert sat.healthy
        assert 'PRN01' in str(sat)
    
    def test_gps_spoofer_initialization(self):
        """Test GPS spoofer initialization"""
        from core.vehicle.gps_spoof import GPSSpoofer
        
        spoofer = GPSSpoofer()
        
        assert spoofer.frequency == 1575.42e6
        assert len(spoofer.get_visible_satellites()) > 0
        assert not spoofer.is_spoofing
    
    def test_gps_ca_code_generation(self):
        """Test GPS C/A code generation"""
        from core.vehicle.gps_spoof import GPSCACodeGenerator
        
        gen = GPSCACodeGenerator(prn=1)
        
        code = gen.code
        assert len(code) == 1023  # C/A code is 1023 chips
        assert all(c in [-1, 1] for c in code)
    
    def test_gps_circular_trajectory(self):
        """Test circular trajectory creation"""
        from core.vehicle.gps_spoof import GPSTrajectory, GPSCoordinate
        
        center = GPSCoordinate(37.7749, -122.4194)
        trajectory = GPSTrajectory.create_circle(center, radius_m=100, num_points=36)
        
        assert len(trajectory.waypoints) == 36
        assert trajectory.loop


# ============================================================================
# Bluetooth/BLE Tests
# ============================================================================

class TestBluetoothVehicle:
    """Tests for Vehicle Bluetooth/BLE Module"""
    
    def test_ble_vulnerability_types(self):
        """Test BLE vulnerability enumeration"""
        from core.vehicle.bluetooth_vehicle import BLEVulnerability
        
        assert BLEVulnerability.BLUEBORNE.value == "blueborne"
        assert BLEVulnerability.KNOB.value == "knob"
        assert BLEVulnerability.FIXED_PIN.value == "fixed_pin"
    
    def test_vehicle_device_types(self):
        """Test vehicle device type enumeration"""
        from core.vehicle.bluetooth_vehicle import VehicleDeviceType
        
        assert VehicleDeviceType.OBD_ADAPTER.value == "obd_adapter"
        assert VehicleDeviceType.INFOTAINMENT.value == "infotainment"
        assert VehicleDeviceType.PHONE_AS_KEY.value == "phone_as_key"
    
    def test_ble_device_creation(self):
        """Test BLE device dataclass"""
        from core.vehicle.bluetooth_vehicle import BLEDevice, VehicleDeviceType
        
        device = BLEDevice(
            address="AA:BB:CC:DD:EE:FF",
            name="OBDII",
            rssi=-65,
            device_type=VehicleDeviceType.OBD_ADAPTER
        )
        
        assert device.address == "AA:BB:CC:DD:EE:FF"
        assert device.is_vehicle_related()
    
    def test_vehicle_ble_scanner(self):
        """Test vehicle BLE scanner initialization"""
        from core.vehicle.bluetooth_vehicle import VehicleBLEScanner
        
        scanner = VehicleBLEScanner()
        
        # Simulated scan
        devices = scanner.scan(duration=1.0)
        
        # Should return simulated devices
        assert isinstance(devices, list)
    
    def test_obd_bluetooth_exploit(self):
        """Test OBD Bluetooth exploit initialization"""
        from core.vehicle.bluetooth_vehicle import OBDBluetoothExploit
        
        exploit = OBDBluetoothExploit()
        
        assert not exploit._connected
    
    def test_infotainment_attack(self):
        """Test infotainment attack initialization"""
        from core.vehicle.bluetooth_vehicle import InfotainmentAttack
        
        attack = InfotainmentAttack()
        
        # Scan for infotainment (simulated)
        systems = attack.scan_infotainment_systems(duration=1.0)
        assert isinstance(systems, list)
    
    def test_ble_device_vehicle_detection(self):
        """Test vehicle-related device detection"""
        from core.vehicle.bluetooth_vehicle import BLEDevice, VehicleDeviceType
        
        # OBD device
        obd = BLEDevice(
            address="AA:BB:CC:DD:EE:01",
            name="VGATE iCar Pro",
            rssi=-70
        )
        assert obd.is_vehicle_related()
        
        # Generic device
        generic = BLEDevice(
            address="AA:BB:CC:DD:EE:02",
            name="Random Device",
            rssi=-70
        )
        assert not generic.is_vehicle_related()


# ============================================================================
# V2X Tests
# ============================================================================

class TestV2XAttack:
    """Tests for V2X Attack Module"""
    
    def test_v2x_protocols(self):
        """Test V2X protocol enumeration"""
        from core.vehicle.v2x import V2XProtocol
        
        assert V2XProtocol.DSRC.value == "dsrc"
        assert V2XProtocol.CV2X.value == "cv2x"
    
    def test_v2x_message_types(self):
        """Test V2X message type enumeration"""
        from core.vehicle.v2x import V2XMessageType
        
        assert V2XMessageType.BSM.value == 0x14
        assert V2XMessageType.EVA.value == 0x16
        assert V2XMessageType.SPAT.value == 0x13
    
    def test_v2x_position_creation(self):
        """Test V2X position creation"""
        from core.vehicle.v2x import V2XPosition
        
        pos = V2XPosition(
            latitude=37.7749,
            longitude=-122.4194,
            elevation=10.0
        )
        
        j2735 = pos.to_j2735()
        assert j2735['latitude'] == int(37.7749 * 1e7)
        assert j2735['longitude'] == int(-122.4194 * 1e7)
    
    def test_v2x_motion_creation(self):
        """Test V2X motion state"""
        from core.vehicle.v2x import V2XMotion
        
        motion = V2XMotion(
            speed=30.0,
            heading=90.0,
            acceleration=1.0
        )
        
        j2735 = motion.to_j2735()
        assert j2735['speed'] == int(30.0 * 50)
    
    def test_bsm_creation(self):
        """Test BSM creation"""
        from core.vehicle.v2x import BasicSafetyMessage, V2XPosition, V2XMotion
        
        bsm = BasicSafetyMessage(
            msg_count=1,
            temporary_id=bytes([0x12, 0x34, 0x56, 0x78]),
            dsecond=1000,
            position=V2XPosition(37.7749, -122.4194, 10.0),
            motion=V2XMotion(30.0, 90.0)
        )
        
        assert bsm.msg_count == 1
        assert bsm.temporary_id == bytes([0x12, 0x34, 0x56, 0x78])
    
    def test_bsm_encode_decode(self):
        """Test BSM encode/decode round trip"""
        from core.vehicle.v2x import BasicSafetyMessage, V2XPosition, V2XMotion
        
        original = BasicSafetyMessage(
            msg_count=5,
            temporary_id=bytes([0xAA, 0xBB, 0xCC, 0xDD]),
            dsecond=30000,
            position=V2XPosition(40.7128, -74.0060, 50.0),
            motion=V2XMotion(25.0, 180.0, 0.5)
        )
        
        encoded = original.encode()
        decoded = BasicSafetyMessage.decode(encoded)
        
        assert decoded is not None
        assert decoded.msg_count == original.msg_count
        assert decoded.temporary_id == original.temporary_id
    
    def test_bsm_spoofer_initialization(self):
        """Test BSM spoofer initialization"""
        from core.vehicle.v2x import BSMSpoofer
        
        spoofer = BSMSpoofer()
        
        assert spoofer.frequency == 5.9e9
    
    def test_bsm_ghost_vehicle(self):
        """Test ghost vehicle generation"""
        from core.vehicle.v2x import BSMSpoofer, V2XPosition, V2XMotion
        
        spoofer = BSMSpoofer()
        
        bsm = spoofer.generate_ghost_vehicle(
            V2XPosition(37.7749, -122.4194, 0.0),
            V2XMotion(0, 0)
        )
        
        assert bsm is not None
        assert len(bsm.temporary_id) == 4
    
    def test_v2x_jammer_initialization(self):
        """Test V2X jammer initialization"""
        from core.vehicle.v2x import V2XJammer
        
        jammer = V2XJammer()
        
        assert jammer.frequency == 5.9e9
    
    def test_dsrc_attack_initialization(self):
        """Test DSRC attack initialization"""
        from core.vehicle.v2x import DSRCAttack
        
        attack = DSRCAttack()
        
        assert attack._bsm_spoofer is not None
        assert attack._jammer is not None
    
    def test_cv2x_attack_initialization(self):
        """Test C-V2X attack initialization"""
        from core.vehicle.v2x import CV2XAttack
        
        attack = CV2XAttack()
        
        assert attack.frequency == 5.9e9
    
    def test_unified_v2x_attack(self):
        """Test unified V2X attack interface"""
        from core.vehicle.v2x import V2XAttack
        
        attack = V2XAttack()
        
        assert attack.dsrc is not None
        assert attack.cv2x is not None
        assert attack.bsm_spoofer is not None
        assert attack.jammer is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestVehicleModuleIntegration:
    """Integration tests for vehicle modules"""
    
    def test_all_imports(self):
        """Test all vehicle module imports"""
        from core.vehicle import (
            CANBusController, CANFrame,
            UDSClient, UDSService,
            KeyFobAttack, KeyFobProtocol,
            TPMSSpoofer, TPMSProtocol,
            GPSSpoofer, GPSCoordinate,
            VehicleBLEScanner, VehicleBLEAttack,
            V2XAttack, BSMSpoofer
        )
        
        assert CANBusController is not None
        assert UDSClient is not None
        assert KeyFobAttack is not None
        assert TPMSSpoofer is not None
        assert GPSSpoofer is not None
        assert VehicleBLEScanner is not None
        assert V2XAttack is not None
    
    def test_can_uds_integration(self):
        """Test CAN bus and UDS integration"""
        from core.vehicle.can_bus import CANBusController
        from core.vehicle.uds import UDSClient
        
        can = CANBusController()
        uds = UDSClient(can)
        
        assert uds.can == can
    
    def test_full_vehicle_attack_workflow(self):
        """Test complete vehicle attack workflow (simulated)"""
        from core.vehicle.can_bus import CANBusController
        from core.vehicle.uds import UDSClient
        from core.vehicle.key_fob import KeyFobAttack
        from core.vehicle.tpms import TPMSSpoofer
        from core.vehicle.gps_spoof import GPSSpoofer
        from core.vehicle.bluetooth_vehicle import VehicleBLEScanner
        from core.vehicle.v2x import V2XAttack
        
        # Initialize all modules
        can = CANBusController()
        uds = UDSClient(can)
        key_fob = KeyFobAttack()
        tpms = TPMSSpoofer()
        gps = GPSSpoofer()
        ble = VehicleBLEScanner()
        v2x = V2XAttack()
        
        # All should be initialized without SDR (simulated mode)
        assert hasattr(can, '_connected')  # Check internal state
        assert uds.current_session.value == 0x01
        assert len(key_fob.get_captures()) == 0
        assert len(tpms.get_discovered_sensors()) == 0
        assert not gps.is_spoofing
        assert v2x.bsm_spoofer is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
