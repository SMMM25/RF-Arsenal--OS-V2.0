#!/usr/bin/env python3
"""
RF Arsenal OS - Mission Profiles System
Pre-configured operation templates for common white hat scenarios

DESIGN PHILOSOPHY:
- Complex operations made accessible through guided workflows
- Step-by-step execution with safety checks at each phase
- Beginner-friendly explanations with expert shortcuts
- Full audit trail for legal documentation
"""

import os
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MissionCategory(Enum):
    """Categories of mission profiles"""
    WIFI_AUDIT = "wifi_audit"
    CELLULAR_AUDIT = "cellular_audit"
    DRONE_DEFENSE = "drone_defense"
    SPECTRUM_RECON = "spectrum_recon"
    GPS_TESTING = "gps_testing"
    IOT_AUDIT = "iot_audit"
    FULL_SWEEP = "full_sweep"
    CUSTOM = "custom"


class StepStatus(Enum):
    """Status of a mission step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class MissionStep:
    """A single step in a mission profile"""
    id: str
    name: str
    description: str
    command: str  # AI command to execute
    explanation: str  # Beginner-friendly explanation
    is_optional: bool = False
    requires_confirmation: bool = False
    estimated_duration_seconds: int = 30
    safety_notes: List[str] = field(default_factory=list)
    success_criteria: str = ""
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class MissionProfile:
    """A complete mission profile template"""
    id: str
    name: str
    category: MissionCategory
    description: str
    objective: str
    difficulty: str  # "beginner", "intermediate", "advanced"
    estimated_duration_minutes: int
    legal_requirements: List[str]
    prerequisites: List[str]
    steps: List[MissionStep]
    cleanup_steps: List[MissionStep] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MissionExecution:
    """Tracks execution of a mission"""
    mission_id: str
    profile: MissionProfile
    started_at: datetime
    current_step_index: int = 0
    is_running: bool = False
    is_paused: bool = False
    is_completed: bool = False
    is_aborted: bool = False
    results: Dict[str, Any] = field(default_factory=dict)
    audit_log: List[Dict] = field(default_factory=list)


class MissionProfileManager:
    """
    Manages mission profiles and guided execution
    
    Features:
    - Pre-built templates for common operations
    - Step-by-step guided execution
    - Safety checks and confirmations
    - Audit logging for documentation
    - Pause/resume/abort capabilities
    """
    
    def __init__(self, ai_command_center=None):
        self.ai = ai_command_center
        self.profiles: Dict[str, MissionProfile] = {}
        self.active_mission: Optional[MissionExecution] = None
        self._step_callback: Optional[Callable] = None
        self._progress_callback: Optional[Callable] = None
        
        # Load built-in profiles
        self._load_builtin_profiles()
        
        logger.info(f"MissionProfileManager initialized with {len(self.profiles)} profiles")
    
    def _load_builtin_profiles(self):
        """Load all built-in mission profiles"""
        
        # ============ WIFI SECURITY AUDIT ============
        self.profiles['wifi_security_audit'] = MissionProfile(
            id='wifi_security_audit',
            name='WiFi Security Audit',
            category=MissionCategory.WIFI_AUDIT,
            description='Comprehensive WiFi network security assessment',
            objective='Identify vulnerable access points, weak encryption, and potential attack vectors',
            difficulty='beginner',
            estimated_duration_minutes=30,
            legal_requirements=[
                'Written authorization from network owner',
                'Scope document defining target networks',
                'Rules of engagement agreement'
            ],
            prerequisites=[
                'WiFi adapter in monitor mode capable',
                'Target network SSID/BSSID (if specific)',
                'Authorization documentation ready'
            ],
            steps=[
                MissionStep(
                    id='wifi_1',
                    name='Pre-flight Check',
                    description='Verify equipment and authorization',
                    command='status',
                    explanation='Before any security testing, we verify our equipment is working and confirm we have proper authorization. This protects you legally.',
                    requires_confirmation=True,
                    estimated_duration_seconds=10,
                    safety_notes=['Confirm you have written authorization', 'Verify you are testing the correct network']
                ),
                MissionStep(
                    id='wifi_2',
                    name='Enable Stealth Mode',
                    description='Randomize MAC address for anonymity',
                    command='randomize mac address',
                    explanation='We change our wireless adapter\'s MAC address to a random one. This prevents our real hardware ID from being logged by the target network.',
                    estimated_duration_seconds=5,
                    safety_notes=['Your original MAC is hidden', 'Reduces forensic footprint']
                ),
                MissionStep(
                    id='wifi_3',
                    name='Passive Network Discovery',
                    description='Scan for all visible WiFi networks',
                    command='scan wifi networks',
                    explanation='This passively listens for WiFi beacon frames. We\'re not transmitting - just listening to what networks are advertising themselves. Completely legal and undetectable.',
                    estimated_duration_seconds=30,
                    success_criteria='List of detected networks with SSID, BSSID, channel, encryption'
                ),
                MissionStep(
                    id='wifi_4',
                    name='Identify Target Network',
                    description='Confirm target network details',
                    command='show wifi scan results',
                    explanation='Review the discovered networks and identify your authorized target. Note the encryption type (WPA2, WPA3, WEP) and channel.',
                    requires_confirmation=True,
                    estimated_duration_seconds=15,
                    safety_notes=['Only proceed with authorized target', 'Document the target BSSID']
                ),
                MissionStep(
                    id='wifi_5',
                    name='Client Detection',
                    description='Identify devices connected to target network',
                    command='scan wifi clients',
                    explanation='We listen for devices communicating with the target access point. This reveals how many clients are connected and their MAC addresses.',
                    estimated_duration_seconds=60,
                    is_optional=True
                ),
                MissionStep(
                    id='wifi_6',
                    name='Handshake Capture',
                    description='Capture WPA handshake for offline analysis',
                    command='capture wifi handshake',
                    explanation='We capture the 4-way WPA handshake when a client connects. This can be analyzed offline to test password strength. Requires a client to connect (or reconnect).',
                    requires_confirmation=True,
                    estimated_duration_seconds=120,
                    safety_notes=['May require deauth to force reconnection', 'Only on authorized networks']
                ),
                MissionStep(
                    id='wifi_7',
                    name='Generate Report',
                    description='Compile findings into security report',
                    command='generate wifi audit report',
                    explanation='All findings are compiled into a professional report documenting discovered networks, encryption types, vulnerabilities, and recommendations.',
                    estimated_duration_seconds=10
                )
            ],
            cleanup_steps=[
                MissionStep(
                    id='wifi_cleanup_1',
                    name='Stop Capture',
                    description='Stop any active captures',
                    command='stop wifi operations',
                    explanation='Cleanly stop all WiFi monitoring operations.',
                    estimated_duration_seconds=5
                ),
                MissionStep(
                    id='wifi_cleanup_2',
                    name='Restore MAC',
                    description='Optionally restore original MAC',
                    command='restore mac address',
                    explanation='Restore original MAC address if needed.',
                    is_optional=True,
                    estimated_duration_seconds=5
                )
            ],
            tags=['wifi', 'wireless', 'audit', 'beginner', 'passive']
        )
        
        # ============ CELLULAR PENETRATION TEST ============
        self.profiles['cellular_pen_test'] = MissionProfile(
            id='cellular_pen_test',
            name='Cellular Network Pen Test',
            category=MissionCategory.CELLULAR_AUDIT,
            description='Authorized cellular network security assessment',
            objective='Test cellular network defenses, identify IMSI exposure risks, and assess encryption',
            difficulty='advanced',
            estimated_duration_minutes=60,
            legal_requirements=[
                'Explicit written authorization from carrier/organization',
                'FCC experimental license or authorized test environment',
                'Isolated test environment or Faraday cage recommended',
                'Legal counsel review of testing scope'
            ],
            prerequisites=[
                'BladeRF SDR connected and functional',
                'Isolated test environment (Faraday cage ideal)',
                'Test SIM cards for controlled devices',
                'Legal authorization documentation'
            ],
            steps=[
                MissionStep(
                    id='cell_1',
                    name='Legal Verification',
                    description='Confirm all legal requirements are met',
                    command='status',
                    explanation='Cellular testing has strict legal requirements. We verify authorization and confirm we\'re in an isolated environment to prevent affecting public networks.',
                    requires_confirmation=True,
                    estimated_duration_seconds=10,
                    safety_notes=[
                        'CRITICAL: Must have explicit written authorization',
                        'Operating without authorization is a federal crime',
                        'Ensure Faraday cage or isolated environment'
                    ]
                ),
                MissionStep(
                    id='cell_2',
                    name='Hardware Check',
                    description='Verify BladeRF SDR is operational',
                    command='show hardware status',
                    explanation='We verify the Software Defined Radio is connected and functioning properly before any transmission.',
                    estimated_duration_seconds=10
                ),
                MissionStep(
                    id='cell_3',
                    name='Spectrum Survey',
                    description='Scan cellular bands for existing signals',
                    command='scan spectrum 700mhz to 2100mhz',
                    explanation='Before any testing, we survey the cellular spectrum to understand what\'s already present and identify our test frequencies.',
                    estimated_duration_seconds=60
                ),
                MissionStep(
                    id='cell_4',
                    name='Passive IMSI Detection',
                    description='Listen for IMSI transmissions (passive only)',
                    command='start passive cellular monitoring',
                    explanation='We passively monitor for IMSI (phone identifiers) being transmitted. This reveals if devices are exposing their identity - a privacy vulnerability.',
                    estimated_duration_seconds=120,
                    safety_notes=['Passive only - no transmission', 'Documents existing vulnerabilities']
                ),
                MissionStep(
                    id='cell_5',
                    name='Test BTS Setup',
                    description='Configure test base station (isolated environment only)',
                    command='configure 4g test base station',
                    explanation='In our isolated test environment, we set up a controlled base station. This simulates a rogue cell tower to test device behavior.',
                    requires_confirmation=True,
                    estimated_duration_seconds=30,
                    safety_notes=[
                        'MUST be in Faraday cage or isolated environment',
                        'Use minimal power settings',
                        'Only test authorized devices'
                    ]
                ),
                MissionStep(
                    id='cell_6',
                    name='Device Response Test',
                    description='Test how authorized devices respond to test BTS',
                    command='start imsi catch mode',
                    explanation='We observe how test devices respond to our base station. Vulnerable devices will connect and reveal their IMSI.',
                    requires_confirmation=True,
                    estimated_duration_seconds=180,
                    safety_notes=['Only test pre-authorized devices', 'Document all captured data']
                ),
                MissionStep(
                    id='cell_7',
                    name='Encryption Analysis',
                    description='Analyze encryption negotiation',
                    command='analyze cellular encryption',
                    explanation='We examine what encryption the devices negotiate. Weak or absent encryption is a critical vulnerability.',
                    estimated_duration_seconds=60
                ),
                MissionStep(
                    id='cell_8',
                    name='Generate Report',
                    description='Compile findings into security report',
                    command='generate cellular audit report',
                    explanation='All findings documented with vulnerabilities, risk ratings, and remediation recommendations.',
                    estimated_duration_seconds=15
                )
            ],
            cleanup_steps=[
                MissionStep(
                    id='cell_cleanup_1',
                    name='Stop BTS',
                    description='Immediately stop test base station',
                    command='stop cellular operations',
                    explanation='Critical: Stop all cellular transmissions immediately.',
                    estimated_duration_seconds=5
                ),
                MissionStep(
                    id='cell_cleanup_2',
                    name='Secure Data',
                    description='Encrypt captured test data',
                    command='encrypt mission data',
                    explanation='All captured IMSI and device data is encrypted for secure storage.',
                    estimated_duration_seconds=10
                )
            ],
            tags=['cellular', 'imsi', 'advanced', 'requires-authorization', 'sdr']
        )
        
        # ============ DRONE DETECTION & DEFENSE ============
        self.profiles['drone_defense'] = MissionProfile(
            id='drone_defense',
            name='Drone Detection & Defense',
            category=MissionCategory.DRONE_DEFENSE,
            description='Detect, track, and optionally neutralize unauthorized drones',
            objective='Establish drone detection perimeter and response capability',
            difficulty='intermediate',
            estimated_duration_minutes=45,
            legal_requirements=[
                'Property owner authorization for detection',
                'Jamming requires specific authorization (varies by jurisdiction)',
                'Document all detected drones for legal purposes'
            ],
            prerequisites=[
                'BladeRF SDR connected',
                'Clear line of sight to sky',
                'Understanding of local drone regulations'
            ],
            steps=[
                MissionStep(
                    id='drone_1',
                    name='System Check',
                    description='Verify detection equipment',
                    command='status',
                    explanation='We verify all detection equipment is operational before establishing our drone detection perimeter.',
                    estimated_duration_seconds=10
                ),
                MissionStep(
                    id='drone_2',
                    name='Frequency Survey',
                    description='Scan drone control frequencies',
                    command='scan spectrum 2.4ghz to 5.8ghz',
                    explanation='Consumer drones typically operate on 2.4GHz and 5.8GHz bands. We survey these frequencies to establish a baseline.',
                    estimated_duration_seconds=30
                ),
                MissionStep(
                    id='drone_3',
                    name='Enable Detection',
                    description='Start continuous drone detection',
                    command='detect drones',
                    explanation='We begin actively monitoring for drone control signals. The system recognizes signatures from DJI, FPV, and other common drones.',
                    estimated_duration_seconds=10,
                    success_criteria='Detection system active, monitoring 2.4/5.8 GHz'
                ),
                MissionStep(
                    id='drone_4',
                    name='Signal Analysis',
                    description='Analyze any detected drone signals',
                    command='analyze drone signals',
                    explanation='When a drone is detected, we analyze its control protocol to identify the drone type, estimate distance, and locate the operator.',
                    estimated_duration_seconds=60,
                    is_optional=True
                ),
                MissionStep(
                    id='drone_5',
                    name='Direction Finding',
                    description='Estimate drone/operator location',
                    command='locate drone signal source',
                    explanation='Using signal strength analysis, we estimate the direction and approximate distance to both the drone and its operator.',
                    estimated_duration_seconds=30,
                    is_optional=True
                ),
                MissionStep(
                    id='drone_6',
                    name='Enable Auto-Defense',
                    description='Activate automatic countermeasures',
                    command='enable drone auto-defend',
                    explanation='CAUTION: This enables automatic jamming when threats are detected. Only use with proper authorization.',
                    requires_confirmation=True,
                    is_optional=True,
                    estimated_duration_seconds=10,
                    safety_notes=[
                        'Jamming may be illegal in your jurisdiction',
                        'Will disrupt all 2.4/5.8 GHz devices in range',
                        'Use only with explicit authorization'
                    ]
                ),
                MissionStep(
                    id='drone_7',
                    name='Generate Detection Log',
                    description='Export detection records',
                    command='export drone detection log',
                    explanation='All detected drones, timestamps, and signal data are exported for documentation and potential legal action.',
                    estimated_duration_seconds=10
                )
            ],
            cleanup_steps=[
                MissionStep(
                    id='drone_cleanup_1',
                    name='Stop Countermeasures',
                    description='Disable any active jamming',
                    command='stop drone jamming',
                    explanation='Stop all active countermeasures.',
                    estimated_duration_seconds=5
                ),
                MissionStep(
                    id='drone_cleanup_2',
                    name='Stop Detection',
                    description='Stop drone detection monitoring',
                    command='stop drone detection',
                    explanation='Cleanly stop detection systems.',
                    estimated_duration_seconds=5
                )
            ],
            tags=['drone', 'uav', 'detection', 'defense', 'intermediate']
        )
        
        # ============ FULL SPECTRUM RECONNAISSANCE ============
        self.profiles['spectrum_recon'] = MissionProfile(
            id='spectrum_recon',
            name='Full Spectrum Reconnaissance',
            category=MissionCategory.SPECTRUM_RECON,
            description='Comprehensive RF environment survey and analysis',
            objective='Map all RF activity in the area for security assessment',
            difficulty='beginner',
            estimated_duration_minutes=20,
            legal_requirements=[
                'Passive reception is generally legal',
                'No transmission involved',
                'Respect privacy - do not record communications content'
            ],
            prerequisites=[
                'BladeRF SDR connected',
                'Antenna appropriate for frequency range'
            ],
            steps=[
                MissionStep(
                    id='spec_1',
                    name='Hardware Verification',
                    description='Verify SDR is connected and calibrated',
                    command='show hardware status',
                    explanation='We verify the Software Defined Radio is properly connected and ready for spectrum analysis.',
                    estimated_duration_seconds=10
                ),
                MissionStep(
                    id='spec_2',
                    name='VHF/UHF Survey',
                    description='Scan 30 MHz - 1 GHz',
                    command='scan spectrum 30mhz to 1ghz',
                    explanation='This range covers FM radio, TV broadcasts, amateur radio, public safety, and many IoT devices. We\'re building a picture of the RF environment.',
                    estimated_duration_seconds=120
                ),
                MissionStep(
                    id='spec_3',
                    name='Cellular Band Survey',
                    description='Scan cellular frequencies',
                    command='scan spectrum 700mhz to 2200mhz',
                    explanation='We examine cellular bands to identify carriers present, signal strengths, and potential rogue base stations.',
                    estimated_duration_seconds=90
                ),
                MissionStep(
                    id='spec_4',
                    name='WiFi/Bluetooth Survey',
                    description='Scan 2.4 GHz and 5 GHz bands',
                    command='scan spectrum 2.4ghz to 5.9ghz',
                    explanation='These bands contain WiFi, Bluetooth, Zigbee, drones, and many consumer devices. High activity expected.',
                    estimated_duration_seconds=60
                ),
                MissionStep(
                    id='spec_5',
                    name='Signal Classification',
                    description='Identify and classify detected signals',
                    command='classify detected signals',
                    explanation='The system attempts to identify what each detected signal is - cellular, WiFi, radio, etc.',
                    estimated_duration_seconds=30
                ),
                MissionStep(
                    id='spec_6',
                    name='Anomaly Detection',
                    description='Identify unusual or suspicious signals',
                    command='detect spectrum anomalies',
                    explanation='We look for signals that don\'t belong - potential bugs, unauthorized transmitters, or suspicious activity.',
                    estimated_duration_seconds=30
                ),
                MissionStep(
                    id='spec_7',
                    name='Generate Spectrum Report',
                    description='Create comprehensive RF environment report',
                    command='generate spectrum report',
                    explanation='All findings compiled into a report with signal maps, identified sources, and potential security concerns.',
                    estimated_duration_seconds=15
                )
            ],
            cleanup_steps=[
                MissionStep(
                    id='spec_cleanup_1',
                    name='Stop Scanning',
                    description='Stop spectrum analysis',
                    command='stop spectrum scan',
                    explanation='Cleanly stop spectrum analysis.',
                    estimated_duration_seconds=5
                )
            ],
            tags=['spectrum', 'reconnaissance', 'passive', 'beginner', 'survey']
        )
        
        # ============ GPS SECURITY TESTING ============
        self.profiles['gps_security_test'] = MissionProfile(
            id='gps_security_test',
            name='GPS Security Testing',
            category=MissionCategory.GPS_TESTING,
            description='Test GPS receiver vulnerability to spoofing attacks',
            objective='Assess GPS-dependent system resilience to spoofing',
            difficulty='intermediate',
            estimated_duration_minutes=25,
            legal_requirements=[
                'MUST be in shielded/isolated environment',
                'Written authorization for all test devices',
                'GPS spoofing in open air is ILLEGAL',
                'Faraday cage or RF-shielded room required'
            ],
            prerequisites=[
                'Faraday cage or RF-shielded test environment',
                'BladeRF SDR with GPS-capable antenna',
                'Test devices with GPS (authorized)',
                'Legal authorization documentation'
            ],
            steps=[
                MissionStep(
                    id='gps_1',
                    name='Environment Verification',
                    description='Confirm isolated test environment',
                    command='status',
                    explanation='CRITICAL: GPS spoofing MUST be done in a shielded environment. Spoofing in open air affects aircraft and is a federal crime.',
                    requires_confirmation=True,
                    estimated_duration_seconds=10,
                    safety_notes=[
                        'MUST be in Faraday cage or shielded room',
                        'Verify NO GPS signal leakage to outside',
                        'Open-air GPS spoofing is a FEDERAL CRIME'
                    ]
                ),
                MissionStep(
                    id='gps_2',
                    name='Baseline GPS Reception',
                    description='Verify test device receives real GPS (or not in cage)',
                    command='show gps status',
                    explanation='We verify our test device\'s GPS behavior before spoofing. In a proper Faraday cage, it should show no GPS fix.',
                    estimated_duration_seconds=30
                ),
                MissionStep(
                    id='gps_3',
                    name='Configure Spoof Location',
                    description='Set target spoofed coordinates',
                    command='configure gps spoof location 37.7749 -122.4194',
                    explanation='We configure a fake GPS location. The test device should report being at this location when spoofing is active.',
                    estimated_duration_seconds=10
                ),
                MissionStep(
                    id='gps_4',
                    name='Start GPS Spoofing',
                    description='Begin transmitting fake GPS signal',
                    command='start gps spoofing',
                    explanation='We transmit a fake GPS signal. Vulnerable devices will lock onto our signal and report the fake location.',
                    requires_confirmation=True,
                    estimated_duration_seconds=10,
                    safety_notes=['Confirm shielded environment', 'Minimal power output']
                ),
                MissionStep(
                    id='gps_5',
                    name='Verify Spoof Success',
                    description='Check if test device reports spoofed location',
                    command='verify gps spoof effectiveness',
                    explanation='We check if the test device is now reporting our fake location. This indicates GPS vulnerability.',
                    estimated_duration_seconds=60,
                    success_criteria='Test device reports spoofed coordinates'
                ),
                MissionStep(
                    id='gps_6',
                    name='Time Spoofing Test',
                    description='Test GPS time manipulation',
                    command='test gps time spoofing',
                    explanation='GPS also provides time. We test if the device accepts manipulated time, which can break security certificates and logs.',
                    is_optional=True,
                    estimated_duration_seconds=30
                ),
                MissionStep(
                    id='gps_7',
                    name='Generate Report',
                    description='Document GPS vulnerability findings',
                    command='generate gps security report',
                    explanation='Results documented with vulnerability assessment and recommendations for GPS security improvements.',
                    estimated_duration_seconds=10
                )
            ],
            cleanup_steps=[
                MissionStep(
                    id='gps_cleanup_1',
                    name='Stop Spoofing',
                    description='Immediately stop GPS transmission',
                    command='stop gps spoofing',
                    explanation='Critical: Stop GPS transmission immediately.',
                    estimated_duration_seconds=5
                )
            ],
            tags=['gps', 'spoofing', 'location', 'intermediate', 'requires-shielding']
        )
    
    def list_profiles(self, category: Optional[MissionCategory] = None, 
                     difficulty: Optional[str] = None) -> List[Dict]:
        """List available mission profiles with optional filtering"""
        profiles = []
        
        for profile in self.profiles.values():
            # Filter by category
            if category and profile.category != category:
                continue
            
            # Filter by difficulty
            if difficulty and profile.difficulty != difficulty:
                continue
            
            profiles.append({
                'id': profile.id,
                'name': profile.name,
                'category': profile.category.value,
                'difficulty': profile.difficulty,
                'duration_minutes': profile.estimated_duration_minutes,
                'description': profile.description,
                'steps': len(profile.steps)
            })
        
        return profiles
    
    def get_profile(self, profile_id: str) -> Optional[MissionProfile]:
        """Get a specific mission profile"""
        return self.profiles.get(profile_id)
    
    def get_profile_details(self, profile_id: str) -> Optional[Dict]:
        """Get detailed information about a profile"""
        profile = self.profiles.get(profile_id)
        if not profile:
            return None
        
        return {
            'id': profile.id,
            'name': profile.name,
            'category': profile.category.value,
            'description': profile.description,
            'objective': profile.objective,
            'difficulty': profile.difficulty,
            'duration_minutes': profile.estimated_duration_minutes,
            'legal_requirements': profile.legal_requirements,
            'prerequisites': profile.prerequisites,
            'steps': [
                {
                    'id': step.id,
                    'name': step.name,
                    'description': step.description,
                    'optional': step.is_optional,
                    'requires_confirmation': step.requires_confirmation,
                    'duration_seconds': step.estimated_duration_seconds
                }
                for step in profile.steps
            ],
            'cleanup_steps': len(profile.cleanup_steps),
            'tags': profile.tags
        }
    
    def start_mission(self, profile_id: str) -> Dict:
        """Start executing a mission profile"""
        if self.active_mission and self.active_mission.is_running:
            return {
                'success': False,
                'error': 'A mission is already in progress. Abort it first.'
            }
        
        profile = self.profiles.get(profile_id)
        if not profile:
            return {
                'success': False,
                'error': f'Mission profile not found: {profile_id}'
            }
        
        # Create execution tracker
        self.active_mission = MissionExecution(
            mission_id=f"{profile_id}_{int(time.time())}",
            profile=profile,
            started_at=datetime.now(),
            is_running=True
        )
        
        # Log mission start
        self._log_audit('mission_started', {
            'profile_id': profile_id,
            'profile_name': profile.name
        })
        
        logger.info(f"Mission started: {profile.name}")
        
        return {
            'success': True,
            'mission_id': self.active_mission.mission_id,
            'profile_name': profile.name,
            'total_steps': len(profile.steps),
            'message': f"Mission '{profile.name}' started.\n"
                      f"Legal requirements:\n" + 
                      '\n'.join([f"  - {req}" for req in profile.legal_requirements]) +
                      f"\n\nSay 'next step' to begin or 'show mission status' for details."
        }
    
    def get_current_step(self) -> Optional[Dict]:
        """Get the current step in the active mission"""
        if not self.active_mission or not self.active_mission.is_running:
            return None
        
        profile = self.active_mission.profile
        idx = self.active_mission.current_step_index
        
        if idx >= len(profile.steps):
            return None
        
        step = profile.steps[idx]
        
        return {
            'step_number': idx + 1,
            'total_steps': len(profile.steps),
            'id': step.id,
            'name': step.name,
            'description': step.description,
            'explanation': step.explanation,
            'command': step.command,
            'is_optional': step.is_optional,
            'requires_confirmation': step.requires_confirmation,
            'safety_notes': step.safety_notes,
            'estimated_duration': step.estimated_duration_seconds
        }
    
    def execute_current_step(self, confirmed: bool = False) -> Dict:
        """Execute the current step of the active mission"""
        if not self.active_mission or not self.active_mission.is_running:
            return {
                'success': False,
                'error': 'No active mission. Start a mission first.'
            }
        
        profile = self.active_mission.profile
        idx = self.active_mission.current_step_index
        
        if idx >= len(profile.steps):
            return {
                'success': False,
                'error': 'Mission completed. No more steps.'
            }
        
        step = profile.steps[idx]
        
        # Check if confirmation required
        if step.requires_confirmation and not confirmed:
            return {
                'success': True,
                'requires_confirmation': True,
                'step_name': step.name,
                'safety_notes': step.safety_notes,
                'message': f"CONFIRMATION REQUIRED for step: {step.name}\n\n"
                          f"Safety notes:\n" +
                          '\n'.join([f"  ! {note}" for note in step.safety_notes]) +
                          f"\n\nSay 'confirm' to proceed or 'skip step' to skip."
            }
        
        # Mark step as in progress
        step.status = StepStatus.IN_PROGRESS
        step.started_at = datetime.now()
        
        # Execute via AI Command Center if available
        result_message = ""
        if self.ai:
            try:
                result = self.ai.process_command(step.command)
                result_message = result.message
                step.result = result_message
            except Exception as e:
                result_message = f"Step execution error: {e}"
                step.status = StepStatus.FAILED
                step.result = str(e)
        else:
            result_message = f"[Simulated] Executed: {step.command}"
            step.result = result_message
        
        # Mark step as completed
        step.status = StepStatus.COMPLETED
        step.completed_at = datetime.now()
        
        # Log to audit
        self._log_audit('step_completed', {
            'step_id': step.id,
            'step_name': step.name,
            'command': step.command,
            'result': step.result
        })
        
        # Move to next step
        self.active_mission.current_step_index += 1
        
        # Check if mission complete
        if self.active_mission.current_step_index >= len(profile.steps):
            return self._complete_mission()
        
        # Get next step preview
        next_step = profile.steps[self.active_mission.current_step_index]
        
        return {
            'success': True,
            'step_completed': step.name,
            'result': result_message,
            'progress': f"{self.active_mission.current_step_index}/{len(profile.steps)}",
            'next_step': {
                'name': next_step.name,
                'description': next_step.description,
                'optional': next_step.is_optional
            },
            'message': f"Step completed: {step.name}\n\n"
                      f"Result: {result_message}\n\n"
                      f"Next: {next_step.name}\n"
                      f"{'(Optional - say \"skip step\" to skip)' if next_step.is_optional else ''}\n"
                      f"Say 'next step' to continue."
        }
    
    def skip_step(self) -> Dict:
        """Skip the current step (if optional)"""
        if not self.active_mission or not self.active_mission.is_running:
            return {'success': False, 'error': 'No active mission.'}
        
        profile = self.active_mission.profile
        idx = self.active_mission.current_step_index
        
        if idx >= len(profile.steps):
            return {'success': False, 'error': 'No more steps.'}
        
        step = profile.steps[idx]
        
        if not step.is_optional:
            return {
                'success': False,
                'error': f"Step '{step.name}' is required and cannot be skipped."
            }
        
        step.status = StepStatus.SKIPPED
        self.active_mission.current_step_index += 1
        
        self._log_audit('step_skipped', {'step_id': step.id, 'step_name': step.name})
        
        # Check if mission complete
        if self.active_mission.current_step_index >= len(profile.steps):
            return self._complete_mission()
        
        next_step = profile.steps[self.active_mission.current_step_index]
        
        return {
            'success': True,
            'skipped': step.name,
            'next_step': next_step.name,
            'message': f"Skipped: {step.name}\nNext: {next_step.name}"
        }
    
    def _complete_mission(self) -> Dict:
        """Mark mission as completed and run cleanup"""
        if not self.active_mission:
            return {'success': False, 'error': 'No active mission.'}
        
        profile = self.active_mission.profile
        
        # Run cleanup steps
        cleanup_results = []
        for step in profile.cleanup_steps:
            if self.ai:
                try:
                    result = self.ai.process_command(step.command)
                    cleanup_results.append(f"{step.name}: OK")
                except:
                    cleanup_results.append(f"{step.name}: SKIPPED")
            else:
                cleanup_results.append(f"{step.name}: [Simulated]")
        
        self.active_mission.is_completed = True
        self.active_mission.is_running = False
        
        self._log_audit('mission_completed', {
            'mission_id': self.active_mission.mission_id,
            'profile_name': profile.name,
            'steps_completed': self.active_mission.current_step_index
        })
        
        duration = datetime.now() - self.active_mission.started_at
        
        return {
            'success': True,
            'mission_completed': True,
            'profile_name': profile.name,
            'duration_seconds': int(duration.total_seconds()),
            'cleanup_results': cleanup_results,
            'message': f"MISSION COMPLETED: {profile.name}\n"
                      f"Duration: {int(duration.total_seconds())} seconds\n"
                      f"Cleanup: {len(cleanup_results)} steps executed\n\n"
                      f"All findings logged for audit trail."
        }
    
    def abort_mission(self, reason: str = "User requested") -> Dict:
        """Abort the current mission and run cleanup"""
        if not self.active_mission:
            return {'success': False, 'error': 'No active mission to abort.'}
        
        profile = self.active_mission.profile
        
        # Run cleanup steps
        for step in profile.cleanup_steps:
            if self.ai:
                try:
                    self.ai.process_command(step.command)
                except:
                    pass
        
        self.active_mission.is_aborted = True
        self.active_mission.is_running = False
        
        self._log_audit('mission_aborted', {
            'mission_id': self.active_mission.mission_id,
            'reason': reason,
            'step_reached': self.active_mission.current_step_index
        })
        
        mission_name = profile.name
        self.active_mission = None
        
        return {
            'success': True,
            'message': f"Mission aborted: {mission_name}\n"
                      f"Reason: {reason}\n"
                      f"Cleanup steps executed."
        }
    
    def get_mission_status(self) -> Dict:
        """Get current mission status"""
        if not self.active_mission:
            return {
                'active': False,
                'message': 'No active mission. Say "list missions" to see available profiles.'
            }
        
        profile = self.active_mission.profile
        idx = self.active_mission.current_step_index
        
        elapsed = datetime.now() - self.active_mission.started_at
        
        status = {
            'active': True,
            'mission_id': self.active_mission.mission_id,
            'profile_name': profile.name,
            'progress': f"{idx}/{len(profile.steps)}",
            'elapsed_seconds': int(elapsed.total_seconds()),
            'is_paused': self.active_mission.is_paused,
            'current_step': None
        }
        
        if idx < len(profile.steps):
            step = profile.steps[idx]
            status['current_step'] = {
                'name': step.name,
                'description': step.description
            }
        
        return status
    
    def _log_audit(self, event: str, data: Dict):
        """Log event to mission audit trail"""
        if self.active_mission:
            self.active_mission.audit_log.append({
                'timestamp': datetime.now().isoformat(),
                'event': event,
                'data': data
            })
        
        logger.info(f"Mission audit: {event} - {data}")


# Global instance
_mission_manager: Optional[MissionProfileManager] = None


def get_mission_manager(ai_command_center=None) -> MissionProfileManager:
    """Get global MissionProfileManager instance"""
    global _mission_manager
    if _mission_manager is None:
        _mission_manager = MissionProfileManager(ai_command_center)
    return _mission_manager


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = get_mission_manager()
    
    print("=" * 60)
    print("RF Arsenal OS - Mission Profiles")
    print("=" * 60)
    
    profiles = manager.list_profiles()
    print(f"\nAvailable Missions ({len(profiles)}):\n")
    
    for p in profiles:
        print(f"  [{p['difficulty'].upper()}] {p['name']}")
        print(f"      {p['description']}")
        print(f"      Duration: ~{p['duration_minutes']} min | Steps: {p['steps']}")
        print()
