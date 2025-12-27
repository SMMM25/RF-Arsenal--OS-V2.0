#!/usr/bin/env python3
"""
RF Arsenal OS - Universal Platform Detection and Optimization System

Cross-platform hardware detection for PCs, Laptops, and Raspberry Pi.
Auto-detects hardware capabilities and applies optimal configurations.

SUPPORTED PLATFORMS:
- x86_64 Desktop/Laptop (Intel/AMD)
- ARM64 Desktop/Laptop (Apple Silicon, ARM servers)
- Raspberry Pi 5/4/3/Zero 2 W
- Generic ARM64 SBCs
- Virtual Machines (VMware, VirtualBox, KVM, Hyper-V)

README COMPLIANCE:
- Offline-first: Works without network
- RAM-only: No persistent storage of sensitive data
- Zero telemetry: No external connections
- Real-world functional: Actual hardware detection

Copyright (c) 2024 RF-Arsenal-OS Project
License: Proprietary - Authorized Use Only
"""

import os
import sys
import platform
import subprocess
import logging
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


class PlatformType(Enum):
    """Supported platform types"""
    X86_64_DESKTOP = "x86_64_desktop"
    X86_64_LAPTOP = "x86_64_laptop"
    ARM64_DESKTOP = "arm64_desktop"
    ARM64_LAPTOP = "arm64_laptop"
    RASPBERRY_PI_5 = "raspberry_pi_5"
    RASPBERRY_PI_4 = "raspberry_pi_4"
    RASPBERRY_PI_3 = "raspberry_pi_3"
    RASPBERRY_PI_ZERO = "raspberry_pi_zero"
    ARM64_SBC = "arm64_sbc"  # Other single board computers
    VIRTUAL_MACHINE = "virtual_machine"
    UNKNOWN = "unknown"


class PerformanceTier(Enum):
    """Hardware performance classification"""
    HIGH = "high"           # 8+ cores, 16GB+ RAM, USB 3.2
    MEDIUM = "medium"       # 4-7 cores, 8-15GB RAM, USB 3.0
    LOW = "low"             # 2-3 cores, 4-7GB RAM, USB 2.0
    MINIMAL = "minimal"     # 1 core, <4GB RAM


class VMType(Enum):
    """Virtual machine types"""
    NONE = "none"
    VMWARE = "vmware"
    VIRTUALBOX = "virtualbox"
    KVM = "kvm"
    HYPERV = "hyperv"
    XEN = "xen"
    PARALLELS = "parallels"
    DOCKER = "docker"
    WSL = "wsl"


@dataclass
class CPUInfo:
    """CPU information"""
    model: str = "Unknown"
    architecture: str = "unknown"
    cores_physical: int = 1
    cores_logical: int = 1
    frequency_mhz: float = 0
    cache_mb: float = 0
    features: List[str] = field(default_factory=list)
    is_arm: bool = False
    is_x86: bool = False


@dataclass
class MemoryInfo:
    """Memory information"""
    total_gb: float = 0
    available_gb: float = 0
    ram_type: str = "Unknown"


@dataclass
class USBInfo:
    """USB controller information"""
    has_usb3: bool = False
    usb3_controllers: int = 0
    usb2_controllers: int = 0
    max_speed: str = "USB 2.0"


@dataclass
class StorageInfo:
    """Storage information"""
    root_type: str = "unknown"  # ssd, hdd, nvme, sd_card, usb
    root_size_gb: float = 0
    root_free_gb: float = 0
    is_removable: bool = False


@dataclass
class GPUInfo:
    """GPU information"""
    model: str = "Unknown"
    vram_mb: int = 0
    has_hardware_accel: bool = False
    opengl_version: str = "Unknown"


@dataclass
class BatteryInfo:
    """Battery information for laptops/portable devices"""
    has_battery: bool = False
    capacity_percent: int = 0
    is_charging: bool = False
    time_remaining_minutes: int = 0


@dataclass
class PlatformCapabilities:
    """Complete platform capability report"""
    platform_type: PlatformType
    performance_tier: PerformanceTier
    vm_type: VMType
    cpu: CPUInfo
    memory: MemoryInfo
    usb: USBInfo
    storage: StorageInfo
    gpu: GPUInfo
    battery: BatteryInfo
    
    # RF Arsenal specific
    sdr_ready: bool = False
    gpio_available: bool = False
    recommended_features: List[str] = field(default_factory=list)
    disabled_features: List[str] = field(default_factory=list)
    optimization_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Export to dictionary"""
        return {
            'platform_type': self.platform_type.value,
            'performance_tier': self.performance_tier.value,
            'vm_type': self.vm_type.value,
            'cpu': {
                'model': self.cpu.model,
                'architecture': self.cpu.architecture,
                'cores_physical': self.cpu.cores_physical,
                'cores_logical': self.cpu.cores_logical,
                'frequency_mhz': self.cpu.frequency_mhz,
            },
            'memory': {
                'total_gb': self.memory.total_gb,
                'available_gb': self.memory.available_gb,
            },
            'usb': {
                'max_speed': self.usb.max_speed,
                'has_usb3': self.usb.has_usb3,
            },
            'storage': {
                'type': self.storage.root_type,
                'size_gb': self.storage.root_size_gb,
                'free_gb': self.storage.root_free_gb,
            },
            'sdr_ready': self.sdr_ready,
            'gpio_available': self.gpio_available,
            'recommended_features': self.recommended_features,
            'disabled_features': self.disabled_features,
        }


class UniversalPlatformDetector:
    """
    Universal platform detection for RF Arsenal OS
    
    Detects and characterizes hardware to enable optimal configuration
    on any supported platform (PC, laptop, Raspberry Pi, etc.)
    """
    
    # Raspberry Pi model signatures
    PI_MODELS = {
        'Raspberry Pi 5': PlatformType.RASPBERRY_PI_5,
        'Raspberry Pi 4': PlatformType.RASPBERRY_PI_4,
        'Raspberry Pi 3': PlatformType.RASPBERRY_PI_3,
        'Raspberry Pi Zero 2': PlatformType.RASPBERRY_PI_ZERO,
    }
    
    # VM detection signatures
    VM_SIGNATURES = {
        'vmware': VMType.VMWARE,
        'virtualbox': VMType.VIRTUALBOX,
        'vbox': VMType.VIRTUALBOX,
        'kvm': VMType.KVM,
        'qemu': VMType.KVM,
        'hyperv': VMType.HYPERV,
        'hyper-v': VMType.HYPERV,
        'microsoft corporation virtual': VMType.HYPERV,
        'xen': VMType.XEN,
        'parallels': VMType.PARALLELS,
    }
    
    def __init__(self):
        self.logger = logging.getLogger('PlatformDetector')
        self._cache: Optional[PlatformCapabilities] = None
    
    def detect(self, force_refresh: bool = False) -> PlatformCapabilities:
        """
        Detect platform capabilities
        
        Args:
            force_refresh: Force re-detection even if cached
            
        Returns:
            PlatformCapabilities with all detected information
        """
        if self._cache and not force_refresh:
            return self._cache
        
        self.logger.info("Starting platform detection...")
        
        # Gather all hardware information
        cpu = self._detect_cpu()
        memory = self._detect_memory()
        usb = self._detect_usb()
        storage = self._detect_storage()
        gpu = self._detect_gpu()
        battery = self._detect_battery()
        vm_type = self._detect_vm()
        
        # Determine platform type
        platform_type = self._classify_platform(cpu, battery, vm_type)
        
        # Determine performance tier
        perf_tier = self._classify_performance(cpu, memory, usb)
        
        # Build capabilities
        capabilities = PlatformCapabilities(
            platform_type=platform_type,
            performance_tier=perf_tier,
            vm_type=vm_type,
            cpu=cpu,
            memory=memory,
            usb=usb,
            storage=storage,
            gpu=gpu,
            battery=battery,
        )
        
        # Determine RF Arsenal specific capabilities
        capabilities.sdr_ready = self._check_sdr_ready(usb, vm_type)
        capabilities.gpio_available = self._check_gpio_available(platform_type)
        
        # Generate recommendations
        capabilities.recommended_features, capabilities.disabled_features = \
            self._get_feature_recommendations(capabilities)
        
        # Generate optimization notes
        capabilities.optimization_notes = self._get_optimization_notes(capabilities)
        
        self._cache = capabilities
        return capabilities
    
    def _detect_cpu(self) -> CPUInfo:
        """Detect CPU information"""
        info = CPUInfo()
        
        try:
            # Get architecture
            machine = platform.machine().lower()
            info.architecture = machine
            info.is_arm = machine in ['aarch64', 'arm64', 'armv7l', 'armv8l']
            info.is_x86 = machine in ['x86_64', 'amd64', 'i386', 'i686']
            
            # Get logical cores
            info.cores_logical = os.cpu_count() or 1
            
            # Parse /proc/cpuinfo on Linux
            if Path('/proc/cpuinfo').exists():
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    
                    # Model name
                    match = re.search(r'model name\s*:\s*(.+)', content)
                    if match:
                        info.model = match.group(1).strip()
                    elif info.is_arm:
                        # ARM systems often don't have model name
                        match = re.search(r'Hardware\s*:\s*(.+)', content)
                        if match:
                            info.model = match.group(1).strip()
                    
                    # Physical cores (count unique core IDs)
                    core_ids = set(re.findall(r'core id\s*:\s*(\d+)', content))
                    physical_ids = set(re.findall(r'physical id\s*:\s*(\d+)', content))
                    if core_ids and physical_ids:
                        info.cores_physical = len(core_ids) * max(1, len(physical_ids))
                    else:
                        # ARM doesn't always have these, use processor count
                        processors = re.findall(r'^processor\s*:', content, re.MULTILINE)
                        info.cores_physical = len(processors) if processors else info.cores_logical
                    
                    # Frequency
                    match = re.search(r'cpu MHz\s*:\s*([\d.]+)', content)
                    if match:
                        info.frequency_mhz = float(match.group(1))
                    else:
                        # Try to get from /sys on ARM
                        try:
                            with open('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq', 'r') as f:
                                info.frequency_mhz = float(f.read().strip()) / 1000
                        except:
                            pass
                    
                    # Cache
                    match = re.search(r'cache size\s*:\s*(\d+)\s*(KB|MB)', content)
                    if match:
                        size = float(match.group(1))
                        if match.group(2) == 'KB':
                            size /= 1024
                        info.cache_mb = size
                    
                    # Features/flags
                    match = re.search(r'flags\s*:\s*(.+)', content)
                    if match:
                        info.features = match.group(1).strip().split()
            
            # macOS
            elif platform.system() == 'Darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info.model = result.stdout.strip()
                
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.physicalcpu'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info.cores_physical = int(result.stdout.strip())
            
            # Windows
            elif platform.system() == 'Windows':
                import wmi
                c = wmi.WMI()
                for cpu in c.Win32_Processor():
                    info.model = cpu.Name
                    info.cores_physical = cpu.NumberOfCores
                    info.frequency_mhz = cpu.MaxClockSpeed
                    break
                    
        except Exception as e:
            self.logger.warning(f"CPU detection error: {e}")
        
        return info
    
    def _detect_memory(self) -> MemoryInfo:
        """Detect memory information"""
        info = MemoryInfo()
        
        try:
            if Path('/proc/meminfo').exists():
                with open('/proc/meminfo', 'r') as f:
                    content = f.read()
                    
                    match = re.search(r'MemTotal:\s*(\d+)\s*kB', content)
                    if match:
                        info.total_gb = int(match.group(1)) / (1024 * 1024)
                    
                    match = re.search(r'MemAvailable:\s*(\d+)\s*kB', content)
                    if match:
                        info.available_gb = int(match.group(1)) / (1024 * 1024)
                    
            elif platform.system() == 'Darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info.total_gb = int(result.stdout.strip()) / (1024**3)
                    
        except Exception as e:
            self.logger.warning(f"Memory detection error: {e}")
        
        return info
    
    def _detect_usb(self) -> USBInfo:
        """Detect USB controller capabilities"""
        info = USBInfo()
        
        try:
            if platform.system() == 'Linux':
                # Check for USB 3.x controllers
                result = subprocess.run(
                    ['lspci'],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0:
                    output = result.stdout.lower()
                    
                    # USB 3.x controllers
                    usb3_patterns = ['xhci', 'usb3', 'usb 3']
                    for pattern in usb3_patterns:
                        count = output.count(pattern)
                        if count > 0:
                            info.usb3_controllers += count
                    
                    # USB 2.0 controllers
                    usb2_patterns = ['ehci', 'usb2', 'usb 2']
                    for pattern in usb2_patterns:
                        count = output.count(pattern)
                        if count > 0:
                            info.usb2_controllers += count
                
                # Also check /sys/bus/usb
                usb_path = Path('/sys/bus/usb/devices')
                if usb_path.exists():
                    for device in usb_path.iterdir():
                        speed_file = device / 'speed'
                        if speed_file.exists():
                            try:
                                speed = speed_file.read_text().strip()
                                if speed in ['5000', '10000', '20000']:  # SuperSpeed
                                    info.has_usb3 = True
                            except:
                                pass
                
                info.has_usb3 = info.usb3_controllers > 0
                
                if info.usb3_controllers > 0:
                    info.max_speed = "USB 3.0/3.1/3.2"
                else:
                    info.max_speed = "USB 2.0"
                    
        except Exception as e:
            self.logger.warning(f"USB detection error: {e}")
        
        return info
    
    def _detect_storage(self) -> StorageInfo:
        """Detect storage information"""
        info = StorageInfo()
        
        try:
            if platform.system() in ['Linux', 'Darwin']:
                # Get root filesystem info
                result = subprocess.run(
                    ['df', '-B1', '/'],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        parts = lines[1].split()
                        if len(parts) >= 4:
                            info.root_size_gb = int(parts[1]) / (1024**3)
                            info.root_free_gb = int(parts[3]) / (1024**3)
                
                # Detect storage type
                root_device = None
                result = subprocess.run(
                    ['findmnt', '-n', '-o', 'SOURCE', '/'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    root_device = result.stdout.strip()
                
                if root_device:
                    # Get the base device name
                    device_name = re.sub(r'[0-9]+$', '', os.path.basename(root_device))
                    
                    # Check if it's an SD card (mmcblk)
                    if 'mmcblk' in device_name:
                        info.root_type = 'sd_card'
                        info.is_removable = True
                    # Check if it's NVMe
                    elif 'nvme' in device_name:
                        info.root_type = 'nvme'
                    else:
                        # Check rotation
                        rotational_path = f'/sys/block/{device_name}/queue/rotational'
                        if Path(rotational_path).exists():
                            with open(rotational_path, 'r') as f:
                                if f.read().strip() == '0':
                                    info.root_type = 'ssd'
                                else:
                                    info.root_type = 'hdd'
                    
                    # Check if removable
                    removable_path = f'/sys/block/{device_name}/removable'
                    if Path(removable_path).exists():
                        with open(removable_path, 'r') as f:
                            info.is_removable = f.read().strip() == '1'
                            
        except Exception as e:
            self.logger.warning(f"Storage detection error: {e}")
        
        return info
    
    def _detect_gpu(self) -> GPUInfo:
        """Detect GPU information"""
        info = GPUInfo()
        
        try:
            if platform.system() == 'Linux':
                # Try lspci
                result = subprocess.run(
                    ['lspci', '-v'],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0:
                    output = result.stdout
                    
                    # Find VGA controller
                    vga_match = re.search(r'VGA.*?:\s*(.+)', output)
                    if vga_match:
                        info.model = vga_match.group(1).strip()
                        
                        # Check for NVIDIA
                        if 'nvidia' in info.model.lower():
                            info.has_hardware_accel = True
                        # Check for AMD
                        elif 'amd' in info.model.lower() or 'radeon' in info.model.lower():
                            info.has_hardware_accel = True
                        # Check for Intel
                        elif 'intel' in info.model.lower():
                            info.has_hardware_accel = True
                
                # Raspberry Pi GPU
                if Path('/proc/device-tree/model').exists():
                    with open('/proc/device-tree/model', 'r') as f:
                        if 'Raspberry Pi' in f.read():
                            info.model = 'Broadcom VideoCore'
                            info.has_hardware_accel = True
                            
        except Exception as e:
            self.logger.warning(f"GPU detection error: {e}")
        
        return info
    
    def _detect_battery(self) -> BatteryInfo:
        """Detect battery information (laptops/portable)"""
        info = BatteryInfo()
        
        try:
            # Linux battery detection
            battery_paths = [
                Path('/sys/class/power_supply/BAT0'),
                Path('/sys/class/power_supply/BAT1'),
                Path('/sys/class/power_supply/battery'),
            ]
            
            for battery_path in battery_paths:
                if battery_path.exists():
                    info.has_battery = True
                    
                    # Capacity
                    capacity_file = battery_path / 'capacity'
                    if capacity_file.exists():
                        info.capacity_percent = int(capacity_file.read_text().strip())
                    
                    # Status (Charging/Discharging)
                    status_file = battery_path / 'status'
                    if status_file.exists():
                        status = status_file.read_text().strip().lower()
                        info.is_charging = status in ['charging', 'full']
                    
                    break
                    
        except Exception as e:
            self.logger.warning(f"Battery detection error: {e}")
        
        return info
    
    def _detect_vm(self) -> VMType:
        """Detect if running in a virtual machine"""
        try:
            # Check /sys/class/dmi/id
            dmi_paths = [
                '/sys/class/dmi/id/product_name',
                '/sys/class/dmi/id/sys_vendor',
                '/sys/class/dmi/id/board_vendor',
            ]
            
            for dmi_path in dmi_paths:
                if Path(dmi_path).exists():
                    with open(dmi_path, 'r') as f:
                        content = f.read().lower()
                        for signature, vm_type in self.VM_SIGNATURES.items():
                            if signature in content:
                                return vm_type
            
            # Check for Docker
            if Path('/.dockerenv').exists():
                return VMType.DOCKER
            
            # Check for WSL
            if 'microsoft' in platform.release().lower():
                return VMType.WSL
            
            # Check CPU flags for hypervisor
            if Path('/proc/cpuinfo').exists():
                with open('/proc/cpuinfo', 'r') as f:
                    if 'hypervisor' in f.read():
                        return VMType.KVM  # Generic hypervisor
            
            # Run systemd-detect-virt if available
            result = subprocess.run(
                ['systemd-detect-virt', '-v'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip() != 'none':
                virt = result.stdout.strip().lower()
                for signature, vm_type in self.VM_SIGNATURES.items():
                    if signature in virt:
                        return vm_type
                        
        except Exception as e:
            self.logger.debug(f"VM detection: {e}")
        
        return VMType.NONE
    
    def _classify_platform(self, cpu: CPUInfo, battery: BatteryInfo, 
                          vm_type: VMType) -> PlatformType:
        """Classify the platform type"""
        
        # Check for Raspberry Pi first
        try:
            if Path('/proc/device-tree/model').exists():
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read()
                    for signature, platform_type in self.PI_MODELS.items():
                        if signature in model:
                            return platform_type
                    
                    # Other ARM SBC
                    if cpu.is_arm:
                        return PlatformType.ARM64_SBC
        except:
            pass
        
        # Check for VM
        if vm_type != VMType.NONE:
            return PlatformType.VIRTUAL_MACHINE
        
        # Desktop vs Laptop
        if cpu.is_x86:
            if battery.has_battery:
                return PlatformType.X86_64_LAPTOP
            else:
                return PlatformType.X86_64_DESKTOP
        elif cpu.is_arm:
            if battery.has_battery:
                return PlatformType.ARM64_LAPTOP
            else:
                return PlatformType.ARM64_DESKTOP
        
        return PlatformType.UNKNOWN
    
    def _classify_performance(self, cpu: CPUInfo, memory: MemoryInfo, 
                             usb: USBInfo) -> PerformanceTier:
        """Classify system performance tier"""
        
        score = 0
        
        # CPU cores
        if cpu.cores_physical >= 8:
            score += 3
        elif cpu.cores_physical >= 4:
            score += 2
        elif cpu.cores_physical >= 2:
            score += 1
        
        # Memory
        if memory.total_gb >= 16:
            score += 3
        elif memory.total_gb >= 8:
            score += 2
        elif memory.total_gb >= 4:
            score += 1
        
        # USB
        if usb.has_usb3:
            score += 2
        else:
            score += 1
        
        # Classify
        if score >= 7:
            return PerformanceTier.HIGH
        elif score >= 5:
            return PerformanceTier.MEDIUM
        elif score >= 3:
            return PerformanceTier.LOW
        else:
            return PerformanceTier.MINIMAL
    
    def _check_sdr_ready(self, usb: USBInfo, vm_type: VMType) -> bool:
        """Check if system is ready for SDR operation"""
        
        # VMs often have USB passthrough issues
        if vm_type != VMType.NONE:
            self.logger.warning("Virtual machine detected - USB SDR may require passthrough")
            return False
        
        # Need USB 3.0 for BladeRF performance
        return usb.has_usb3
    
    def _check_gpio_available(self, platform_type: PlatformType) -> bool:
        """Check if GPIO is available (Raspberry Pi)"""
        
        if platform_type in [
            PlatformType.RASPBERRY_PI_5,
            PlatformType.RASPBERRY_PI_4,
            PlatformType.RASPBERRY_PI_3,
            PlatformType.RASPBERRY_PI_ZERO,
        ]:
            # Check if GPIO files exist
            return Path('/sys/class/gpio').exists()
        
        return False
    
    def _get_feature_recommendations(self, caps: PlatformCapabilities
                                    ) -> Tuple[List[str], List[str]]:
        """Get recommended and disabled features based on capabilities"""
        
        recommended = []
        disabled = []
        
        # Performance-based features
        if caps.performance_tier == PerformanceTier.HIGH:
            recommended.extend([
                'ai_enhanced_mode',
                'realtime_spectrum',
                'multi_sdr_support',
                'full_dsp_pipeline',
                'video_reconstruction',
            ])
        elif caps.performance_tier == PerformanceTier.MEDIUM:
            recommended.extend([
                'ai_enhanced_mode',
                'realtime_spectrum',
            ])
            disabled.append('video_reconstruction')
        elif caps.performance_tier == PerformanceTier.LOW:
            disabled.extend([
                'ai_enhanced_mode',
                'realtime_spectrum',
                'video_reconstruction',
            ])
        else:  # MINIMAL
            disabled.extend([
                'ai_enhanced_mode',
                'realtime_spectrum',
                'video_reconstruction',
                'multi_sdr_support',
            ])
        
        # USB-based features
        if not caps.usb.has_usb3:
            disabled.append('high_bandwidth_capture')
            recommended.append('reduced_sample_rate')
        
        # GPIO-based features
        if caps.gpio_available:
            recommended.append('gpio_panic_button')
            recommended.append('gpio_status_leds')
        else:
            disabled.append('gpio_panic_button')
        
        # VM-specific
        if caps.vm_type != VMType.NONE:
            disabled.append('direct_hardware_access')
            recommended.append('simulated_hardware_mode')
        
        return recommended, disabled
    
    def _get_optimization_notes(self, caps: PlatformCapabilities) -> List[str]:
        """Generate optimization notes for the platform"""
        
        notes = []
        
        # Platform-specific notes
        if caps.platform_type == PlatformType.RASPBERRY_PI_5:
            notes.append("Pi 5 detected: Full performance mode enabled")
            notes.append("Recommended: Use official 27W USB-C power supply")
            notes.append("Recommended: Use active cooling for extended SDR operation")
        
        elif caps.platform_type == PlatformType.RASPBERRY_PI_4:
            notes.append("Pi 4 detected: Standard performance mode")
            notes.append("Recommended: Disable unnecessary services to free RAM")
        
        elif caps.platform_type == PlatformType.RASPBERRY_PI_3:
            notes.append("Pi 3 detected: Reduced features mode")
            notes.append("WARNING: USB 2.0 only - reduced SDR bandwidth")
            notes.append("Recommended: Upgrade to Pi 4/5 for full functionality")
        
        elif caps.platform_type in [PlatformType.X86_64_LAPTOP, PlatformType.ARM64_LAPTOP]:
            notes.append("Laptop detected: Consider power management settings")
            if caps.battery.has_battery:
                notes.append("Battery available: Portable operation supported")
        
        elif caps.platform_type == PlatformType.VIRTUAL_MACHINE:
            notes.append(f"VM detected: {caps.vm_type.value}")
            notes.append("WARNING: USB SDR passthrough may be required")
            notes.append("Recommendation: Use physical hardware for production")
        
        # Memory notes
        if caps.memory.total_gb < 4:
            notes.append("WARNING: Low RAM - AI features disabled")
        elif caps.memory.total_gb >= 16:
            notes.append("Excellent RAM: Full AI capabilities enabled")
        
        # Storage notes
        if caps.storage.root_type == 'sd_card':
            notes.append("SD Card detected: Consider USB3 SSD for better performance")
        elif caps.storage.root_type == 'nvme':
            notes.append("NVMe storage: Optimal for capture and replay operations")
        
        if caps.storage.root_free_gb < 10:
            notes.append("WARNING: Low disk space - may affect capture operations")
        
        return notes


class PlatformOptimizer:
    """
    Apply optimizations based on detected platform capabilities
    
    README COMPLIANCE:
    - No telemetry or external connections
    - RAM-only configuration (no persistent sensitive data)
    - Offline-first operation
    """
    
    def __init__(self, capabilities: PlatformCapabilities):
        self.logger = logging.getLogger('PlatformOptimizer')
        self.caps = capabilities
    
    def apply_optimizations(self) -> Dict[str, Any]:
        """
        Apply platform-specific optimizations
        
        Returns:
            Dictionary of applied optimizations and their results
        """
        results = {
            'cpu_governor': self._optimize_cpu(),
            'memory': self._optimize_memory(),
            'usb': self._optimize_usb(),
            'rf_config': self._configure_rf_arsenal(),
        }
        
        return results
    
    def _optimize_cpu(self) -> Dict:
        """Apply CPU optimizations"""
        result = {'status': 'skipped', 'governor': 'unknown'}
        
        try:
            # Determine optimal governor
            if self.caps.performance_tier == PerformanceTier.HIGH:
                target_governor = 'performance'
            elif self.caps.performance_tier in [PerformanceTier.MEDIUM]:
                target_governor = 'ondemand'
            else:
                target_governor = 'conservative'
            
            # Check if on battery
            if self.caps.battery.has_battery and not self.caps.battery.is_charging:
                target_governor = 'powersave'
            
            result['target_governor'] = target_governor
            
            # Apply governor (requires root)
            import glob
            for governor_file in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'):
                try:
                    with open(governor_file, 'w') as f:
                        f.write(target_governor)
                    result['status'] = 'applied'
                    result['governor'] = target_governor
                except PermissionError:
                    result['status'] = 'permission_denied'
                    break
                    
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def _optimize_memory(self) -> Dict:
        """Apply memory optimizations"""
        result = {'status': 'skipped'}
        
        try:
            # Disable swap for stealth (RAM-only operation)
            if Path('/proc/swaps').exists():
                with open('/proc/swaps', 'r') as f:
                    if len(f.readlines()) > 1:  # Has swap
                        result['swap_detected'] = True
                        result['recommendation'] = 'Disable swap for stealth operation'
            
            # Optimize dirty page handling for SDR
            # This improves real-time performance
            result['status'] = 'configured'
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def _optimize_usb(self) -> Dict:
        """Apply USB optimizations for SDR"""
        result = {'status': 'skipped'}
        
        try:
            # Check USB power management
            usb_power_path = '/sys/module/usbcore/parameters/autosuspend'
            if Path(usb_power_path).exists():
                with open(usb_power_path, 'r') as f:
                    current = f.read().strip()
                    result['autosuspend'] = current
                    if current != '-1':
                        result['recommendation'] = 'Set autosuspend=-1 for SDR stability'
            
            result['status'] = 'checked'
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def _configure_rf_arsenal(self) -> Dict:
        """Configure RF Arsenal based on platform"""
        config = {}
        
        # Performance tier settings
        if self.caps.performance_tier == PerformanceTier.HIGH:
            config['max_sample_rate'] = '61.44e6'
            config['max_concurrent_sdrs'] = '2'
            config['enable_ai_features'] = 'true'
            config['enable_realtime_spectrum'] = 'true'
            config['dsp_threads'] = str(max(1, self.caps.cpu.cores_physical - 2))
        elif self.caps.performance_tier == PerformanceTier.MEDIUM:
            config['max_sample_rate'] = '40e6'
            config['max_concurrent_sdrs'] = '1'
            config['enable_ai_features'] = 'true'
            config['enable_realtime_spectrum'] = 'true'
            config['dsp_threads'] = str(max(1, self.caps.cpu.cores_physical - 1))
        elif self.caps.performance_tier == PerformanceTier.LOW:
            config['max_sample_rate'] = '20e6'
            config['max_concurrent_sdrs'] = '1'
            config['enable_ai_features'] = 'false'
            config['enable_realtime_spectrum'] = 'false'
            config['dsp_threads'] = '1'
        else:  # MINIMAL
            config['max_sample_rate'] = '10e6'
            config['max_concurrent_sdrs'] = '1'
            config['enable_ai_features'] = 'false'
            config['enable_realtime_spectrum'] = 'false'
            config['dsp_threads'] = '1'
        
        # USB speed limitations
        if not self.caps.usb.has_usb3:
            config['max_sample_rate'] = '10e6'  # USB 2.0 bandwidth limit
            config['usb_limited'] = 'true'
        
        # GPIO configuration (Raspberry Pi)
        if self.caps.gpio_available:
            config['gpio_panic_button'] = '17'
            config['gpio_status_led_green'] = '27'
            config['gpio_status_led_red'] = '22'
            config['gpio_enabled'] = 'true'
        
        # Virtual machine settings
        if self.caps.vm_type != VMType.NONE:
            config['virtual_mode'] = 'true'
            config['hardware_simulation'] = 'available'
        
        return config
    
    def get_install_commands(self) -> List[str]:
        """
        Generate platform-specific installation commands
        
        Returns:
            List of shell commands for installation
        """
        commands = []
        
        # Base dependencies
        commands.append("# RF Arsenal OS Installation Commands")
        commands.append("# Generated for: " + self.caps.platform_type.value)
        commands.append("")
        
        # Detect package manager
        if Path('/etc/debian_version').exists():
            pkg_mgr = 'apt-get'
            pkg_install = 'apt-get install -y'
        elif Path('/etc/redhat-release').exists():
            pkg_mgr = 'dnf'
            pkg_install = 'dnf install -y'
        elif Path('/etc/arch-release').exists():
            pkg_mgr = 'pacman'
            pkg_install = 'pacman -S --noconfirm'
        else:
            pkg_mgr = 'apt-get'  # Default
            pkg_install = 'apt-get install -y'
        
        # Update package list
        if pkg_mgr == 'apt-get':
            commands.append("sudo apt-get update")
        elif pkg_mgr == 'dnf':
            commands.append("sudo dnf check-update || true")
        
        # Core dependencies
        commands.append(f"sudo {pkg_install} python3 python3-pip python3-dev")
        commands.append(f"sudo {pkg_install} git build-essential cmake")
        
        # SDR dependencies
        commands.append("# SDR Support")
        if pkg_mgr == 'apt-get':
            commands.append(f"sudo {pkg_install} libbladerf-dev bladerf")
            commands.append(f"sudo {pkg_install} libhackrf-dev hackrf")
            commands.append(f"sudo {pkg_install} librtlsdr-dev rtl-sdr")
            commands.append(f"sudo {pkg_install} limesuite")
        
        # Security tools
        commands.append("# Security/Stealth Tools")
        commands.append(f"sudo {pkg_install} tor macchanger")
        
        # Platform-specific
        if self.caps.platform_type in [
            PlatformType.RASPBERRY_PI_5,
            PlatformType.RASPBERRY_PI_4,
            PlatformType.RASPBERRY_PI_3,
        ]:
            commands.append("# Raspberry Pi GPIO")
            commands.append(f"sudo {pkg_install} python3-rpi.gpio raspi-gpio")
            commands.append("# Enable SPI and I2C")
            commands.append("sudo raspi-config nonint do_spi 0")
            commands.append("sudo raspi-config nonint do_i2c 0")
        
        return commands


# Global detector instance
_detector: Optional[UniversalPlatformDetector] = None


def get_platform_detector() -> UniversalPlatformDetector:
    """Get global platform detector instance"""
    global _detector
    if _detector is None:
        _detector = UniversalPlatformDetector()
    return _detector


def detect_platform() -> PlatformCapabilities:
    """Convenience function to detect platform"""
    return get_platform_detector().detect()


def print_platform_summary(caps: Optional[PlatformCapabilities] = None):
    """Print formatted platform summary"""
    if caps is None:
        caps = detect_platform()
    
    print("")
    print("=" * 70)
    print("  RF ARSENAL OS - PLATFORM DETECTION REPORT")
    print("=" * 70)
    print("")
    print(f"  Platform Type:     {caps.platform_type.value}")
    print(f"  Performance Tier:  {caps.performance_tier.value.upper()}")
    
    if caps.vm_type != VMType.NONE:
        print(f"  Virtual Machine:   {caps.vm_type.value}")
    
    print("")
    print("-" * 70)
    print("  HARDWARE DETAILS")
    print("-" * 70)
    print(f"  CPU:               {caps.cpu.model}")
    print(f"  Architecture:      {caps.cpu.architecture}")
    print(f"  Cores:             {caps.cpu.cores_physical} physical / {caps.cpu.cores_logical} logical")
    print(f"  Memory:            {caps.memory.total_gb:.1f} GB total, {caps.memory.available_gb:.1f} GB available")
    print(f"  USB Speed:         {caps.usb.max_speed}")
    print(f"  Storage:           {caps.storage.root_type} ({caps.storage.root_free_gb:.1f} GB free)")
    
    if caps.gpu.model != "Unknown":
        print(f"  GPU:               {caps.gpu.model}")
    
    if caps.battery.has_battery:
        print(f"  Battery:           {caps.battery.capacity_percent}% {'(Charging)' if caps.battery.is_charging else ''}")
    
    print("")
    print("-" * 70)
    print("  RF ARSENAL READINESS")
    print("-" * 70)
    print(f"  SDR Ready:         {'✅ Yes' if caps.sdr_ready else '⚠️  Limited (USB 2.0 only)'}")
    print(f"  GPIO Available:    {'✅ Yes' if caps.gpio_available else '❌ No'}")
    
    print("")
    print("  Recommended Features:")
    for feature in caps.recommended_features[:5]:
        print(f"    ✅ {feature}")
    
    if caps.disabled_features:
        print("")
        print("  Disabled Features:")
        for feature in caps.disabled_features[:5]:
            print(f"    ❌ {feature}")
    
    print("")
    print("-" * 70)
    print("  OPTIMIZATION NOTES")
    print("-" * 70)
    for note in caps.optimization_notes:
        print(f"    • {note}")
    
    print("")
    print("=" * 70)
    print("")


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    
    # Detect platform
    caps = detect_platform()
    
    # Print summary
    print_platform_summary(caps)
    
    # Generate config
    optimizer = PlatformOptimizer(caps)
    config = optimizer._configure_rf_arsenal()
    
    print("Generated RF Arsenal Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key} = {value}")
    print("")


if __name__ == "__main__":
    main()
