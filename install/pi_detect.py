#!/usr/bin/env python3
"""
Raspberry Pi Hardware Detection and Optimization
Detects Pi model (5/4/3) and applies appropriate optimizations

Copyright (c) 2024 RF-Arsenal-OS Project
License: MIT
"""

import subprocess
import os
from pathlib import Path
from typing import Dict, Tuple


class RaspberryPiDetector:
    """Detect and optimize for Raspberry Pi hardware"""
    
    def __init__(self):
        self.model = self.detect_model()
        self.specs = self.get_specs()
    
    def detect_model(self) -> str:
        """Detect Raspberry Pi model"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model_str = f.read().strip()
                
                if 'Raspberry Pi 5' in model_str:
                    return 'pi5'
                elif 'Raspberry Pi 4' in model_str:
                    return 'pi4'
                elif 'Raspberry Pi 3' in model_str:
                    return 'pi3'
                else:
                    return 'unknown'
        except FileNotFoundError:
            return 'not_pi'
    
    def get_specs(self) -> Dict:
        """Get hardware specifications"""
        specs = {
            'model': self.model,
            'cpu': self.get_cpu_info(),
            'memory': self.get_memory_gb(),
            'usb_version': self.get_usb_version(),
            'recommended': False,
            'features': 'Unknown'  # Default value for unknown platforms
        }
        
        if self.model == 'pi5':
            specs['recommended'] = True
            specs['features'] = 'All features enabled (OPTIMAL)'
        elif self.model == 'pi4':
            specs['recommended'] = True
            specs['features'] = 'All features enabled (GOOD)'
        elif self.model == 'pi3':
            specs['recommended'] = False
            specs['features'] = 'Reduced features (MINIMUM)'
        elif self.model == 'not_pi':
            specs['features'] = 'Not a Raspberry Pi'
        elif self.model == 'unknown':
            specs['features'] = 'Unknown Raspberry Pi model'
        
        return specs
    
    def get_cpu_info(self) -> str:
        """Get CPU information"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
            
            return "Unknown CPU"
        except:
            return "Unknown CPU"
    
    def get_memory_gb(self) -> int:
        """Get total memory in GB"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        return mem_kb // (1024 * 1024)
            return 0
        except:
            return 0
    
    def get_usb_version(self) -> str:
        """Detect USB version"""
        if self.model in ['pi5', 'pi4']:
            return 'USB 3.0'
        elif self.model == 'pi3':
            return 'USB 2.0'
        else:
            return 'Unknown'
    
    def apply_optimizations(self) -> bool:
        """Apply hardware-specific optimizations"""
        print(f"\n🔧 Applying optimizations for {self.model.upper()}...")
        
        if self.model == 'pi5':
            return self.optimize_pi5()
        elif self.model == 'pi4':
            return self.optimize_pi4()
        elif self.model == 'pi3':
            return self.optimize_pi3()
        else:
            print("  ℹ️  No optimizations for this hardware")
            return True
    
    def optimize_pi5(self) -> bool:
        """Optimize for Raspberry Pi 5"""
        try:
            print("  🚀 Pi 5 detected - Enabling performance mode")
            
            # Set CPU governor to performance
            self.set_cpu_governor('performance')
            
            # Increase GPU memory (Pi 5 has plenty of RAM)
            self.set_config_value('gpu_mem', '256')
            
            # Enable all features
            self.set_rf_config('ENABLE_ALL_FEATURES', 'true')
            self.set_rf_config('ENABLE_AI_FEATURES', 'true')
            self.set_rf_config('ENABLE_REALTIME_SPECTRUM', 'true')
            self.set_rf_config('MAX_CONCURRENT_SDRS', '2')
            
            print("  ✅ Pi 5 optimizations applied")
            return True
        
        except Exception as e:
            print(f"  ⚠️  Optimization error: {e}")
            return False
    
    def optimize_pi4(self) -> bool:
        """Optimize for Raspberry Pi 4"""
        try:
            print("  ✅ Pi 4 detected - Standard optimization")
            
            # Set CPU governor to performance
            self.set_cpu_governor('performance')
            
            # Set GPU memory
            mem_gb = self.specs['memory']
            gpu_mem = '128' if mem_gb >= 4 else '64'
            self.set_config_value('gpu_mem', gpu_mem)
            
            # Enable all features
            self.set_rf_config('ENABLE_ALL_FEATURES', 'true')
            self.set_rf_config('ENABLE_AI_FEATURES', 'true')
            self.set_rf_config('MAX_CONCURRENT_SDRS', '2')
            
            print("  ✅ Pi 4 optimizations applied")
            return True
        
        except Exception as e:
            print(f"  ⚠️  Optimization error: {e}")
            return False
    
    def optimize_pi3(self) -> bool:
        """Optimize for Raspberry Pi 3"""
        try:
            print("  ⚠️  Pi 3 detected - Reduced features for compatibility")
            
            # Use ondemand governor (save power)
            self.set_cpu_governor('ondemand')
            
            # Minimal GPU memory
            self.set_config_value('gpu_mem', '64')
            
            # Disable heavy features
            self.set_rf_config('ENABLE_AI_FEATURES', 'false')
            self.set_rf_config('ENABLE_REALTIME_SPECTRUM', 'false')
            self.set_rf_config('MAX_CONCURRENT_SDRS', '1')
            
            print("  ✅ Pi 3 optimizations applied (limited features)")
            return True
        
        except Exception as e:
            print(f"  ⚠️  Optimization error: {e}")
            return False
    
    def set_cpu_governor(self, governor: str):
        """Set CPU frequency governor"""
        from core.validation import InputValidator
        
        # Validate governor name to prevent injection
        valid_governors = ['performance', 'powersave', 'ondemand', 'conservative', 'schedutil']
        if governor not in valid_governors:
            print(f"  ⚠️  Invalid CPU governor: {governor}")
            return
        
        try:
            # SECURITY FIX: Removed shell=True, using Python file operations instead
            import glob
            for governor_file in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'):
                try:
                    with open(governor_file, 'w') as f:
                        f.write(governor)
                except PermissionError:
                    # Try with sudo if we don't have permissions
                    subprocess.run(
                        ['sudo', 'tee', governor_file],
                        input=governor.encode(),
                        check=True,
                        capture_output=True
                    )
        except Exception as e:
            print(f"  ⚠️  Could not set CPU governor: {e}")
    
    def set_config_value(self, key: str, value: str):
        """Set value in /boot/config.txt"""
        config_file = Path('/boot/config.txt')
        
        if not config_file.exists():
            return
        
        try:
            # Read existing config
            with open(config_file, 'r') as f:
                lines = f.readlines()
            
            # Update or add key
            found = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    found = True
                    break
            
            if not found:
                lines.append(f"{key}={value}\n")
            
            # Write back (requires sudo)
            with open(config_file, 'w') as f:
                f.writelines(lines)
        
        except Exception as e:
            print(f"  ⚠️  Could not update config.txt: {e}")
    
    def set_rf_config(self, key: str, value: str):
        """Set RF Arsenal configuration value"""
        config_dir = Path('/etc/rf-arsenal')
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / 'hardware.conf'
        
        try:
            # Read existing config or create new
            lines = []
            if config_file.exists():
                with open(config_file, 'r') as f:
                    lines = f.readlines()
            
            # Update or add key
            found = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    found = True
                    break
            
            if not found:
                lines.append(f"{key}={value}\n")
            
            # Write config
            with open(config_file, 'w') as f:
                f.writelines(lines)
        
        except Exception as e:
            print(f"  ⚠️  Could not update RF config: {e}")
    
    def print_summary(self):
        """Print hardware summary"""
        print("\n" + "="*60)
        print("🖥️  RASPBERRY PI HARDWARE SUMMARY")
        print("="*60)
        print(f"Model:       {self.model.upper()}")
        print(f"CPU:         {self.specs['cpu']}")
        print(f"Memory:      {self.specs['memory']} GB")
        print(f"USB:         {self.specs['usb_version']}")
        print(f"Features:    {self.specs['features']}")
        print(f"Recommended: {'✅ YES' if self.specs['recommended'] else '⚠️  MINIMUM'}")
        print("="*60)


def main():
    """Main entry point"""
    detector = RaspberryPiDetector()
    
    detector.print_summary()
    
    if detector.model == 'not_pi':
        print("\nℹ️  Not running on Raspberry Pi - Skipping optimizations")
        return
    
    # Apply optimizations
    if detector.apply_optimizations():
        print("\n✅ Hardware optimization complete!")
    else:
        print("\n⚠️  Some optimizations failed")


if __name__ == "__main__":
    main()
