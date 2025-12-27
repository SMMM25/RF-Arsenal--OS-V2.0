#!/usr/bin/env python3
"""
RF Arsenal OS - Emergency Protocols
Real panic button and emergency wipe
"""

import os
import sys
import subprocess
import threading
import time
import logging
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

class EmergencySystem:
    """Production emergency protocols"""
    
    def __init__(self, bladerf_controller=None):
        self.logger = logging.getLogger('Emergency')
        self.bladerf = bladerf_controller
        self.panic_gpio = 17  # Physical button on GPIO 17
        self.deadman_timeout = 3600  # 1 hour
        self.last_activity = time.time()
        self.armed = True
        
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.panic_gpio, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
    def setup_panic_button(self):
        """Setup physical panic button"""
        if not GPIO_AVAILABLE:
            self.logger.warning("GPIO not available - panic button disabled")
            return False
            
        try:
            GPIO.add_event_detect(
                self.panic_gpio,
                GPIO.FALLING,
                callback=self.panic_button_pressed,
                bouncetime=300
            )
            self.logger.info("Panic button armed on GPIO 17")
            return True
        except Exception as e:
            self.logger.error(f"Failed to setup panic button: {e}")
            return False
            
    def panic_button_pressed(self, channel):
        """Handle panic button press"""
        self.logger.critical("PANIC BUTTON PRESSED!")
        self.emergency_wipe("panic_button")
        
    def start_deadman_switch(self):
        """Start deadman timer"""
        def monitor():
            while self.armed:
                if time.time() - self.last_activity > self.deadman_timeout:
                    self.logger.critical("DEADMAN SWITCH TRIGGERED!")
                    self.emergency_wipe("deadman_switch")
                    break
                time.sleep(60)
                
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        self.logger.info(f"Deadman switch armed ({self.deadman_timeout}s)")
        
    def update_activity(self):
        """Reset deadman timer"""
        self.last_activity = time.time()
        
    def emergency_wipe(self, trigger):
        """Execute emergency wipe procedure"""
        self.logger.critical(f"EMERGENCY WIPE INITIATED: {trigger}")
        
        # 1. Stop all RF activity immediately
        if self.bladerf:
            try:
                self.bladerf.emergency_stop()
                self.bladerf.close()
            except:
                pass
        
        # 2. Wipe packet captures (NEW)
        try:
            from modules.network.packet_capture import WiresharkCapture
            pcap = WiresharkCapture()
            self.logger.info("Stopping active packet captures...")
            if pcap.is_capturing:
                pcap.stop_capture()
            self.logger.info("Wiping packet capture files...")
            result = pcap.cleanup(secure_delete=True)
            self.logger.info(f"✅ Wiped {len(result.get('deleted', []))} capture files")
        except Exception as e:
            self.logger.error(f"Packet capture cleanup failed: {e}")
        
        # 2b. Wipe phone targeting data (NEW)
        try:
            from modules.cellular.phone_targeting import PhoneNumberTargeting
            # Access targeting system if available
            if hasattr(self, 'phone_targeting') and self.phone_targeting:
                self.logger.info("Wiping phone targeting data...")
                self.phone_targeting.emergency_cleanup()
                self.logger.info("✅ Phone targeting data wiped")
        except Exception as e:
            self.logger.error(f"Phone targeting cleanup failed: {e}")
        
        # 2c. Wipe VoLTE interception data (NEW)
        try:
            from modules.cellular.volte_interceptor import VoLTEInterceptor
            # Access VoLTE interceptor if available
            if hasattr(self, 'volte_interceptor') and self.volte_interceptor:
                self.logger.info("Wiping VoLTE interception data...")
                self.volte_interceptor.emergency_cleanup()
                self.logger.info("✅ VoLTE interception data wiped")
        except Exception as e:
            self.logger.error(f"VoLTE cleanup failed: {e}")
                
        # 3. Wipe RAM
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3')
        except:
            pass
            
        # 4. Secure delete tmpfs data
        try:
            # SECURITY FIX: Removed shell=True to prevent command injection
            import shutil
            ram_path = '/tmp/rfarsenal_ram'
            if os.path.exists(ram_path):
                shutil.rmtree(ram_path, ignore_errors=True)
        except:
            pass
            
        # 4. Overwrite storage (if not SD card - would damage it)
        # Only do this on production with proper storage
        # subprocess.run(['shred', '-vfz', '-n', '1', '/dev/mmcblk0'],
        #                check=False)
        
        # 5. Log the wipe
        self.logger.critical("EMERGENCY WIPE COMPLETED")
        
        # 6. Power off
        subprocess.run(['poweroff', '-f'], check=False)
        
    def geofence_check(self, current_lat, current_lon, safe_zones):
        """Check if device is in authorized zone"""
        def distance(lat1, lon1, lat2, lon2):
            from math import radians, sin, cos, sqrt, atan2
            R = 6371000  # Earth radius in meters
            
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c
            
        # Check if in any safe zone
        for zone in safe_zones:
            dist = distance(current_lat, current_lon, 
                          zone['lat'], zone['lon'])
            if dist <= zone['radius']:
                return True
                
        # Outside all safe zones
        self.logger.critical("GEOFENCE BREACH DETECTED!")
        self.emergency_wipe("geofence_breach")
        return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Setup emergency system
    emergency = EmergencySystem()
    emergency.setup_panic_button()
    emergency.start_deadman_switch()
    
    # Simulate activity
    while True:
        time.sleep(30)
        emergency.update_activity()
        print("System active...")
