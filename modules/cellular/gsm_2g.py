#!/usr/bin/env python3
"""
2G/GSM Base Station using OpenBTS
Production implementation
"""

import subprocess
import os
import signal
import logging
import time
import sqlite3
from pathlib import Path

class GSM2GBaseStation:
    """Production 2G/GSM base station controller"""
    
    def __init__(self, bladerf_controller):
        self.bladerf = bladerf_controller
        self.logger = logging.getLogger('GSM-2G')
        self.process = None
        self.config_dir = Path("/etc/OpenBTS")
        self.running = False
        
    def configure_openbts(self, config):
        """Configure OpenBTS for BladeRF"""
        from core.validation import InputValidator
        
        # Validate inputs to prevent SQL injection
        arfcn = config.get('arfcn', 51)
        mcc = config.get('mcc', '001')
        mnc = config.get('mnc', '01')
        name = config.get('name', 'RF-Arsenal')
        tx_atten = config.get('tx_atten', 0)
        
        # Validate integer values
        if not isinstance(arfcn, int) or not (0 <= arfcn <= 1023):
            self.logger.error(f"Invalid ARFCN: {arfcn}")
            return False
        
        if not isinstance(tx_atten, (int, float)) or not (-50 <= tx_atten <= 50):
            self.logger.error(f"Invalid TX attenuation: {tx_atten}")
            return False
        
        # Validate string values (prevent SQL injection)
        valid, error = InputValidator.validate_string(str(mcc), max_length=10, allow_special=False)
        if not valid:
            self.logger.error(f"Invalid MCC: {error}")
            return False
        
        valid, error = InputValidator.validate_string(str(mnc), max_length=10, allow_special=False)
        if not valid:
            self.logger.error(f"Invalid MNC: {error}")
            return False
        
        valid, error = InputValidator.validate_string(name, max_length=50, allow_special=False)
        if not valid:
            self.logger.error(f"Invalid name: {error}")
            return False
        
        openbts_config = {
            'GSM.Radio.C0': arfcn,
            'GSM.Identity.MCC': mcc,
            'GSM.Identity.MNC': mnc,
            'GSM.Identity.ShortName': name,
            'TRX.TxAttenOffset': tx_atten,
            'TRX.RadioFrequencyOffset': 0,
        }
        
        config_file = self.config_dir / "OpenBTS.db"
        
        # SECURITY FIX: Use Python sqlite3 with parameterized queries
        # This prevents SQL injection attacks completely
        try:
            conn = sqlite3.connect(str(config_file))
            cursor = conn.cursor()
            
            for key, value in openbts_config.items():
                try:
                    # Use parameterized query (? placeholders) to prevent SQL injection
                    cursor.execute(
                        "UPDATE CONFIG SET VALUESTRING = ? WHERE KEYSTRING = ?",
                        (str(value), key)
                    )
                except sqlite3.Error as e:
                    self.logger.error(f"Failed to update {key}: {e}")
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return False
            
        self.logger.info("OpenBTS configured")
        return True
        
    def start_base_station(self, config=None):
        """Start 2G base station"""
        if self.running:
            self.logger.warning("Base station already running")
            return False
            
        if config is None:
            config = {
                'arfcn': 51,
                'mcc': '001',
                'mnc': '01',
                'name': 'TEST-GSM'
            }
            
        try:
            self.configure_openbts(config)
            
            self.process = subprocess.Popen(
                ['OpenBTS'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd='/OpenBTS'
            )
            
            time.sleep(5)
            
            if self.process.poll() is None:
                self.running = True
                self.logger.info("2G base station started")
                return True
            else:
                self.logger.error("Failed to start OpenBTS")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting base station: {e}")
            return False
            
    def stop_base_station(self):
        """Stop base station"""
        if not self.running:
            return True
            
        try:
            if self.process:
                self.process.send_signal(signal.SIGTERM)
                self.process.wait(timeout=10)
                
            self.running = False
            self.logger.info("2G base station stopped")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping base station: {e}")
            if self.process:
                self.process.kill()
            return False
            
    def imsi_catch_mode(self):
        """Passive IMSI collection mode"""
        config = {
            'arfcn': 51,
            'mcc': '001',
            'mnc': '01',
            'name': 'Vodafone',
            'tx_atten': -10
        }
        
        self.start_base_station(config)
        self.logger.info("IMSI catcher mode active")
        
    def get_connected_devices(self):
        """Get list of connected devices"""
        try:
            result = subprocess.run(
                ['OpenBTSCLI', '-c', 'tmsis'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            devices = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    devices.append({
                        'imsi': parts[0],
                        'tmsi': parts[1],
                        'age': parts[2] if len(parts) > 2 else 'N/A'
                    })
                    
            return devices
        except Exception as e:
            self.logger.error(f"Error getting devices: {e}")
            return []
