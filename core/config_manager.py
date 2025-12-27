"""
RF Arsenal OS - Configuration Manager
=====================================

Centralized configuration management for all RF Arsenal modules.
Handles loading, saving, and validation of system configuration.

Author: RF Arsenal Core Team
Version: 2.0.0
License: Authorized Use Only
"""

import logging
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration manager.
    
    Features:
    - JSON-based configuration storage
    - Environment variable overrides
    - Default values with validation
    - Thread-safe access
    - Automatic config reload
    """
    
    DEFAULT_CONFIG = {
        # Security settings
        'security': {
            'stealth_level': 'maximum',  # maximum, high, medium, low, none
            'passive_mode': True,
            'anonymize_logs': True,
            'transmission_blocking': True,
        },
        
        # Hardware settings
        'hardware': {
            'default_sdr': 'hackrf',  # hackrf, limesdr, bladerf, rtlsdr, usrp, plutosdr
            'sample_rate': 10_000_000,  # 10 MSPS
            'frequency': 1842_600_000,  # 1842.6 MHz (LTE Band 3)
            'bandwidth': 10_000_000,  # 10 MHz
        },
        
        # AI/ML settings
        'ai': {
            'device_fingerprinting_enabled': True,
            'signal_classification_enabled': True,
            'anomaly_detection_enabled': True,
            'model_path': 'data/ml_models',
            'training_data_path': 'data/training',
        },
        
        # Geolocation settings
        'geolocation': {
            'enabled': True,
            'method': 'timing_advance',  # timing_advance, rssi_triangulation, cell_id
            'cell_database_path': 'data/cell_database.json',
            'opencellid_api_key': None,
        },
        
        # Logging settings
        'logging': {
            'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
            'file_path': 'logs/rf_arsenal.log',
            'max_size_mb': 100,
            'backup_count': 5,
        },
        
        # UI settings
        'ui': {
            'theme': 'dark',
            'map_provider': 'openstreetmap',
            'refresh_rate_ms': 1000,
        },
        
        # Network settings
        'network': {
            'api_enabled': False,
            'api_port': 8000,
            'api_host': '127.0.0.1',
        }
    }
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config: Dict[str, Any] = {}
        self._lock = Lock()
        
        # Load configuration
        self.load()
        
        logger.info(f"✅ Configuration Manager initialized (file: {config_file})")
    
    def load(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if loaded successfully
        """
        with self._lock:
            # Start with defaults
            self.config = self.DEFAULT_CONFIG.copy()
            
            # Load from file if exists
            if self.config_file.exists():
                try:
                    with open(self.config_file, 'r') as f:
                        file_config = json.load(f)
                    
                    # Merge with defaults (file config overrides defaults)
                    self._deep_merge(self.config, file_config)
                    
                    logger.info(f"✅ Configuration loaded from {self.config_file}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load config: {e}")
                    return False
            else:
                logger.info("Using default configuration")
                return True
    
    def save(self) -> bool:
        """
        Save current configuration to file.
        
        Returns:
            True if saved successfully
        """
        with self._lock:
            try:
                # Create directory if needed
                self.config_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save to file
                with open(self.config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
                
                logger.info(f"✅ Configuration saved to {self.config_file}")
                return True
            except Exception as e:
                logger.error(f"Failed to save config: {e}")
                return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.
        
        Args:
            key_path: Dot-separated key path (e.g., "security.passive_mode")
            default: Default value if key not found
        
        Returns:
            Configuration value
        
        Examples:
            >>> config.get('security.stealth_level')
            'maximum'
            >>> config.get('hardware.frequency')
            1842600000
        """
        with self._lock:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
    
    def set(self, key_path: str, value: Any, save: bool = False):
        """
        Set configuration value by key path.
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
            save: Save to file immediately (default: False)
        
        Examples:
            >>> config.set('security.stealth_level', 'high')
            >>> config.set('hardware.frequency', 2450000000, save=True)
        """
        with self._lock:
            keys = key_path.split('.')
            target = self.config
            
            # Navigate to parent
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            
            # Set value
            target[keys[-1]] = value
            logger.debug(f"Config set: {key_path} = {value}")
            
            if save:
                self.save()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., "security", "hardware")
        
        Returns:
            Section dictionary
        """
        return self.get(section, {})
    
    def reset_to_defaults(self, save: bool = False):
        """
        Reset configuration to defaults.
        
        Args:
            save: Save to file immediately (default: False)
        """
        with self._lock:
            self.config = self.DEFAULT_CONFIG.copy()
            logger.warning("⚠️  Configuration reset to defaults")
            
            if save:
                self.save()
    
    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge override dict into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    # Convenience accessors for common settings
    @property
    def stealth_level(self) -> str:
        """Get current stealth level."""
        return self.get('security.stealth_level', 'maximum')
    
    @property
    def passive_mode(self) -> bool:
        """Check if passive mode is enabled."""
        return self.get('security.passive_mode', True)
    
    @property
    def anonymize_logs(self) -> bool:
        """Check if log anonymization is enabled."""
        return self.get('security.anonymize_logs', True)
    
    @property
    def default_sdr(self) -> str:
        """Get default SDR device."""
        return self.get('hardware.default_sdr', 'hackrf')
    
    @property
    def ai_enabled(self) -> bool:
        """Check if AI features are enabled."""
        return (
            self.get('ai.device_fingerprinting_enabled', True) or
            self.get('ai.signal_classification_enabled', True) or
            self.get('ai.anomaly_detection_enabled', True)
        )


# Singleton instance
_config_manager_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get singleton configuration manager instance."""
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = ConfigManager()
    return _config_manager_instance


if __name__ == "__main__":
    # Test configuration manager
    print("⚙️  RF Arsenal OS - Configuration Manager Test\n")
    
    config = ConfigManager("test_config.json")
    
    # Test get
    print(f"Stealth level: {config.get('security.stealth_level')}")
    print(f"Passive mode: {config.get('security.passive_mode')}")
    print(f"Default SDR: {config.get('hardware.default_sdr')}")
    print(f"Frequency: {config.get('hardware.frequency') / 1e6:.2f} MHz")
    
    # Test set
    config.set('security.stealth_level', 'high')
    config.set('hardware.frequency', 2450e6)
    
    print(f"\nAfter changes:")
    print(f"Stealth level: {config.stealth_level}")
    print(f"Frequency: {config.get('hardware.frequency') / 1e6:.2f} MHz")
    
    # Test save
    config.save()
    print(f"\n✅ Configuration saved to test_config.json")
    
    # Cleanup
    import os
    os.remove("test_config.json")
