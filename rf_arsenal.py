#!/usr/bin/env python3
"""
RF Arsenal OS - Main Entry Point
White Hat Edition - Authorized Use Only

This is the main executable that initializes all systems and launches the GUI.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the entire application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup file handler
    log_file = log_dir / "rf_arsenal.log"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"RF Arsenal OS starting... Log level: {log_level}")
    
    return logger

def check_root_privileges() -> bool:
    """
    Check if the application is running with root privileges
    
    Returns:
        True if running as root, False otherwise
    """
    return os.geteuid() == 0

def validate_environment() -> tuple[bool, Optional[str]]:
    """
    Validate that the environment meets all requirements
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check Python version
    if sys.version_info < (3, 8):
        return False, "Python 3.8 or higher is required"
    
    # Check for required directories
    required_dirs = ['core', 'modules', 'security', 'ui']
    for dir_name in required_dirs:
        if not (PROJECT_ROOT / dir_name).exists():
            return False, f"Required directory missing: {dir_name}"
    
    # Check for hardware interface
    try:
        from core.hardware_interface import HardwareInterface
    except ImportError as e:
        return False, f"Hardware interface not available: {e}"
    
    return True, None

def initialize_stealth_mode(logger: logging.Logger) -> bool:
    """
    Initialize stealth and OPSEC features
    
    Args:
        logger: Logger instance
    
    Returns:
        True if stealth mode initialized successfully
    """
    try:
        from core.stealth import StealthSystem
        
        logger.info("Initializing stealth mode...")
        stealth = StealthSystem()
        
        # Enable RAM-only operation
        if stealth.enable_ram_only():
            logger.info("RAM-only operation enabled")
        else:
            logger.warning("Failed to enable RAM-only operation")
        
        return True
        
    except ImportError:
        logger.warning("Stealth system not available")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize stealth mode: {e}")
        return False

def main() -> int:
    """
    Main entry point for RF Arsenal OS
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Setup logging
    logger = setup_logging()
    
    try:
        # Display banner
        logger.info("=" * 60)
        logger.info("RF Arsenal OS - White Hat Edition")
        logger.info("AUTHORIZED USE ONLY - For Security Testing")
        logger.info("=" * 60)
        
        # Check root privileges
        if not check_root_privileges():
            logger.warning("Not running as root. Some features may be unavailable.")
            logger.warning("Run with: sudo python3 rf_arsenal.py")
        
        # Validate environment
        is_valid, error_msg = validate_environment()
        if not is_valid:
            logger.error(f"Environment validation failed: {error_msg}")
            return 1
        
        logger.info("Environment validation passed")
        
        # Initialize stealth mode
        initialize_stealth_mode(logger)
        
        # Import and initialize hardware
        try:
            from core.hardware import BladeRFController
            
            logger.info("Initializing hardware controller...")
            hw_controller = BladeRFController.get_instance()
            logger.info("Hardware controller initialized")
            
        except Exception as e:
            logger.warning(f"Hardware initialization failed: {e}")
            logger.warning("Running in mock mode")
        
        # Launch GUI
        try:
            from PyQt6.QtWidgets import QApplication
            from ui.main_gui import MainWindow
            
            logger.info("Launching GUI application...")
            
            app = QApplication(sys.argv)
            app.setApplicationName("RF Arsenal OS")
            app.setOrganizationName("RF Arsenal")
            
            main_window = MainWindow()
            main_window.show()
            
            logger.info("GUI application started successfully")
            
            # Run application event loop
            exit_code = app.exec()
            
            logger.info(f"Application exiting with code: {exit_code}")
            return exit_code
            
        except ImportError as e:
            logger.error(f"Failed to import GUI components: {e}")
            logger.error("Please ensure PyQt6 is installed: pip install PyQt6")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
