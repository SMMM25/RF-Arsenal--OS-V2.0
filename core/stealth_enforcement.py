"""
RF Arsenal OS - Stealth Enforcement & Operational Security
==========================================================

CRITICAL SECURITY MODULE: Enforces stealth-only operations and blocks
any actions that could compromise operational anonymity.

Author: RF Arsenal Security Team
Version: 2.0.0
License: Authorized Use Only
"""

import logging
from typing import List, Optional, Set, Callable
from enum import Enum
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be performed."""
    RECEIVE = "receive"           # Passive signal reception
    TRANSMIT = "transmit"         # Active transmission (BREAKS STEALTH)
    ANALYZE = "analyze"           # Local signal analysis
    LOG = "log"                   # Data logging
    NETWORK = "network"           # Network communication
    STORAGE = "storage"           # Data storage
    IMSI_CATCH = "imsi_catch"     # IMSI catcher (BREAKS STEALTH)
    ACTIVE_GEO = "active_geo"     # Active geolocation (BREAKS STEALTH)
    GPS_SPOOF = "gps_spoof"       # GPS spoofing (BREAKS STEALTH)
    JAMMING = "jamming"           # Signal jamming (BREAKS STEALTH)


class StealthLevel(Enum):
    """Stealth operation levels."""
    MAXIMUM = "maximum"           # Zero RF emissions, passive only
    HIGH = "high"                 # Minimal RF emissions, mostly passive
    MEDIUM = "medium"             # Limited active operations
    LOW = "low"                   # Active operations allowed
    NONE = "none"                 # No stealth restrictions


class StealthViolationError(RuntimeError):
    """Raised when an operation violates stealth requirements."""
    pass


class StealthEnforcer:
    """
    Enforces stealth operational security across all RF Arsenal modules.
    
    Features:
    - Operation validation before execution
    - Real-time stealth violation detection
    - Automatic operation blocking
    - Audit logging of blocked operations
    - Stealth level management
    """
    
    def __init__(self, stealth_level: StealthLevel = StealthLevel.MAXIMUM):
        """
        Initialize stealth enforcer.
        
        Args:
            stealth_level: Default stealth level (default: MAXIMUM)
        """
        self.stealth_level = stealth_level
        self._blocked_operations: Set[OperationType] = set()
        self._operation_log: List[dict] = []
        self._lock = Lock()
        self._violation_callbacks: List[Callable] = []
        
        # Configure blocked operations based on stealth level
        self._configure_stealth_level(stealth_level)
        
        logger.info(f"🔒 Stealth Enforcer initialized (Level: {stealth_level.value})")
    
    def _configure_stealth_level(self, level: StealthLevel):
        """Configure blocked operations based on stealth level."""
        self._blocked_operations.clear()
        
        if level == StealthLevel.MAXIMUM:
            # Block ALL transmission and active operations
            self._blocked_operations.update([
                OperationType.TRANSMIT,
                OperationType.IMSI_CATCH,
                OperationType.ACTIVE_GEO,
                OperationType.GPS_SPOOF,
                OperationType.JAMMING
            ])
        elif level == StealthLevel.HIGH:
            # Block most active operations
            self._blocked_operations.update([
                OperationType.TRANSMIT,
                OperationType.IMSI_CATCH,
                OperationType.JAMMING
            ])
        elif level == StealthLevel.MEDIUM:
            # Block only high-risk operations
            self._blocked_operations.update([
                OperationType.IMSI_CATCH,
                OperationType.JAMMING
            ])
        elif level == StealthLevel.LOW:
            # Block only illegal operations
            self._blocked_operations.add(OperationType.JAMMING)
        # NONE level blocks nothing
        
        logger.info(f"🔒 Stealth level configured: {level.value} (blocking {len(self._blocked_operations)} operations)")
    
    def set_stealth_level(self, level: StealthLevel):
        """
        Change stealth level at runtime.
        
        Args:
            level: New stealth level
        """
        with self._lock:
            old_level = self.stealth_level
            self.stealth_level = level
            self._configure_stealth_level(level)
            logger.warning(f"⚠️  Stealth level changed: {old_level.value} → {level.value}")
    
    def validate_operation(self, 
                          operation: OperationType, 
                          context: Optional[dict] = None,
                          raise_on_violation: bool = True) -> bool:
        """
        Validate if an operation is allowed under current stealth level.
        
        Args:
            operation: Type of operation to validate
            context: Additional context for validation (optional)
            raise_on_violation: Raise exception on violation (default: True)
        
        Returns:
            True if operation is allowed, False otherwise
        
        Raises:
            StealthViolationError: If operation is blocked and raise_on_violation=True
        """
        with self._lock:
            is_blocked = operation in self._blocked_operations
            
            # Log the validation attempt
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation.value,
                'stealth_level': self.stealth_level.value,
                'blocked': is_blocked,
                'context': context or {}
            }
            self._operation_log.append(log_entry)
            
            if is_blocked:
                error_msg = (
                    f"🔒 STEALTH VIOLATION: Operation '{operation.value}' is BLOCKED "
                    f"(Current level: {self.stealth_level.value})\n"
                    f"❌ This operation would compromise operational stealth.\n"
                    f"💡 To enable: Set stealth level to {self._required_level_for(operation).value} or lower."
                )
                
                logger.error(error_msg)
                
                # Trigger violation callbacks
                for callback in self._violation_callbacks:
                    try:
                        callback(operation, context)
                    except Exception as e:
                        logger.error(f"Violation callback failed: {e}")
                
                if raise_on_violation:
                    raise StealthViolationError(error_msg)
                
                return False
            
            logger.debug(f"✅ Operation validated: {operation.value}")
            return True
    
    def _required_level_for(self, operation: OperationType) -> StealthLevel:
        """Get minimum stealth level required for an operation."""
        if operation in [OperationType.IMSI_CATCH, OperationType.JAMMING]:
            return StealthLevel.LOW
        elif operation in [OperationType.TRANSMIT, OperationType.GPS_SPOOF]:
            return StealthLevel.MEDIUM
        elif operation == OperationType.ACTIVE_GEO:
            return StealthLevel.HIGH
        else:
            return StealthLevel.MAXIMUM
    
    def is_passive_only(self) -> bool:
        """Check if enforcer is in passive-only mode."""
        return OperationType.TRANSMIT in self._blocked_operations
    
    def validate_transmit(self, context: Optional[dict] = None):
        """Convenience method to validate transmission operations."""
        self.validate_operation(OperationType.TRANSMIT, context)
    
    def validate_imsi_catcher(self, context: Optional[dict] = None):
        """Convenience method to validate IMSI catcher operations."""
        self.validate_operation(OperationType.IMSI_CATCH, context)
    
    def validate_active_geolocation(self, context: Optional[dict] = None):
        """Convenience method to validate active geolocation."""
        self.validate_operation(OperationType.ACTIVE_GEO, context)
    
    def register_violation_callback(self, callback: Callable):
        """
        Register callback to be notified of stealth violations.
        
        Args:
            callback: Function(operation, context) to call on violations
        """
        self._violation_callbacks.append(callback)
        logger.info(f"✅ Violation callback registered ({len(self._violation_callbacks)} total)")
    
    def get_operation_log(self, limit: int = 100) -> List[dict]:
        """
        Get recent operation validation log.
        
        Args:
            limit: Maximum number of entries to return
        
        Returns:
            List of log entries
        """
        with self._lock:
            return self._operation_log[-limit:]
    
    def get_statistics(self) -> dict:
        """
        Get stealth enforcement statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_ops = len(self._operation_log)
            blocked_ops = sum(1 for log in self._operation_log if log['blocked'])
            
            return {
                'stealth_level': self.stealth_level.value,
                'passive_only': self.is_passive_only(),
                'total_validations': total_ops,
                'blocked_operations': blocked_ops,
                'allowed_operations': total_ops - blocked_ops,
                'block_rate': blocked_ops / total_ops if total_ops > 0 else 0,
                'blocked_operation_types': list(op.value for op in self._blocked_operations)
            }
    
    def generate_stealth_report(self) -> str:
        """
        Generate human-readable stealth status report.
        
        Returns:
            Formatted report string
        """
        stats = self.get_statistics()
        
        report = [
            "=" * 60,
            "🔒 RF ARSENAL OS - STEALTH ENFORCEMENT REPORT",
            "=" * 60,
            "",
            f"Current Stealth Level: {stats['stealth_level'].upper()}",
            f"Passive-Only Mode:     {'✅ ENABLED' if stats['passive_only'] else '❌ DISABLED'}",
            "",
            "📊 Operation Statistics:",
            f"  Total Validations:   {stats['total_validations']}",
            f"  Blocked Operations:  {stats['blocked_operations']} ({stats['block_rate']:.1%})",
            f"  Allowed Operations:  {stats['allowed_operations']}",
            "",
            "🚫 Blocked Operation Types:",
        ]
        
        for op_type in stats['blocked_operation_types']:
            report.append(f"  ❌ {op_type}")
        
        if not stats['blocked_operation_types']:
            report.append("  ✅ None (stealth restrictions disabled)")
        
        report.append("=" * 60)
        
        return "\n".join(report)


# Singleton instance for system-wide enforcement
_enforcer_instance: Optional[StealthEnforcer] = None


def get_stealth_enforcer(stealth_level: Optional[StealthLevel] = None) -> StealthEnforcer:
    """
    Get singleton stealth enforcer instance.
    
    Args:
        stealth_level: Override stealth level (only on first call)
    
    Returns:
        StealthEnforcer instance
    """
    global _enforcer_instance
    if _enforcer_instance is None:
        level = stealth_level or StealthLevel.MAXIMUM
        _enforcer_instance = StealthEnforcer(stealth_level=level)
    return _enforcer_instance


# Convenience functions
def validate_transmit(context: Optional[dict] = None):
    """Quick transmit validation using singleton."""
    get_stealth_enforcer().validate_transmit(context)


def validate_imsi_catcher(context: Optional[dict] = None):
    """Quick IMSI catcher validation using singleton."""
    get_stealth_enforcer().validate_imsi_catcher(context)


def is_passive_only() -> bool:
    """Check if system is in passive-only mode."""
    return get_stealth_enforcer().is_passive_only()


if __name__ == "__main__":
    # Test stealth enforcement
    print("🔒 RF Arsenal OS - Stealth Enforcement Test\n")
    
    enforcer = StealthEnforcer(stealth_level=StealthLevel.MAXIMUM)
    
    print(enforcer.generate_stealth_report())
    print()
    
    # Test allowed operation
    print("Testing RECEIVE operation (should be allowed):")
    try:
        enforcer.validate_operation(OperationType.RECEIVE)
        print("✅ RECEIVE allowed\n")
    except StealthViolationError as e:
        print(f"❌ {e}\n")
    
    # Test blocked operation
    print("Testing TRANSMIT operation (should be blocked):")
    try:
        enforcer.validate_operation(OperationType.TRANSMIT)
        print("✅ TRANSMIT allowed\n")
    except StealthViolationError as e:
        print(f"❌ Blocked as expected\n")
    
    # Test stealth level change
    print("Changing stealth level to LOW:")
    enforcer.set_stealth_level(StealthLevel.LOW)
    
    print("\nTesting TRANSMIT after level change:")
    try:
        enforcer.validate_operation(OperationType.TRANSMIT)
        print("✅ TRANSMIT now allowed\n")
    except StealthViolationError as e:
        print(f"❌ {e}\n")
    
    print(enforcer.generate_stealth_report())
