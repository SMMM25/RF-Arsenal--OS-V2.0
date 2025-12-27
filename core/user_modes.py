#!/usr/bin/env python3
"""
RF Arsenal OS - User Mode System
Beginner/Expert modes that scale with user skill level

DESIGN PHILOSOPHY:
- Beginner mode: AI explains everything, confirms dangerous actions, hides complexity
- Expert mode: Full control, minimal prompts, all features exposed
- Smooth transition as users gain experience
- Never compromise security for convenience
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class UserMode(Enum):
    """User experience modes"""
    BEGINNER = "beginner"       # Full guidance, explanations, confirmations
    INTERMEDIATE = "intermediate"  # Some guidance, key confirmations
    EXPERT = "expert"           # Minimal prompts, full control


class FeatureAccess(Enum):
    """Feature access levels"""
    HIDDEN = "hidden"           # Not shown at all
    VISIBLE = "visible"         # Shown but requires confirmation
    ENABLED = "enabled"         # Fully accessible


@dataclass
class UserProfile:
    """User profile with mode and preferences"""
    mode: UserMode = UserMode.BEGINNER
    commands_executed: int = 0
    missions_completed: int = 0
    show_explanations: bool = True
    confirm_dangerous: bool = True
    show_safety_tips: bool = True
    auto_opsec_check: bool = True
    verbose_output: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Feature-specific overrides
    feature_overrides: Dict[str, FeatureAccess] = field(default_factory=dict)


@dataclass 
class ModeConfiguration:
    """Configuration for a user mode"""
    mode: UserMode
    name: str
    description: str
    
    # UI/UX settings
    show_explanations: bool
    confirm_all_commands: bool
    confirm_dangerous_only: bool
    show_safety_warnings: bool
    show_legal_warnings: bool
    verbose_output: bool
    show_progress_tips: bool
    
    # Feature access defaults
    feature_access: Dict[str, FeatureAccess]
    
    # Command behavior
    require_mission_for_ops: bool  # Must use mission profiles
    auto_suggest_commands: bool
    show_command_preview: bool


class UserModeManager:
    """
    Manages user experience modes
    
    Features:
    - Beginner: Maximum hand-holding, everything explained
    - Intermediate: Balance of guidance and efficiency  
    - Expert: Full control, minimal interruptions
    - Automatic mode suggestions based on experience
    - Per-feature access control
    """
    
    # Default configurations for each mode
    MODE_CONFIGS = {
        UserMode.BEGINNER: ModeConfiguration(
            mode=UserMode.BEGINNER,
            name="Beginner Mode",
            description="Full guidance with explanations. Perfect for learning RF Arsenal OS.",
            show_explanations=True,
            confirm_all_commands=False,
            confirm_dangerous_only=True,
            show_safety_warnings=True,
            show_legal_warnings=True,
            verbose_output=True,
            show_progress_tips=True,
            feature_access={
                # Dangerous features hidden or require confirmation
                'jamming': FeatureAccess.VISIBLE,
                'gps_spoofing': FeatureAccess.VISIBLE,
                'cellular_bts': FeatureAccess.VISIBLE,
                'imsi_catch': FeatureAccess.VISIBLE,
                'drone_hijack': FeatureAccess.HIDDEN,
                'emergency_wipe': FeatureAccess.VISIBLE,
                # Safe features enabled
                'wifi_scan': FeatureAccess.ENABLED,
                'spectrum_scan': FeatureAccess.ENABLED,
                'drone_detect': FeatureAccess.ENABLED,
                'status': FeatureAccess.ENABLED,
                'opsec_check': FeatureAccess.ENABLED,
                'missions': FeatureAccess.ENABLED,
            },
            require_mission_for_ops=True,  # Encourage using guided missions
            auto_suggest_commands=True,
            show_command_preview=True
        ),
        
        UserMode.INTERMEDIATE: ModeConfiguration(
            mode=UserMode.INTERMEDIATE,
            name="Intermediate Mode",
            description="Balanced guidance. Confirmations for dangerous operations only.",
            show_explanations=False,  # Only on request
            confirm_all_commands=False,
            confirm_dangerous_only=True,
            show_safety_warnings=True,
            show_legal_warnings=True,
            verbose_output=False,
            show_progress_tips=False,
            feature_access={
                'jamming': FeatureAccess.VISIBLE,
                'gps_spoofing': FeatureAccess.VISIBLE,
                'cellular_bts': FeatureAccess.VISIBLE,
                'imsi_catch': FeatureAccess.VISIBLE,
                'drone_hijack': FeatureAccess.VISIBLE,
                'emergency_wipe': FeatureAccess.VISIBLE,
                'wifi_scan': FeatureAccess.ENABLED,
                'spectrum_scan': FeatureAccess.ENABLED,
                'drone_detect': FeatureAccess.ENABLED,
                'status': FeatureAccess.ENABLED,
                'opsec_check': FeatureAccess.ENABLED,
                'missions': FeatureAccess.ENABLED,
            },
            require_mission_for_ops=False,
            auto_suggest_commands=True,
            show_command_preview=False
        ),
        
        UserMode.EXPERT: ModeConfiguration(
            mode=UserMode.EXPERT,
            name="Expert Mode",
            description="Full control. Minimal prompts. All features accessible.",
            show_explanations=False,
            confirm_all_commands=False,
            confirm_dangerous_only=False,  # No confirmations
            show_safety_warnings=False,     # User knows what they're doing
            show_legal_warnings=True,       # Always show legal warnings
            verbose_output=False,
            show_progress_tips=False,
            feature_access={
                'jamming': FeatureAccess.ENABLED,
                'gps_spoofing': FeatureAccess.ENABLED,
                'cellular_bts': FeatureAccess.ENABLED,
                'imsi_catch': FeatureAccess.ENABLED,
                'drone_hijack': FeatureAccess.ENABLED,
                'emergency_wipe': FeatureAccess.ENABLED,
                'wifi_scan': FeatureAccess.ENABLED,
                'spectrum_scan': FeatureAccess.ENABLED,
                'drone_detect': FeatureAccess.ENABLED,
                'status': FeatureAccess.ENABLED,
                'opsec_check': FeatureAccess.ENABLED,
                'missions': FeatureAccess.ENABLED,
            },
            require_mission_for_ops=False,
            auto_suggest_commands=False,
            show_command_preview=False
        )
    }
    
    # Dangerous operations that should show warnings/confirmations
    DANGEROUS_OPERATIONS = [
        'jamming', 'jam',
        'gps_spoofing', 'spoof gps',
        'cellular_bts', 'start bts', 'base station',
        'imsi_catch', 'imsi',
        'drone_hijack', 'hijack',
        'emergency_wipe', 'wipe', 'panic',
        'transmit', 'broadcast',
        'deauth', 'evil twin',
        'intercept',
    ]
    
    # Experience thresholds for mode suggestions
    EXPERIENCE_THRESHOLDS = {
        UserMode.INTERMEDIATE: {
            'commands_executed': 50,
            'missions_completed': 3
        },
        UserMode.EXPERT: {
            'commands_executed': 200,
            'missions_completed': 10
        }
    }
    
    def __init__(self, profile_path: Optional[str] = None):
        self.profile_path = Path(profile_path) if profile_path else Path.home() / '.rf_arsenal' / 'user_profile.json'
        self.profile: UserProfile = UserProfile()
        self.config: ModeConfiguration = self.MODE_CONFIGS[UserMode.BEGINNER]
        
        # Load existing profile
        self._load_profile()
        
        logger.info(f"UserModeManager initialized - Mode: {self.profile.mode.value}")
    
    def _load_profile(self):
        """Load user profile from disk"""
        try:
            if self.profile_path.exists():
                with open(self.profile_path, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct profile
                self.profile = UserProfile(
                    mode=UserMode(data.get('mode', 'beginner')),
                    commands_executed=data.get('commands_executed', 0),
                    missions_completed=data.get('missions_completed', 0),
                    show_explanations=data.get('show_explanations', True),
                    confirm_dangerous=data.get('confirm_dangerous', True),
                    show_safety_tips=data.get('show_safety_tips', True),
                    auto_opsec_check=data.get('auto_opsec_check', True),
                    verbose_output=data.get('verbose_output', True),
                )
                
                # Load config for mode
                self.config = self.MODE_CONFIGS[self.profile.mode]
                
                logger.info(f"Loaded user profile: mode={self.profile.mode.value}, commands={self.profile.commands_executed}")
            else:
                logger.info("No existing profile, using defaults (Beginner mode)")
                
        except Exception as e:
            logger.warning(f"Failed to load profile: {e}. Using defaults.")
    
    def _save_profile(self):
        """Save user profile to disk"""
        try:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'mode': self.profile.mode.value,
                'commands_executed': self.profile.commands_executed,
                'missions_completed': self.profile.missions_completed,
                'show_explanations': self.profile.show_explanations,
                'confirm_dangerous': self.profile.confirm_dangerous,
                'show_safety_tips': self.profile.show_safety_tips,
                'auto_opsec_check': self.profile.auto_opsec_check,
                'verbose_output': self.profile.verbose_output,
                'last_activity': datetime.now().isoformat()
            }
            
            with open(self.profile_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save profile: {e}")
    
    def get_current_mode(self) -> UserMode:
        """Get current user mode"""
        return self.profile.mode
    
    def get_config(self) -> ModeConfiguration:
        """Get current mode configuration"""
        return self.config
    
    def set_mode(self, mode: UserMode) -> Dict:
        """Set user mode"""
        old_mode = self.profile.mode
        self.profile.mode = mode
        self.config = self.MODE_CONFIGS[mode]
        
        # Update profile preferences based on mode
        self.profile.show_explanations = self.config.show_explanations
        self.profile.verbose_output = self.config.verbose_output
        
        self._save_profile()
        
        logger.info(f"Mode changed: {old_mode.value} -> {mode.value}")
        
        return {
            'success': True,
            'old_mode': old_mode.value,
            'new_mode': mode.value,
            'description': self.config.description,
            'message': self._get_mode_welcome_message(mode)
        }
    
    def _get_mode_welcome_message(self, mode: UserMode) -> str:
        """Get welcome message for mode"""
        config = self.MODE_CONFIGS[mode]
        
        messages = {
            UserMode.BEGINNER: (
                f"BEGINNER MODE ACTIVATED\n\n"
                f"Welcome! I'll guide you through everything.\n\n"
                f"What's enabled:\n"
                f"  ✓ Detailed explanations for all commands\n"
                f"  ✓ Safety warnings before dangerous operations\n"
                f"  ✓ Guided mission profiles recommended\n"
                f"  ✓ Auto OPSEC monitoring\n\n"
                f"Tip: Start with 'list missions' to see guided operations.\n"
                f"Say 'help' anytime for assistance."
            ),
            UserMode.INTERMEDIATE: (
                f"INTERMEDIATE MODE ACTIVATED\n\n"
                f"Good progress! Reduced prompts, more efficiency.\n\n"
                f"What's changed:\n"
                f"  • Explanations only when requested\n"
                f"  • Confirmations for dangerous ops only\n"
                f"  • Direct command execution available\n"
                f"  • All features accessible\n\n"
                f"Say 'explain' before a command for details."
            ),
            UserMode.EXPERT: (
                f"EXPERT MODE ACTIVATED\n\n"
                f"Full control. Minimal interruptions.\n\n"
                f"What's enabled:\n"
                f"  • No confirmations (except legal warnings)\n"
                f"  • Direct execution of all commands\n"
                f"  • Concise output\n"
                f"  • All advanced features\n\n"
                f"OPSEC is your responsibility. Stay sharp."
            )
        }
        
        return messages.get(mode, "Mode changed.")
    
    def record_command(self):
        """Record that a command was executed"""
        self.profile.commands_executed += 1
        self.profile.last_activity = datetime.now()
        self._save_profile()
        
        # Check if mode upgrade suggested
        return self._check_mode_suggestion()
    
    def record_mission_complete(self):
        """Record that a mission was completed"""
        self.profile.missions_completed += 1
        self._save_profile()
        
        return self._check_mode_suggestion()
    
    def _check_mode_suggestion(self) -> Optional[Dict]:
        """Check if user qualifies for mode upgrade"""
        current = self.profile.mode
        
        # Check for intermediate
        if current == UserMode.BEGINNER:
            threshold = self.EXPERIENCE_THRESHOLDS[UserMode.INTERMEDIATE]
            if (self.profile.commands_executed >= threshold['commands_executed'] and
                self.profile.missions_completed >= threshold['missions_completed']):
                return {
                    'suggest_upgrade': True,
                    'suggested_mode': UserMode.INTERMEDIATE.value,
                    'message': (
                        f"You've executed {self.profile.commands_executed} commands and "
                        f"completed {self.profile.missions_completed} missions.\n"
                        f"Ready to try INTERMEDIATE mode? Say 'set mode intermediate'."
                    )
                }
        
        # Check for expert
        elif current == UserMode.INTERMEDIATE:
            threshold = self.EXPERIENCE_THRESHOLDS[UserMode.EXPERT]
            if (self.profile.commands_executed >= threshold['commands_executed'] and
                self.profile.missions_completed >= threshold['missions_completed']):
                return {
                    'suggest_upgrade': True,
                    'suggested_mode': UserMode.EXPERT.value,
                    'message': (
                        f"Impressive! {self.profile.commands_executed} commands, "
                        f"{self.profile.missions_completed} missions.\n"
                        f"Ready for EXPERT mode? Say 'set mode expert'."
                    )
                }
        
        return None
    
    def should_confirm(self, command: str) -> bool:
        """Check if command requires confirmation in current mode"""
        command_lower = command.lower()
        
        # In expert mode, never confirm (except we still show legal warnings)
        if self.profile.mode == UserMode.EXPERT:
            return False
        
        # Check if it's a dangerous operation
        is_dangerous = any(op in command_lower for op in self.DANGEROUS_OPERATIONS)
        
        # In beginner mode, confirm dangerous operations
        if self.profile.mode == UserMode.BEGINNER:
            return is_dangerous or self.config.confirm_all_commands
        
        # In intermediate mode, only confirm dangerous
        if self.profile.mode == UserMode.INTERMEDIATE:
            return is_dangerous and self.config.confirm_dangerous_only
        
        return False
    
    def should_show_explanation(self, command: str) -> bool:
        """Check if explanation should be shown"""
        return self.config.show_explanations
    
    def should_show_safety_warning(self, command: str) -> bool:
        """Check if safety warning should be shown"""
        if not self.config.show_safety_warnings:
            return False
        
        command_lower = command.lower()
        return any(op in command_lower for op in self.DANGEROUS_OPERATIONS)
    
    def get_feature_access(self, feature: str) -> FeatureAccess:
        """Get access level for a feature"""
        # Check user overrides first
        if feature in self.profile.feature_overrides:
            return self.profile.feature_overrides[feature]
        
        # Fall back to mode defaults
        return self.config.feature_access.get(feature, FeatureAccess.ENABLED)
    
    def is_feature_accessible(self, feature: str) -> Tuple[bool, Optional[str]]:
        """Check if feature is accessible and return reason if not"""
        access = self.get_feature_access(feature)
        
        if access == FeatureAccess.HIDDEN:
            return False, f"Feature '{feature}' is not available in {self.profile.mode.value} mode."
        
        if access == FeatureAccess.VISIBLE:
            return True, f"Feature '{feature}' requires confirmation in {self.profile.mode.value} mode."
        
        return True, None
    
    def format_response(self, response: str, command: str, explanation: Optional[str] = None) -> str:
        """Format response based on user mode"""
        output = []
        
        # Add explanation in beginner mode
        if self.config.show_explanations and explanation:
            output.append(f"EXPLANATION:\n{explanation}\n")
        
        # Main response
        output.append(response)
        
        # Add suggestions in beginner mode
        if self.config.auto_suggest_commands:
            suggestions = self._get_command_suggestions(command)
            if suggestions:
                output.append(f"\nSUGGESTIONS: {', '.join(suggestions)}")
        
        # Add progress tips occasionally
        if self.config.show_progress_tips and self.profile.commands_executed % 10 == 0:
            output.append(f"\nTIP: You've run {self.profile.commands_executed} commands. Great progress!")
        
        return '\n'.join(output)
    
    def _get_command_suggestions(self, last_command: str) -> List[str]:
        """Get suggested follow-up commands"""
        last_lower = last_command.lower()
        
        suggestions_map = {
            'scan wifi': ['show wifi results', 'capture handshake', 'stop wifi'],
            'detect drone': ['show drone signals', 'jam drone', 'stop drone'],
            'spectrum': ['analyze signal', 'save spectrum data'],
            'status': ['show opsec', 'list missions'],
            'go online': ['show network status', 'go offline'],
        }
        
        for trigger, suggestions in suggestions_map.items():
            if trigger in last_lower:
                return suggestions
        
        return []
    
    def get_beginner_tips(self, context: str) -> List[str]:
        """Get contextual tips for beginners"""
        if self.profile.mode != UserMode.BEGINNER:
            return []
        
        tips = {
            'startup': [
                "Start with 'show status' to see system state",
                "Run 'show opsec' to check your security posture",
                "'list missions' shows guided operations"
            ],
            'network': [
                "System is OFFLINE by default for your protection",
                "Going online requires explicit confirmation",
                "Always check OPSEC score before online operations"
            ],
            'wifi': [
                "Passive scanning doesn't transmit - it's just listening",
                "Deauth attacks affect real users - only use on authorized networks",
                "Always have written authorization before testing"
            ],
            'cellular': [
                "Cellular testing has strict legal requirements",
                "Use a Faraday cage for isolated testing",
                "IMSI catching without authorization is illegal"
            ],
            'gps': [
                "GPS spoofing MUST be done in shielded environment",
                "Open-air GPS spoofing is a federal crime",
                "Test only on devices you own or are authorized to test"
            ]
        }
        
        return tips.get(context, tips['startup'])
    
    def get_status(self) -> Dict:
        """Get current user mode status"""
        return {
            'mode': self.profile.mode.value,
            'mode_name': self.config.name,
            'description': self.config.description,
            'commands_executed': self.profile.commands_executed,
            'missions_completed': self.profile.missions_completed,
            'settings': {
                'show_explanations': self.config.show_explanations,
                'confirm_dangerous': self.config.confirm_dangerous_only,
                'show_safety_warnings': self.config.show_safety_warnings,
                'verbose_output': self.config.verbose_output
            }
        }
    
    def get_mode_comparison(self) -> str:
        """Get comparison of all modes"""
        output = []
        output.append("USER MODES COMPARISON")
        output.append("=" * 60)
        
        for mode in UserMode:
            config = self.MODE_CONFIGS[mode]
            current = "← CURRENT" if mode == self.profile.mode else ""
            
            output.append(f"\n{config.name.upper()} {current}")
            output.append(f"  {config.description}")
            output.append(f"  • Explanations: {'Yes' if config.show_explanations else 'On request'}")
            output.append(f"  • Confirmations: {'All dangerous' if config.confirm_dangerous_only else 'None'}")
            output.append(f"  • Safety warnings: {'Yes' if config.show_safety_warnings else 'No'}")
            output.append(f"  • Verbose output: {'Yes' if config.verbose_output else 'No'}")
        
        output.append("\n" + "=" * 60)
        output.append("Change mode: 'set mode beginner/intermediate/expert'")
        
        return '\n'.join(output)


# Global instance
_user_mode_manager: Optional[UserModeManager] = None


def get_user_mode_manager() -> UserModeManager:
    """Get global UserModeManager instance"""
    global _user_mode_manager
    if _user_mode_manager is None:
        _user_mode_manager = UserModeManager()
    return _user_mode_manager


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = get_user_mode_manager()
    
    print("=" * 60)
    print("RF Arsenal OS - User Mode System")
    print("=" * 60)
    
    print(f"\nCurrent: {manager.get_current_mode().value}")
    print(f"Commands executed: {manager.profile.commands_executed}")
    print(f"Missions completed: {manager.profile.missions_completed}")
    
    print("\n" + manager.get_mode_comparison())
