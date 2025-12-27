"""
RF Arsenal OS - Automation Module
Complete automation framework including session recording, mission scripting,
scheduled tasks, and trigger-based actions.

All operations support offline mode and RAM-only execution.
"""

from .session_recorder import SessionRecorder, SessionPlayback, SessionEvent
from .mission_scripting import MissionScript, MissionEngine, MissionStep
from .scheduler import TaskScheduler, ScheduledTask, ScheduleType
from .triggers import TriggerEngine, TriggerAction, TriggerCondition

__all__ = [
    'SessionRecorder',
    'SessionPlayback',
    'SessionEvent',
    'MissionScript',
    'MissionEngine',
    'MissionStep',
    'TaskScheduler',
    'ScheduledTask',
    'ScheduleType',
    'TriggerEngine',
    'TriggerAction',
    'TriggerCondition'
]

__version__ = "1.0.0"
