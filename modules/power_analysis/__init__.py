"""
RF Arsenal OS - Power Analysis Side-Channel Module

Side-channel attack capabilities:
- Simple Power Analysis (SPA)
- Differential Power Analysis (DPA)
- Correlation Power Analysis (CPA)
- Electromagnetic Analysis (EMA)
- Timing attacks

Hardware: Oscilloscope/Logic Analyzer + BladeRF for EM capture

WARNING: Use only on devices you own or have explicit authorization to test.
"""

from .power_attacks import (
    PowerAnalysisController,
    PowerTrace,
    KeyHypothesis,
    AttackResult,
    AttackMode,
    TargetAlgorithm,
    get_power_analysis_controller
)

__all__ = [
    'PowerAnalysisController',
    'PowerTrace',
    'KeyHypothesis',
    'AttackResult',
    'AttackMode',
    'TargetAlgorithm',
    'get_power_analysis_controller'
]

__version__ = '1.0.0'
