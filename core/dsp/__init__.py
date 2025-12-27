"""
RF Arsenal OS - Digital Signal Processing Engine

Production-grade DSP algorithms for RF operations.
Implements OFDM, modulation, synchronization, and channel coding.
"""

from .primitives import (
    DSPConfig,
    DSPEngine,
    FilterType,
    FilterSpec,
    FilterDesign,  # Direct export
    FilterDesign as FilterDesigner,  # Backward compatibility alias
    WindowType,
    WindowFunctions as WindowFunction,
    Resampler,
    DCBlocker,
    AGC,
    IQCorrector,
)

# Aliases for backward compatibility
FFTProcessor = DSPEngine  # DSPEngine includes FFT functionality
PowerMeter = DSPEngine  # DSPEngine includes power measurement
NoiseEstimator = DSPEngine  # DSPEngine includes noise estimation

from .modulation import (
    ModulationType,
    ModulationConfig,
    ModulationEngine as Modulator,  # Alias for compatibility
    Demodulator,
    PSKModulator as QPSKModem,  # Alias
    QAMModulator as QAM16Modem,  # Alias
    QAMModulator as QAM64Modem,  # Alias
    QAMModulator as QAM256Modem,  # Alias
    GMSKModulator as GMSKModem,  # Alias
    OFDMModulator
)

from .ofdm import (
    LTEOFDMParams as OFDMConfig,  # Alias for compatibility
    LTEOFDMParams as OFDMSymbol,  # Alias
    OFDMEngine,  # Direct export
    OFDMEngine as LTEOFDMProcessor,  # Alias for compatibility
    OFDMEngine as NROFDMProcessor,  # Alias
    ResourceGrid as CellSearcher,  # Alias
    ResourceGrid as LTECellSearcher,  # Alias
    ResourceGrid as NRCellSearcher,  # Alias
    ResourceGrid as ResourceMapper,  # Alias
)

from .synchronization import (
    TimingSync,
    FrequencySync,
    FrameSync,
    CellSearch as PSSSynchronizer,  # Alias
    CellSearch as SSSSynchronizer,  # Alias
    TimingSync as LTETimingAdvance,  # Alias
    FrequencySync as CFOEstimator,  # Alias
    TimingSync as SymbolTimingRecovery,  # Alias
)

from .channel_coding import (
    CRCCalculator,
    ConvolutionalCoder,
    TurboCoder,
    LDPCCoder,
    PolarCoder,
    ChannelCoder as RateMatching,  # Alias
    ConvolutionalCoder as InterleaverLTE,  # Alias
    ConvolutionalCoder as InterleaverNR,  # Alias
)

__all__ = [
    # Primitives
    'DSPConfig',
    'FilterDesign',
    'FilterDesigner',  # Alias
    'WindowFunction',
    'OFDMEngine',
    'FFTProcessor',
    'Resampler',
    'DCBlocker',
    'AGC',
    'PowerMeter',
    'NoiseEstimator',
    # Modulation
    'ModulationType',
    'ModulationConfig',
    'Modulator',
    'Demodulator',
    'QPSKModem',
    'QAM16Modem',
    'QAM64Modem',
    'QAM256Modem',
    'GMSKModem',
    'OFDMModulator',
    # OFDM
    'OFDMConfig',
    'OFDMSymbol',
    'LTEOFDMProcessor',
    'NROFDMProcessor',
    'CellSearcher',
    'LTECellSearcher',
    'NRCellSearcher',
    'ResourceMapper',
    # Synchronization
    'TimingSync',
    'FrequencySync',
    'FrameSync',
    'PSSSynchronizer',
    'SSSSynchronizer',
    'LTETimingAdvance',
    'CFOEstimator',
    'SymbolTimingRecovery',
    # Channel Coding
    'CRCCalculator',
    'ConvolutionalCoder',
    'TurboCoder',
    'LDPCCoder',
    'PolarCoder',
    'RateMatching',
    'InterleaverLTE',
    'InterleaverNR',
]

__version__ = '1.0.0'
