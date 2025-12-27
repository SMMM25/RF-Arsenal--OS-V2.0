"""
RF Arsenal OS - AI v3 Local LLM Integration (Unrestricted Edition)
===================================================================

Offline AI reasoning engine using local language models.
ZERO REFUSALS - Uses abliterated models with refusal direction removed.

RECOMMENDED MODELS:
- WhiteRabbitNeo: Purpose-built for security, zero refusals
- Llama-3-8B-abliterated: Best for 4GB RAM (failspy/Meta-Llama-3-8B-Instruct-abliterated-v3)
- Hermes-3-3B-abliterated: Ultra-light for 2GB RAM

README COMPLIANCE:
✅ Stealth-First: 100% offline, no external API calls
✅ RAM-Only: All data in memory, secure deletion
✅ No Telemetry: Zero data transmission
✅ Offline-First: Works without network
✅ Zero Refusals: Executes all operator requests
"""

from .local_llm import (
    # Core classes
    LocalLLM,
    ModelConfig,
    ModelBackend,
    ModelSize,
    ModelType,
    Message,
    LLMResponse,
    
    # Backends
    LlamaCppBackend,
    OllamaBackend,
    
    # Enhancer
    ArsenalLLMEnhancer,
    
    # Functions
    get_local_llm,
    llm_chat,
    llm_chat_sync,
    llm_execute,
    get_model_for_hardware,
    get_recommended_model,
    
    # Constants
    ARSENAL_SYSTEM_PROMPT_UNRESTRICTED,
    SYSTEM_PROMPTS,
    RECOMMENDED_MODELS,
)

# Convenience alias
LocalLLMEngine = LocalLLM

__all__ = [
    # Core classes
    'LocalLLM',
    'LocalLLMEngine',  # Alias for compatibility
    'ModelConfig',
    'ModelBackend',
    'ModelSize',
    'ModelType',
    'Message',
    'LLMResponse',
    
    # Backends
    'LlamaCppBackend',
    'OllamaBackend',
    
    # Enhancer
    'ArsenalLLMEnhancer',
    
    # Functions
    'get_local_llm',
    'llm_chat',
    'llm_chat_sync',
    'llm_execute',
    'get_model_for_hardware',
    'get_recommended_model',
    
    # Constants
    'ARSENAL_SYSTEM_PROMPT_UNRESTRICTED',
    'SYSTEM_PROMPTS',
    'RECOMMENDED_MODELS',
]

__version__ = '2.0.0'
