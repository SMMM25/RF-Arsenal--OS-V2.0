#!/usr/bin/env python3
"""
RF Arsenal OS - Voice AI Interface
Speech recognition and synthesis for hands-free operation

Requirements:
- whisper (OpenAI Whisper for speech recognition)
- pyttsx3 or gTTS (for text-to-speech)
- sounddevice or pyaudio (for audio capture)
- numpy
"""

import os
import sys
import time
import queue
import logging
import threading
from typing import Optional, Callable
from dataclasses import dataclass

# Try importing optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

# Import AI controller
try:
    from modules.ai.ai_controller import AIController
except ImportError:
    AIController = None

logger = logging.getLogger(__name__)


@dataclass
class VoiceConfig:
    """Voice AI configuration"""
    # Speech recognition settings
    whisper_model: str = "tiny"  # tiny, base, small, medium, large
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    
    # Voice activation detection
    silence_threshold: float = 0.02
    min_speech_duration: float = 0.5  # seconds
    max_speech_duration: float = 10.0  # seconds
    silence_duration: float = 1.0  # seconds of silence to end recording
    
    # Text-to-speech settings
    tts_rate: int = 150  # words per minute
    tts_volume: float = 1.0
    tts_voice: Optional[str] = None  # Default system voice
    
    # Wake word
    wake_word: str = "arsenal"
    require_wake_word: bool = True


class AudioCapture:
    """Audio capture from microphone"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.logger = logging.getLogger('AudioCapture')
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            self.logger.warning(f"Audio status: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.copy())
    
    def start_stream(self):
        """Start audio input stream"""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice not installed. Run: pip install sounddevice")
        
        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            callback=self._audio_callback
        )
        self.stream.start()
        self.logger.info("Audio stream started")
    
    def stop_stream(self):
        """Stop audio input stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.logger.info("Audio stream stopped")
    
    def record_speech(self) -> Optional[np.ndarray]:
        """Record speech with voice activity detection"""
        if not HAS_NUMPY:
            raise RuntimeError("numpy not installed")
        
        self.is_recording = True
        audio_chunks = []
        silence_count = 0
        speech_detected = False
        start_time = time.time()
        
        silence_frames = int(self.config.silence_duration * self.config.sample_rate / 1024)
        max_frames = int(self.config.max_speech_duration * self.config.sample_rate / 1024)
        
        self.logger.debug("Recording started, listening for speech...")
        
        while True:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_chunks.append(chunk)
                
                # Calculate RMS amplitude
                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) / 32768.0
                
                if rms > self.config.silence_threshold:
                    speech_detected = True
                    silence_count = 0
                elif speech_detected:
                    silence_count += 1
                    if silence_count >= silence_frames:
                        self.logger.debug("End of speech detected")
                        break
                
                # Max duration check
                if len(audio_chunks) >= max_frames:
                    self.logger.debug("Max duration reached")
                    break
                    
            except queue.Empty:
                if time.time() - start_time > self.config.max_speech_duration + 2:
                    break
        
        self.is_recording = False
        
        if not speech_detected or len(audio_chunks) < 5:
            return None
        
        # Concatenate and convert to float32 for Whisper
        audio_data = np.concatenate(audio_chunks)
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        return audio_float


class SpeechRecognizer:
    """Speech recognition using OpenAI Whisper"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.logger = logging.getLogger('SpeechRecognizer')
        self.model = None
        
    def load_model(self):
        """Load Whisper model"""
        if not HAS_WHISPER:
            raise RuntimeError("whisper not installed. Run: pip install openai-whisper")
        
        self.logger.info(f"Loading Whisper model: {self.config.whisper_model}")
        self.model = whisper.load_model(self.config.whisper_model)
        self.logger.info("Whisper model loaded")
    
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text"""
        if self.model is None:
            self.load_model()
        
        # Ensure correct sample rate (Whisper expects 16kHz)
        result = self.model.transcribe(
            audio,
            language="en",
            fp16=False  # Use FP32 for CPU compatibility
        )
        
        text = result["text"].strip()
        self.logger.info(f"Transcribed: {text}")
        
        return text


class TextToSpeech:
    """Text-to-speech synthesis"""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.logger = logging.getLogger('TTS')
        self.engine = None
        self._lock = threading.Lock()
        
    def initialize(self):
        """Initialize TTS engine"""
        if not HAS_PYTTSX3:
            self.logger.warning("pyttsx3 not installed. TTS disabled.")
            return False
        
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.config.tts_rate)
            self.engine.setProperty('volume', self.config.tts_volume)
            
            # Set voice if specified
            if self.config.tts_voice:
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if self.config.tts_voice.lower() in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            self.logger.info("TTS engine initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"TTS initialization failed: {e}")
            return False
    
    def speak(self, text: str, blocking: bool = True):
        """Speak text"""
        if self.engine is None:
            print(f"[TTS] {text}")
            return
        
        with self._lock:
            try:
                self.engine.say(text)
                if blocking:
                    self.engine.runAndWait()
                else:
                    # Start in thread
                    thread = threading.Thread(target=self.engine.runAndWait)
                    thread.daemon = True
                    thread.start()
            except Exception as e:
                self.logger.error(f"TTS error: {e}")
                print(f"[TTS] {text}")


class VoiceAIInterface:
    """
    Voice-controlled AI interface for RF Arsenal OS
    
    Features:
    - Wake word detection
    - Continuous listening
    - Speech recognition (Whisper)
    - Text-to-speech responses
    - Natural language command processing
    """
    
    def __init__(self, config: Optional[VoiceConfig] = None, main_controller=None):
        """
        Initialize voice AI interface
        
        Args:
            config: Voice configuration
            main_controller: Main system controller (optional)
        """
        self.config = config or VoiceConfig()
        self.logger = logging.getLogger('VoiceAI')
        
        # Components
        self.audio_capture = AudioCapture(self.config)
        self.speech_recognizer = SpeechRecognizer(self.config)
        self.tts = TextToSpeech(self.config)
        
        # AI Controller
        if AIController:
            self.controller = AIController(main_controller)
        else:
            self.controller = None
            self.logger.warning("AIController not available")
        
        # State
        self.is_running = False
        self.is_listening = False
        self.command_callback: Optional[Callable] = None
        
    def check_dependencies(self) -> dict:
        """Check if all dependencies are available"""
        return {
            'numpy': HAS_NUMPY,
            'sounddevice': HAS_SOUNDDEVICE,
            'whisper': HAS_WHISPER,
            'pyttsx3': HAS_PYTTSX3,
            'ai_controller': self.controller is not None
        }
    
    def initialize(self) -> bool:
        """Initialize all components"""
        deps = self.check_dependencies()
        
        # Check critical dependencies
        if not deps['numpy']:
            self.logger.error("numpy is required. Run: pip install numpy")
            return False
        
        if not deps['sounddevice']:
            self.logger.error("sounddevice is required. Run: pip install sounddevice")
            return False
        
        if not deps['whisper']:
            self.logger.error("whisper is required. Run: pip install openai-whisper")
            return False
        
        # Load Whisper model
        try:
            self.speech_recognizer.load_model()
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            return False
        
        # Initialize TTS (optional)
        self.tts.initialize()
        
        self.logger.info("Voice AI initialized")
        return True
    
    def detect_wake_word(self, text: str) -> bool:
        """Check if wake word is in text"""
        wake_word = self.config.wake_word.lower()
        return wake_word in text.lower()
    
    def extract_command(self, text: str) -> str:
        """Extract command after wake word"""
        wake_word = self.config.wake_word.lower()
        text_lower = text.lower()
        
        if wake_word in text_lower:
            # Get text after wake word
            idx = text_lower.find(wake_word) + len(wake_word)
            command = text[idx:].strip()
            # Clean up common artifacts
            command = command.lstrip(',').lstrip('.').strip()
            return command
        
        return text.strip()
    
    def process_command(self, command: str) -> str:
        """Process voice command"""
        if not command:
            return "I didn't catch that. Please try again."
        
        self.logger.info(f"Processing command: {command}")
        
        # Handle built-in commands
        command_lower = command.lower()
        
        if any(w in command_lower for w in ['stop listening', 'sleep', 'goodbye']):
            self.is_running = False
            return "Going to sleep. Say the wake word to activate."
        
        if 'help' in command_lower:
            return "I can help with cellular, WiFi, GPS, drone, spectrum, and jamming operations. Just tell me what you need."
        
        if 'status' in command_lower or 'what can you do' in command_lower:
            return "RF Arsenal ready. I can start base stations, scan networks, spoof GPS, detect drones, analyze spectrum, and more."
        
        # Process through AI controller
        if self.controller:
            response = self.controller.execute_command(command)
            return response
        else:
            return f"Command received: {command}. AI controller not available."
    
    def listen_once(self) -> Optional[str]:
        """Listen for one utterance and return transcription"""
        if not self.is_running:
            return None
        
        self.logger.debug("Listening for speech...")
        self.is_listening = True
        
        try:
            # Record speech
            audio = self.audio_capture.record_speech()
            
            if audio is None:
                self.is_listening = False
                return None
            
            # Transcribe
            text = self.speech_recognizer.transcribe(audio)
            self.is_listening = False
            
            return text if text else None
            
        except Exception as e:
            self.logger.error(f"Listen error: {e}")
            self.is_listening = False
            return None
    
    def start_continuous_listening(self):
        """Start continuous listening mode"""
        if not self.initialize():
            print("❌ Failed to initialize voice AI")
            return
        
        self.is_running = True
        
        # Start audio stream
        self.audio_capture.start_stream()
        
        print("")
        print("╔═══════════════════════════════════════════════════════╗")
        print("║       RF Arsenal OS - Voice Control Interface         ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print("")
        
        if self.config.require_wake_word:
            print(f"🎤 Listening for wake word: '{self.config.wake_word}'")
        else:
            print("🎤 Listening for commands...")
        
        print("   Say 'help' for assistance, 'stop listening' to exit")
        print("")
        
        self.tts.speak("RF Arsenal voice control active", blocking=False)
        
        try:
            while self.is_running:
                text = self.listen_once()
                
                if not text:
                    continue
                
                print(f"📢 Heard: {text}")
                
                # Check wake word if required
                if self.config.require_wake_word:
                    if not self.detect_wake_word(text):
                        continue
                    command = self.extract_command(text)
                else:
                    command = text
                
                if not command:
                    continue
                
                print(f"🔧 Command: {command}")
                
                # Process command
                response = self.process_command(command)
                print(f"✓ {response}")
                
                # Speak response
                self.tts.speak(response)
                
                # Callback if set
                if self.command_callback:
                    self.command_callback(command, response)
                
        except KeyboardInterrupt:
            print("\n🛑 Voice control interrupted")
        finally:
            self.stop()
    
    def stop(self):
        """Stop voice AI"""
        self.is_running = False
        self.audio_capture.stop_stream()
        self.logger.info("Voice AI stopped")
    
    def set_command_callback(self, callback: Callable[[str, str], None]):
        """Set callback for command processing"""
        self.command_callback = callback


class SimulatedVoiceAI:
    """
    Simulated voice AI for testing without audio hardware
    Uses text input to simulate voice commands
    """
    
    def __init__(self, main_controller=None):
        self.logger = logging.getLogger('SimVoiceAI')
        self.config = VoiceConfig()
        
        if AIController:
            self.controller = AIController(main_controller)
        else:
            self.controller = None
    
    def start_cli(self):
        """Start simulated voice command interface"""
        print("")
        print("╔═══════════════════════════════════════════════════════╗")
        print("║    RF Arsenal OS - Simulated Voice Interface          ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print("")
        print("Type commands as if speaking. Prefix with wake word.")
        print(f"Wake word: '{self.config.wake_word}'")
        print("")
        print("Examples:")
        print(f"  > {self.config.wake_word} start 5g base station")
        print(f"  > {self.config.wake_word} scan wifi networks")
        print(f"  > {self.config.wake_word} detect drones")
        print("")
        print("Type 'exit' to quit")
        print("")
        
        while True:
            try:
                text = input("🎤 [Simulated Voice] > ").strip()
                
                if not text:
                    continue
                
                if text.lower() in ['exit', 'quit', 'q']:
                    print("✓ Exiting voice simulation...")
                    break
                
                # Check wake word
                wake_word = self.config.wake_word.lower()
                if wake_word not in text.lower():
                    print(f"💤 (Wake word '{wake_word}' not detected)")
                    continue
                
                # Extract command after wake word
                idx = text.lower().find(wake_word) + len(wake_word)
                command = text[idx:].strip().lstrip(',').lstrip('.').strip()
                
                if not command:
                    print("❓ No command after wake word")
                    continue
                
                print(f"🔧 Processing: {command}")
                
                # Process through controller
                if self.controller:
                    response = self.controller.execute_command(command)
                else:
                    response = f"[Simulated] Command: {command}"
                
                print(f"🔊 [TTS] {response}")
                print("")
                
            except KeyboardInterrupt:
                print("\n✓ Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RF Arsenal OS Voice AI")
    parser.add_argument('--simulate', '-s', action='store_true',
                        help='Use simulated voice input (text-based)')
    parser.add_argument('--model', '-m', default='tiny',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size')
    parser.add_argument('--no-wake-word', action='store_true',
                        help='Disable wake word requirement')
    parser.add_argument('--wake-word', '-w', default='arsenal',
                        help='Custom wake word')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.simulate:
        # Use simulated voice interface
        sim = SimulatedVoiceAI()
        sim.config.wake_word = args.wake_word
        sim.start_cli()
    else:
        # Use real voice interface
        config = VoiceConfig(
            whisper_model=args.model,
            wake_word=args.wake_word,
            require_wake_word=not args.no_wake_word
        )
        
        voice_ai = VoiceAIInterface(config)
        
        # Check dependencies
        deps = voice_ai.check_dependencies()
        print("Dependency check:")
        for dep, available in deps.items():
            status = "✓" if available else "✗"
            print(f"  {status} {dep}")
        
        missing = [k for k, v in deps.items() if not v and k != 'pyttsx3']
        if missing:
            print(f"\n❌ Missing required dependencies: {', '.join(missing)}")
            print("Run: pip install numpy sounddevice openai-whisper")
            print("\nUse --simulate for text-based testing")
            return
        
        voice_ai.start_continuous_listening()


if __name__ == "__main__":
    main()
