#!/bin/bash
#
# RF Arsenal OS - AI Module Installer
# Installs lightweight AI components for Raspberry Pi
#

set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║   RF Arsenal OS - AI Module Installation              ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "[!] Please run as root (use sudo)"
    exit 1
fi

# Detect system
echo "[*] Detecting system..."
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    echo "    Detected: $MODEL"
fi

# Install system dependencies
echo ""
echo "[*] Installing system dependencies..."
apt-get update -qq
apt-get install -y \
    python3-pip \
    python3-dev \
    portaudio19-dev \
    espeak \
    espeak-ng \
    beep \
    ffmpeg \
    git \
    build-essential \
    cmake

echo "    ✓ System dependencies installed"

# Upgrade pip
echo ""
echo "[*] Upgrading pip..."
pip3 install --upgrade pip setuptools wheel

# Install Python packages for text-only AI (minimal)
echo ""
echo "[*] Installing Python packages (text-only mode)..."
pip3 install --quiet \
    numpy \
    scipy

echo "    ✓ Python packages installed"

# Optional: Install Whisper for voice control
echo ""
read -p "[?] Install voice control? (Whisper + Audio libs) [y/N]: " install_voice
if [[ $install_voice =~ ^[Yy]$ ]]; then
    echo "[*] Installing voice control components..."
    
    # Install audio libraries
    pip3 install --quiet \
        openai-whisper \
        sounddevice \
        soundfile \
        scipy
    
    # Download Whisper Tiny model (39MB)
    echo "[*] Downloading Whisper Tiny model (39MB)..."
    python3 -c "import whisper; whisper.load_model('tiny')"
    
    echo "    ✓ Voice control installed"
    VOICE_ENABLED=true
else
    echo "    ⊘ Skipping voice control (text-only mode)"
    VOICE_ENABLED=false
fi

# Optional: Install local LLM
echo ""
read -p "[?] Install local LLM? (Llama.cpp + 700MB model) [y/N]: " install_llm
if [[ $install_llm =~ ^[Yy]$ ]]; then
    echo "[*] Installing llama.cpp..."
    
    cd /tmp
    
    if [ ! -d "llama.cpp" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git
    fi
    
    cd llama.cpp
    make clean
    make -j$(nproc)
    cp llama-cli /usr/local/bin/
    
    echo "    ✓ llama.cpp installed"
    
    # Download LLM model
    echo ""
    echo "[*] Choose LLM model:"
    echo "    1) Llama 3.2 1B Q4 (700MB) - Recommended"
    echo "    2) TinyLlama 1.1B Q4 (600MB) - Lighter"
    echo "    3) Skip model download"
    read -p "[?] Choice [1-3]: " model_choice
    
    mkdir -p /opt/rfarsenal/models
    
    case $model_choice in
        1)
            echo "[*] Downloading Llama 3.2 1B model (700MB)..."
            cd /opt/rfarsenal/models
            wget -q --show-progress \
                https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
                -O llama-3.2-1b-instruct.Q4_K_M.gguf
            echo "    ✓ Llama 3.2 1B model downloaded"
            ;;
        2)
            echo "[*] Downloading TinyLlama 1.1B model (600MB)..."
            cd /opt/rfarsenal/models
            wget -q --show-progress \
                https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
                -O tinyllama.gguf
            echo "    ✓ TinyLlama model downloaded"
            ;;
        3)
            echo "    ⊘ Skipping model download"
            ;;
    esac
    
    LLM_ENABLED=true
else
    echo "    ⊘ Skipping LLM (basic parsing only)"
    LLM_ENABLED=false
fi

# Create AI module directory
echo ""
echo "[*] Setting up AI module..."
mkdir -p /opt/rfarsenal/modules/ai

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║          RF Arsenal AI Installation Complete           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "Installation Summary:"
echo "  ✓ Text-only AI:    Enabled"

if [ "$VOICE_ENABLED" = true ]; then
    echo "  ✓ Voice control:   Enabled (Whisper Tiny - 39MB)"
else
    echo "  ⊘ Voice control:   Disabled"
fi

if [ "$LLM_ENABLED" = true ]; then
    echo "  ✓ Local LLM:       Enabled"
else
    echo "  ⊘ Local LLM:       Disabled (basic parsing)"
fi

echo ""
echo "Usage:"
echo "  Text interface:  python3 modules/ai/text_ai.py"
if [ "$VOICE_ENABLED" = true ]; then
    echo "  Voice interface: python3 modules/ai/voice_ai.py"
fi
echo ""

if [ "$VOICE_ENABLED" = true ]; then
    echo "Estimated resource usage:"
    echo "  RAM:  2-3 GB (with voice + LLM)"
    echo "  Disk: ~800 MB (models)"
else
    echo "Estimated resource usage:"
    echo "  RAM:  < 100 MB (text-only)"
    echo "  Disk: < 50 MB"
fi

echo ""
echo "Test the AI interface:"
echo "  cd /opt/rfarsenal"
echo "  python3 modules/ai/text_ai.py"
echo ""
echo "[*] Installation complete!"
