# VoLTE/VoNR Voice Interception Module

**RF Arsenal OS - Professional Voice Interception System**

## Overview

The VoLTE (Voice over LTE) and VoNR (Voice over New Radio/5G) interception module provides professional-grade voice call interception capabilities for authorized penetration testing.

### Key Capabilities

- **4G/5G Voice Interception**: Intercept voice calls on modern networks
- **Forced Downgrade**: Force phones to downgrade from VoLTE/VoNR to unencrypted 2G/3G calls
- **SIP Monitoring**: Capture call metadata from SIP signaling (VoLTE calls)
- **Audio Capture**: Record full audio from downgraded calls
- **Call Logging**: Track call history with metadata

## Technical Approach

### Method 1: Forced Downgrade (Full Audio)

This is the primary interception method that provides full audio capture:

1. **Advertise LTE network without VoLTE/VoNR capability**
   - Configure rogue LTE base station with `ims_support=False`
   - Set `voice_domain='CS'` (Circuit-Switched only)
   - Enable `eps_fallback=True`

2. **Force phones to fall back to 2G/3G**
   - When phone attempts voice call, it sees no VoLTE support
   - Phone automatically downgrades to 3G UMTS or 2G GSM
   - Connects to rogue 2G/3G base station

3. **Intercept unencrypted calls**
   - 2G GSM uses weak A5/1 or A5/3 encryption (easily broken)
   - 3G UMTS uses Kasumi algorithm (known vulnerabilities)
   - Capture and decode audio frames in real-time

4. **Audio Decoding**
   - GSM Full Rate (FR): 13 kbps, 33 bytes → 160 samples
   - GSM Enhanced FR: 12.2 kbps
   - AMR (Adaptive Multi-Rate): 4.75-12.2 kbps
   - Output: 8 kHz, 16-bit PCM WAV files

### Method 2: SIP Monitoring (Metadata Only)

This method captures call metadata without audio:

1. **Advertise full VoLTE support**
   - Configure LTE with `ims_support=True`
   - Set `voice_domain='PS'` (Packet-Switched/VoLTE)

2. **Monitor SIP signaling**
   - Capture SIP INVITE messages (call setup)
   - Capture SIP BYE messages (call teardown)
   - Extract caller, callee, duration

3. **Limitations**
   - Audio is IPSec encrypted (cannot decrypt)
   - Metadata only: who called whom, when, duration
   - No audio capture

## STEALTH FEATURES

The module is designed for maximum stealth:

### Storage Security
- **Covert paths**: `/tmp/.rf_arsenal_data/voice/`
- **Obfuscated filenames**: MD5 hashes instead of phone numbers
- **Restrictive permissions**: 0o700 (owner only)
- **RAM-only mode**: Optional volatile storage

### Anti-Forensics
- **Encrypted RAM overlay**: Secure volatile storage
- **3-pass secure deletion**: DoD 5220.22-M standard
- **Emergency cleanup**: Panic button integration
- **No external connections**: 100% local operation

### Minimal Logging
- **Obfuscated numbers**: `+15551234567` → `+15***67`
- **Stealth mode logging**: Minimal console output
- **Local only**: No network logging

### Emergency Integration
- **Panic button**: Secure wipe on button press
- **Deadman switch**: Auto-wipe on timeout
- **Secure deletion**: All audio files wiped

## Installation

### Hardware Requirements

- **BladeRF 2.0 micro xA9** (LTE base station)
- **HackRF One or USRP** (2G/3G base station)
- **Raspberry Pi 4+** or Linux PC
- **Dual SDR setup** (one for LTE, one for GSM)

### Software Dependencies

```bash
# Core dependencies
sudo apt-get install -y build-essential cmake pkg-config \
    libgsm1-dev libopencore-amrnb-dev tshark

# Python packages
pip3 install numpy

# GSM codec (for audio decoding)
pip3 install pygsm

# AMR codec (for audio decoding)  
pip3 install opencore-amr

# srsRAN (LTE base station)
cd /opt
git clone https://github.com/srsran/srsRAN_4G.git
cd srsRAN_4G
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# OsmocomBB/OpenBTS (2G/3G base station)
sudo apt-get install -y openbts osmocom-bb
```

### Module Integration

The module is automatically loaded when cellular controllers are initialized:

```python
from modules.cellular.volte_interceptor import VoLTEInterceptor

# Initialize with LTE and GSM controllers
interceptor = VoLTEInterceptor(
    lte_controller=lte,
    gsm_controller=gsm,
    stealth_mode=True  # Enable stealth features
)
```

## Usage

### Command Line Interface

#### Start Voice Interception (Downgrade Mode)

```bash
# Via AI controller
intercept voice

# Full audio capture via forced downgrade
```

#### Start SIP Monitoring (Metadata Only)

```bash
# Via AI controller
intercept voice sip

# Metadata only, audio encrypted
```

#### Stop Interception

```bash
stop voice
```

#### List Captured Calls

```bash
# List all calls
list calls

# List active calls only
list calls active
```

#### Export Call Log

```bash
export calls
# Saves to: /tmp/.rf_arsenal_data/call_log_<timestamp>.json
```

### Python API

#### Basic Usage

```python
# Start downgrade-based interception (full audio)
interceptor.start_interception(mode='downgrade')

# Run for desired duration
time.sleep(300)  # 5 minutes

# Stop interception
interceptor.stop_interception()

# List captured calls
interceptor.list_calls()

# Export call log
interceptor.export_call_log('/tmp/call_log.json')
```

#### SIP Monitoring (Metadata Only)

```python
# Start SIP monitoring
interceptor.start_interception(mode='sip')

# Captures call metadata, audio encrypted
```

#### Emergency Cleanup

```python
# Secure wipe (called on panic button)
interceptor.emergency_cleanup()
```

## Output Files

### Audio Captures

**Location**: `/tmp/.rf_arsenal_data/voice/`

**Filename Format**: `{call_hash}.wav`
- Obfuscated MD5 hash
- Example: `a4b8c2d1.wav`

**Audio Format**:
- Format: WAV (PCM)
- Sample Rate: 8000 Hz (GSM standard)
- Bit Depth: 16-bit
- Channels: 1 (Mono)

### Call Logs

**Location**: `/tmp/.rf_arsenal_data/call_log_<timestamp>.json`

**Format**:
```json
{
  "statistics": {
    "calls_intercepted": 5,
    "volte_calls": 2,
    "downgraded_calls": 3,
    "failed_intercepts": 0
  },
  "calls": [
    {
      "call_id": "abc123",
      "caller": "+15***67",
      "callee": "+19***01",
      "protocol": "CS-GSM",
      "start_time": "2024-12-21T10:30:00",
      "end_time": "2024-12-21T10:32:15",
      "duration": 135.0,
      "audio_file": "/tmp/.rf_arsenal_data/voice/a4b8c2d1.wav",
      "sip_metadata": {}
    }
  ]
}
```

## Technical Details

### Audio Codec Support

#### GSM Full Rate (FR)
- Bitrate: 13 kbps
- Frame size: 33 bytes
- Output: 160 samples per frame (20ms @ 8kHz)
- Algorithm: RPE-LTP (Regular Pulse Excitation - Long Term Prediction)
- Library: `libgsm`

#### AMR (Adaptive Multi-Rate)
- Bitrate: 4.75 - 12.2 kbps (8 modes)
- Frame size: Variable (6-31 bytes)
- Output: 160 samples per frame (20ms @ 8kHz)
- Algorithm: ACELP (Algebraic Code Excited Linear Prediction)
- Library: `opencore-amr`

### Call States

- `IDLE`: No active call
- `RINGING`: Incoming/outgoing call setup
- `ACTIVE`: Call in progress
- `HELD`: Call on hold
- `TERMINATED`: Call ended

### Voice Protocols

- `VoLTE`: Voice over LTE (4G) - IPSec encrypted
- `VoNR`: Voice over NR (5G) - IPSec encrypted
- `CS-UMTS`: Circuit-Switched 3G - Kasumi encryption (weak)
- `CS-GSM`: Circuit-Switched 2G - A5/1 encryption (broken)

## Security Considerations

### Physical Security

- **Range**: ~500m typical (depends on SDR TX power)
- **Detection risk**: Rogue base stations can be detected
- **Legal compliance**: Requires authorization

### Data Security

- **Encrypted storage**: Optional RAM overlay encryption
- **Secure deletion**: 3-pass DoD standard
- **Obfuscated filenames**: No identifiable information
- **Local only**: No network transmission

### Operational Security

- **Stealth mode**: Minimal logging and output
- **Emergency wipe**: Panic button integration
- **Anti-forensics**: Secure cleanup on shutdown
- **Covert paths**: Hidden storage directories

## Troubleshooting

### No Audio Captured

**Problem**: Calls detected but no audio files

**Solutions**:
1. Install GSM codec: `pip3 install pygsm`
2. Install AMR codec: `pip3 install opencore-amr`
3. Check audio capture thread logs
4. Verify 2G base station is running

### Phones Not Downgrading

**Problem**: Phones stay on VoLTE

**Solutions**:
1. Ensure `ims_support=False` in LTE config
2. Set `voice_domain='CS'` (Circuit-Switched)
3. Enable `eps_fallback=True`
4. Check LTE signal strength (must be stronger than real network)
5. Some phones prefer VoLTE (carrier settings)

### SIP Packets Not Captured

**Problem**: No SIP metadata

**Solutions**:
1. Install tshark: `sudo apt-get install tshark`
2. Run as root or configure permissions
3. Check network interface (`tshark -i any`)
4. Verify port 5060 traffic

### Failed Interception

**Problem**: High `failed_intercepts` count

**Solutions**:
1. Check SDR hardware connections
2. Verify frequencies (carrier-specific)
3. Increase TX power (within legal limits)
4. Check for interference
5. Verify base station configuration

## Advanced Configuration

### Custom Storage Location

```python
interceptor = VoLTEInterceptor(lte, gsm, stealth_mode=False)
interceptor.output_dir = Path('/custom/path')
```

### RAM-Only Operation

```python
from security.anti_forensics import EncryptedRAMOverlay

# Enable RAM overlay
interceptor.ram_overlay = EncryptedRAMOverlay()
```

### Custom Codec Parameters

```python
# Custom audio format (default: 8kHz, 16-bit, mono)
# Modify in _init_audio_file() method
with wave.open(filename, 'wb') as wav:
    wav.setnchannels(1)      # Mono
    wav.setsampwidth(2)      # 16-bit
    wav.setframerate(16000)  # 16 kHz (custom)
```

## Performance

### Resource Usage

- **CPU**: ~30% per active call (audio decoding)
- **Memory**: ~50 MB base + ~2 MB per call
- **Storage**: ~600 KB per minute of audio (8 kHz, mono)
- **Network**: Local only, no external connections

### Scalability

- **Concurrent calls**: Up to 20 simultaneous calls
- **Call duration**: Unlimited
- **Storage limit**: Depends on available disk space

## Integration

### Emergency System

```python
# In core/emergency.py
def emergency_wipe(self, trigger):
    # VoLTE cleanup integration
    if hasattr(self, 'volte_interceptor'):
        self.volte_interceptor.emergency_cleanup()
```

### AI Controller

```python
# In modules/ai/ai_controller.py
from modules.cellular.volte_interceptor import VoLTEInterceptor, parse_volte_command

# Commands: intercept voice, stop voice, list calls, export calls
```

## Legal Notice

This module is designed for **AUTHORIZED PENETRATION TESTING ONLY**.

Unauthorized voice interception is **ILLEGAL** in most jurisdictions:
- USA: 18 U.S.C. § 2511 (Wiretap Act) - up to 5 years imprisonment
- EU: GDPR Article 6 violations - up to €20M fines
- UK: Investigatory Powers Act 2016 - up to 2 years imprisonment

**Requirements**:
- Written authorization from network owner
- Penetration testing engagement contract
- Legal compliance documentation

**This system proves vulnerabilities. Do not use for illegal surveillance.**

## References

- [VoLTE Security White Paper](https://www.gsma.com/security/)
- [srsRAN Documentation](https://docs.srsran.com/)
- [GSM Codec Specifications](https://www.etsi.org/technologies/mobile/gsm)
- [AMR Codec Specifications](https://www.3gpp.org/technologies/keywords-acronyms/103-amr)
- [DoD 5220.22-M Secure Deletion](https://www.dss.mil/isp/odaa/documents/nispom.pdf)

## Support

For technical issues or questions:
- Check system logs: `journalctl -u rf-arsenal -f`
- Review module logs: `/var/log/rf-arsenal/volte.log`
- GitHub Issues: [RF Arsenal OS Repository](https://github.com/SMMM25/RF-Arsenal-OS)

---

**Version**: 1.0.5  
**Status**: PRODUCTION READY  
**Last Updated**: 2024-12-21
