# RF Arsenal OS - Stealth Operations Guide
**🔒 Maximum Stealth & Anonymity for ML Device Fingerprinting**

---

## ⚠️ CRITICAL: What "Stealth" Means

**Stealth Operation:** RF monitoring that is **undetectable** to targets, network operators, and detection systems.

### ✅ Stealth-Safe Activities (Passive-Only)
- **Receiving RF signals** (cellular downlink from towers)
- **Analyzing RF characteristics** (timing, power, modulation)
- **Machine learning classification** (local processing only)
- **Network profiling** (statistical analysis of observed devices)

### ❌ Stealth-Breaking Activities (Active/Detectable)
- **IMSI catching** (requires fake cell tower transmission)
- **Active geolocation** (requires ranging query transmissions)
- **Device probing** (requires direct queries to devices)
- **Network injection** (requires packet transmission)
- **Jamming** (requires active RF interference)

---

## 🎯 Stealth Threat Model

### Who Can Detect You?

| Threat Actor | Detection Method | Risk Level | Mitigation |
|--------------|------------------|------------|------------|
| **Cellular Carriers** | Anomaly detection, rogue tower identification | 🔴 **High** | Use passive-only mode, never transmit |
| **Target Devices** | SnoopSnitch app, GSMK CryptoPhone | 🔴 **High** | Disable IMSI catcher features |
| **RF Monitoring** | Spectrum analyzers, direction finding | 🟡 **Medium** | Passive-only, physical concealment |
| **Network Security** | SIEM/IDS on corporate networks | 🟡 **Medium** | No active probing or injection |
| **Law Enforcement** | Specialized RF detection equipment | 🔴 **High** | Legal authorization required |

### Detection Signatures to Avoid

❌ **Rogue Cell Tower Signatures:**
- Unusual cell IDs not in carrier database
- Missing encryption (IMSI catchers often downgrade to 2G)
- Sudden signal strength changes
- Devices losing connectivity then reconnecting

❌ **Active Transmission Signatures:**
- Unexpected RF energy at non-standard times/frequencies
- Burst transmissions that don't match normal network patterns
- High-power transmissions from mobile equipment
- Frequency hopping patterns inconsistent with legitimate use

❌ **Behavioral Anomalies:**
- Multiple devices simultaneously connecting to unknown tower
- Network performance degradation in specific area
- Unusual handover patterns
- Devices exhibiting abnormal timing advance values

---

## 🔒 ML Fingerprinting Stealth Configuration

### Stealth Mode Architecture

```python
# 🔒 STEALTH-SAFE CONFIGURATION
fingerprinter = MLDeviceFingerprinting(
    passive_mode=True,       # ✅ Enforce receive-only operation
    anonymize_logs=True      # ✅ Hash all IMSI/IMEI identifiers
)

# Verify stealth mode is active
assert fingerprinter.is_passive_mode() == True
assert fingerprinter.transmit_enabled == False
assert fingerprinter.active_probing_enabled == False

# ✅ SAFE: Passive RF signature classification
signature = capture_passive_signature()  # Receive only
profile = fingerprinter.identify_device(signature)

# ❌ UNSAFE: These operations would raise RuntimeError in passive mode
# fingerprinter.validate_stealth_operation('imsi_catcher')  # Raises error
# fingerprinter.validate_stealth_operation('active_geolocation')  # Raises error
```

### Command-Line Stealth Enforcement

```bash
# 🔒 STEALTH MODE: Passive-only operation
sudo python3 rf_arsenal_os.py \
    --ml-classify-live \
    --passive-only \           # Enforce no transmission
    --anonymize-logs \         # Hash identifiers
    --frequency 1842.6e6 \
    --model pretrained.pkl     # Use pre-trained model (no field training)

# ❌ BLOCKED: System prevents stealth-breaking features
sudo python3 rf_arsenal_os.py \
    --ml-classify-live \
    --passive-only \
    --imsi-catcher             # ERROR: Incompatible with passive mode

# ERROR: --imsi-catcher incompatible with --passive-only
# Passive mode disables all transmission capabilities
```

### SDR Hardware Configuration

```python
# Configure SDR for receive-only (hardware-enforced)
sdr_config = {
    'tx_enabled': False,              # ✅ Disable transmitter
    'rx_enabled': True,               # ✅ Enable receiver
    'tx_power': 0,                    # ✅ Zero transmit power
    'tx_antenna': None,               # ✅ No TX antenna connected
    'hardware_tx_lockout': True       # ✅ Hardware TX disable (BladeRF, USRP)
}

# Verify TX is disabled
assert sdr.tx_enabled == False, "Transmitter must be disabled for stealth"

# Physical verification (spectrum analyzer test)
# Observe SDR for 60 seconds - should show ZERO RF output
verify_no_transmission(sdr, duration=60)
```

---

## 📚 Training Data: Stealth-Safe Workflow

### ❌ Traditional Training (BREAKS STEALTH)

```bash
# ❌ BAD: Capturing real device signatures in the field
sudo python3 rf_arsenal_os.py \
    --ml-capture-signature \        # ⚠️ May require active probing
    --device-label "iPhone_14" \
    --frequency 1842.6e6

# Why this breaks stealth:
# - May transmit probing signals to elicit responses
# - Generates unusual RF activity patterns
# - Creates detectable timing anomalies
# - Risk of detection by spectrum monitoring
```

### ✅ Stealth-Safe Training (100% SYNTHETIC)

```python
# 🔒 STEALTH-SAFE: Use only synthetic training data
from modules.ai.training_data_generator import TrainingDataGenerator

generator = TrainingDataGenerator()

# Generate stealth-safe training dataset (no field captures)
dataset = generator.generate_stealth_safe_dataset(
    samples_per_device=200,  # 200 synthetic samples per device type
    output_path='data/training/stealth_safe_dataset.json'
)

# Validate dataset contains no real captures
assert generator.validate_no_real_captures(dataset), \
    "Dataset must be 100% synthetic for stealth operations"

# ✅ RESULT:
# - Zero RF transmissions performed
# - Zero field captures required
# - Model trained entirely offline
# - Safe for operational deployment
```

### Training Workflow Comparison

| Method | Stealth Level | Accuracy | Detection Risk |
|--------|---------------|----------|----------------|
| **Real Field Captures** | ❌ Low | 90-95% | **Very High** (RF transmissions) |
| **Lab Captures (Controlled)** | 🟡 Medium | 88-93% | Low (isolated environment) |
| **Synthetic Data Only** | ✅ **High** | 85-92% | **Zero** (no transmissions) |
| **Hybrid (Synthetic + Lab)** | 🟡 Medium | 90-94% | Low (if lab isolated) |

**Recommendation:** Use **synthetic data only** for maximum stealth. Accept 3-5% accuracy reduction in exchange for zero detection risk.

---

## 🛡️ Identifier Anonymization

### Why Anonymize IMSI/IMEI?

- **Privacy Law Compliance:** GDPR, CCPA require pseudonymization of personal identifiers
- **Operational Security:** Reduces damage if logs are compromised
- **Legal Protection:** Demonstrates good-faith privacy efforts
- **Stealth Enhancement:** Limits exposure if system is discovered

### Anonymization Implementation

```python
# Automatic anonymization with passive mode
fingerprinter = MLDeviceFingerprinting(
    passive_mode=True,
    anonymize_logs=True  # ✅ Hash all identifiers
)

# IMSI is automatically hashed before storage
original_imsi = "310260123456789"  # Raw IMSI (never stored)
hashed_imsi = fingerprinter._anonymize_identifier(original_imsi)
# Result: "8a3f7c2b4d1e"  # First 12 chars of SHA-256

# All logs and reports use hashed values
print(f"Device detected: {hashed_imsi[:8]}")  # Prints: "8a3f7c2b"
# Original IMSI is NOT stored anywhere
```

### Hash Properties

- **Algorithm:** SHA-256 (cryptographically secure)
- **Output:** First 12 characters of hex digest
- **Collision Resistance:** ~2^48 unique hashes (sufficient for network scale)
- **Irreversibility:** Cannot recover original IMSI from hash

### Log Example (Anonymized)

```
# ✅ ANONYMIZED LOG (safe to store/share)
[2024-03-15 14:23:11] Device detected: 8a3f7c2b - Apple iPhone 14 Pro (iOS 17.2) - 94% confidence
[2024-03-15 14:24:33] Device detected: 9f2d4e6a - Samsung Galaxy S23 (Android 14) - 89% confidence

# ❌ NON-ANONYMIZED LOG (privacy violation risk)
# [2024-03-15 14:23:11] Device detected: 310260123456789 - Apple iPhone 14 Pro (iOS 17.2) - 94% confidence
# ⚠️ Raw IMSI is PII (Personally Identifiable Information) - do NOT log!
```

---

## 🗺️ Geolocation: Passive vs Active

### ❌ Active Geolocation (BREAKS STEALTH)

```python
# ❌ UNSAFE: Active ranging requires transmission
geolocation = CellularGeolocation(method='active')

# This will transmit ranging queries to cells
position = geolocation.calculate_position(measurements)
# ⚠️ DETECTABLE: Devices and towers will see unusual ranging signals
```

**Why Active Geolocation Breaks Stealth:**
1. Requires **transmission of ranging queries** to multiple cell towers
2. Creates **unusual RF signatures** (non-standard measurement reports)
3. May trigger **network anomaly detection** (SIEM alerts)
4. Can be detected by **RF monitoring equipment**
5. Devices may **log unusual measurement requests**

### ✅ Passive Geolocation (STEALTH-SAFE)

```python
# ✅ SAFE: Passive timing advance analysis (receive-only)
geolocation = CellularGeolocation(method='passive')

# Uses timing advance from overheard device transmissions
measurements = capture_passive_measurements()
position = geolocation.passive_triangulation(measurements)

# No transmission required - pure observation
# Accuracy: ±100-500m (acceptable for most use cases)
```

**Passive Geolocation Methods (Stealth-Safe):**
1. **Timing Advance (TA):** Calculate distance from cell tower timing (GSM: ±550m accuracy)
2. **RSSI Triangulation:** Use signal strength from multiple observations (±100-1000m)
3. **Cell ID Lookup:** Match Cell ID to database coordinates (±500-5000m)

### Geolocation Accuracy Comparison

| Method | Accuracy | Stealth Level | Requirements |
|--------|----------|---------------|--------------|
| **Active Ranging** | ±10-50m | ❌ Low | RF transmission (detectable) |
| **Timing Advance** | ±100-550m | ✅ High | Passive observation only |
| **RSSI Triangulation** | ±100-1000m | ✅ High | Multiple cell observations |
| **Cell ID Lookup** | ±500-5000m | ✅ High | Cell database access |

**Recommendation:** Use **Timing Advance** or **RSSI Triangulation** for stealth operations. Accept reduced accuracy (~500m) in exchange for zero detection risk.

---

## 🚫 IMSI Catcher: Why It Breaks Stealth

### How IMSI Catchers Work

```
Normal Cell Tower:  Device <---> Legitimate Tower <---> Carrier Network
                           (Encrypted, Authenticated)

IMSI Catcher:       Device <---> Fake Tower <---> [YOU]
                           (Often unencrypted, forces 2G downgrade)
```

### Detection Signatures

**SnoopSnitch Detection (Android App):**
- ❌ Missing encryption
- ❌ Unusual cell ID (not in carrier database)
- ❌ Forced 2G downgrade (IMSI catchers can't do 4G/5G often)
- ❌ Unusual LAC/CellID combinations
- ❌ Timing advance anomalies

**GSMK CryptoPhone Detection:**
- ❌ IMSI request without authentication
- ❌ Missing ciphering
- ❌ Abnormal broadcast parameters
- ❌ Cell reselection manipulation

**Carrier-Side Detection:**
- ❌ Devices report connection to unknown cell ID
- ❌ Network performance degradation in area
- ❌ Unusual handover patterns
- ❌ Multiple devices offline simultaneously

### Stealth-Safe Alternative

```bash
# ❌ IMSI CATCHER (ACTIVE - BREAKS STEALTH)
# sudo python3 rf_arsenal_os.py --imsi-catcher --ml-classify

# ✅ PASSIVE IMSI OBSERVATION (STEALTH-SAFE)
sudo python3 rf_arsenal_os.py \
    --observe-imsi-from-downlink \  # Listen to tower broadcasts
    --ml-classify-live \
    --passive-only

# How it works:
# 1. Listen to downlink (tower → devices) transmissions
# 2. Extract IMSI/TMSI from broadcast paging messages
# 3. Classify devices using ML fingerprinting
# 4. ZERO transmission required (completely passive)
```

**Passive IMSI Observation Limitations:**
- Only observes devices actively communicating
- Cannot capture IMSI on first connection (uses TMSI)
- Requires decoding downlink messages (no encryption break required)
- Lower capture rate (~30-50% vs 100% for IMSI catcher)

**Trade-off:** Accept lower capture rate (50% vs 100%) for zero detection risk.

---

## 🔬 Stealth Validation Testing

### Pre-Deployment Checklist

```bash
# 1. Verify passive mode enforced
python3 -c "
from modules.ai.device_fingerprinting import MLDeviceFingerprinting
fp = MLDeviceFingerprinting(passive_mode=True)
assert fp.is_passive_mode() == True
assert fp.transmit_enabled == False
print('✅ Passive mode verified')
"

# 2. Test stealth operation validation
python3 -c "
from modules.ai.device_fingerprinting import MLDeviceFingerprinting
fp = MLDeviceFingerprinting(passive_mode=True)
try:
    fp.validate_stealth_operation('imsi_catcher')
    print('❌ FAIL: Should have raised error')
except RuntimeError:
    print('✅ Stealth validation working')
"

# 3. Verify SDR in receive-only mode
sudo python3 rf_arsenal_os.py --sdr-test --verify-no-tx

# 4. Spectrum analyzer test (manual)
# - Connect spectrum analyzer to SDR output
# - Run system for 60 seconds
# - Verify ZERO RF transmissions
```

### Spectrum Analyzer Validation

```bash
# Test protocol:
# 1. Connect SDR to spectrum analyzer
# 2. Start ML fingerprinting in passive mode
# 3. Monitor spectrum for 60 seconds
# 4. Verify no transmissions above noise floor

sudo python3 rf_arsenal_os.py \
    --ml-classify-live \
    --passive-only \
    --frequency 1842.6e6 \
    --duration 60 \
    --spectrum-analyzer-test

# Expected result:
# ✅ No RF emissions detected above -95 dBm
# ✅ SDR operating in receive-only mode
# ✅ Stealth validation PASSED
```

### Penetration Testing Detection Resistance

**Red Team Test Scenarios:**

1. **SnoopSnitch Scan:** Run SnoopSnitch app on test device in area → Should detect NOTHING
2. **Carrier SIEM Monitoring:** Check carrier logs for anomalies → Should show ZERO alerts
3. **Spectrum Analyzer Sweep:** Monitor frequency bands → Should detect ONLY normal traffic
4. **Direction Finding:** Attempt to locate RF source → Should find NOTHING (passive mode)
5. **Network Performance Test:** Measure latency/throughput → Should see ZERO degradation

**Pass Criteria:** All tests show ZERO detection signatures.

---

## 📋 Operational Security Checklist

### Before Deployment

- [ ] ✅ Model trained using **synthetic data only** (no field captures)
- [ ] ✅ `--passive-only` mode enforced in code
- [ ] ✅ `--anonymize-logs` enabled (IMSI/IMEI hashing)
- [ ] ✅ IMSI catcher features **disabled/removed**
- [ ] ✅ Geolocation set to `method='passive'`
- [ ] ✅ SDR verified in **RX-only mode** (hardware test)
- [ ] ✅ Spectrum analyzer test **passed** (no TX detected)
- [ ] ✅ All logs stored **locally** (no network exfiltration)
- [ ] ✅ Physical SDR **concealment** planned
- [ ] ✅ **Legal authorization** obtained in writing
- [ ] ✅ Incident response plan prepared (if discovered)
- [ ] ✅ Data destruction plan ready (if compromised)

### During Operation

- [ ] ✅ Monitor for detection attempts (unusual network behavior)
- [ ] ✅ Periodic stealth validation tests
- [ ] ✅ Secure log storage (encrypted, access-controlled)
- [ ] ✅ Physical security of SDR equipment
- [ ] ✅ No network connectivity (air-gapped operation if possible)

### After Operation

- [ ] ✅ Secure data storage or destruction (per policy)
- [ ] ✅ Remove all traces from operational environment
- [ ] ✅ Debrief with authorization authority
- [ ] ✅ Lessons learned documentation

---

## ⚖️ Legal Considerations

### When Stealth Operations Are Legal

✅ **Your own corporate network** (with employee policy/consent)  
✅ **Penetration testing** with written authorization  
✅ **Law enforcement** with proper warrant/legal order  
✅ **Academic research** with IRB approval and consent  
✅ **Red team exercises** with explicit authorization

### When Stealth Operations Are ILLEGAL

❌ **Public spaces** without authorization  
❌ **Competitor networks** (corporate espionage)  
❌ **Stalking/harassment** of individuals  
❌ **Unauthorized surveillance** of any kind  
❌ **Foreign espionage** (violates national security laws)

### Legal Framework (United States)

- **FCC Part 15:** Limits on RF transmission (passive reception is legal)
- **ECPA (Electronic Communications Privacy Act):** Prohibits unauthorized interception
- **CFAA (Computer Fraud and Abuse Act):** Prohibits unauthorized network access
- **State Wiretap Laws:** Vary by state (California has strictest rules)

### Legal Framework (European Union)

- **GDPR:** Requires legal basis for processing personal data (IMSI is PII)
- **ePrivacy Directive:** Protects confidentiality of electronic communications
- **Telecommunications Regulations:** Vary by member state

### Legal Framework (Other Jurisdictions)

- **UK:** Investigatory Powers Act 2016, Regulation of Investigatory Powers Act 2000
- **Australia:** Telecommunications (Interception and Access) Act 1979
- **Canada:** Criminal Code Section 184 (interception of communications)

**⚠️ ALWAYS consult a lawyer before conducting RF security assessments.**

---

## 🎓 Stealth Best Practices Summary

### DO ✅

1. **Train models offline** using synthetic data
2. **Use passive-only mode** (receive only, never transmit)
3. **Anonymize all identifiers** (hash IMSI/IMEI/MAC)
4. **Verify with spectrum analyzer** (confirm no TX)
5. **Physical concealment** of SDR equipment
6. **Legal authorization** in writing
7. **Local processing** (no network exfiltration)
8. **Secure log storage** (encrypted, access-controlled)
9. **Incident response plan** (if detected)
10. **Regular stealth validation** testing

### DON'T ❌

1. **Never use IMSI catchers** (highly detectable)
2. **Never use active geolocation** (requires transmission)
3. **Never probe devices** actively
4. **Never log raw IMSI/IMEI** (privacy violation)
5. **Never transmit** in operational environment
6. **Never use in public** without authorization
7. **Never exfiltrate data** over network (air-gapped best)
8. **Never operate without** legal authorization
9. **Never ignore** detection indicators
10. **Never assume** you're undetectable (defense in depth)

---

## 📞 Questions? Read This First

**Q: Is passive RF monitoring legal?**  
A: Generally yes (like listening to radio), BUT processing personal data (IMSI) requires legal basis. Authorization is ALWAYS safest.

**Q: Can carriers detect passive monitoring?**  
A: No, if you never transmit. They cannot detect a receiver (it's physically impossible without quantum entanglement).

**Q: Will SnoopSnitch detect me in passive mode?**  
A: No. SnoopSnitch detects fake cell towers (IMSI catchers). Passive monitoring doesn't create a cell tower.

**Q: How accurate is passive geolocation?**  
A: ±100-500m using timing advance. Active is ±10-50m but breaks stealth. Choose: accuracy OR stealth, not both.

**Q: Can I combine passive monitoring with IMSI catcher?**  
A: **NO**. IMSI catcher breaks stealth completely. You must choose one or the other.

**Q: What if my operation is discovered?**  
A: Stop immediately. Secure data. Contact legal counsel. Cooperate with authorities if authorized operation.

---

## 🔐 Final Warning

**Stealth is NOT a license to break the law.**

Even perfectly stealthy operations can be illegal if conducted without authorization. The fact that you can't be detected doesn't mean you have permission to conduct surveillance.

**ALWAYS obtain legal authorization before conducting RF security assessments.**

---

**RF Arsenal OS - Democratizing RF Security Testing**  
*"Know Your Network. Secure Your Airwaves. Respect Privacy."*
