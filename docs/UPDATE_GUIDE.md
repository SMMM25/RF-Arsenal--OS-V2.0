# RF Arsenal OS - Update Guide

**Version**: 1.0.0  
**Last Updated**: 2024-12-20  
**Update System**: Tor anonymity, GPG verification, automatic rollback

---

## 🔄 UPDATE SYSTEM OVERVIEW

RF Arsenal OS features a **secure, anonymous update system**:

### Key Features
- ✅ **Anonymous Update Checks** via Tor network
- ✅ **Cryptographic Verification** (GPG + SHA-256)
- ✅ **Automatic Backups** before every update
- ✅ **Rollback Support** on failure
- ✅ **User Approval Required** (never forced)
- ✅ **Offline Support** for air-gapped systems

### Update Process Flow
```
Check for Updates (Tor/HTTPS)
         ↓
Display Release Notes
         ↓
User Approval Required
         ↓
Download Update Package
         ↓
Verify SHA-256 Checksum
         ↓
Verify GPG Signature
         ↓
Create System Backup
         ↓
Apply Update
         ↓
Success → Cleanup
Failure → Automatic Rollback
🤖 AUTOMATIC UPDATES
Checking for Updates
Quick Check (No Installation):

Copypython3 update_manager.py --check
Output Example:

🔍 Checking for updates...
  🔒 Using Tor for anonymous connection...
  ℹ️  Current version: 1.0.0
  ℹ️  Latest version:  1.0.1
  ✅ New version available: 1.0.1
Automatic Check on Startup
Enabled by default when launching RF Arsenal OS:

Copypython3 rf_arsenal_os.py
To skip update check:

Copypython3 rf_arsenal_os.py --no-update-check
Installing Updates Interactively
Launch the interactive update wizard:

Copypython3 update_manager.py --install
Interactive Process Steps:
Step 1: Update Check

🔍 Checking for updates...
  ℹ️  Current version: 1.0.0
  ✅ New version available: 1.0.1
Step 2: Release Notes

📋 Release Information:
============================================================
Version: 1.0.1
Published: 2024-12-25

Release Notes:
• Bug fixes for BladeRF initialization
• Performance improvements
• Security update: patched CVE-2024-XXXXX
============================================================
Step 3: User Approval

❓ Download and install this update? (yes/no): yes
Step 4: Download & Verification

📥 Downloading update...
  ✅ Downloaded: 8.45 MB
  🔐 Verifying SHA-256 checksum...
  ✅ Checksum verified
  🔐 Verifying GPG signature...
  ✅ GPG signature OK
Step 5: Backup & Installation

🔧 Applying update...
  💾 Creating backup...
  ✅ Backup: /var/backups/rf-arsenal/rf-arsenal-1.0.0.tar.gz
  📋 Installing new files...
  ✅ Update complete! Now running v1.0.1

🎉 Update completed successfully!
   Restart RF Arsenal OS to use new version
Restarting After Update
Copy# Restart system (recommended)
sudo reboot

# Or just restart RF Arsenal OS
python3 rf_arsenal_os.py
🛠️ MANUAL UPDATES
Method 1: Git Pull (Developer Mode)
Copycd RF-Arsenal-OS

# Stash local changes (if any)
git stash

# Pull latest
git pull origin main

# Update dependencies
pip3 install --upgrade -r install/requirements.txt

# Reapply local changes
git stash pop
Method 2: Download Release Package
Download from GitHub Releases
Transfer to Raspberry Pi
Verify Checksum:

Copysha256sum -c rf-arsenal-os-v1.0.1.tar.gz.sha256
Backup Current System:

Copytar -czf ~/rf-arsenal-backup-$(date +%Y%m%d).tar.gz /path/to/RF-Arsenal-OS
Extract Update:

Copytar -xzf rf-arsenal-os-v1.0.1.tar.gz -C /tmp/
Apply Update:

Copycd /path/to/RF-Arsenal-OS
sudo pkill -f rf_arsenal_os.py
cp -r /tmp/rf-arsenal-os-1.0.1/* .
pip3 install --upgrade -r install/requirements.txt
Verify:

Copypython3 rf_arsenal_os.py --check
Method 3: Fresh Clone (Clean Install)
Copy# Backup configs
cp -r ~/RF-Arsenal-OS/configs ~/configs-backup

# Remove old
rm -rf ~/RF-Arsenal-OS

# Clone latest
git clone https://github.com/SMMM25/RF-Arsenal-OS.git
cd RF-Arsenal-OS

# Install
pip3 install -r install/requirements.txt

# Restore configs
cp -r ~/configs-backup/* configs/
⚙️ UPDATE CONFIGURATION
Configuration File: /etc/rf-arsenal/update.conf

Default Configuration:

Copy# RF Arsenal OS - Update Configuration

# Check for updates on startup
AUTO_CHECK_UPDATES=true

# Check interval (seconds, 86400 = 24 hours)
UPDATE_CHECK_INTERVAL=86400

# Auto-install without approval (NOT RECOMMENDED)
AUTO_INSTALL_UPDATES=false

# Use Tor for anonymous checks
ANONYMOUS_UPDATES=true

# Create backup before updating
BACKUP_BEFORE_UPDATE=true

# Max backups to keep
MAX_BACKUPS=3
Modifying Configuration:

Copy# Edit config
sudo nano /etc/rf-arsenal/update.conf

# Example: Disable auto-check
AUTO_CHECK_UPDATES=false

# Example: Check weekly
UPDATE_CHECK_INTERVAL=604800

# Save and exit
Configuration Options
Option	Values	Description
AUTO_CHECK_UPDATES	true/false	Check automatically
UPDATE_CHECK_INTERVAL	Seconds	Check frequency
AUTO_INSTALL_UPDATES	true/false	Install without asking (⚠️ dangerous)
ANONYMOUS_UPDATES	true/false	Use Tor network
BACKUP_BEFORE_UPDATE	true/false	Create backup
MAX_BACKUPS	Number	Backups to keep
🔒 SECURITY FEATURES
1. HTTPS Transport Security
All updates use HTTPS (TLS 1.2+) to prevent MITM attacks.

2. Tor Anonymity (Optional)
When enabled, routes through Tor network:

Copy# Check Tor status
systemctl status tor

# Enable Tor for updates
# Edit /etc/rf-arsenal/update.conf
ANONYMOUS_UPDATES=true

# Disable Tor (direct HTTPS)
python3 update_manager.py --no-tor
Benefits:

Hides IP address
Prevents tracking
Adds anonymity layer
Note: Tor is slower (~30-60s for checks)

3. SHA-256 Checksum Verification
Every release includes .sha256 file:

Copy# Automatic during update
# Or manual:
sha256sum rf-arsenal-os-v1.0.1.tar.gz
If checksums don't match, update is rejected.

4. GPG Signature Verification
Releases are GPG signed:

Copy# Automatic during update
# Or manual:
gpg --verify rf-arsenal-os-v1.0.1.tar.gz.sig
Note: Full GPG verification coming in v1.1.0

5. Automatic Backup
Before any update:

Backed up to /var/backups/rf-arsenal/
Timestamped backups
Old backups cleaned (keeps last 3)
6. Rollback on Failure
If update fails:

Automatic restore from backup
Original version preserved
No data loss
🔄 ROLLBACK & RECOVERY
Automatic Rollback
If update fails:

❌ Update failed

🔄 Attempting rollback...
  ✅ Rollback successful
Manual Rollback
List available backups:

Copyls -lh /var/backups/rf-arsenal/
Restore from backup:

Copycd /path/to/RF-Arsenal-OS
sudo tar -xzf /var/backups/rf-arsenal/rf-arsenal-1.0.0.tar.gz -C .
Verify:

Copypython3 rf_arsenal_os.py --check
Emergency Recovery
If system is broken:

Copy# Boot from different SD card
# Mount RF Arsenal drive
sudo mount /dev/sdX1 /mnt

# Copy backup
cp /mnt/var/backups/rf-arsenal/rf-arsenal-1.0.0.tar.gz ~/

# Reinstall
🔧 TROUBLESHOOTING
Issue: Update Check Fails
Symptoms:

❌ Network error: Connection refused
Solutions:

Copy# Check internet
ping -c 3 github.com

# Try without Tor
python3 update_manager.py --no-tor --check

# Restart Tor (if using)
sudo systemctl restart tor
Issue: Checksum Verification Fails
Symptoms:

❌ Checksum mismatch!
Solutions:

Copy# Re-download
python3 update_manager.py --install

# Or manual verification
sha256sum rf-arsenal-os-v1.0.1.tar.gz
⚠️ Security Warning: If checksums fail repeatedly, release may be compromised. DO NOT INSTALL.

Issue: Update Fails to Apply
Symptoms:

❌ Update failed
🔄 Rollback successful
Solutions:

Copy# Check disk space
df -h /
# Need at least 2GB free

# Check permissions
ls -la /path/to/RF-Arsenal-OS

# Try manual update
Issue: Backup Creation Fails
Symptoms:

⚠️  Failed to create backup
Solutions:

Copy# Create backup directory
sudo mkdir -p /var/backups/rf-arsenal
sudo chown $USER:$USER /var/backups/rf-arsenal

# Free up space
sudo apt clean

# Check disk space
df -h /var/backups
📊 UPDATE HISTORY
View update history:

Copy# Check update log
cat /var/log/rf-arsenal/update.log

# List backups (with dates)
ls -lh /var/backups/rf-arsenal/
🎓 BEST PRACTICES
Recommendations:
Always backup before manual updates

Copytar -czf ~/rf-arsenal-backup.tar.gz /path/to/RF-Arsenal-OS
Enable Tor for anonymity (if privacy matters)

CopyANONYMOUS_UPDATES=true
Keep auto-check enabled, but manual installation

CopyAUTO_CHECK_UPDATES=true
AUTO_INSTALL_UPDATES=false  # Require approval
Test updates in non-production first

Read release notes carefully

Keep 2-3 backups

CopyMAX_BACKUPS=3
When to Update:
✅ Update immediately for:

Security vulnerabilities (CVE patches)
Critical bug fixes
Hardware compatibility issues
⏸️ Delay updates when:

In middle of important operations
Testing/research in progress
Need to verify stability
❌ Don't update if:

System is stable and working
Breaking changes you're not ready for
No internet (wait for offline method)
📞 GETTING HELP
Update Issues:

Check log: cat /var/log/rf-arsenal/update.log
Search issues: https://github.com/SMMM25/RF-Arsenal-OS/issues
Open new issue with details
🔮 FUTURE ENHANCEMENTS (v1.1+)
Planned improvements:

✅ Full GPG signature verification (mandatory)
✅ Delta updates (only changed files)
✅ Update scheduling (time windows)
✅ Auto-rollback on boot failure
✅ GUI update notifications
✅ Release channels (stable/beta)
Version: 1.0.0
Last Updated: 2024-12-20
Status: Production Ready
License: MIT

🔄 Stay updated, stay secure! 🔄
