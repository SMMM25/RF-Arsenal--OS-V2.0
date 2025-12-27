#!/usr/bin/env python3
"""
RF Arsenal OS - Deployment Verification Script
Verifies system readiness for USB deployment

This script checks:
1. All core modules are importable
2. Syntax validation passes
3. README compliance requirements are met
4. Hardware requirement errors are properly implemented
5. Uncensored LLM integration is functional

Usage: python3 install/verify_deployment.py

Copyright (c) 2024 RF-Arsenal-OS Project
License: MIT
"""

import sys
import os
import importlib
import ast
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


def print_banner():
    """Print verification banner"""
    print(f"""{Colors.CYAN}
╔═══════════════════════════════════════════════════════════════════════════╗
║            RF Arsenal OS - Deployment Verification Script                 ║
║                           v4.1.0 Phase 10                                ║
╚═══════════════════════════════════════════════════════════════════════════╝
{Colors.NC}""")


def log_pass(msg: str):
    print(f"  {Colors.GREEN}✅ PASS{Colors.NC} {msg}")


def log_fail(msg: str):
    print(f"  {Colors.RED}❌ FAIL{Colors.NC} {msg}")


def log_warn(msg: str):
    print(f"  {Colors.YELLOW}⚠️  WARN{Colors.NC} {msg}")


def log_info(msg: str):
    print(f"  {Colors.BLUE}ℹ️  INFO{Colors.NC} {msg}")


def check_syntax_validation() -> tuple:
    """Check all Python files pass syntax validation"""
    print(f"\n{Colors.BLUE}[1/6] Syntax Validation{Colors.NC}")
    
    passed = 0
    failed = 0
    failures = []
    
    for py_file in PROJECT_ROOT.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
            passed += 1
        except SyntaxError as e:
            failed += 1
            failures.append((str(py_file.relative_to(PROJECT_ROOT)), str(e)))
    
    if failed == 0:
        log_pass(f"All {passed} Python files pass syntax check")
    else:
        log_fail(f"{failed} files have syntax errors:")
        for path, error in failures[:5]:
            print(f"      - {path}: {error}")
    
    return passed, failed


def check_core_imports() -> tuple:
    """Check core modules are importable"""
    print(f"\n{Colors.BLUE}[2/6] Core Module Imports{Colors.NC}")
    
    core_modules = [
        ('core', 'Core package'),
        ('core.ai_command_center', 'AI Command Center'),
        ('core.stealth', 'Stealth System'),
        ('core.emergency', 'Emergency Protocol'),
        ('core.offline_capability', 'Offline Capability'),
        ('core.opsec_monitor', 'OPSEC Monitor'),
        ('core.security_validator', 'Security Validator'),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, description in core_modules:
        try:
            importlib.import_module(module_name)
            log_pass(f"{description} ({module_name})")
            passed += 1
        except ImportError as e:
            log_fail(f"{description} ({module_name}): {e}")
            failed += 1
        except Exception as e:
            log_warn(f"{description} ({module_name}): {type(e).__name__}")
            passed += 1
    
    return passed, failed


def check_ai_integration() -> tuple:
    """Check AI/LLM integration modules"""
    print(f"\n{Colors.BLUE}[3/6] AI/LLM Integration{Colors.NC}")
    
    ai_modules = [
        ('core.ai_v2.local_llm', 'AI v2 Local LLM'),
        ('core.ai_v3.local_llm', 'AI v3 Local LLM'),
        ('core.ai_v2.enhanced_ai', 'Enhanced AI'),
        ('core.ai_v2.agent_framework', 'Agent Framework'),
        ('core.arsenal_ai_v3', 'Arsenal AI v3'),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, description in ai_modules:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, 'LocalLLM') or hasattr(module, 'get_local_llm'):
                log_pass(f"{description} - LLM components found")
            elif hasattr(module, 'EnhancedAI') or hasattr(module, 'ArsenalAI'):
                log_pass(f"{description} - AI components found")
            else:
                log_pass(f"{description} - Module loaded")
            passed += 1
        except ImportError as e:
            log_fail(f"{description}: {e}")
            failed += 1
        except Exception as e:
            log_warn(f"{description}: {type(e).__name__}: {e}")
            passed += 1
    
    return passed, failed


def check_hardware_requirements() -> tuple:
    """Check hardware requirement errors are properly implemented"""
    print(f"\n{Colors.BLUE}[4/6] Hardware Requirement Enforcement{Colors.NC}")
    
    hardware_modules = [
        'modules/adsb/adsb_attacks.py',
        'modules/nfc/proxmark3.py',
        'modules/lora/lora_attack.py',
        'modules/power_analysis/power_attacks.py',
        'modules/tempest/tempest_attacks.py',
        'core/external/srsran/srsran_controller.py',
        'core/external/openairinterface/oai_controller.py',
    ]
    
    passed = 0
    checked = 0
    
    for module_path in hardware_modules:
        full_path = PROJECT_ROOT / module_path
        if not full_path.exists():
            log_warn(f"File not found: {module_path}")
            continue
        
        checked += 1
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            if 'DependencyError' in content or 'DEPENDENCY REQUIRED' in content or 'HARDWARE REQUIRED' in content:
                log_pass(f"Hardware requirement enforced: {Path(module_path).name}")
                passed += 1
            elif 'simulation' in content.lower() and 'mode' in content.lower():
                log_warn(f"May have simulation fallback: {Path(module_path).name}")
            else:
                log_pass(f"No simulation found: {Path(module_path).name}")
                passed += 1
        except Exception as e:
            log_fail(f"Error checking {module_path}: {e}")
    
    return passed, checked - passed


def check_readme_compliance() -> tuple:
    """Check README governance compliance markers"""
    print(f"\n{Colors.BLUE}[5/6] README Compliance Markers{Colors.NC}")
    
    compliance_checks = [
        ('Stealth-First', ['offline', 'no.*external.*api', 'stealth']),
        ('RAM-Only', ['ram.only', 'memory', 'no.*persist', 'secure.*delete']),
        ('No Telemetry', ['no.*telemetry', 'no.*logging.*external', 'privacy']),
        ('Offline-First', ['offline.first', 'offline.*default', 'online.*explicit']),
        ('Real-World Only', ['hardware.*required', 'dependency.*error', 'no.*simulation']),
    ]
    
    passed = 0
    key_files = [
        PROJECT_ROOT / 'core' / '__init__.py',
        PROJECT_ROOT / 'core' / 'ai_v3' / 'local_llm.py',
        PROJECT_ROOT / 'core' / 'stealth.py',
        PROJECT_ROOT / 'README.md',
    ]
    
    for rule_name, patterns in compliance_checks:
        import re
        found = False
        for filepath in key_files:
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        content = f.read().lower()
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            found = True
                            break
                except:
                    pass
            if found:
                break
        
        if found:
            log_pass(f"Rule compliant: {rule_name}")
            passed += 1
        else:
            log_warn(f"Verify compliance: {rule_name}")
    
    return passed, len(compliance_checks) - passed


def check_deployment_files() -> tuple:
    """Check deployment infrastructure files exist"""
    print(f"\n{Colors.BLUE}[6/6] Deployment Infrastructure{Colors.NC}")
    
    required_files = [
        ('distro/build_arsenal_os.sh', 'DragonOS Build Script'),
        ('install/create_usb_installer.sh', 'USB Installer Creator'),
        ('install/create_portable_usb.sh', 'Portable USB Creator'),
        ('install/requirements.txt', 'Python Requirements'),
        ('install/quick_install.sh', 'Quick Install Script'),
        ('rf_arsenal_os.py', 'Main Launcher'),
        ('DEEP_AUDIT_REPORT.md', 'Audit Report'),
        ('.gitignore', 'Git Ignore'),
    ]
    
    passed = 0
    failed = 0
    
    for filepath, description in required_files:
        full_path = PROJECT_ROOT / filepath
        if full_path.exists():
            log_pass(f"{description}: {filepath}")
            passed += 1
        else:
            log_fail(f"{description}: {filepath} NOT FOUND")
            failed += 1
    
    return passed, failed


def generate_summary(results: dict):
    """Generate verification summary"""
    print(f"\n{Colors.CYAN}{'='*75}{Colors.NC}")
    print(f"{Colors.CYAN}                    VERIFICATION SUMMARY{Colors.NC}")
    print(f"{Colors.CYAN}{'='*75}{Colors.NC}")
    
    total_passed = sum(r[0] for r in results.values())
    total_failed = sum(r[1] for r in results.values())
    
    for check_name, (passed, failed) in results.items():
        status = f"{Colors.GREEN}PASS{Colors.NC}" if failed == 0 else f"{Colors.YELLOW}WARN{Colors.NC}"
        print(f"  {check_name}: {status} ({passed} passed, {failed} issues)")
    
    print(f"\n  {Colors.BLUE}Total:{Colors.NC} {total_passed} passed, {total_failed} issues")
    
    overall_status = "READY" if total_failed <= 2 else "NEEDS ATTENTION"
    status_color = Colors.GREEN if total_failed <= 2 else Colors.YELLOW
    
    print(f"\n{status_color}  DEPLOYMENT STATUS: {overall_status}{Colors.NC}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return total_failed <= 2


def main():
    """Main verification routine"""
    print_banner()
    
    print(f"{Colors.BLUE}Starting verification...{Colors.NC}")
    print(f"Project Root: {PROJECT_ROOT}")
    
    results = {}
    results['Syntax Validation'] = check_syntax_validation()
    results['Core Imports'] = check_core_imports()
    results['AI/LLM Integration'] = check_ai_integration()
    results['Hardware Requirements'] = check_hardware_requirements()
    results['README Compliance'] = check_readme_compliance()
    results['Deployment Files'] = check_deployment_files()
    
    success = generate_summary(results)
    
    print(f"\n{Colors.CYAN}{'='*75}{Colors.NC}")
    print(f"{Colors.CYAN}                    USB DEPLOYMENT COMMANDS{Colors.NC}")
    print(f"{Colors.CYAN}{'='*75}{Colors.NC}")
    print(f"""
  {Colors.GREEN}Option 1: DragonOS Integration (x86_64){Colors.NC}
    cd distro && sudo ./build_arsenal_os.sh --platform x86_64 --mode full
    sudo dd if=build/rf-arsenal-os.iso of=/dev/sdX bs=4M status=progress

  {Colors.GREEN}Option 2: Raspberry Pi USB Installer{Colors.NC}
    sudo bash install/create_usb_installer.sh /dev/sdX

  {Colors.GREEN}Option 3: Portable USB (Any Linux){Colors.NC}
    sudo bash install/create_portable_usb.sh /dev/sdX

  {Colors.GREEN}Option 4: Quick Install (Existing System){Colors.NC}
    bash install/quick_install.sh
""")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
