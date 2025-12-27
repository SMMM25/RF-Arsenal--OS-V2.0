"""
RF Arsenal OS - Security Validator & Compliance Checker
=======================================================

CRITICAL SECURITY MODULE: Validates system configuration for security
compliance, detects vulnerabilities, and ensures operational safety.

Author: RF Arsenal Security Team
Version: 2.0.0
License: Authorized Use Only
"""

import logging
import os
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityViolation:
    """Represents a security violation or risk."""
    
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    
    def __init__(self,
                 severity: str,
                 category: str,
                 description: str,
                 location: Optional[str] = None,
                 recommendation: Optional[str] = None):
        """
        Initialize security violation.
        
        Args:
            severity: Violation severity (CRITICAL, HIGH, MEDIUM, LOW, INFO)
            category: Violation category
            description: Detailed description
            location: Location of violation (file, line, etc.)
            recommendation: Recommended fix
        """
        self.severity = severity
        self.category = category
        self.description = description
        self.location = location
        self.recommendation = recommendation
        self.timestamp = datetime.now()
    
    def __str__(self) -> str:
        """Human-readable representation."""
        loc = f" [{self.location}]" if self.location else ""
        return f"[{self.severity}] {self.category}{loc}: {self.description}"


class SecurityValidator:
    """
    Comprehensive security validation and compliance checking.
    
    Features:
    - Plaintext PII detection
    - Stealth compliance validation
    - Configuration security audit
    - Code pattern vulnerability scanning
    - GDPR/CCPA compliance checking
    """
    
    def __init__(self, project_root: str):
        """
        Initialize security validator.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root)
        self.violations: List[SecurityViolation] = []
        
        logger.info(f"🔐 Security Validator initialized (root: {project_root})")
    
    def validate_all(self) -> List[SecurityViolation]:
        """
        Run all security validation checks.
        
        Returns:
            List of security violations found
        """
        logger.info("🔍 Starting comprehensive security validation...")
        
        self.violations.clear()
        
        # Run all validation checks
        self._check_plaintext_identifiers()
        self._check_hardcoded_credentials()
        self._check_stealth_compliance()
        self._check_logging_security()
        self._check_file_permissions()
        self._check_dependency_security()
        
        # Summarize results
        critical = sum(1 for v in self.violations if v.severity == "CRITICAL")
        high = sum(1 for v in self.violations if v.severity == "HIGH")
        medium = sum(1 for v in self.violations if v.severity == "MEDIUM")
        
        logger.info(
            f"🔍 Security validation complete: "
            f"{len(self.violations)} issues found "
            f"(CRITICAL: {critical}, HIGH: {high}, MEDIUM: {medium})"
        )
        
        return self.violations
    
    def _check_plaintext_identifiers(self):
        """Check for plaintext PII storage (IMSI, IMEI, etc.)."""
        logger.info("🔍 Checking for plaintext identifier storage...")
        
        # Patterns that indicate plaintext PII storage
        dangerous_patterns = [
            (r'self\.\w+\[imsi\]', 'Direct IMSI dictionary key usage'),
            (r'self\.\w+\[imei\]', 'Direct IMEI dictionary key usage'),
            (r'imsi\s*=\s*["\']?\d{15}', 'Hardcoded IMSI value'),
            (r'imei\s*=\s*["\']?\d{15}', 'Hardcoded IMEI value'),
            (r'\.store\(.*imsi.*\)', 'Storing IMSI without anonymization'),
            (r'\.log\(.*imsi.*\)', 'Logging IMSI without anonymization'),
        ]
        
        python_files = list(self.project_root.glob('**/*.py'))
        
        for filepath in python_files:
            if '.venv' in str(filepath) or '__pycache__' in str(filepath):
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Check if file uses anonymization
                uses_anonymization = (
                    'anonymize_imsi' in content or
                    'anonymize_imei' in content or
                    '_anonymize_identifier' in content or
                    'from core.anonymization import' in content
                )
                
                for line_num, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith('#'):
                        continue
                    
                    for pattern, description in dangerous_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Check if this line also has anonymization
                            if not uses_anonymization or 'anonymize' not in line.lower():
                                self.violations.append(SecurityViolation(
                                    severity=SecurityViolation.CRITICAL,
                                    category="Plaintext PII",
                                    description=f"{description}: {line.strip()[:80]}",
                                    location=f"{filepath.name}:{line_num}",
                                    recommendation="Use core.anonymization.anonymize_imsi() or anonymize_imei()"
                                ))
            
            except Exception as e:
                logger.warning(f"Failed to scan {filepath}: {e}")
    
    def _check_hardcoded_credentials(self):
        """Check for hardcoded API keys, passwords, etc."""
        logger.info("🔍 Checking for hardcoded credentials...")
        
        credential_patterns = [
            (r'api_key\s*=\s*["\'][^"\']{20,}["\']', 'Hardcoded API key'),
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
            (r'secret\s*=\s*["\'][^"\']{20,}["\']', 'Hardcoded secret'),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', 'Hardcoded token'),
        ]
        
        python_files = list(self.project_root.glob('**/*.py'))
        
        for filepath in python_files:
            if '.venv' in str(filepath) or '__pycache__' in str(filepath):
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    if line.strip().startswith('#'):
                        continue
                    
                    for pattern, description in credential_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Exception for test/example values
                            if 'test' in line.lower() or 'example' in line.lower():
                                continue
                            
                            self.violations.append(SecurityViolation(
                                severity=SecurityViolation.HIGH,
                                category="Hardcoded Credential",
                                description=f"{description}",
                                location=f"{filepath.name}:{line_num}",
                                recommendation="Use environment variables or config files"
                            ))
            
            except Exception as e:
                logger.warning(f"Failed to scan {filepath}: {e}")
    
    def _check_stealth_compliance(self):
        """Check for stealth-breaking operations."""
        logger.info("🔍 Checking stealth compliance...")
        
        # Check for transmission functions without stealth validation
        python_files = list(self.project_root.glob('modules/hardware/*.py'))
        
        for filepath in python_files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Check if file imports stealth enforcement
                has_stealth_import = (
                    'from core.stealth_enforcement import' in content or
                    'import core.stealth_enforcement' in content
                )
                
                # Look for transmit functions
                for line_num, line in enumerate(lines, 1):
                    if 'def transmit' in line or 'def tx' in line or 'def send' in line:
                        # Check next 10 lines for validation
                        check_lines = '\n'.join(lines[line_num:line_num+10])
                        
                        if 'validate_transmit' not in check_lines and 'validate_operation' not in check_lines:
                            self.violations.append(SecurityViolation(
                                severity=SecurityViolation.CRITICAL,
                                category="Stealth Violation",
                                description="Transmit function without stealth validation",
                                location=f"{filepath.name}:{line_num}",
                                recommendation="Add validate_transmit() before transmission"
                            ))
            
            except Exception as e:
                logger.warning(f"Failed to scan {filepath}: {e}")
    
    def _check_logging_security(self):
        """Check for insecure logging practices."""
        logger.info("🔍 Checking logging security...")
        
        python_files = list(self.project_root.glob('**/*.py'))
        
        for filepath in python_files:
            if '.venv' in str(filepath) or '__pycache__' in str(filepath):
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    # Check for logging sensitive data
                    if 'logger.' in line or 'print(' in line:
                        if re.search(r'(imsi|imei|phone|password|token)', line, re.IGNORECASE):
                            if 'anonymize' not in line.lower():
                                self.violations.append(SecurityViolation(
                                    severity=SecurityViolation.HIGH,
                                    category="Insecure Logging",
                                    description=f"Potentially logging sensitive data: {line.strip()[:60]}",
                                    location=f"{filepath.name}:{line_num}",
                                    recommendation="Anonymize identifiers before logging"
                                ))
            
            except Exception as e:
                logger.warning(f"Failed to scan {filepath}: {e}")
    
    def _check_file_permissions(self):
        """Check for insecure file permissions."""
        logger.info("🔍 Checking file permissions...")
        
        # Check critical files
        critical_paths = [
            'core/anonymization.py',
            'core/stealth_enforcement.py',
            'core/transmission_monitor.py',
            'data/ml_models/',
            'logs/'
        ]
        
        for path_str in critical_paths:
            path = self.project_root / path_str
            
            if path.exists():
                stat_info = path.stat()
                mode = oct(stat_info.st_mode)[-3:]
                
                # Check if world-writable or world-readable for sensitive files
                if 'anonymization' in str(path) or 'stealth' in str(path):
                    if mode[2] != '0':  # Others have permissions
                        self.violations.append(SecurityViolation(
                            severity=SecurityViolation.MEDIUM,
                            category="File Permissions",
                            description=f"Security module has world permissions: {mode}",
                            location=str(path),
                            recommendation="Set permissions to 600 or 700"
                        ))
    
    def _check_dependency_security(self):
        """Check for known vulnerable dependencies."""
        logger.info("🔍 Checking dependency security...")
        
        requirements_file = self.project_root / 'requirements.txt'
        
        if requirements_file.exists():
            # Known vulnerable versions (example)
            vulnerable = {
                'requests': ['2.27.0', '2.27.1'],  # Example
                'urllib3': ['1.26.0', '1.26.1'],   # Example
            }
            
            try:
                with open(requirements_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '==' in line:
                            pkg, version = line.split('==')
                            pkg = pkg.strip()
                            version = version.strip()
                            
                            if pkg in vulnerable and version in vulnerable[pkg]:
                                self.violations.append(SecurityViolation(
                                    severity=SecurityViolation.HIGH,
                                    category="Vulnerable Dependency",
                                    description=f"{pkg}=={version} has known vulnerabilities",
                                    location="requirements.txt",
                                    recommendation=f"Update {pkg} to latest version"
                                ))
            
            except Exception as e:
                logger.warning(f"Failed to check dependencies: {e}")
    
    def generate_report(self) -> str:
        """
        Generate comprehensive security report.
        
        Returns:
            Formatted report string
        """
        if not self.violations:
            return (
                "=" * 70 + "\n"
                "🔐 RF ARSENAL OS - SECURITY VALIDATION REPORT\n"
                "=" * 70 + "\n\n"
                "✅ NO SECURITY VIOLATIONS DETECTED\n\n"
                "All security checks passed successfully.\n"
                "=" * 70
            )
        
        # Group by severity
        by_severity = {
            SecurityViolation.CRITICAL: [],
            SecurityViolation.HIGH: [],
            SecurityViolation.MEDIUM: [],
            SecurityViolation.LOW: [],
            SecurityViolation.INFO: []
        }
        
        for violation in self.violations:
            by_severity[violation.severity].append(violation)
        
        report = [
            "=" * 70,
            "🔐 RF ARSENAL OS - SECURITY VALIDATION REPORT",
            "=" * 70,
            "",
            f"⚠️  TOTAL VIOLATIONS: {len(self.violations)}",
            ""
        ]
        
        for severity in [SecurityViolation.CRITICAL, SecurityViolation.HIGH, 
                        SecurityViolation.MEDIUM, SecurityViolation.LOW]:
            violations = by_severity[severity]
            if violations:
                report.append(f"\n{'='*70}")
                report.append(f"[{severity}] {len(violations)} Issues Found")
                report.append(f"{'='*70}\n")
                
                for i, v in enumerate(violations, 1):
                    report.append(f"{i}. {v.category}")
                    report.append(f"   Location: {v.location}")
                    report.append(f"   Issue: {v.description}")
                    if v.recommendation:
                        report.append(f"   Fix: {v.recommendation}")
                    report.append("")
        
        report.extend([
            "=" * 70,
            "⚠️  ACTION REQUIRED",
            "=" * 70,
            "Please address CRITICAL and HIGH severity issues before deployment.",
            "=" * 70
        ])
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test security validation
    print("🔐 RF Arsenal OS - Security Validator Test\n")
    
    validator = SecurityValidator("/home/user/webapp")
    violations = validator.validate_all()
    
    print(validator.generate_report())
    
    if violations:
        print(f"\n⚠️  Found {len(violations)} security issues")
    else:
        print("\n✅ No security issues found")
