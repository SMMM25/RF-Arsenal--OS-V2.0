#!/usr/bin/env python3
"""
Unit tests for Cryptocurrency Security Assessment & Recovery Toolkit
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestWalletSecurityScanner:
    """Test wallet security scanner module."""
    
    def test_import(self):
        """Test module imports correctly."""
        from modules.superhero.wallet_security_scanner import (
            WalletSecurityScanner,
            VulnerabilitySeverity,
            WalletType,
            get_scanner
        )
        assert WalletSecurityScanner is not None
        assert VulnerabilitySeverity is not None
    
    def test_create_scanner(self):
        """Test scanner instantiation."""
        from modules.superhero.wallet_security_scanner import get_scanner
        scanner = get_scanner()
        assert scanner is not None
        assert scanner.ram_only == True
    
    def test_scanner_stealth_mode(self):
        """Test scanner respects RAM-only mode."""
        from modules.superhero.wallet_security_scanner import WalletSecurityScanner
        scanner = WalletSecurityScanner(ram_only=True)
        assert scanner.ram_only == True


class TestKeyDerivationAnalyzer:
    """Test key derivation analyzer module."""
    
    def test_import(self):
        """Test module imports correctly."""
        from modules.superhero.key_derivation_analyzer import (
            KeyDerivationAnalyzer,
            DerivationStandard,
            get_analyzer
        )
        assert KeyDerivationAnalyzer is not None
        assert DerivationStandard is not None
    
    def test_create_analyzer(self):
        """Test analyzer instantiation."""
        from modules.superhero.key_derivation_analyzer import get_analyzer
        analyzer = get_analyzer()
        assert analyzer is not None
    
    def test_analyzer_methods(self):
        """Test analyzer has expected methods."""
        from modules.superhero.key_derivation_analyzer import KeyDerivationAnalyzer
        analyzer = KeyDerivationAnalyzer()
        
        # Verify key methods exist
        assert hasattr(analyzer, 'analyze_mnemonic')
        assert hasattr(analyzer, 'analyze_derivation_path')
        assert hasattr(analyzer, 'analyze_key_stretching')
        assert callable(analyzer.analyze_mnemonic)


class TestSmartContractAuditor:
    """Test smart contract auditor module."""
    
    def test_import(self):
        """Test module imports correctly."""
        from modules.superhero.smart_contract_auditor import (
            SmartContractAuditor,
            VulnerabilitySeverity,
            ContractVulnerability,
            get_auditor
        )
        assert SmartContractAuditor is not None
        assert VulnerabilitySeverity is not None
    
    def test_create_auditor(self):
        """Test auditor instantiation."""
        from modules.superhero.smart_contract_auditor import get_auditor
        auditor = get_auditor()
        assert auditor is not None
    
    def test_auditor_methods(self):
        """Test auditor has expected methods."""
        from modules.superhero.smart_contract_auditor import SmartContractAuditor
        auditor = SmartContractAuditor()
        
        # Verify key methods exist  
        assert hasattr(auditor, 'audit_contract')
        assert hasattr(auditor, 'get_audit_report')
        assert callable(auditor.audit_contract)


class TestRecoveryToolkit:
    """Test recovery toolkit module."""
    
    def test_import(self):
        """Test module imports correctly."""
        from modules.superhero.recovery_toolkit import (
            RecoveryToolkit,
            RecoveryMethod,
            RecoveryStatus,
            WalletType,
            AuthorizationLevel,
            create_recovery_toolkit
        )
        assert RecoveryToolkit is not None
        assert RecoveryMethod is not None
        assert AuthorizationLevel is not None
    
    def test_create_toolkit(self):
        """Test toolkit instantiation."""
        from modules.superhero.recovery_toolkit import create_recovery_toolkit
        toolkit = create_recovery_toolkit(stealth_mode=True)
        assert toolkit is not None
        assert toolkit.stealth_mode == True
    
    def test_supported_wallets(self):
        """Test supported wallet types."""
        from modules.superhero.recovery_toolkit import create_recovery_toolkit
        toolkit = create_recovery_toolkit()
        wallets = toolkit.get_supported_wallets()
        assert 'software_wallets' in wallets
        assert 'hardware_wallets' in wallets
        assert 'multisig_wallets' in wallets
    
    def test_authorization_required(self):
        """Test that recovery requires authorization."""
        from modules.superhero.recovery_toolkit import (
            create_recovery_toolkit,
            RecoveryMethod,
            WalletType
        )
        toolkit = create_recovery_toolkit()
        
        # Attempt recovery without authorization should fail
        result = toolkit.initiate_recovery(
            wallet_address="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            wallet_type=WalletType.METAMASK,
            recovery_method=RecoveryMethod.SEED_PHRASE_RECONSTRUCTION,
            authorization_id="invalid_auth"
        )
        assert result['success'] == False
    
    def test_seed_reconstructor(self):
        """Test seed phrase reconstructor."""
        from modules.superhero.recovery_toolkit import SeedPhraseReconstructor
        reconstructor = SeedPhraseReconstructor()
        
        # Test word validation
        is_valid, _ = reconstructor.verify_seed_phrase(['abandon'] * 12)
        # Words are valid but checksum won't match
        assert is_valid == True  # Words are valid


class TestMaliciousAddressDatabase:
    """Test malicious address database module."""
    
    def test_import(self):
        """Test module imports correctly."""
        from modules.superhero.malicious_address_db import (
            MaliciousAddressDatabase,
            ThreatLevel,
            AddressCategory,
            Chain,
            create_database
        )
        assert MaliciousAddressDatabase is not None
        assert ThreatLevel is not None
    
    def test_create_database(self):
        """Test database instantiation."""
        from modules.superhero.malicious_address_db import create_database
        db = create_database(ram_only=True)
        assert db is not None
        assert db.ram_only == True
    
    def test_add_and_lookup_address(self):
        """Test adding and looking up addresses."""
        from modules.superhero.malicious_address_db import (
            create_database,
            Chain,
            AddressCategory,
            DataSource
        )
        
        db = create_database(ram_only=True)
        
        # Add a malicious address
        address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
        result = db.add_address(
            address=address,
            chain=Chain.ETHEREUM,
            categories=[AddressCategory.SCAM],
            source=DataSource.MANUAL_ENTRY,
            description="Test scam address",
            stolen_amount=10000.0,
            victim_count=5
        )
        
        assert result is not None
        assert result.address == address.lower()
        
        # Lookup the address
        check_result = db.check_address(address)
        assert check_result['known_malicious'] == True
        assert 'scam' in check_result['categories']
    
    def test_threat_level_calculation(self):
        """Test threat level auto-calculation."""
        from modules.superhero.malicious_address_db import (
            MaliciousAddress,
            Chain,
            AddressCategory,
            ThreatLevel
        )
        from datetime import datetime
        
        address = MaliciousAddress(
            address="0x123",
            chain=Chain.ETHEREUM,
            threat_level=ThreatLevel.UNKNOWN,
            categories=[AddressCategory.RANSOMWARE],
            first_seen=datetime.now(),
            last_updated=datetime.now(),
            total_stolen_usd=15_000_000,  # $15M
            known_victims=2000,
            active=True
        )
        
        calculated = address.calculate_threat_level()
        assert calculated == ThreatLevel.CRITICAL
    
    def test_statistics(self):
        """Test database statistics."""
        from modules.superhero.malicious_address_db import create_database
        db = create_database(ram_only=True)
        stats = db.get_statistics()
        assert 'total_addresses' in stats
        assert 'by_chain' in stats


class TestAuthorityReportGenerator:
    """Test authority report generator module."""
    
    def test_import(self):
        """Test module imports correctly."""
        from modules.superhero.authority_report_generator import (
            AuthorityReportGenerator,
            ReportType,
            ClassificationLevel,
            create_report_generator
        )
        assert AuthorityReportGenerator is not None
        assert ReportType is not None
    
    def test_create_generator(self):
        """Test generator instantiation."""
        from modules.superhero.authority_report_generator import create_report_generator
        generator = create_report_generator(stealth_mode=True)
        assert generator is not None
        assert generator.stealth_mode == True
    
    def test_create_case(self):
        """Test case creation."""
        from modules.superhero.authority_report_generator import create_report_generator
        from datetime import datetime
        
        generator = create_report_generator()
        case = generator.create_case(
            case_name="Test Investigation",
            case_type="Cryptocurrency Theft",
            incident_date=datetime.now(),
            total_loss_usd=100000.0,
            jurisdiction="International"
        )
        
        assert case is not None
        assert case.case_name == "Test Investigation"
        assert case.total_loss_usd == 100000.0
    
    def test_add_evidence(self):
        """Test evidence addition."""
        from modules.superhero.authority_report_generator import create_report_generator
        
        generator = create_report_generator()
        
        # Add transaction evidence
        evidence = generator.add_transaction_evidence(
            tx_hash="0x123abc",
            from_address="0xsender",
            to_address="0xreceiver",
            value=10.0,
            value_usd=25000.0,
            chain="ethereum",
            block_number=12345678,
            title="Suspicious Transfer",
            description="Large transfer to known mixer",
            source="Blockchain Data"
        )
        
        assert evidence is not None
        assert evidence.tx_hash == "0x123abc"
    
    def test_report_generation(self):
        """Test report generation."""
        from modules.superhero.authority_report_generator import (
            create_report_generator,
            ReportType
        )
        from datetime import datetime
        
        generator = create_report_generator()
        
        # Create case
        case = generator.create_case(
            case_name="Test Report",
            case_type="Test",
            incident_date=datetime.now(),
            total_loss_usd=50000.0,
            jurisdiction="Test Jurisdiction"
        )
        
        # Generate report
        report, output = generator.generate_report(
            case_info=case,
            report_type=ReportType.LAW_ENFORCEMENT,
            output_format="markdown"
        )
        
        assert report is not None
        assert report.report_id.startswith("RPT-")
        assert "Test Report" in output
    
    def test_available_formats(self):
        """Test available output formats."""
        from modules.superhero.authority_report_generator import create_report_generator
        generator = create_report_generator()
        formats = generator.get_available_formats()
        assert 'markdown' in formats
        assert 'json' in formats
        assert 'html' in formats


class TestStealthCompliance:
    """Test stealth compliance across all modules."""
    
    def test_wallet_scanner_ram_only(self):
        """Test wallet scanner RAM-only mode."""
        from modules.superhero.wallet_security_scanner import WalletSecurityScanner
        scanner = WalletSecurityScanner(ram_only=True)
        assert scanner.ram_only == True
    
    def test_recovery_toolkit_ram_only(self):
        """Test recovery toolkit stealth mode."""
        from modules.superhero.recovery_toolkit import create_recovery_toolkit
        toolkit = create_recovery_toolkit(stealth_mode=True)
        assert toolkit.stealth_mode == True
    
    def test_malicious_db_ram_only(self):
        """Test malicious DB RAM-only mode."""
        from modules.superhero.malicious_address_db import create_database
        db = create_database(ram_only=True)
        assert db.ram_only == True
    
    def test_report_generator_stealth(self):
        """Test report generator stealth mode."""
        from modules.superhero.authority_report_generator import create_report_generator
        generator = create_report_generator(stealth_mode=True)
        assert generator.stealth_mode == True


class TestIntegration:
    """Integration tests for toolkit components."""
    
    def test_full_workflow(self):
        """Test complete investigation workflow."""
        from modules.superhero.malicious_address_db import (
            create_database,
            Chain,
            AddressCategory,
            DataSource
        )
        from modules.superhero.authority_report_generator import (
            create_report_generator,
            ReportType
        )
        from datetime import datetime
        
        # 1. Create malicious address database
        db = create_database(ram_only=True)
        
        # 2. Add known malicious address
        suspect_address = "0xbad1234567890abcdef1234567890abcdef1234"
        db.add_address(
            address=suspect_address,
            chain=Chain.ETHEREUM,
            categories=[AddressCategory.SCAM, AddressCategory.RUG_PULL],
            source=DataSource.COMMUNITY_REPORTS,
            description="Known rug pull perpetrator",
            stolen_amount=500000.0,
            victim_count=150
        )
        
        # 3. Check address
        check = db.check_address(suspect_address)
        assert check['known_malicious'] == True
        
        # 4. Create report
        generator = create_report_generator(stealth_mode=True)
        case = generator.create_case(
            case_name="Rug Pull Investigation",
            case_type="DeFi Fraud",
            incident_date=datetime.now(),
            total_loss_usd=500000.0,
            jurisdiction="International"
        )
        
        # 5. Add evidence
        generator.add_address_evidence(
            address=suspect_address,
            chain="ethereum",
            title="Suspect Wallet",
            description="Main perpetrator wallet",
            source="Malicious Address Database",
            risk_score=95.0
        )
        
        # 6. Generate report
        report, output = generator.generate_report(
            case_info=case,
            report_type=ReportType.LAW_ENFORCEMENT,
            output_format="json"
        )
        
        assert report is not None
        assert "Rug Pull Investigation" in case.case_name


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
