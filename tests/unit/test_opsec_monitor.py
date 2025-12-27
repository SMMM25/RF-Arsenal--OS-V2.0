#!/usr/bin/env python3
"""
Unit Tests for OPSEC Monitor
Tests OPSEC scoring, issue detection, and remediation
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.opsec_monitor import (
    OPSECMonitor,
    OPSECCategory,
    ThreatLevel,
    OPSECIssue,
    OPSECScore,
    get_opsec_monitor
)


class TestOPSECMonitor(unittest.TestCase):
    """Test OPSEC Monitor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = OPSECMonitor()
    
    def test_initial_score_calculation(self):
        """Test that score can be calculated"""
        score = self.monitor.get_score()
        self.assertIsInstance(score, OPSECScore)
        self.assertGreaterEqual(score.total_score, 0)
        self.assertLessEqual(score.total_score, 100)
    
    def test_get_score_summary(self):
        """Test getting score summary"""
        summary = self.monitor.get_score_summary()
        
        self.assertIn('score', summary)
        self.assertIn('threat_level', summary)
        self.assertIn('issues_count', summary)
        self.assertIn('categories', summary)
    
    def test_run_all_checks(self):
        """Test running all OPSEC checks"""
        issues = self.monitor.run_all_checks()
        self.assertIsInstance(issues, list)
        # All items should be OPSECIssue instances
        for issue in issues:
            self.assertIsInstance(issue, OPSECIssue)
    
    def test_get_active_issues(self):
        """Test getting active issues"""
        issues = self.monitor.get_active_issues()
        self.assertIsInstance(issues, list)
        # All returned issues should be active
        for issue in issues:
            self.assertTrue(issue.is_active)
    
    def test_score_has_threat_level(self):
        """Test that score includes threat level"""
        score = self.monitor.get_score()
        self.assertIn(score.threat_level, list(ThreatLevel))
    
    def test_score_has_categories(self):
        """Test that score includes category breakdown"""
        score = self.monitor.get_score()
        self.assertIsInstance(score.category_scores, dict)
        self.assertGreater(len(score.category_scores), 0)
    
    def test_set_network_mode(self):
        """Test setting network mode"""
        self.monitor.set_network_mode("offline")
        self.assertEqual(self.monitor._network_mode, "offline")
        
        self.monitor.set_network_mode("online")
        self.assertEqual(self.monitor._network_mode, "online")
    
    def test_set_stealth_mode(self):
        """Test setting stealth mode"""
        self.monitor.set_stealth_mode(True)
        self.assertTrue(self.monitor._stealth_mode)
        
        self.monitor.set_stealth_mode(False)
        self.assertFalse(self.monitor._stealth_mode)
    
    def test_set_ram_only_mode(self):
        """Test setting RAM-only mode"""
        self.monitor.set_ram_only_mode(True)
        self.assertTrue(self.monitor._ram_only_mode)
        
        self.monitor.set_ram_only_mode(False)
        self.assertFalse(self.monitor._ram_only_mode)
    
    def test_register_callback(self):
        """Test registering callback"""
        callback = Mock()
        self.monitor.register_callback(callback)
        self.assertIn(callback, self.monitor._callbacks)
    
    def test_detailed_report(self):
        """Test generating detailed report"""
        report = self.monitor.get_detailed_report()
        self.assertIsInstance(report, str)
        self.assertIn("OPSEC", report)  # Should contain title
    
    def test_category_weights_sum_to_100(self):
        """Test that category weights sum to 100"""
        total_weight = sum(OPSECMonitor.CATEGORY_WEIGHTS.values())
        self.assertEqual(total_weight, 100)


class TestOPSECCategories(unittest.TestCase):
    """Test OPSEC category enums"""
    
    def test_all_categories_exist(self):
        """Test that expected categories exist"""
        expected = [
            'NETWORK', 'IDENTITY', 'FORENSICS',
            'HARDWARE', 'BEHAVIOR', 'LOCATION'
        ]
        for cat in expected:
            self.assertTrue(hasattr(OPSECCategory, cat), f"Missing category: {cat}")
    
    def test_category_values(self):
        """Test that categories have lowercase string values"""
        for cat in OPSECCategory:
            self.assertEqual(cat.value, cat.name.lower())


class TestThreatLevels(unittest.TestCase):
    """Test threat level enums"""
    
    def test_all_threat_levels_exist(self):
        """Test that expected threat levels exist"""
        expected = ['SECURE', 'GOOD', 'WARNING', 'DANGER', 'CRITICAL']
        for level in expected:
            self.assertTrue(hasattr(ThreatLevel, level), f"Missing level: {level}")
    
    def test_threat_level_values(self):
        """Test that threat levels have string values"""
        for level in ThreatLevel:
            self.assertIsInstance(level.value, str)


class TestOPSECIssue(unittest.TestCase):
    """Test OPSECIssue dataclass"""
    
    def test_create_issue(self):
        """Test creating an OPSEC issue"""
        issue = OPSECIssue(
            id="test_001",
            category=OPSECCategory.NETWORK,
            severity=5,
            title="Test Issue",
            description="Test violation",
            recommendation="Fix the issue"
        )
        
        self.assertEqual(issue.id, "test_001")
        self.assertEqual(issue.category, OPSECCategory.NETWORK)
        self.assertEqual(issue.description, "Test violation")
        self.assertEqual(issue.severity, 5)
        self.assertEqual(issue.title, "Test Issue")
        self.assertTrue(issue.is_active)
        self.assertFalse(issue.auto_fixable)
    
    def test_issue_defaults(self):
        """Test issue default values"""
        issue = OPSECIssue(
            id="test_002",
            category=OPSECCategory.IDENTITY,
            severity=3,
            title="Default Test",
            description="Testing defaults",
            recommendation="None needed"
        )
        
        self.assertTrue(issue.is_active)
        self.assertFalse(issue.auto_fixable)
        self.assertIsNone(issue.fix_command)
        self.assertIsInstance(issue.detected_at, datetime)
    
    def test_issue_with_auto_fix(self):
        """Test issue with auto-fix capability"""
        issue = OPSECIssue(
            id="test_003",
            category=OPSECCategory.NETWORK,
            severity=7,
            title="Fixable Issue",
            description="Can be fixed automatically",
            recommendation="Run the fix command",
            auto_fixable=True,
            fix_command="ip link set eth0 down"
        )
        
        self.assertTrue(issue.auto_fixable)
        self.assertIsNotNone(issue.fix_command)


class TestOPSECScore(unittest.TestCase):
    """Test OPSECScore dataclass"""
    
    def test_create_score(self):
        """Test creating an OPSEC score"""
        score = OPSECScore(
            total_score=85,
            threat_level=ThreatLevel.GOOD,
            category_scores={"network": 80, "identity": 90},
            active_issues=[],
            recommendations=["Keep monitoring"]
        )
        
        self.assertEqual(score.total_score, 85)
        self.assertEqual(score.threat_level, ThreatLevel.GOOD)
        self.assertEqual(score.category_scores["network"], 80)


class TestGlobalInstance(unittest.TestCase):
    """Test global singleton pattern"""
    
    def test_singleton(self):
        """Test that get_opsec_monitor returns same instance"""
        monitor1 = get_opsec_monitor()
        monitor2 = get_opsec_monitor()
        self.assertIs(monitor1, monitor2)
    
    def test_singleton_is_opsec_monitor(self):
        """Test that singleton is an OPSECMonitor instance"""
        monitor = get_opsec_monitor()
        self.assertIsInstance(monitor, OPSECMonitor)


class TestMonitoring(unittest.TestCase):
    """Test monitoring functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = OPSECMonitor()
    
    def tearDown(self):
        """Clean up after tests"""
        self.monitor.stop_monitoring()
    
    def test_start_monitoring(self):
        """Test starting monitoring"""
        self.monitor.start_monitoring(interval_seconds=60)
        self.assertTrue(self.monitor._monitoring)
    
    def test_stop_monitoring(self):
        """Test stopping monitoring"""
        self.monitor.start_monitoring()
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor._monitoring)
    
    def test_monitoring_not_started_by_default(self):
        """Test that monitoring is not started by default"""
        self.assertFalse(self.monitor._monitoring)


if __name__ == '__main__':
    unittest.main(verbosity=2)
