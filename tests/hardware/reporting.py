"""
Comprehensive Hardware Test Reporting Framework.

Provides detailed reporting capabilities for hardware integration tests
including HTML, JSON, JUnit, and PDF-ready report generation.

Features:
- Multi-format report generation (HTML, JSON, JUnit XML, CSV)
- Test metrics collection and analysis
- Test coverage tracking
- Historical trend analysis
- Visual report generation
- CI/CD integration support

Author: RF Arsenal Development Team
License: Proprietary
"""

import csv
import hashlib
import json
import logging
import os
import statistics
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import numpy as np

from .framework import (
    TestResult,
    TestStatus,
    TestCategory,
    TestPriority,
    TestEnvironment,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Test Metrics
# ============================================================================

@dataclass
class TestMetrics:
    """
    Comprehensive test metrics collection.
    
    Collects and analyzes metrics from test executions including
    pass rates, timing statistics, and trend analysis.
    """
    
    # Basic counts
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    timeout_tests: int = 0
    
    # Timing metrics
    total_duration_seconds: float = 0.0
    min_duration_seconds: float = 0.0
    max_duration_seconds: float = 0.0
    avg_duration_seconds: float = 0.0
    std_duration_seconds: float = 0.0
    
    # Category breakdowns
    by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_priority: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Failure analysis
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    flaky_tests: List[str] = field(default_factory=list)
    
    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        if self.total_tests == 0:
            return 0.0
        return ((self.failed_tests + self.error_tests) / self.total_tests) * 100
    
    @property
    def skip_rate(self) -> float:
        """Calculate skip rate percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.skipped_tests / self.total_tests) * 100
    
    @property
    def execution_rate(self) -> float:
        """Calculate execution rate (tests per second)."""
        if self.total_duration_seconds == 0:
            return 0.0
        return self.total_tests / self.total_duration_seconds
    
    def calculate_from_results(self, results: List[TestResult]) -> None:
        """
        Calculate metrics from test results.
        
        Args:
            results: List of test results to analyze
        """
        self.total_tests = len(results)
        self.passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
        self.failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)
        self.skipped_tests = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        self.error_tests = sum(1 for r in results if r.status == TestStatus.ERROR)
        self.timeout_tests = sum(1 for r in results if r.status == TestStatus.TIMEOUT)
        
        # Timing analysis
        durations = [r.duration_seconds for r in results if r.duration_seconds > 0]
        if durations:
            self.total_duration_seconds = sum(durations)
            self.min_duration_seconds = min(durations)
            self.max_duration_seconds = max(durations)
            self.avg_duration_seconds = statistics.mean(durations)
            if len(durations) > 1:
                self.std_duration_seconds = statistics.stdev(durations)
        
        # Category breakdown
        self.by_category = defaultdict(lambda: defaultdict(int))
        for r in results:
            cat = r.category.value
            self.by_category[cat][r.status.value] += 1
        
        # Priority breakdown
        self.by_priority = defaultdict(lambda: defaultdict(int))
        for r in results:
            pri = str(r.priority.value)
            self.by_priority[pri][r.status.value] += 1
        
        # Failure analysis
        for r in results:
            if r.status in [TestStatus.FAILED, TestStatus.ERROR]:
                reason = r.error_type or 'Unknown'
                self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'skipped_tests': self.skipped_tests,
            'error_tests': self.error_tests,
            'timeout_tests': self.timeout_tests,
            'pass_rate': self.pass_rate,
            'failure_rate': self.failure_rate,
            'skip_rate': self.skip_rate,
            'total_duration_seconds': self.total_duration_seconds,
            'avg_duration_seconds': self.avg_duration_seconds,
            'min_duration_seconds': self.min_duration_seconds,
            'max_duration_seconds': self.max_duration_seconds,
            'execution_rate': self.execution_rate,
            'by_category': dict(self.by_category),
            'by_priority': dict(self.by_priority),
            'failure_reasons': self.failure_reasons,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class TestCoverage:
    """
    Test coverage tracking for hardware components.
    
    Tracks which hardware capabilities, devices, and test categories
    have been covered by test execution.
    """
    
    # Device coverage
    devices_tested: List[str] = field(default_factory=list)
    devices_available: List[str] = field(default_factory=list)
    
    # Capability coverage
    capabilities_tested: List[str] = field(default_factory=list)
    capabilities_available: List[str] = field(default_factory=list)
    
    # Test category coverage
    categories_executed: List[str] = field(default_factory=list)
    categories_available: List[str] = field(default_factory=list)
    
    # Frequency coverage
    frequency_ranges_tested: List[Tuple[float, float]] = field(default_factory=list)
    
    # Feature coverage
    features_tested: Dict[str, bool] = field(default_factory=dict)
    
    @property
    def device_coverage_percent(self) -> float:
        """Calculate device coverage percentage."""
        if not self.devices_available:
            return 0.0
        return len(self.devices_tested) / len(self.devices_available) * 100
    
    @property
    def capability_coverage_percent(self) -> float:
        """Calculate capability coverage percentage."""
        if not self.capabilities_available:
            return 0.0
        return len(self.capabilities_tested) / len(self.capabilities_available) * 100
    
    @property
    def category_coverage_percent(self) -> float:
        """Calculate category coverage percentage."""
        if not self.categories_available:
            return 0.0
        return len(self.categories_executed) / len(self.categories_available) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert coverage to dictionary."""
        return {
            'device_coverage': {
                'tested': self.devices_tested,
                'available': self.devices_available,
                'percent': self.device_coverage_percent
            },
            'capability_coverage': {
                'tested': self.capabilities_tested,
                'available': self.capabilities_available,
                'percent': self.capability_coverage_percent
            },
            'category_coverage': {
                'executed': self.categories_executed,
                'available': self.categories_available,
                'percent': self.category_coverage_percent
            },
            'frequency_ranges_tested': self.frequency_ranges_tested,
            'features_tested': self.features_tested
        }


# ============================================================================
# Report Generation
# ============================================================================

class TestReporter(ABC):
    """
    Abstract base class for test reporters.
    
    Provides common interface for generating test reports
    in various formats.
    """
    
    def __init__(self, output_directory: str = "."):
        """Initialize reporter."""
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def generate_report(
        self,
        results: List[TestResult],
        metrics: TestMetrics,
        environment: TestEnvironment,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate test report.
        
        Args:
            results: Test results to report
            metrics: Calculated test metrics
            environment: Test environment info
            filename: Output filename (optional)
        
        Returns:
            Path to generated report
        """
        pass
    
    def _generate_summary(
        self,
        results: List[TestResult],
        metrics: TestMetrics
    ) -> Dict[str, Any]:
        """Generate summary data for report."""
        return {
            'total': metrics.total_tests,
            'passed': metrics.passed_tests,
            'failed': metrics.failed_tests,
            'skipped': metrics.skipped_tests,
            'errors': metrics.error_tests,
            'timeouts': metrics.timeout_tests,
            'pass_rate': metrics.pass_rate,
            'duration_seconds': metrics.total_duration_seconds,
            'avg_duration_seconds': metrics.avg_duration_seconds
        }


class HTMLReporter(TestReporter):
    """
    HTML test report generator.
    
    Generates comprehensive HTML reports with:
    - Summary dashboard
    - Test result tables
    - Charts and visualizations
    - Detailed test information
    """
    
    def generate_report(
        self,
        results: List[TestResult],
        metrics: TestMetrics,
        environment: TestEnvironment,
        filename: Optional[str] = None
    ) -> str:
        """Generate HTML report."""
        filename = filename or f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = self.output_directory / filename
        
        html_content = self._generate_html(results, metrics, environment)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self._logger.info(f"Generated HTML report: {output_path}")
        return str(output_path)
    
    def _generate_html(
        self,
        results: List[TestResult],
        metrics: TestMetrics,
        environment: TestEnvironment
    ) -> str:
        """Generate HTML content."""
        summary = self._generate_summary(results, metrics)
        
        # Determine overall status color
        if metrics.pass_rate >= 95:
            status_color = "#28a745"
            status_text = "EXCELLENT"
        elif metrics.pass_rate >= 80:
            status_color = "#17a2b8"
            status_text = "GOOD"
        elif metrics.pass_rate >= 60:
            status_color = "#ffc107"
            status_text = "FAIR"
        else:
            status_color = "#dc3545"
            status_text = "NEEDS ATTENTION"
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hardware Integration Test Report</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--dark-color);
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary-color), #3498db);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header p {{
            opacity: 0.9;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 10px 25px;
            background-color: {status_color};
            color: white;
            border-radius: 25px;
            font-weight: bold;
            margin-top: 15px;
            font-size: 1.1em;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        
        .metric-passed {{ color: var(--success-color); }}
        .metric-failed {{ color: var(--danger-color); }}
        .metric-skipped {{ color: var(--warning-color); }}
        .metric-total {{ color: var(--primary-color); }}
        
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: var(--primary-color);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--light-color);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background-color: #f5f5f5;
        }}
        
        .status-passed {{
            background-color: rgba(40, 167, 69, 0.1);
            color: var(--success-color);
            font-weight: bold;
        }}
        
        .status-failed {{
            background-color: rgba(220, 53, 69, 0.1);
            color: var(--danger-color);
            font-weight: bold;
        }}
        
        .status-skipped {{
            background-color: rgba(255, 193, 7, 0.1);
            color: #856404;
            font-weight: bold;
        }}
        
        .status-error {{
            background-color: rgba(220, 53, 69, 0.1);
            color: var(--danger-color);
            font-weight: bold;
        }}
        
        .progress-bar {{
            height: 30px;
            background-color: var(--light-color);
            border-radius: 15px;
            overflow: hidden;
            display: flex;
        }}
        
        .progress-segment {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .progress-passed {{ background-color: var(--success-color); }}
        .progress-failed {{ background-color: var(--danger-color); }}
        .progress-skipped {{ background-color: var(--warning-color); }}
        
        .env-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .env-item {{
            background-color: var(--light-color);
            padding: 15px;
            border-radius: 5px;
        }}
        
        .env-item strong {{
            color: var(--primary-color);
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .dashboard {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Hardware Integration Test Report</h1>
            <p>RF Arsenal - Comprehensive Hardware Testing</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <span class="status-badge">{status_text} - {metrics.pass_rate:.1f}% Pass Rate</span>
        </header>
        
        <div class="dashboard">
            <div class="metric-card">
                <div class="metric-value metric-total">{metrics.total_tests}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-passed">{metrics.passed_tests}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-failed">{metrics.failed_tests + metrics.error_tests}</div>
                <div class="metric-label">Failed/Errors</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-skipped">{metrics.skipped_tests}</div>
                <div class="metric-label">Skipped</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-total">{metrics.total_duration_seconds:.1f}s</div>
                <div class="metric-label">Duration</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-total">{metrics.avg_duration_seconds:.2f}s</div>
                <div class="metric-label">Avg Test Time</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Test Results Overview</h2>
            <div class="progress-bar">
                <div class="progress-segment progress-passed" style="width: {metrics.pass_rate}%">
                    {metrics.passed_tests} Passed
                </div>
                <div class="progress-segment progress-failed" style="width: {metrics.failure_rate}%">
                    {metrics.failed_tests + metrics.error_tests} Failed
                </div>
                <div class="progress-segment progress-skipped" style="width: {metrics.skip_rate}%">
                    {metrics.skipped_tests} Skipped
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📋 Test Results Detail</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Status</th>
                        <th>Category</th>
                        <th>Priority</th>
                        <th>Duration</th>
                        <th>Message</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add test results
        for result in results:
            status_class = f"status-{result.status.value}"
            message = result.message or result.error_message or '-'
            if len(message) > 100:
                message = message[:100] + '...'
            
            html += f"""                    <tr class="{status_class}">
                        <td>{result.test_name}</td>
                        <td>{result.status.value.upper()}</td>
                        <td>{result.category.value}</td>
                        <td>{result.priority.value}</td>
                        <td>{result.duration_seconds:.2f}s</td>
                        <td>{message}</td>
                    </tr>
"""
        
        html += """                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>🖥️ Test Environment</h2>
            <div class="env-grid">
"""
        
        html += f"""                <div class="env-item"><strong>Hostname:</strong> {environment.hostname}</div>
                <div class="env-item"><strong>Platform:</strong> {environment.platform}</div>
                <div class="env-item"><strong>Architecture:</strong> {environment.architecture}</div>
                <div class="env-item"><strong>Python:</strong> {environment.python_version}</div>
                <div class="env-item"><strong>CPUs:</strong> {environment.cpu_count}</div>
                <div class="env-item"><strong>Run ID:</strong> {environment.run_id[:8]}...</div>
"""
        
        html += """            </div>
        </div>
"""
        
        # Add connected devices section if any
        if environment.connected_devices:
            html += """
        <div class="section">
            <h2>📡 Connected Devices</h2>
            <table>
                <thead>
                    <tr>
                        <th>Device ID</th>
                        <th>Manufacturer</th>
                        <th>Model</th>
                        <th>Serial</th>
                    </tr>
                </thead>
                <tbody>
"""
            for device in environment.connected_devices:
                html += f"""                    <tr>
                        <td>{device.device_id}</td>
                        <td>{device.manufacturer}</td>
                        <td>{device.model}</td>
                        <td>{device.serial_number}</td>
                    </tr>
"""
            html += """                </tbody>
            </table>
        </div>
"""
        
        html += f"""
        <footer>
            <p>RF Arsenal Hardware Integration Test Framework</p>
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>"""
        
        return html


class JSONReporter(TestReporter):
    """
    JSON test report generator.
    
    Generates machine-readable JSON reports suitable for
    CI/CD integration and automated processing.
    """
    
    def generate_report(
        self,
        results: List[TestResult],
        metrics: TestMetrics,
        environment: TestEnvironment,
        filename: Optional[str] = None
    ) -> str:
        """Generate JSON report."""
        filename = filename or f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.output_directory / filename
        
        report_data = {
            'report_info': {
                'title': 'Hardware Integration Test Report',
                'generated_at': datetime.now().isoformat(),
                'format_version': '1.0'
            },
            'summary': self._generate_summary(results, metrics),
            'metrics': metrics.to_dict(),
            'environment': environment.to_dict(),
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self._logger.info(f"Generated JSON report: {output_path}")
        return str(output_path)


class JUnitReporter(TestReporter):
    """
    JUnit XML test report generator.
    
    Generates JUnit-compatible XML reports for CI/CD systems
    like Jenkins, GitLab CI, and GitHub Actions.
    """
    
    def generate_report(
        self,
        results: List[TestResult],
        metrics: TestMetrics,
        environment: TestEnvironment,
        filename: Optional[str] = None
    ) -> str:
        """Generate JUnit XML report."""
        filename = filename or f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
        output_path = self.output_directory / filename
        
        xml_content = self._generate_junit_xml(results, metrics, environment)
        
        with open(output_path, 'w') as f:
            f.write(xml_content)
        
        self._logger.info(f"Generated JUnit report: {output_path}")
        return str(output_path)
    
    def _generate_junit_xml(
        self,
        results: List[TestResult],
        metrics: TestMetrics,
        environment: TestEnvironment
    ) -> str:
        """Generate JUnit XML content."""
        # Create root element
        root = ET.Element('testsuites')
        root.set('name', 'Hardware Integration Tests')
        root.set('tests', str(metrics.total_tests))
        root.set('failures', str(metrics.failed_tests))
        root.set('errors', str(metrics.error_tests))
        root.set('skipped', str(metrics.skipped_tests))
        root.set('time', str(metrics.total_duration_seconds))
        root.set('timestamp', datetime.now().isoformat())
        
        # Group results by category
        by_category = defaultdict(list)
        for result in results:
            by_category[result.category.value].append(result)
        
        # Create test suite for each category
        for category, cat_results in by_category.items():
            suite = ET.SubElement(root, 'testsuite')
            suite.set('name', category)
            suite.set('tests', str(len(cat_results)))
            suite.set('failures', str(sum(1 for r in cat_results if r.status == TestStatus.FAILED)))
            suite.set('errors', str(sum(1 for r in cat_results if r.status == TestStatus.ERROR)))
            suite.set('skipped', str(sum(1 for r in cat_results if r.status == TestStatus.SKIPPED)))
            suite.set('time', str(sum(r.duration_seconds for r in cat_results)))
            
            # Add test cases
            for result in cat_results:
                testcase = ET.SubElement(suite, 'testcase')
                testcase.set('name', result.test_name)
                testcase.set('classname', f"hardware.{category}")
                testcase.set('time', str(result.duration_seconds))
                
                if result.status == TestStatus.FAILED:
                    failure = ET.SubElement(testcase, 'failure')
                    failure.set('message', result.error_message or 'Test failed')
                    failure.set('type', result.error_type or 'AssertionError')
                    if result.stack_trace:
                        failure.text = result.stack_trace
                
                elif result.status == TestStatus.ERROR:
                    error = ET.SubElement(testcase, 'error')
                    error.set('message', result.error_message or 'Test error')
                    error.set('type', result.error_type or 'Exception')
                    if result.stack_trace:
                        error.text = result.stack_trace
                
                elif result.status == TestStatus.SKIPPED:
                    skipped = ET.SubElement(testcase, 'skipped')
                    skipped.set('message', result.message or 'Test skipped')
                
                # Add system output
                if result.logs:
                    system_out = ET.SubElement(testcase, 'system-out')
                    system_out.text = '\n'.join(result.logs)
        
        # Convert to string
        xml_str = ET.tostring(root, encoding='unicode', method='xml')
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_str}'


class CSVReporter(TestReporter):
    """
    CSV test report generator.
    
    Generates CSV reports for spreadsheet analysis.
    """
    
    def generate_report(
        self,
        results: List[TestResult],
        metrics: TestMetrics,
        environment: TestEnvironment,
        filename: Optional[str] = None
    ) -> str:
        """Generate CSV report."""
        filename = filename or f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = self.output_directory / filename
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Test ID', 'Test Name', 'Status', 'Category', 'Priority',
                'Duration (s)', 'Device ID', 'Message', 'Error Type', 'Error Message'
            ])
            
            # Write test results
            for result in results:
                writer.writerow([
                    result.test_id,
                    result.test_name,
                    result.status.value,
                    result.category.value,
                    result.priority.value,
                    f"{result.duration_seconds:.3f}",
                    result.device_id or '-',
                    result.message or '-',
                    result.error_type or '-',
                    result.error_message or '-'
                ])
        
        self._logger.info(f"Generated CSV report: {output_path}")
        return str(output_path)


class ConsolidatedReporter:
    """
    Generates multiple report formats simultaneously.
    
    Convenience class to generate all report formats at once.
    """
    
    def __init__(self, output_directory: str = "."):
        """Initialize consolidated reporter."""
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.html_reporter = HTMLReporter(str(self.output_directory))
        self.json_reporter = JSONReporter(str(self.output_directory))
        self.junit_reporter = JUnitReporter(str(self.output_directory))
        self.csv_reporter = CSVReporter(str(self.output_directory))
    
    def generate_all_reports(
        self,
        results: List[TestResult],
        metrics: TestMetrics,
        environment: TestEnvironment,
        base_filename: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate all report formats.
        
        Returns:
            Dictionary mapping format to output path
        """
        base = base_filename or f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            'html': self.html_reporter.generate_report(results, metrics, environment, f"{base}.html"),
            'json': self.json_reporter.generate_report(results, metrics, environment, f"{base}.json"),
            'junit': self.junit_reporter.generate_report(results, metrics, environment, f"{base}.xml"),
            'csv': self.csv_reporter.generate_report(results, metrics, environment, f"{base}.csv")
        }


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'TestMetrics',
    'TestCoverage',
    'TestReporter',
    'HTMLReporter',
    'JSONReporter',
    'JUnitReporter',
    'CSVReporter',
    'ConsolidatedReporter',
]
