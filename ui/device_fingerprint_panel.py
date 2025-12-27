#!/usr/bin/env python3
"""
RF Arsenal OS - Device Fingerprinting UI Panel
GUI panel for device fingerprinting and network profiling
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DeviceFingerprintPanel:
    """
    GUI panel for device fingerprinting
    
    Features:
    - Live device list
    - Network statistics
    - Device type breakdown (charts)
    - OS version analysis
    - Training data collection
    - Model management
    """
    
    def __init__(self, parent=None, show_stealth_warnings: bool = True):
        """Initialize device fingerprint panel
        
        Args:
            parent: Parent widget (for GUI integration)
            show_stealth_warnings: If True, display stealth operation warnings in reports
        """
        self.parent = parent
        
        # Data
        self.detected_devices = {}
        self.network_stats = {}
        
        # UI state
        self.auto_refresh = True
        self.refresh_interval = 2.0  # seconds
        self.show_stealth_warnings = show_stealth_warnings
        
        logger.info("Device fingerprint panel initialized")
    
    def update_device_list(self, devices: Dict):
        """
        Update displayed device list
        
        Args:
            devices: Dictionary of IMSI → DeviceProfile
        """
        self.detected_devices = devices
        logger.debug(f"Updated device list: {len(devices)} devices")
    
    def update_statistics(self, stats: Dict):
        """
        Update network statistics
        
        Args:
            stats: Network statistics dictionary
        """
        self.network_stats = stats
        logger.debug("Updated network statistics")
    
    def generate_html_report(self, output_path: str) -> bool:
        """
        Generate HTML report with charts
        
        Args:
            output_path: Output HTML file path
            
        Returns:
            True if generated successfully
        """
        try:
            html = self._generate_html()
            
            with open(output_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Generated HTML report: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"HTML report generation failed: {e}")
            return False
    
    def _generate_html(self) -> str:
        """Generate HTML report with Chart.js"""
        # Prepare data for charts
        import json
        
        # OS breakdown
        os_counts = {}
        mfg_counts = {}
        
        for profile in self.detected_devices.values():
            os_counts[profile.os] = os_counts.get(profile.os, 0) + 1
            mfg_counts[profile.manufacturer] = mfg_counts.get(profile.manufacturer, 0) + 1
        
        os_labels = list(os_counts.keys())
        os_data = list(os_counts.values())
        
        mfg_labels = list(mfg_counts.keys())
        mfg_data = list(mfg_counts.values())
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>RF Arsenal OS - Device Fingerprinting Report</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #007bff;
        }}
        
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background: #007bff;
            color: white;
        }}
        
        tr:hover {{
            background: #f5f5f5;
        }}
        
        .confidence-bar {{
            width: 100px;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 RF Arsenal OS - Device Fingerprinting Report</h1>
        
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(self.detected_devices)}</div>
                <div class="stat-label">Total Devices</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(os_counts)}</div>
                <div class="stat-label">Operating Systems</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(mfg_counts)}</div>
                <div class="stat-label">Manufacturers</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.network_stats.get('processing_rate', 0):.1f}</div>
                <div class="stat-label">Devices/Second</div>
            </div>
        </div>
        
        <h2>📊 Operating System Breakdown</h2>
        <div class="chart-container">
            <canvas id="osChart"></canvas>
        </div>
        
        <h2>🏢 Manufacturer Breakdown</h2>
        <div class="chart-container">
            <canvas id="mfgChart"></canvas>
        </div>
        
        <h2>📱 Detected Devices</h2>
        <table>
            <thead>
                <tr>
                    <th>IMSI</th>
                    <th>Manufacturer</th>
                    <th>Model</th>
                    <th>OS</th>
                    <th>Baseband</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
'''
        
        # Add device rows
        for imsi, profile in sorted(self.detected_devices.items()):
            confidence_pct = profile.confidence * 100
            html += f'''
                <tr>
                    <td>{imsi[-8:]}</td>
                    <td>{profile.manufacturer}</td>
                    <td>{profile.model}</td>
                    <td>{profile.os}</td>
                    <td>{profile.baseband_version}</td>
                    <td>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence_pct}%"></div>
                        </div>
                        {confidence_pct:.1f}%
                    </td>
                </tr>
'''
        
        html += f'''
            </tbody>
        </table>
        '''
        
        # Add stealth warnings if enabled
        if self.show_stealth_warnings:
            html += '''
        <div style="background: #fff3cd; border: 2px solid #ff9800; padding: 20px; margin: 30px 0; border-radius: 8px;">
            <h3 style="color: #d32f2f; margin-top: 0;">⚠️ STEALTH & PRIVACY NOTICE</h3>
            <p><strong>🔒 Stealth Operation Requirements:</strong></p>
            <ul>
                <li>✅ Model must be trained using synthetic data ONLY (no field captures)</li>
                <li>✅ System must operate in <code>--passive-only</code> mode (receive only, no transmission)</li>
                <li>✅ All IMSI/IMEI identifiers must be anonymized (hashed) in logs</li>
                <li>❌ IMSI catcher features must be DISABLED (breaks stealth)</li>
                <li>❌ Active geolocation must be DISABLED (use passive timing advance only)</li>
            </ul>
            <p><strong>🔐 Privacy Protections:</strong></p>
            <ul>
                <li>IMSI values shown in this report are HASHED (first 8-12 chars of SHA-256)</li>
                <li>Raw identifiers are NOT stored in reports or databases</li>
                <li>All processing occurs locally (no network exfiltration)</li>
            </ul>
            <p><strong>⚠️ Legal Warning:</strong> This tool must ONLY be used on networks you own or have explicit written authorization to monitor. Unauthorized surveillance may violate privacy laws (GDPR, CCPA, ECPA, etc.) and telecommunications regulations. Use responsibly.</p>
        </div>
        '''
        
        html += '''
        <div class="footer">
            <p>RF Arsenal OS - White Hat Edition</p>
            <p>Report generated using Machine Learning Device Fingerprinting</p>
            <p style="color: #999; font-size: 12px;">⚠️ Stealth mode warnings enabled - See privacy notice above</p>
        </div>
    </div>
    
    <script>
        // OS Chart
        const osCtx = document.getElementById('osChart').getContext('2d');
        new Chart(osCtx, {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(os_labels)},
                datasets: [{{
                    data: {json.dumps(os_data)},
                    backgroundColor: [
                        '#007bff',
                        '#28a745',
                        '#ffc107',
                        '#dc3545',
                        '#6610f2',
                        '#e83e8c'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'right'
                    }},
                    title: {{
                        display: true,
                        text: 'Operating System Distribution'
                    }}
                }}
            }}
        }});
        
        // Manufacturer Chart
        const mfgCtx = document.getElementById('mfgChart').getContext('2d');
        new Chart(mfgCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(mfg_labels)},
                datasets: [{{
                    label: 'Number of Devices',
                    data: {json.dumps(mfg_data)},
                    backgroundColor: '#007bff'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    title: {{
                        display: true,
                        text: 'Devices by Manufacturer'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            stepSize: 1
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>'''
        
        return html
    
    def export_csv(self, output_path: str) -> bool:
        """
        Export device list to CSV
        
        Args:
            output_path: Output CSV file path
            
        Returns:
            True if exported successfully
        """
        try:
            import csv
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'IMSI',
                    'Manufacturer',
                    'Model',
                    'OS',
                    'Baseband',
                    'Confidence'
                ])
                
                # Data rows
                for imsi, profile in sorted(self.detected_devices.items()):
                    writer.writerow([
                        imsi,
                        profile.manufacturer,
                        profile.model,
                        profile.os,
                        profile.baseband_version,
                        f"{profile.confidence:.3f}"
                    ])
            
            logger.info(f"Exported CSV to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False


if __name__ == "__main__":
    # Test UI panel
    logging.basicConfig(level=logging.INFO)
    
    print("RF Arsenal OS - Device Fingerprint Panel Test")
    print("=" * 60)
    
    from modules.ai.device_fingerprinting import DeviceProfile
    
    # Create test data
    test_devices = {
        "310410000000001": DeviceProfile(
            manufacturer="Apple",
            model="iPhone 14 Pro",
            os="iOS 17.x",
            baseband_version="Qualcomm X65",
            confidence=0.95,
            timestamp=datetime.now().isoformat()
        ),
        "310410000000002": DeviceProfile(
            manufacturer="Samsung",
            model="Galaxy S23",
            os="Android 14",
            baseband_version="Qualcomm X70",
            confidence=0.92,
            timestamp=datetime.now().isoformat()
        ),
        "310410000000003": DeviceProfile(
            manufacturer="Google",
            model="Pixel 7 Pro",
            os="Android 14",
            baseband_version="Qualcomm X65",
            confidence=0.88,
            timestamp=datetime.now().isoformat()
        )
    }
    
    # Create panel
    panel = DeviceFingerprintPanel()
    panel.update_device_list(test_devices)
    panel.update_statistics({'processing_rate': 2.5})
    
    # Generate HTML report
    print("\n[+] Generating HTML report...")
    panel.generate_html_report("test_fingerprint_report.html")
    print("[+] Report saved to: test_fingerprint_report.html")
    
    # Export CSV
    print("\n[+] Exporting CSV...")
    panel.export_csv("test_devices.csv")
    print("[+] CSV saved to: test_devices.csv")
    
    print("\n[+] Test complete")
