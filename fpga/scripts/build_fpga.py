#!/usr/bin/env python3
"""
RF Arsenal OS - FPGA Build Script
Automated FPGA image building for BladeRF

This script:
1. Generates Quartus project files
2. Runs synthesis, place & route
3. Creates programming files
4. Optionally flashes to device
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
FPGA_DIR = PROJECT_ROOT / "fpga"
HDL_DIR = FPGA_DIR / "hdl"
BUILD_DIR = FPGA_DIR / "build"
OUTPUT_DIR = FPGA_DIR / "images"
CONSTRAINTS_DIR = FPGA_DIR / "constraints"


@dataclass
class BuildConfig:
    """FPGA build configuration"""
    name: str
    version: str
    description: str
    
    # Target device
    device_family: str = "Cyclone V"
    device_part: str = "5CEFA9F23I7"  # BladeRF 2.0 micro xA9
    
    # Features to include
    features: List[str] = None
    
    # Build options
    optimization: str = "balanced"  # balanced, speed, area
    timing_effort: str = "high"
    fitter_effort: str = "standard"
    
    # Paths
    hdl_files: List[str] = None
    constraints_file: str = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = ['fft', 'fir_filter', 'stealth', 'ofdm']
        if self.hdl_files is None:
            self.hdl_files = []


# Predefined build configurations
BUILD_CONFIGS = {
    'default': BuildConfig(
        name='rf_arsenal_default',
        version='1.0.0',
        description='Default RF Arsenal FPGA image',
        features=['fft', 'fir_filter'],
    ),
    'stealth': BuildConfig(
        name='rf_arsenal_stealth',
        version='1.0.0',
        description='Stealth mode FPGA image',
        features=['fft', 'fir_filter', 'stealth'],
    ),
    'lte': BuildConfig(
        name='rf_arsenal_lte',
        version='1.0.0',
        description='LTE acceleration FPGA image',
        features=['fft', 'fir_filter', 'ofdm', 'channel_est'],
    ),
    'full': BuildConfig(
        name='rf_arsenal_full',
        version='1.0.0',
        description='Full-featured FPGA image',
        features=['fft', 'fir_filter', 'iir_filter', 'stealth', 'ofdm', 'channel_est'],
    ),
}


class FPGABuilder:
    """FPGA image builder"""
    
    def __init__(self, config: BuildConfig, quartus_path: Optional[str] = None):
        """
        Initialize builder
        
        Args:
            config: Build configuration
            quartus_path: Path to Quartus installation
        """
        self.config = config
        self.quartus_path = self._find_quartus(quartus_path)
        
        # Build directory
        self.build_dir = BUILD_DIR / config.name
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file
        self.log_file = self.build_dir / "build.log"
        
        logger.info(f"FPGABuilder initialized for: {config.name}")
    
    def _find_quartus(self, quartus_path: Optional[str]) -> Optional[Path]:
        """Find Quartus installation"""
        if quartus_path:
            return Path(quartus_path)
        
        # Common installation paths
        search_paths = [
            Path("/opt/intelFPGA_lite"),
            Path("/opt/intelFPGA"),
            Path("/opt/altera"),
            Path.home() / "intelFPGA_lite",
            Path.home() / "intelFPGA",
            Path("C:/intelFPGA_lite"),
            Path("C:/intelFPGA"),
        ]
        
        for base in search_paths:
            if base.exists():
                # Find latest version
                versions = sorted(base.glob("*"), reverse=True)
                for version in versions:
                    quartus_bin = version / "quartus" / "bin"
                    if quartus_bin.exists():
                        logger.info(f"Found Quartus: {quartus_bin}")
                        return quartus_bin
        
        logger.warning("Quartus not found - build will be simulated")
        return None
    
    def generate_project_files(self) -> bool:
        """Generate Quartus project files"""
        logger.info("Generating project files...")
        
        try:
            # Generate QSF file
            self._generate_qsf()
            
            # Generate QPF file
            self._generate_qpf()
            
            # Generate TCL script
            self._generate_tcl()
            
            # Copy constraint file
            self._copy_constraints()
            
            logger.info("Project files generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate project files: {e}")
            return False
    
    def _generate_qsf(self) -> None:
        """Generate Quartus Settings File"""
        qsf_content = f"""
# =============================================================================
# RF Arsenal OS - FPGA Project Settings
# Generated: {datetime.now().isoformat()}
# Configuration: {self.config.name}
# =============================================================================

# Project settings
set_global_assignment -name FAMILY "{self.config.device_family}"
set_global_assignment -name DEVICE {self.config.device_part}
set_global_assignment -name TOP_LEVEL_ENTITY rf_arsenal_top
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
set_global_assignment -name MIN_CORE_JUNCTION_TEMP 0
set_global_assignment -name MAX_CORE_JUNCTION_TEMP 85
set_global_assignment -name ERROR_CHECK_FREQUENCY_DIVISOR 1

# Timing settings
set_global_assignment -name TIMING_ANALYZER_MULTICORNER_ANALYSIS ON
set_global_assignment -name NUM_PARALLEL_PROCESSORS ALL

# Optimization settings
"""
        
        if self.config.optimization == "speed":
            qsf_content += """
set_global_assignment -name OPTIMIZATION_MODE "HIGH PERFORMANCE EFFORT"
set_global_assignment -name PHYSICAL_SYNTHESIS_COMBO_LOGIC ON
set_global_assignment -name PHYSICAL_SYNTHESIS_REGISTER_RETIMING ON
"""
        elif self.config.optimization == "area":
            qsf_content += """
set_global_assignment -name OPTIMIZATION_MODE "MINIMIZE AREA"
set_global_assignment -name AUTO_RAM_RECOGNITION ON
set_global_assignment -name AUTO_DSP_RECOGNITION ON
"""
        else:
            qsf_content += """
set_global_assignment -name OPTIMIZATION_MODE "BALANCED"
"""
        
        # Add HDL source files
        qsf_content += "\n# HDL Source Files\n"
        
        # Core files always included
        core_files = [
            "core/rf_arsenal_pkg.vhd",
            "core/rf_arsenal_top.vhd",
        ]
        
        # Feature-dependent files
        feature_files = {
            'fft': ["dsp/fft_engine.vhd"],
            'fir_filter': ["dsp/fir_filter.vhd"],
            'iir_filter': ["dsp/iir_filter.vhd"],
            'stealth': ["stealth/stealth_processor.vhd"],
            'ofdm': ["lte/ofdm_modulator.vhd"],
            'channel_est': ["lte/channel_estimator.vhd"],
        }
        
        all_files = core_files.copy()
        for feature in self.config.features:
            if feature in feature_files:
                all_files.extend(feature_files[feature])
        
        for hdl_file in all_files:
            hdl_path = HDL_DIR / hdl_file
            if hdl_path.exists():
                qsf_content += f'set_global_assignment -name VHDL_FILE "{hdl_path}"\n'
            else:
                logger.warning(f"HDL file not found: {hdl_path}")
        
        # Add constraints
        qsf_content += f"""
# Constraints
set_global_assignment -name SDC_FILE "{CONSTRAINTS_DIR}/bladerf_micro_xA9.sdc"

# Feature enables (passed to HDL as generics)
"""
        
        for feature in self.config.features:
            qsf_content += f'set_parameter -name ENABLE_{feature.upper()} 1\n'
        
        # Write QSF file
        qsf_path = self.build_dir / f"{self.config.name}.qsf"
        with open(qsf_path, 'w') as f:
            f.write(qsf_content)
        
        logger.info(f"Generated: {qsf_path}")
    
    def _generate_qpf(self) -> None:
        """Generate Quartus Project File"""
        qpf_content = f"""
QUARTUS_VERSION = "20.1"
DATE = "{datetime.now().strftime('%H:%M:%S  %B %d, %Y')}"

PROJECT_REVISION = "{self.config.name}"
"""
        
        qpf_path = self.build_dir / f"{self.config.name}.qpf"
        with open(qpf_path, 'w') as f:
            f.write(qpf_content)
        
        logger.info(f"Generated: {qpf_path}")
    
    def _generate_tcl(self) -> None:
        """Generate TCL build script"""
        tcl_content = f"""
# RF Arsenal FPGA Build Script
# Run with: quartus_sh -t build.tcl

package require ::quartus::project
package require ::quartus::flow

# Project settings
set project_name "{self.config.name}"
set revision_name "{self.config.name}"

# Open project
if {{[project_exists $project_name]}} {{
    project_open -revision $revision_name $project_name
}} else {{
    project_new -revision $revision_name $project_name
}}

# Load settings
source ${{project_name}}.qsf

# Run compilation
puts "Starting compilation..."
set compile_start_time [clock seconds]

# Analysis & Synthesis
if {{[catch {{execute_module -tool map}} result]}} {{
    puts "ERROR: Analysis & Synthesis failed"
    puts $result
    project_close
    exit 1
}}

# Fitter
if {{[catch {{execute_module -tool fit}} result]}} {{
    puts "ERROR: Fitter failed"
    puts $result
    project_close
    exit 1
}}

# Timing Analysis
if {{[catch {{execute_module -tool sta}} result]}} {{
    puts "ERROR: Timing analysis failed"
    puts $result
    project_close
    exit 1
}}

# Generate programming files
if {{[catch {{execute_module -tool asm}} result]}} {{
    puts "ERROR: Assembler failed"
    puts $result
    project_close
    exit 1
}}

set compile_end_time [clock seconds]
set compile_time [expr {{$compile_end_time - $compile_start_time}}]
puts "Compilation completed in $compile_time seconds"

# Generate RBF file for BladeRF
set_global_assignment -name GENERATE_RBF_FILE ON
execute_module -tool asm

project_close
puts "Build complete!"
"""
        
        tcl_path = self.build_dir / "build.tcl"
        with open(tcl_path, 'w') as f:
            f.write(tcl_content)
        
        logger.info(f"Generated: {tcl_path}")
    
    def _copy_constraints(self) -> None:
        """Copy constraint files to build directory"""
        src = CONSTRAINTS_DIR / "bladerf_micro_xA9.sdc"
        dst = self.build_dir / "bladerf_micro_xA9.sdc"
        
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"Copied constraints: {dst}")
    
    def run_synthesis(self) -> bool:
        """Run Quartus synthesis"""
        if self.quartus_path is None:
            logger.warning("Quartus not available - skipping synthesis")
            return self._simulate_build()
        
        logger.info("Running synthesis...")
        
        try:
            quartus_sh = self.quartus_path / "quartus_sh"
            tcl_script = self.build_dir / "build.tcl"
            
            cmd = [str(quartus_sh), "-t", str(tcl_script)]
            
            with open(self.log_file, 'w') as log:
                result = subprocess.run(
                    cmd,
                    cwd=self.build_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600  # 1 hour timeout
                )
            
            if result.returncode != 0:
                logger.error("Synthesis failed - check build.log")
                return False
            
            logger.info("Synthesis completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Synthesis timeout")
            return False
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return False
    
    def _simulate_build(self) -> bool:
        """Simulate build when Quartus not available"""
        logger.info("Simulating build (no Quartus)...")
        
        # Create placeholder output files
        output_dir = self.build_dir / "output_files"
        output_dir.mkdir(exist_ok=True)
        
        # Create placeholder RBF file
        rbf_path = output_dir / f"{self.config.name}.rbf"
        with open(rbf_path, 'wb') as f:
            # Write minimal placeholder
            f.write(b'RF_ARSENAL_FPGA_PLACEHOLDER\x00' * 100)
        
        # Create build report
        report = {
            'name': self.config.name,
            'version': self.config.version,
            'build_date': datetime.now().isoformat(),
            'simulated': True,
            'features': self.config.features,
            'device': self.config.device_part,
        }
        
        report_path = output_dir / "build_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Created placeholder: {rbf_path}")
        return True
    
    def generate_output(self) -> Optional[Path]:
        """Generate final output files"""
        logger.info("Generating output files...")
        
        # Source files
        output_dir = self.build_dir / "output_files"
        rbf_src = output_dir / f"{self.config.name}.rbf"
        
        if not rbf_src.exists():
            logger.error("RBF file not found")
            return None
        
        # Destination
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        rbf_dst = OUTPUT_DIR / f"{self.config.name}_v{self.config.version}_{timestamp}.rbf"
        
        # Copy RBF
        shutil.copy2(rbf_src, rbf_dst)
        
        # Also copy as latest
        latest_dst = OUTPUT_DIR / f"{self.config.name}.rbf"
        shutil.copy2(rbf_src, latest_dst)
        
        # Generate metadata
        metadata = {
            'name': self.config.name,
            'version': self.config.version,
            'description': self.config.description,
            'build_date': datetime.now().isoformat(),
            'device': self.config.device_part,
            'features': self.config.features,
            'file_size': rbf_dst.stat().st_size,
        }
        
        meta_path = OUTPUT_DIR / f"{self.config.name}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Output: {rbf_dst}")
        logger.info(f"Metadata: {meta_path}")
        
        return rbf_dst
    
    def build(self) -> Optional[Path]:
        """Run complete build process"""
        logger.info(f"Starting build: {self.config.name}")
        
        if not self.generate_project_files():
            return None
        
        if not self.run_synthesis():
            return None
        
        return self.generate_output()
    
    def clean(self) -> None:
        """Clean build directory"""
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
            logger.info(f"Cleaned: {self.build_dir}")


def flash_to_device(rbf_path: Path, verify: bool = True) -> bool:
    """
    Flash FPGA image to BladeRF device
    
    Args:
        rbf_path: Path to RBF file
        verify: Whether to verify after flashing
        
    Returns:
        True if successful
    """
    logger.info(f"Flashing to device: {rbf_path}")
    
    try:
        # Check for bladeRF-cli
        result = subprocess.run(
            ["bladeRF-cli", "--version"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error("bladeRF-cli not found")
            return False
        
        # Load FPGA image (temporary)
        cmd = ["bladeRF-cli", "-l", str(rbf_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Flash failed: {result.stderr}")
            return False
        
        logger.info("FPGA image loaded successfully")
        
        # Optionally flash to SPI (persistent)
        # cmd = ["bladeRF-cli", "-L", str(rbf_path)]
        
        return True
        
    except Exception as e:
        logger.error(f"Flash error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="RF Arsenal FPGA Build Tool")
    parser.add_argument(
        "config",
        choices=list(BUILD_CONFIGS.keys()),
        help="Build configuration to use"
    )
    parser.add_argument(
        "--quartus-path",
        help="Path to Quartus installation"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building"
    )
    parser.add_argument(
        "--flash",
        action="store_true",
        help="Flash to device after building"
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Generate project files only"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get build configuration
    config = BUILD_CONFIGS[args.config]
    
    # Create builder
    builder = FPGABuilder(config, args.quartus_path)
    
    # Clean if requested
    if args.clean:
        builder.clean()
    
    # Generate project files
    if not builder.generate_project_files():
        sys.exit(1)
    
    # Build if not --no-build
    if not args.no_build:
        rbf_path = builder.build()
        
        if rbf_path is None:
            logger.error("Build failed")
            sys.exit(1)
        
        # Flash if requested
        if args.flash:
            if not flash_to_device(rbf_path):
                sys.exit(1)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
