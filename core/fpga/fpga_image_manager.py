#!/usr/bin/env python3
"""
RF Arsenal OS - FPGA Image Manager
Manages custom FPGA images for BladeRF acceleration

Features:
- Image inventory and versioning
- Verification and integrity checking
- Build automation integration
- Flash/load management
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ImageType(Enum):
    """FPGA image type classification"""
    STANDARD = "standard"  # Stock BladeRF image
    DSP_ACCEL = "dsp_accel"  # DSP acceleration features
    STEALTH = "stealth"  # Stealth mode features
    LTE = "lte"  # LTE acceleration
    NR = "nr"  # 5G NR acceleration
    CUSTOM = "custom"  # User custom image
    DEVELOPMENT = "development"  # Development/debug image


class ImageFormat(Enum):
    """FPGA image file format"""
    RBF = "rbf"  # Raw Binary Format (Altera/Intel)
    BIT = "bit"  # Bitstream (Xilinx)
    HEX = "hex"  # Intel HEX format
    COMPRESSED = "rbf.gz"  # Compressed RBF


@dataclass
class ImageMetadata:
    """FPGA image metadata"""
    name: str
    version: str
    image_type: ImageType
    description: str = ""
    
    # Build information
    build_date: str = ""
    build_hash: str = ""
    source_commit: str = ""
    
    # Target information
    target_device: str = "bladerf-micro"
    target_fpga: str = "cyclone5"
    min_firmware_version: str = "2.0.0"
    
    # Feature flags
    features: List[str] = field(default_factory=list)
    
    # Checksums
    sha256: str = ""
    crc32: str = ""
    file_size: int = 0
    
    # Compatibility
    compatible_with: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'version': self.version,
            'type': self.image_type.value,
            'description': self.description,
            'build_date': self.build_date,
            'build_hash': self.build_hash,
            'source_commit': self.source_commit,
            'target_device': self.target_device,
            'target_fpga': self.target_fpga,
            'min_firmware_version': self.min_firmware_version,
            'features': self.features,
            'sha256': self.sha256,
            'crc32': self.crc32,
            'file_size': self.file_size,
            'compatible_with': self.compatible_with,
            'incompatible_with': self.incompatible_with,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageMetadata':
        """Create from dictionary"""
        return cls(
            name=data.get('name', ''),
            version=data.get('version', '0.0.0'),
            image_type=ImageType(data.get('type', 'custom')),
            description=data.get('description', ''),
            build_date=data.get('build_date', ''),
            build_hash=data.get('build_hash', ''),
            source_commit=data.get('source_commit', ''),
            target_device=data.get('target_device', 'bladerf-micro'),
            target_fpga=data.get('target_fpga', 'cyclone5'),
            min_firmware_version=data.get('min_firmware_version', '2.0.0'),
            features=data.get('features', []),
            sha256=data.get('sha256', ''),
            crc32=data.get('crc32', ''),
            file_size=data.get('file_size', 0),
            compatible_with=data.get('compatible_with', []),
            incompatible_with=data.get('incompatible_with', []),
        )


@dataclass
class FPGAImage:
    """FPGA image representation"""
    path: Path
    metadata: ImageMetadata
    is_valid: bool = False
    validation_message: str = ""
    
    @property
    def filename(self) -> str:
        return self.path.name
    
    @property
    def format(self) -> ImageFormat:
        suffix = ''.join(self.path.suffixes).lower()
        if suffix == '.rbf.gz':
            return ImageFormat.COMPRESSED
        elif suffix == '.rbf':
            return ImageFormat.RBF
        elif suffix == '.bit':
            return ImageFormat.BIT
        elif suffix == '.hex':
            return ImageFormat.HEX
        return ImageFormat.RBF  # Default


class FPGAImageManager:
    """
    Manages FPGA images for RF Arsenal OS
    
    Features:
    - Image inventory management
    - Build automation
    - Verification and integrity checking
    - Flash/load utilities
    """
    
    # Default directories
    DEFAULT_IMAGE_DIR = Path(__file__).parent.parent.parent / "fpga" / "images"
    DEFAULT_HDL_DIR = Path(__file__).parent.parent.parent / "fpga" / "hdl"
    DEFAULT_BUILD_DIR = Path(__file__).parent.parent.parent / "fpga" / "build"
    
    # Built-in image configurations
    BUILTIN_IMAGES = {
        'rf_arsenal_default': {
            'name': 'RF Arsenal Default',
            'version': '1.0.0',
            'type': 'standard',
            'description': 'Default RF Arsenal FPGA image with basic DSP acceleration',
            'features': ['fft', 'fir_filter', 'basic_modulation'],
        },
        'rf_arsenal_stealth': {
            'name': 'RF Arsenal Stealth',
            'version': '1.0.0',
            'type': 'stealth',
            'description': 'Stealth mode with frequency hopping and power ramping',
            'features': ['fft', 'fir_filter', 'freq_hopping', 'power_ramping', 'burst_control'],
        },
        'rf_arsenal_lte': {
            'name': 'RF Arsenal LTE',
            'version': '1.0.0',
            'type': 'lte',
            'description': 'LTE/5G acceleration with OFDM modulator/demodulator',
            'features': ['fft', 'fir_filter', 'ofdm_mod', 'ofdm_demod', 'channel_estimator'],
        },
        'rf_arsenal_full': {
            'name': 'RF Arsenal Full',
            'version': '1.0.0',
            'type': 'custom',
            'description': 'Full-featured image with all accelerators',
            'features': ['fft', 'fir_filter', 'iir_filter', 'ofdm_mod', 'ofdm_demod',
                        'freq_hopping', 'power_ramping', 'channel_estimator', 'sync_detector'],
        },
    }
    
    def __init__(
        self,
        image_dir: Optional[Path] = None,
        hdl_dir: Optional[Path] = None,
        build_dir: Optional[Path] = None
    ):
        """
        Initialize FPGA Image Manager
        
        Args:
            image_dir: Directory containing FPGA images
            hdl_dir: Directory containing HDL source files
            build_dir: Directory for build artifacts
        """
        self._image_dir = Path(image_dir) if image_dir else self.DEFAULT_IMAGE_DIR
        self._hdl_dir = Path(hdl_dir) if hdl_dir else self.DEFAULT_HDL_DIR
        self._build_dir = Path(build_dir) if build_dir else self.DEFAULT_BUILD_DIR
        
        # Image inventory
        self._images: Dict[str, FPGAImage] = {}
        
        # Ensure directories exist
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._build_dir.mkdir(parents=True, exist_ok=True)
        
        # Load inventory
        self._load_inventory()
        
        logger.info(f"FPGAImageManager initialized: {len(self._images)} images found")
    
    @property
    def image_dir(self) -> Path:
        return self._image_dir
    
    @property
    def available_images(self) -> List[str]:
        """Get list of available image names"""
        return list(self._images.keys())
    
    def get_image(self, name: str) -> Optional[FPGAImage]:
        """
        Get FPGA image by name
        
        Args:
            name: Image name (without extension)
            
        Returns:
            FPGAImage if found, None otherwise
        """
        return self._images.get(name)
    
    def get_image_by_type(self, image_type: ImageType) -> List[FPGAImage]:
        """
        Get all images of a specific type
        
        Args:
            image_type: Type to filter by
            
        Returns:
            List of matching images
        """
        return [
            img for img in self._images.values()
            if img.metadata.image_type == image_type
        ]
    
    def add_image(
        self,
        source_path: Path,
        metadata: ImageMetadata,
        verify: bool = True
    ) -> bool:
        """
        Add new FPGA image to inventory
        
        Args:
            source_path: Path to source image file
            metadata: Image metadata
            verify: Whether to verify image integrity
            
        Returns:
            True if added successfully
        """
        try:
            source_path = Path(source_path)
            
            if not source_path.exists():
                logger.error(f"Source image not found: {source_path}")
                return False
            
            # Generate filename
            filename = f"{metadata.name.lower().replace(' ', '_')}.rbf"
            dest_path = self._image_dir / filename
            
            # Copy file
            shutil.copy2(source_path, dest_path)
            
            # Calculate checksums
            metadata.sha256 = self._calculate_sha256(dest_path)
            metadata.crc32 = self._calculate_crc32(dest_path)
            metadata.file_size = dest_path.stat().st_size
            
            # Create image object
            image = FPGAImage(
                path=dest_path,
                metadata=metadata,
                is_valid=True
            )
            
            # Verify if requested
            if verify:
                image.is_valid, image.validation_message = self._verify_image(dest_path)
            
            # Add to inventory
            image_name = dest_path.stem
            self._images[image_name] = image
            
            # Save metadata
            self._save_metadata(image)
            
            logger.info(f"Added FPGA image: {image_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add image: {e}")
            return False
    
    def remove_image(self, name: str) -> bool:
        """
        Remove FPGA image from inventory
        
        Args:
            name: Image name
            
        Returns:
            True if removed successfully
        """
        try:
            if name not in self._images:
                logger.warning(f"Image not found: {name}")
                return False
            
            image = self._images[name]
            
            # Remove files
            if image.path.exists():
                image.path.unlink()
            
            meta_path = image.path.with_suffix('.json')
            if meta_path.exists():
                meta_path.unlink()
            
            # Remove from inventory
            del self._images[name]
            
            logger.info(f"Removed FPGA image: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove image: {e}")
            return False
    
    def verify_image(self, name: str) -> Tuple[bool, str]:
        """
        Verify FPGA image integrity
        
        Args:
            name: Image name
            
        Returns:
            (is_valid, message) tuple
        """
        if name not in self._images:
            return False, f"Image not found: {name}"
        
        image = self._images[name]
        return self._verify_image(image.path, image.metadata)
    
    def verify_all_images(self) -> Dict[str, Tuple[bool, str]]:
        """
        Verify all images in inventory
        
        Returns:
            Dictionary of name -> (is_valid, message)
        """
        results = {}
        for name in self._images:
            results[name] = self.verify_image(name)
        return results
    
    def build_image(
        self,
        config_name: str,
        output_name: Optional[str] = None,
        quartus_path: Optional[str] = None
    ) -> Optional[Path]:
        """
        Build FPGA image from HDL sources
        
        Args:
            config_name: Build configuration name (from BUILTIN_IMAGES)
            output_name: Output image name (default: config_name)
            quartus_path: Path to Quartus tools (auto-detected if None)
            
        Returns:
            Path to built image, or None on failure
        """
        if config_name not in self.BUILTIN_IMAGES:
            logger.error(f"Unknown build configuration: {config_name}")
            return None
        
        config = self.BUILTIN_IMAGES[config_name]
        output_name = output_name or config_name
        
        try:
            logger.info(f"Building FPGA image: {config_name}")
            
            # Create build directory
            build_path = self._build_dir / config_name
            build_path.mkdir(parents=True, exist_ok=True)
            
            # Generate project file
            self._generate_project_file(build_path, config)
            
            # Run synthesis (would use actual Quartus tools)
            output_path = build_path / f"{output_name}.rbf"
            
            # For now, create placeholder
            logger.warning("Actual synthesis requires Quartus tools - creating placeholder")
            output_path.touch()
            
            # Create metadata
            metadata = ImageMetadata(
                name=config['name'],
                version=config['version'],
                image_type=ImageType(config['type']),
                description=config['description'],
                features=config['features'],
                build_date=datetime.now().isoformat(),
            )
            
            # Add to inventory
            self.add_image(output_path, metadata, verify=False)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Build failed: {e}")
            return None
    
    def get_recommended_image(
        self,
        requirements: List[str]
    ) -> Optional[FPGAImage]:
        """
        Get recommended image based on feature requirements
        
        Args:
            requirements: List of required features
            
        Returns:
            Best matching image, or None
        """
        best_match = None
        best_score = -1
        
        for image in self._images.values():
            if not image.is_valid:
                continue
            
            # Calculate match score
            score = 0
            features = image.metadata.features
            
            for req in requirements:
                if req in features:
                    score += 2  # Feature present
                else:
                    score -= 1  # Feature missing
            
            # Prefer smaller images (fewer unused features)
            score -= len(features) * 0.1
            
            if score > best_score:
                best_score = score
                best_match = image
        
        return best_match
    
    def export_image(
        self,
        name: str,
        dest_path: Path,
        include_metadata: bool = True
    ) -> bool:
        """
        Export FPGA image to destination
        
        Args:
            name: Image name
            dest_path: Destination path
            include_metadata: Whether to include metadata file
            
        Returns:
            True if exported successfully
        """
        try:
            if name not in self._images:
                logger.error(f"Image not found: {name}")
                return False
            
            image = self._images[name]
            dest_path = Path(dest_path)
            
            # Create destination directory
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy image
            shutil.copy2(image.path, dest_path)
            
            # Copy metadata if requested
            if include_metadata:
                meta_dest = dest_path.with_suffix('.json')
                meta_data = image.metadata.to_dict()
                with open(meta_dest, 'w') as f:
                    json.dump(meta_data, f, indent=2)
            
            logger.info(f"Exported image: {name} -> {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def import_image(
        self,
        source_path: Path,
        metadata_path: Optional[Path] = None
    ) -> bool:
        """
        Import FPGA image from external source
        
        Args:
            source_path: Path to image file
            metadata_path: Path to metadata file (optional)
            
        Returns:
            True if imported successfully
        """
        try:
            source_path = Path(source_path)
            
            if not source_path.exists():
                logger.error(f"Source not found: {source_path}")
                return False
            
            # Load or create metadata
            if metadata_path and Path(metadata_path).exists():
                with open(metadata_path) as f:
                    meta_dict = json.load(f)
                metadata = ImageMetadata.from_dict(meta_dict)
            else:
                # Create default metadata
                metadata = ImageMetadata(
                    name=source_path.stem,
                    version='0.0.0',
                    image_type=ImageType.CUSTOM,
                    description='Imported FPGA image',
                )
            
            return self.add_image(source_path, metadata)
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False
    
    def get_inventory(self) -> Dict[str, Dict[str, Any]]:
        """
        Get complete image inventory
        
        Returns:
            Dictionary of image information
        """
        inventory = {}
        for name, image in self._images.items():
            inventory[name] = {
                'path': str(image.path),
                'metadata': image.metadata.to_dict(),
                'is_valid': image.is_valid,
                'validation_message': image.validation_message,
                'format': image.format.value,
            }
        return inventory
    
    def refresh_inventory(self) -> None:
        """Refresh image inventory from disk"""
        self._images.clear()
        self._load_inventory()
        logger.info(f"Inventory refreshed: {len(self._images)} images")
    
    # =========================================================================
    # Flash Utilities
    # =========================================================================
    
    def prepare_flash_package(
        self,
        image_names: List[str],
        output_dir: Path
    ) -> Optional[Path]:
        """
        Prepare flash package with multiple images
        
        Args:
            image_names: List of image names to include
            output_dir: Output directory for package
            
        Returns:
            Path to package, or None on failure
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            package_info = {
                'created': datetime.now().isoformat(),
                'images': [],
            }
            
            for name in image_names:
                if name not in self._images:
                    logger.warning(f"Image not found, skipping: {name}")
                    continue
                
                image = self._images[name]
                
                # Copy image
                dest = output_dir / image.filename
                shutil.copy2(image.path, dest)
                
                # Add to package info
                package_info['images'].append({
                    'name': name,
                    'filename': image.filename,
                    'metadata': image.metadata.to_dict(),
                })
            
            # Write package manifest
            manifest_path = output_dir / 'manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump(package_info, f, indent=2)
            
            logger.info(f"Flash package created: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Failed to create flash package: {e}")
            return None
    
    def create_bootloader_image(
        self,
        base_image: str,
        autoload: bool = True
    ) -> Optional[Path]:
        """
        Create bootloader-compatible image for auto-loading
        
        Args:
            base_image: Base image name
            autoload: Whether to enable auto-loading on boot
            
        Returns:
            Path to bootloader image, or None on failure
        """
        if base_image not in self._images:
            logger.error(f"Base image not found: {base_image}")
            return None
        
        try:
            image = self._images[base_image]
            
            # Create output path
            boot_name = f"{base_image}_bootloader.rbf"
            boot_path = self._image_dir / boot_name
            
            # Copy and modify for bootloader
            shutil.copy2(image.path, boot_path)
            
            # Add bootloader header (placeholder implementation)
            # Actual implementation would add proper BladeRF boot header
            
            logger.info(f"Bootloader image created: {boot_path}")
            return boot_path
            
        except Exception as e:
            logger.error(f"Failed to create bootloader image: {e}")
            return None
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _load_inventory(self) -> None:
        """Load image inventory from disk"""
        if not self._image_dir.exists():
            return
        
        # Scan for image files
        for path in self._image_dir.glob('*.rbf'):
            try:
                name = path.stem
                
                # Load metadata if available
                meta_path = path.with_suffix('.json')
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta_dict = json.load(f)
                    metadata = ImageMetadata.from_dict(meta_dict)
                else:
                    # Create basic metadata
                    metadata = ImageMetadata(
                        name=name,
                        version='0.0.0',
                        image_type=ImageType.CUSTOM,
                        file_size=path.stat().st_size,
                    )
                
                # Create image object
                image = FPGAImage(
                    path=path,
                    metadata=metadata,
                    is_valid=True,  # Assume valid until verified
                )
                
                self._images[name] = image
                
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
    
    def _save_metadata(self, image: FPGAImage) -> None:
        """Save image metadata to disk"""
        meta_path = image.path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(image.metadata.to_dict(), f, indent=2)
    
    def _verify_image(
        self,
        path: Path,
        metadata: Optional[ImageMetadata] = None
    ) -> Tuple[bool, str]:
        """Verify image file integrity"""
        try:
            if not path.exists():
                return False, "File not found"
            
            # Check file size
            size = path.stat().st_size
            if size == 0:
                return False, "File is empty"
            
            if metadata and metadata.file_size > 0:
                if size != metadata.file_size:
                    return False, f"Size mismatch: {size} != {metadata.file_size}"
            
            # Verify checksum
            if metadata and metadata.sha256:
                actual_sha256 = self._calculate_sha256(path)
                if actual_sha256 != metadata.sha256:
                    return False, "SHA256 checksum mismatch"
            
            # Basic format validation
            with open(path, 'rb') as f:
                header = f.read(4)
                # RBF files typically start with specific patterns
                # This is a simplified check
                if len(header) < 4:
                    return False, "Invalid file format"
            
            return True, "Verification passed"
            
        except Exception as e:
            return False, f"Verification error: {e}"
    
    def _calculate_sha256(self, path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _calculate_crc32(self, path: Path) -> str:
        """Calculate CRC32 of file"""
        import binascii
        crc = 0
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                crc = binascii.crc32(chunk, crc)
        return f"{crc & 0xFFFFFFFF:08x}"
    
    def _generate_project_file(
        self,
        build_dir: Path,
        config: Dict[str, Any]
    ) -> None:
        """Generate Quartus project file for build"""
        # Create Quartus project settings (simplified)
        qsf_content = f"""
# RF Arsenal FPGA Project
# Auto-generated configuration

set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CEFA9F23I7
set_global_assignment -name TOP_LEVEL_ENTITY rf_arsenal_top

# Source files
set_global_assignment -name VHDL_FILE {self._hdl_dir}/core/rf_arsenal_pkg.vhd
set_global_assignment -name VHDL_FILE {self._hdl_dir}/core/rf_arsenal_top.vhd
set_global_assignment -name VHDL_FILE {self._hdl_dir}/dsp/fft_engine.vhd
set_global_assignment -name VHDL_FILE {self._hdl_dir}/dsp/fir_filter.vhd
set_global_assignment -name VHDL_FILE {self._hdl_dir}/stealth/stealth_processor.vhd
set_global_assignment -name VHDL_FILE {self._hdl_dir}/lte/ofdm_modulator.vhd

# Timing constraints
set_global_assignment -name SDC_FILE {self._hdl_dir}/../constraints/bladerf_timing.sdc

# Features: {', '.join(config.get('features', []))}
"""
        
        qsf_path = build_dir / "rf_arsenal.qsf"
        with open(qsf_path, 'w') as f:
            f.write(qsf_content)
        
        logger.info(f"Generated project file: {qsf_path}")
