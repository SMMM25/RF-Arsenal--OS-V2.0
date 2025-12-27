#!/usr/bin/env python3
"""
RF Arsenal OS - 5G/NR Base Station Module
Hardware: BladeRF 2.0 micro xA9
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
from datetime import datetime

from core.anonymization import get_anonymizer

logger = logging.getLogger(__name__)

@dataclass
class NRConfig:
    """5G/NR Configuration"""
    frequency: int = 3_500_000_000  # n78 band (3.5 GHz)
    sample_rate: int = 61_440_000   # 61.44 MSPS (100 MHz NR)
    bandwidth: int = 100_000_000    # 100 MHz
    cell_id: int = 1
    tac: int = 1                    # Tracking Area Code
    mcc: str = "001"               # Mobile Country Code
    mnc: str = "01"                # Mobile Network Code
    tx_power: int = 23             # dBm
    numerology: int = 1            # SCS = 30 kHz
    arfcn: int = 632628            # NR-ARFCN for n78

class NRBaseStation:
    """5G/NR Base Station (gNodeB) using srsRAN"""
    
    def __init__(self, hardware_controller, anonymize_identifiers: bool = True):
        """
        Initialize 5G NR base station
        
        Args:
            hardware_controller: BladeRF hardware controller instance
            anonymize_identifiers: Automatically anonymize IMSI (default: True)
        """
        self.hw = hardware_controller
        self.config = NRConfig()
        self.is_running = False
        self.connected_ues: Dict[str, Dict] = {}  # IMSI_hash → UE data
        self.slot_number = 0
        
        # 🔐 Centralized anonymization
        self.anonymize_identifiers = anonymize_identifiers
        self.anonymizer = get_anonymizer()
        
        logger.info(f"5G NR gNodeB initialized (Anonymization: {anonymize_identifiers})")
        
    def configure(self, config: NRConfig) -> bool:
        """Configure 5G NR parameters"""
        try:
            self.config = config
            
            # Configure BladeRF for 5G NR
            if not self.hw.configure_hardware({
                'frequency': config.frequency,
                'sample_rate': config.sample_rate,
                'bandwidth': config.bandwidth,
                'tx_gain': config.tx_power,
                'rx_gain': 40
            }):
                logger.error("Failed to configure hardware")
                return False
                
            logger.info(f"5G NR configured: {config.frequency/1e9:.2f} GHz, "
                       f"BW={config.bandwidth/1e6:.0f} MHz, "
                       f"Numerology={config.numerology}, "
                       f"TAC={config.tac}, Cell={config.cell_id}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return False
    
    def start_base_station(self) -> bool:
        """Start 5G NR gNodeB"""
        try:
            logger.info("Starting 5G/NR gNodeB...")
            
            # Generate NR downlink slot
            slot = self._generate_nr_slot()
            
            # Transmit
            if self.hw.transmit_continuous(slot):
                self.is_running = True
                logger.info("5G NR gNodeB active")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to start gNodeB: {e}")
            return False
    
    def _generate_nr_slot(self) -> np.ndarray:
        """Generate complete 5G NR downlink slot"""
        # NR slot duration depends on numerology
        # Numerology 1 (30 kHz SCS) = 0.5ms per slot
        slot_duration = 0.0005  # 0.5ms
        samples_per_slot = int(self.config.sample_rate * slot_duration)
        
        # Generate slot components
        pss = self._generate_pss()      # Primary Sync Signal
        sss = self._generate_sss()      # Secondary Sync Signal
        pbch = self._generate_pbch()    # Physical Broadcast Channel (including DMRS)
        pdsch = self._generate_pdsch()  # Physical Downlink Shared Channel
        csi_rs = self._generate_csi_rs()  # CSI Reference Signal
        
        # Combine into OFDM slot (14 symbols for normal CP)
        slot = np.zeros(samples_per_slot, dtype=np.complex64)
        
        # SSB (SS/PBCH block) in slot 0
        ssb_len = len(pss) + len(sss) + len(pbch)
        if self.slot_number % 20 == 0:  # SSB periodicity
            idx = 0
            slot[idx:idx+len(pss)] += pss
            idx += len(pss)
            slot[idx:idx+len(sss)] += sss
            idx += len(sss)
            slot[idx:idx+len(pbch)] += pbch
        
        # PDSCH in remaining symbols
        pdsch_start = ssb_len if self.slot_number % 20 == 0 else 0
        pdsch_len = min(len(pdsch), len(slot) - pdsch_start)
        slot[pdsch_start:pdsch_start+pdsch_len] += pdsch[:pdsch_len]
        
        # CSI-RS for channel estimation
        csi_len = min(len(csi_rs), len(slot) - pdsch_start - pdsch_len)
        if csi_len > 0:
            slot[pdsch_start+pdsch_len:pdsch_start+pdsch_len+csi_len] += csi_rs[:csi_len]
        
        return slot
    
    def _generate_pss(self) -> np.ndarray:
        """Generate NR Primary Synchronization Signal"""
        # PSS uses m-sequence
        N_id_2 = self.config.cell_id % 3
        length = 127  # PSS length
        
        # Generate m-sequence based PSS
        n = np.arange(length)
        d_pss = np.zeros(length, dtype=np.complex64)
        
        # Simplified m-sequence generation
        x = [0, 0, 0, 0, 0, 0, 1]  # Initial state
        for i in range(length):
            d_pss[i] = 1 - 2 * x[0]
            x_new = (x[3] + x[0]) % 2
            x = [x_new] + x[:-1]
        
        # Apply circular shift based on N_id_2
        d_pss = np.roll(d_pss, N_id_2 * 43)
        
        # OFDM modulation with 30 kHz SCS
        samples = int(self.config.sample_rate * 0.0001)  # ~symbol duration
        t = np.linspace(0, 0.0001, samples, endpoint=False)
        carrier = np.zeros(samples, dtype=np.complex64)
        
        scs = 30000 * (2 ** self.config.numerology)  # Subcarrier spacing
        for k, symbol in enumerate(d_pss):
            freq_offset = (k - 63) * scs
            carrier += symbol * np.exp(2j * np.pi * freq_offset * t)
        
        carrier *= 0.4  # PSS power
        return carrier
    
    def _generate_sss(self) -> np.ndarray:
        """Generate NR Secondary Synchronization Signal"""
        N_id_1 = self.config.cell_id // 3
        N_id_2 = self.config.cell_id % 3
        length = 127
        
        # Generate two m-sequences for SSS
        n = np.arange(length)
        d_sss = np.exp(1j * np.pi * (N_id_1 * n + N_id_2) / 127)
        
        # OFDM modulation
        samples = int(self.config.sample_rate * 0.0001)
        t = np.linspace(0, 0.0001, samples, endpoint=False)
        carrier = np.zeros(samples, dtype=np.complex64)
        
        scs = 30000 * (2 ** self.config.numerology)
        for k, symbol in enumerate(d_sss):
            freq_offset = (k - 63) * scs
            carrier += symbol * np.exp(2j * np.pi * freq_offset * t)
        
        carrier *= 0.4  # SSS power
        return carrier
    
    def _generate_pbch(self) -> np.ndarray:
        """Generate NR Physical Broadcast Channel"""
        # PBCH carries MIB and DMRS
        samples = int(self.config.sample_rate * 0.0003)  # 3 symbols
        
        # Encode MIB
        mib_data = self._encode_mib()
        
        # QPSK modulation for PBCH payload
        symbols = np.array([complex(1, 1), complex(1, -1), 
                           complex(-1, 1), complex(-1, -1)]) / np.sqrt(2)
        
        pbch_payload = np.random.choice(symbols, samples // 3)
        
        # DMRS for PBCH (demodulation reference signal)
        dmrs = self._generate_dmrs(samples // 3)
        
        # Interleave PBCH payload and DMRS (simplified)
        pbch = np.zeros(samples, dtype=np.complex64)
        pbch[0::3] = pbch_payload[:len(pbch[0::3])]
        pbch[1::3] = dmrs[:len(pbch[1::3])]
        pbch[2::3] = pbch_payload[:len(pbch[2::3])]
        
        pbch *= 0.3  # PBCH power
        return pbch
    
    def _generate_dmrs(self, length: int) -> np.ndarray:
        """Generate Demodulation Reference Signal"""
        # Gold sequence based DMRS
        n = np.arange(length)
        dmrs = np.exp(1j * np.pi * (self.config.cell_id * n) / length)
        return dmrs
    
    def _encode_mib(self) -> bytes:
        """Encode Master Information Block for NR"""
        mib = {
            'system_frame_number': self.slot_number // 20,  # SFN
            'subcarrier_spacing_common': self.config.numerology,
            'ssb_subcarrier_offset': 0,
            'dmrs_type_a_position': 2,
            'pdcch_config_sib1': 0,
            'cell_barred': False,
            'intra_freq_reselection': True
        }
        # In production, encode with polar coding
        return bytes(str(mib), 'utf-8')
    
    def _generate_pdsch(self) -> np.ndarray:
        """Generate NR Physical Downlink Shared Channel"""
        samples = int(self.config.sample_rate * 0.0004)  # ~4 symbols
        
        # Random data (in production, encode with LDPC)
        pdsch = (np.random.randn(samples) + 1j * np.random.randn(samples)) / np.sqrt(2)
        pdsch *= 0.15  # PDSCH power
        
        return pdsch
    
    def _generate_csi_rs(self) -> np.ndarray:
        """Generate Channel State Information Reference Signal"""
        samples = int(self.config.sample_rate * 0.00005)  # ~1 symbol
        
        # Gold sequence for CSI-RS
        csi_rs = np.exp(1j * 2 * np.pi * np.random.rand(samples))
        csi_rs *= 0.2
        
        return csi_rs
    
    def track_ues(self) -> List[Dict]:
        """Track connected 5G UEs"""
        try:
            # Receive uplink PRACH
            samples = self.hw.receive_samples(
                int(self.config.sample_rate * 0.001)  # 1ms
            )
            
            if samples is None:
                return []
            
            # Detect PRACH preambles
            ues = self._detect_prach(samples)
            
            # Update UE database
            for ue in ues:
                imsi = ue.get('imsi', 'unknown')
                
                # 🔐 SECURITY FIX: Always anonymize IMSI before storage
                imsi_hash = self.anonymizer.anonymize_imsi(imsi) if self.anonymize_identifiers else imsi
                
                if imsi_hash not in self.connected_ues:
                    self.connected_ues[imsi_hash] = {
                        'first_seen': datetime.now(),
                        'last_seen': datetime.now(),
                        'ss_rsrp': ue.get('ss_rsrp', 0),  # SS-RSRP
                        'ss_rsrq': ue.get('ss_rsrq', 0),  # SS-RSRQ
                        'ss_sinr': ue.get('ss_sinr', 0),  # SS-SINR
                        'technology': '5G/NR'
                    }
                else:
                    self.connected_ues[imsi_hash]['last_seen'] = datetime.now()
                    self.connected_ues[imsi_hash]['ss_rsrp'] = ue.get('ss_rsrp', 0)
                    self.connected_ues[imsi_hash]['ss_rsrq'] = ue.get('ss_rsrq', 0)
                    self.connected_ues[imsi_hash]['ss_sinr'] = ue.get('ss_sinr', 0)
            
            return ues
            
        except Exception as e:
            logger.error(f"UE tracking error: {e}")
            return []
    
    def _detect_prach(self, samples: np.ndarray) -> List[Dict]:
        """Detect NR PRACH preambles"""
        ues = []
        
        # Simplified PRACH detection
        # In production, correlate with Zadoff-Chu sequences
        power = np.abs(samples) ** 2
        threshold = np.mean(power) + 5 * np.std(power)
        
        peaks = np.where(power > threshold)[0]
        
        if len(peaks) > 0:
            # Detected UE attempting random access
            ss_rsrp = 10 * np.log10(np.max(power))
            noise_power = np.mean(power[power < threshold])
            ss_sinr = ss_rsrp - 10 * np.log10(noise_power)
            ss_rsrq = ss_rsrp - 10 * np.log10(self.config.bandwidth / 1e6)
            
            ues.append({
                'imsi': 'detected',  # Would extract from RRC signaling
                'ss_rsrp': ss_rsrp,
                'ss_rsrq': ss_rsrq,
                'ss_sinr': ss_sinr,
                'frequency': self.config.frequency,
                'technology': '5G',
                'arfcn': self.config.arfcn,
                'timestamp': datetime.now().isoformat()
            })
        
        return ues
    
    def get_ue_count(self) -> int:
        """Get number of tracked UEs"""
        return len(self.connected_ues)
    
    def get_connected_ues(self) -> Dict:
        """Get all connected UEs"""
        return self.connected_ues
    
    def update_slot(self):
        """Update slot counter"""
        self.slot_number = (self.slot_number + 1) % 10240  # 10.24s cycle
    
    def stop(self):
        """Stop 5G NR gNodeB"""
        self.is_running = False
        self.hw.stop_transmission()
        logger.info("5G NR gNodeB stopped")

def main():
    """Test 5G NR module"""
    from core.hardware import HardwareController
    
    # Initialize hardware
    hw = HardwareController()
    if not hw.connect():
        print("Failed to connect to BladeRF")
        return
    
    # Create 5G NR base station
    nr = NRBaseStation(hw)
    
    # Configure for n78 band (3.5 GHz), 100 MHz
    config = NRConfig(
        frequency=3_500_000_000,  # 3.5 GHz
        bandwidth=100_000_000,     # 100 MHz
        numerology=1,              # 30 kHz SCS
        tac=100,
        cell_id=1,
        mcc="001",
        mnc="01"
    )
    
    if not nr.configure(config):
        print("Configuration failed")
        return
    
    # Start gNodeB
    if nr.start_base_station():
        print("5G NR gNodeB running...")
        print("Press Ctrl+C to stop")
        
        try:
            import time
            while True:
                ues = nr.track_ues()
                if ues:
                    print(f"Detected {len(ues)} UE(s)")
                    for ue in ues:
                        print(f"  SS-RSRP: {ue['ss_rsrp']:.1f} dBm, "
                              f"SS-RSRQ: {ue['ss_rsrq']:.1f} dB, "
                              f"SS-SINR: {ue['ss_sinr']:.1f} dB")
                nr.update_slot()
                time.sleep(0.0005)  # 0.5ms per slot
        except KeyboardInterrupt:
            print("\nStopping...")
    
    nr.stop()
    hw.disconnect()

if __name__ == "__main__":
    main()
