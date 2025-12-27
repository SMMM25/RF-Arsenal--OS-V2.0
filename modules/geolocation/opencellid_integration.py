#!/usr/bin/env python3
"""
RF Arsenal OS - OpenCellID Database Integration
Cell tower location database access
"""

import logging
import requests
import json
import sqlite3
import os
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class OpenCellIDIntegration:
    """
    Integration with OpenCellID database
    
    Provides cell tower GPS coordinates for:
    - MCC (Mobile Country Code)
    - MNC (Mobile Network Code)
    - LAC (Location Area Code)
    - Cell ID
    
    Data sources:
    - OpenCellID.org (free API with token)
    - Mozilla Location Services
    - Local cached database
    """
    
    def __init__(self, api_token: Optional[str] = None, db_path: str = "data/cells.db"):
        """
        Initialize OpenCellID integration
        
        Args:
            api_token: OpenCellID API token (optional, for live queries)
            db_path: Path to local SQLite database
        """
        self.api_token = api_token
        self.db_path = db_path
        self.base_url = "https://opencellid.org/ajax"
        self.mozilla_url = "https://location.services.mozilla.com/v1/geolocate"
        
        # Initialize local database
        self._init_database()
        
        logger.info(f"OpenCellID integration initialized (DB: {db_path})")
    
    def _init_database(self):
        """Initialize local SQLite database for caching"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create cells table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cells (
                cell_key TEXT PRIMARY KEY,
                mcc INTEGER,
                mnc INTEGER,
                lac INTEGER,
                cell_id TEXT,
                latitude REAL,
                longitude REAL,
                range INTEGER,
                samples INTEGER,
                updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Cell database initialized")
    
    def _make_cell_key(self, mcc: int, mnc: int, lac: int, cell_id: str) -> str:
        """Generate unique key for cell tower"""
        return f"{mcc}-{mnc}-{lac}-{cell_id}"
    
    def lookup_cell(self, mcc: int, mnc: int, lac: int, cell_id: str) -> Optional[Dict]:
        """
        Look up cell tower location
        
        Tries:
        1. Local cache
        2. OpenCellID API (if token available)
        3. Mozilla Location Services
        
        Args:
            mcc: Mobile Country Code
            mnc: Mobile Network Code
            lac: Location Area Code
            cell_id: Cell ID
            
        Returns:
            Dict with lat, lon, range, or None if not found
        """
        cell_key = self._make_cell_key(mcc, mnc, lac, cell_id)
        
        # Try local cache first
        cached = self._lookup_cache(cell_key)
        if cached:
            logger.debug(f"Cell {cell_key} found in cache")
            return cached
        
        # Try OpenCellID API
        if self.api_token:
            result = self._lookup_opencellid(mcc, mnc, lac, cell_id)
            if result:
                self._cache_cell(cell_key, mcc, mnc, lac, cell_id, result)
                return result
        
        # Try Mozilla Location Services
        result = self._lookup_mozilla(mcc, mnc, lac, cell_id)
        if result:
            self._cache_cell(cell_key, mcc, mnc, lac, cell_id, result)
            return result
        
        logger.warning(f"Cell {cell_key} not found in any database")
        return None
    
    def _lookup_cache(self, cell_key: str) -> Optional[Dict]:
        """Look up cell in local cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT latitude, longitude, range, samples
                FROM cells
                WHERE cell_key = ?
            ''', (cell_key,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'lat': row[0],
                    'lon': row[1],
                    'range': row[2],
                    'samples': row[3]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
            return None
    
    def _lookup_opencellid(self, mcc: int, mnc: int, lac: int, cell_id: str) -> Optional[Dict]:
        """Look up cell using OpenCellID API"""
        if not self.api_token:
            return None
        
        try:
            params = {
                'key': self.api_token,
                'mcc': mcc,
                'mnc': mnc,
                'lac': lac,
                'cellid': cell_id,
                'format': 'json'
            }
            
            response = requests.get(
                f"{self.base_url}/searchCell.php",
                params=params,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('lat') and data.get('lon'):
                    logger.info(f"OpenCellID: Found cell {mcc}-{mnc}-{lac}-{cell_id}")
                    return {
                        'lat': float(data['lat']),
                        'lon': float(data['lon']),
                        'range': int(data.get('range', 5000)),
                        'samples': int(data.get('samples', 1))
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"OpenCellID lookup failed: {e}")
            return None
    
    def _lookup_mozilla(self, mcc: int, mnc: int, lac: int, cell_id: str) -> Optional[Dict]:
        """Look up cell using Mozilla Location Services"""
        try:
            payload = {
                'radioType': 'gsm',
                'cellTowers': [{
                    'mobileCountryCode': mcc,
                    'mobileNetworkCode': mnc,
                    'locationAreaCode': lac,
                    'cellId': int(cell_id) if cell_id.isdigit() else 0
                }]
            }
            
            response = requests.post(
                f"{self.mozilla_url}?key=test",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'location' in data:
                    logger.info(f"Mozilla: Found cell {mcc}-{mnc}-{lac}-{cell_id}")
                    return {
                        'lat': data['location']['lat'],
                        'lon': data['location']['lng'],
                        'range': int(data.get('accuracy', 5000)),
                        'samples': 1
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"Mozilla lookup failed: {e}")
            return None
    
    def _cache_cell(self, cell_key: str, mcc: int, mnc: int, lac: int, 
                   cell_id: str, data: Dict):
        """Cache cell tower data locally"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO cells
                (cell_key, mcc, mnc, lac, cell_id, latitude, longitude, range, samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cell_key,
                mcc,
                mnc,
                lac,
                cell_id,
                data['lat'],
                data['lon'],
                data.get('range', 5000),
                data.get('samples', 1)
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Cached cell {cell_key}")
            
        except Exception as e:
            logger.error(f"Cache write failed: {e}")
    
    def import_csv(self, csv_path: str, limit: Optional[int] = None) -> int:
        """
        Import OpenCellID CSV database
        
        Download from: https://opencellid.org/downloads.php
        
        Args:
            csv_path: Path to CSV file
            limit: Optional limit on rows to import
            
        Returns:
            Number of cells imported
        """
        try:
            import csv
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            count = 0
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    if limit and count >= limit:
                        break
                    
                    try:
                        mcc = int(row['mcc'])
                        mnc = int(row['net'])
                        lac = int(row['area'])
                        cell_id = row['cell']
                        lat = float(row['lat'])
                        lon = float(row['lon'])
                        range_m = int(row.get('range', 5000))
                        samples = int(row.get('samples', 1))
                        
                        cell_key = self._make_cell_key(mcc, mnc, lac, cell_id)
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO cells
                            (cell_key, mcc, mnc, lac, cell_id, latitude, longitude, range, samples)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (cell_key, mcc, mnc, lac, cell_id, lat, lon, range_m, samples))
                        
                        count += 1
                        
                        if count % 10000 == 0:
                            conn.commit()
                            logger.info(f"Imported {count} cells...")
                        
                    except Exception as e:
                        logger.debug(f"Skipped invalid row: {e}")
                        continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Imported {count} cells from CSV")
            return count
            
        except Exception as e:
            logger.error(f"CSV import failed: {e}")
            return 0
    
    def get_database_stats(self) -> Dict:
        """Get statistics about local database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total cells
            cursor.execute('SELECT COUNT(*) FROM cells')
            total_cells = cursor.fetchone()[0]
            
            # Cells by country
            cursor.execute('''
                SELECT mcc, COUNT(*) 
                FROM cells 
                GROUP BY mcc 
                ORDER BY COUNT(*) DESC 
                LIMIT 10
            ''')
            top_countries = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_cells': total_cells,
                'top_countries': [{'mcc': mcc, 'count': count} for mcc, count in top_countries],
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            }
            
        except Exception as e:
            logger.error(f"Stats query failed: {e}")
            return {}
    
    def search_nearby_cells(self, lat: float, lon: float, radius_km: float = 10) -> List[Dict]:
        """
        Search for cells near a location
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            
        Returns:
            List of nearby cells
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Simple bounding box search
            # 1 degree latitude ≈ 111 km
            lat_delta = radius_km / 111.0
            lon_delta = radius_km / (111.0 * abs(np.cos(np.radians(lat))))
            
            cursor.execute('''
                SELECT cell_key, mcc, mnc, lac, cell_id, latitude, longitude, range
                FROM cells
                WHERE latitude BETWEEN ? AND ?
                AND longitude BETWEEN ? AND ?
            ''', (
                lat - lat_delta,
                lat + lat_delta,
                lon - lon_delta,
                lon + lon_delta
            ))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'cell_key': row[0],
                    'mcc': row[1],
                    'mnc': row[2],
                    'lac': row[3],
                    'cell_id': row[4],
                    'lat': row[5],
                    'lon': row[6],
                    'range': row[7]
                })
            
            conn.close()
            
            logger.info(f"Found {len(results)} cells within {radius_km}km of {lat:.4f}, {lon:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Nearby search failed: {e}")
            return []


if __name__ == "__main__":
    # Test OpenCellID integration
    logging.basicConfig(level=logging.INFO)
    
    print("RF Arsenal OS - OpenCellID Integration Test")
    print("=" * 50)
    
    # Initialize (no API token for test)
    opencell = OpenCellIDIntegration()
    
    # Test lookup
    print("\n[+] Testing cell lookup...")
    result = opencell.lookup_cell(
        mcc=310,
        mnc=410,
        lac=7033,
        cell_id="20033"
    )
    
    if result:
        print(f"[+] Cell found:")
        print(f"    Location: {result['lat']:.6f}, {result['lon']:.6f}")
        print(f"    Range: {result['range']}m")
    else:
        print("[-] Cell not found (expected without API token)")
    
    # Get stats
    print("\n[+] Database statistics:")
    stats = opencell.get_database_stats()
    print(f"    Total cells: {stats.get('total_cells', 0)}")
    print(f"    Database size: {stats.get('database_size_mb', 0):.2f} MB")
    
    print("\n[+] Test complete")
