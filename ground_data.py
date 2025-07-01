import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, Optional, List

class GroundDataCollector:
    """
    Handles collection of ground-based air quality measurements from CPCB and other sources
    Collects PM2.5, PM10, NO2, SO2, CO, and other pollutants
    """
    
    def __init__(self):
        # API endpoints for ground monitoring data
        self.base_urls = {
            'cpcb': 'https://api.cpcb.nic.in/air',
            'openaq': 'https://api.openaq.org/v2',
            'aqicn': 'https://api.waqi.info'
        }
        
        # API keys from environment
        self.api_keys = {
            'cpcb': os.getenv('CPCB_API_KEY', ''),
            'openaq': os.getenv('OPENAQ_API_KEY', ''),
            'aqicn': os.getenv('AQICN_API_KEY', 'demo')
        }
        
        # Station mapping for different cities
        self.station_mapping = {
            'Patna, Bihar': {
                'cpcb_id': 'PB001',
                'lat': 25.5941, 'lon': 85.1376,
                'stations': ['Patna Collectorate', 'Patna University', 'IGSC Patna']
            },
            'Varanasi, UP': {
                'cpcb_id': 'UP001',
                'lat': 25.3176, 'lon': 82.9739,
                'stations': ['Varanasi Cantonment', 'BHU Varanasi']
            },
            'Guwahati, Assam': {
                'cpcb_id': 'AS001',
                'lat': 26.1445, 'lon': 91.7362,
                'stations': ['Guwahati Central', 'IIT Guwahati']
            },
            'Shimla, HP': {
                'cpcb_id': 'HP001',
                'lat': 31.1048, 'lon': 77.1734,
                'stations': ['Shimla Mall Road', 'Shimla Ridge']
            },
            'Dehradun, UK': {
                'cpcb_id': 'UK001',
                'lat': 30.3165, 'lon': 78.0322,
                'stations': ['Dehradun Clock Tower', 'FRI Dehradun']
            }
        }
        
        # Cache for recent data
        self.cache = {}
        self.cache_timeout = 900  # 15 minutes for ground data
    
    def get_current_data(self, location_coords: Dict[str, float]) -> Optional[Dict]:
        """
        Get current air quality measurements for the specified location
        
        Args:
            location_coords: Dictionary with 'lat' and 'lon' keys
            
        Returns:
            Dictionary with current pollutant measurements
        """
        cache_key = f"current_ground_{location_coords['lat']}_{location_coords['lon']}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Try CPCB data first
            ground_data = self._fetch_cpcb_data(location_coords)
            
            if not ground_data:
                # Fallback to OpenAQ
                ground_data = self._fetch_openaq_data(location_coords)
            
            if not ground_data:
                # Fallback to AQI.cn
                ground_data = self._fetch_aqicn_data(location_coords)
            
            if not ground_data:
                # Generate realistic data for demonstration
                ground_data = self._generate_realistic_ground_data(location_coords)
            
            # Cache the result
            self.cache[cache_key] = {
                'data': ground_data,
                'timestamp': datetime.now()
            }
            
            return ground_data
            
        except Exception as e:
            print(f"Error fetching current ground data: {e}")
            return self._generate_realistic_ground_data(location_coords)
    
    def get_historical_data(self, location_coords: Dict[str, float], hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Get historical air quality measurements
        
        Args:
            location_coords: Dictionary with 'lat' and 'lon' keys
            hours: Number of hours of historical data to fetch
            
        Returns:
            DataFrame with timestamp and pollutant columns
        """
        cache_key = f"historical_ground_{location_coords['lat']}_{location_coords['lon']}_{hours}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Try to fetch from CPCB historical API
            historical_data = self._fetch_cpcb_historical(location_coords, hours)
            
            if historical_data is None or historical_data.empty:
                # Generate realistic historical data
                historical_data = self._generate_historical_ground_data(location_coords, hours)
            
            # Cache the result
            self.cache[cache_key] = {
                'data': historical_data,
                'timestamp': datetime.now()
            }
            
            return historical_data
            
        except Exception as e:
            print(f"Error fetching historical ground data: {e}")
            return self._generate_historical_ground_data(location_coords, hours)
    
    def get_station_info(self, location_coords: Dict[str, float]) -> Optional[Dict]:
        """
        Get information about monitoring stations near the location
        """
        try:
            # Find the closest city from our mapping
            closest_city = self._find_closest_city(location_coords)
            
            if closest_city:
                station_info = self.station_mapping[closest_city].copy()
                station_info['city'] = closest_city
                return station_info
            
            return None
            
        except Exception as e:
            print(f"Error getting station info: {e}")
            return None
    
    def _fetch_cpcb_data(self, location_coords: Dict[str, float]) -> Optional[Dict]:
        """
        Fetch data from CPCB (Central Pollution Control Board) API
        """
        try:
            # Find closest station
            station_info = self.get_station_info(location_coords)
            if not station_info:
                return None
            
            # CPCB API endpoint (note: actual CPCB API may require authentication)
            url = f"{self.base_urls['cpcb']}/latest"
            
            params = {
                'station_id': station_info['cpcb_id'],
                'api_key': self.api_keys['cpcb']
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'measurements' in data:
                    measurements = data['measurements']
                    return {
                        'pm25': measurements.get('pm25', 0),
                        'pm10': measurements.get('pm10', 0),
                        'no2': measurements.get('no2', 0),
                        'so2': measurements.get('so2', 0),
                        'co': measurements.get('co', 0),
                        'ozone': measurements.get('o3', 0),
                        'timestamp': datetime.now(),
                        'station': station_info['stations'][0],
                        'source': 'CPCB'
                    }
            
            return None
            
        except Exception as e:
            print(f"CPCB data fetch error: {e}")
            return None
    
    def _fetch_openaq_data(self, location_coords: Dict[str, float]) -> Optional[Dict]:
        """
        Fetch data from OpenAQ API
        """
        try:
            url = f"{self.base_urls['openaq']}/latest"
            
            params = {
                'coordinates': f"{location_coords['lat']},{location_coords['lon']}",
                'radius': '50000',  # 50km radius
                'limit': '1'
            }
            
            headers = {
                'X-API-Key': self.api_keys['openaq']
            } if self.api_keys['openaq'] else {}
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    result = data['results'][0]
                    measurements = {}
                    
                    for measurement in result.get('measurements', []):
                        param = measurement['parameter']
                        value = measurement['value']
                        
                        if param == 'pm25':
                            measurements['pm25'] = value
                        elif param == 'pm10':
                            measurements['pm10'] = value
                        elif param == 'no2':
                            measurements['no2'] = value
                        elif param == 'so2':
                            measurements['so2'] = value
                        elif param == 'co':
                            measurements['co'] = value / 1000  # Convert to mg/m³
                    
                    return {
                        'pm25': measurements.get('pm25', 0),
                        'pm10': measurements.get('pm10', 0),
                        'no2': measurements.get('no2', 0),
                        'so2': measurements.get('so2', 0),
                        'co': measurements.get('co', 0),
                        'ozone': 0,
                        'timestamp': datetime.now(),
                        'station': result.get('location', 'Unknown'),
                        'source': 'OpenAQ'
                    }
            
            return None
            
        except Exception as e:
            print(f"OpenAQ data fetch error: {e}")
            return None
    
    def _fetch_aqicn_data(self, location_coords: Dict[str, float]) -> Optional[Dict]:
        """
        Fetch data from AQI.cn (World Air Quality Index) API
        """
        try:
            url = f"{self.base_urls['aqicn']}/feed/geo:{location_coords['lat']};{location_coords['lon']}/"
            
            params = {
                'token': self.api_keys['aqicn']
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'ok' and 'data' in data:
                    iaqi = data['data'].get('iaqi', {})
                    
                    return {
                        'pm25': iaqi.get('pm25', {}).get('v', 0),
                        'pm10': iaqi.get('pm10', {}).get('v', 0),
                        'no2': iaqi.get('no2', {}).get('v', 0),
                        'so2': iaqi.get('so2', {}).get('v', 0),
                        'co': iaqi.get('co', {}).get('v', 0) / 10 if iaqi.get('co') else 0,  # Convert to mg/m³
                        'ozone': iaqi.get('o3', {}).get('v', 0),
                        'timestamp': datetime.now(),
                        'station': data['data'].get('city', {}).get('name', 'Unknown'),
                        'source': 'AQI.cn'
                    }
            
            return None
            
        except Exception as e:
            print(f"AQI.cn data fetch error: {e}")
            return None
    
    def _fetch_cpcb_historical(self, location_coords: Dict[str, float], hours: int) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from CPCB API
        """
        try:
            # This would require proper CPCB historical data API
            # For now, return None to trigger fallback
            return None
            
        except Exception as e:
            print(f"CPCB historical fetch error: {e}")
            return None
    
    def _generate_realistic_ground_data(self, location_coords: Dict[str, float]) -> Dict:
        """
        Generate realistic ground pollution data based on location and time patterns
        """
        return self._generate_realistic_ground_data_for_time(location_coords, datetime.now())
    
    def _generate_realistic_ground_data_for_time(self, location_coords: Dict[str, float], target_time: datetime) -> Dict:
        """
        Generate realistic ground pollution data based on location and time patterns
        """
        lat, lon = location_coords['lat'], location_coords['lon']
        current_time = target_time
        
        # Base pollution levels for different regions
        if 25 <= lat <= 30 and 75 <= lon <= 85:  # Northern plains
            base_pm25 = 65
            base_pm10 = 110
            base_no2 = 45
        elif 20 <= lat <= 25:  # Central India
            base_pm25 = 50
            base_pm10 = 85
            base_no2 = 35
        elif lat > 30:  # Himalayan region
            base_pm25 = 25
            base_pm10 = 45
            base_no2 = 20
        else:  # Southern/coastal regions
            base_pm25 = 40
            base_pm10 = 70
            base_no2 = 30
        
        # Seasonal variation
        month = current_time.month
        if month in [11, 12, 1, 2]:  # Winter - higher pollution
            seasonal_factor = 1.8
        elif month in [3, 4, 5]:  # Pre-monsoon
            seasonal_factor = 1.4
        elif month in [6, 7, 8, 9]:  # Monsoon - cleaner air
            seasonal_factor = 0.5
        else:  # Post-monsoon
            seasonal_factor = 1.1
        
        # Daily variation
        hour = current_time.hour
        if 7 <= hour <= 10 or 18 <= hour <= 22:  # Peak traffic hours
            daily_factor = 1.5
        elif 11 <= hour <= 16:  # Afternoon
            daily_factor = 1.0
        else:  # Night/early morning
            daily_factor = 0.7
        
        # Weekend effect (slightly lower on weekends)
        weekend_factor = 0.8 if current_time.weekday() >= 5 else 1.0
        
        # Random variation
        random_factor = np.random.uniform(0.7, 1.3)
        
        # Calculate pollutant values
        pm25 = base_pm25 * seasonal_factor * daily_factor * weekend_factor * random_factor
        pm10 = base_pm10 * seasonal_factor * daily_factor * weekend_factor * random_factor
        no2 = base_no2 * seasonal_factor * daily_factor * weekend_factor * random_factor
        
        # Other pollutants (correlated with primary pollutants)
        so2 = no2 * 0.3 + np.random.uniform(5, 15)
        co = pm25 * 0.04 + np.random.uniform(0.5, 2.0)  # in mg/m³
        ozone = max(0, 80 - pm25 * 0.5 + np.random.uniform(-20, 20))  # Inverse correlation with PM
        
        return {
            'pm25': max(5, min(500, pm25)),
            'pm10': max(10, min(1000, pm10)),
            'no2': max(5, min(200, no2)),
            'so2': max(2, min(100, so2)),
            'co': max(0.1, min(10, co)),
            'ozone': max(0, min(300, ozone)),
            'timestamp': current_time,
            'station': self._get_station_name(location_coords),
            'source': 'Synthetic'
        }
    
    def _generate_historical_ground_data(self, location_coords: Dict[str, float], hours: int) -> pd.DataFrame:
        """
        Generate realistic historical ground pollution data
        """
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=hours),
            end=datetime.now(),
            freq='H'
        )
        
        data_points = []
        for date in dates:
            # Generate data for specific time
            data_point = self._generate_realistic_ground_data_for_time(location_coords, date)
            data_point['timestamp'] = date
            data_points.append(data_point)
        
        return pd.DataFrame(data_points)
    
    def _find_closest_city(self, location_coords: Dict[str, float]) -> Optional[str]:
        """
        Find the closest city from our station mapping
        """
        min_distance = float('inf')
        closest_city = None
        
        for city, info in self.station_mapping.items():
            # Calculate distance using simple Euclidean distance
            distance = ((location_coords['lat'] - info['lat']) ** 2 + 
                       (location_coords['lon'] - info['lon']) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_city = city
        
        return closest_city if min_distance < 1.0 else None  # Within ~100km
    
    def _get_station_name(self, location_coords: Dict[str, float]) -> str:
        """
        Get a station name for the location
        """
        station_info = self.get_station_info(location_coords)
        if station_info and station_info['stations']:
            return station_info['stations'][0]
        return "Local Monitoring Station"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        age = (datetime.now() - self.cache[cache_key]['timestamp']).total_seconds()
        return age < self.cache_timeout
