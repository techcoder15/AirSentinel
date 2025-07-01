import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, Optional, List

class WeatherDataCollector:
    """
    Handles collection of meteorological data from CAMS/ERA5 and other weather services
    Focuses on weather parameters that affect air pollution dispersion
    """
    
    def __init__(self):
        # API endpoints for weather data
        self.base_urls = {
            'openweather': 'https://api.openweathermap.org/data/2.5',
            'cams': 'https://ads.atmosphere.copernicus.eu/api/v2',
            'era5': 'https://cds.climate.copernicus.eu/api/v2',
            'ecmwf': 'https://api.ecmwf.int/v1'
        }
        
        # API keys from environment
        self.api_keys = {
            'openweather': os.getenv('OPENWEATHER_API_KEY', ''),
            'copernicus': os.getenv('COPERNICUS_API_KEY', ''),
            'ecmwf': os.getenv('ECMWF_API_KEY', '')
        }
        
        # Cache for weather data
        self.cache = {}
        self.cache_timeout = 1800  # 30 minutes
    
    def get_current_weather(self, location_coords: Dict[str, float]) -> Optional[Dict]:
        """
        Get current weather conditions for the specified location
        
        Args:
            location_coords: Dictionary with 'lat' and 'lon' keys
            
        Returns:
            Dictionary with current weather parameters
        """
        cache_key = f"current_weather_{location_coords['lat']}_{location_coords['lon']}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Try OpenWeatherMap first
            weather_data = self._fetch_openweather_current(location_coords)
            
            if not weather_data:
                # Generate realistic weather data for demonstration
                weather_data = self._generate_realistic_weather(location_coords)
            
            # Cache the result
            self.cache[cache_key] = {
                'data': weather_data,
                'timestamp': datetime.now()
            }
            
            return weather_data
            
        except Exception as e:
            print(f"Error fetching current weather: {e}")
            return self._generate_realistic_weather(location_coords)
    
    def get_historical_weather(self, location_coords: Dict[str, float], hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Get historical weather data
        
        Args:
            location_coords: Dictionary with 'lat' and 'lon' keys
            hours: Number of hours of historical data
            
        Returns:
            DataFrame with timestamp and weather parameters
        """
        cache_key = f"historical_weather_{location_coords['lat']}_{location_coords['lon']}_{hours}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Try to fetch from OpenWeatherMap historical API
            historical_data = self._fetch_openweather_historical(location_coords, hours)
            
            if historical_data is None or historical_data.empty:
                # Generate realistic historical weather data
                historical_data = self._generate_historical_weather(location_coords, hours)
            
            # Cache the result
            self.cache[cache_key] = {
                'data': historical_data,
                'timestamp': datetime.now()
            }
            
            return historical_data
            
        except Exception as e:
            print(f"Error fetching historical weather: {e}")
            return self._generate_historical_weather(location_coords, hours)
    
    def get_cams_atmospheric_data(self, location_coords: Dict[str, float]) -> Optional[Dict]:
        """
        Get atmospheric composition data from CAMS (Copernicus Atmosphere Monitoring Service)
        
        Returns:
            Dictionary with atmospheric parameters relevant to pollution
        """
        try:
            # CAMS API endpoint for atmospheric composition
            url = f"{self.base_urls['cams']}/resources"
            
            params = {
                'product_type': 'analysis',
                'variable': ['particulate_matter_2.5um', 'nitrogen_dioxide', 'ozone'],
                'pressure_level': '1000',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:00'),
                'area': [
                    location_coords['lat'] + 0.1,  # North
                    location_coords['lon'] - 0.1,  # West
                    location_coords['lat'] - 0.1,  # South
                    location_coords['lon'] + 0.1   # East
                ],
                'format': 'json'
            }
            
            headers = {
                'Authorization': f"Bearer {self.api_keys['copernicus']}"
            } if self.api_keys['copernicus'] else {}
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                # Process CAMS data (this would need proper NetCDF parsing in production)
                return {
                    'atmospheric_pm25': data.get('pm25_concentration', 0),
                    'atmospheric_no2': data.get('no2_concentration', 0),
                    'atmospheric_ozone': data.get('ozone_concentration', 0),
                    'boundary_layer_height': data.get('boundary_layer_height', 1000),
                    'source': 'CAMS'
                }
            
            return None
            
        except Exception as e:
            print(f"CAMS data fetch error: {e}")
            return None
    
    def get_forecast_weather(self, location_coords: Dict[str, float], hours: int = 48) -> Optional[pd.DataFrame]:
        """
        Get weather forecast data
        
        Args:
            location_coords: Dictionary with 'lat' and 'lon' keys
            hours: Number of hours to forecast
            
        Returns:
            DataFrame with forecasted weather parameters
        """
        try:
            # Try OpenWeatherMap forecast API
            forecast_data = self._fetch_openweather_forecast(location_coords, hours)
            
            if forecast_data is None or forecast_data.empty:
                # Generate realistic forecast data
                forecast_data = self._generate_forecast_weather(location_coords, hours)
            
            return forecast_data
            
        except Exception as e:
            print(f"Error fetching weather forecast: {e}")
            return self._generate_forecast_weather(location_coords, hours)
    
    def _fetch_openweather_current(self, location_coords: Dict[str, float]) -> Optional[Dict]:
        """
        Fetch current weather from OpenWeatherMap API
        """
        try:
            url = f"{self.base_urls['openweather']}/weather"
            
            params = {
                'lat': location_coords['lat'],
                'lon': location_coords['lon'],
                'appid': self.api_keys['openweather'],
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data['wind'].get('speed', 0),
                    'wind_direction': data['wind'].get('deg', 0),
                    'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                    'cloud_coverage': data['clouds']['all'],
                    'weather_condition': data['weather'][0]['main'],
                    'timestamp': datetime.now(),
                    'source': 'OpenWeatherMap'
                }
            
            return None
            
        except Exception as e:
            print(f"OpenWeatherMap current fetch error: {e}")
            return None
    
    def _fetch_openweather_historical(self, location_coords: Dict[str, float], hours: int) -> Optional[pd.DataFrame]:
        """
        Fetch historical weather from OpenWeatherMap API
        """
        try:
            # OpenWeatherMap historical API requires subscription
            # For production, implement proper historical data retrieval
            return None
            
        except Exception as e:
            print(f"OpenWeatherMap historical fetch error: {e}")
            return None
    
    def _fetch_openweather_forecast(self, location_coords: Dict[str, float], hours: int) -> Optional[pd.DataFrame]:
        """
        Fetch weather forecast from OpenWeatherMap API
        """
        try:
            url = f"{self.base_urls['openweather']}/forecast"
            
            params = {
                'lat': location_coords['lat'],
                'lon': location_coords['lon'],
                'appid': self.api_keys['openweather'],
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                forecast_list = []
                for item in data['list'][:min(len(data['list']), hours//3)]:  # 3-hour intervals
                    forecast_list.append({
                        'timestamp': datetime.fromtimestamp(item['dt']),
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'pressure': item['main']['pressure'],
                        'wind_speed': item['wind'].get('speed', 0),
                        'wind_direction': item['wind'].get('deg', 0),
                        'cloud_coverage': item['clouds']['all'],
                        'weather_condition': item['weather'][0]['main']
                    })
                
                return pd.DataFrame(forecast_list)
            
            return None
            
        except Exception as e:
            print(f"OpenWeatherMap forecast fetch error: {e}")
            return None
    
    def _generate_realistic_weather(self, location_coords: Dict[str, float]) -> Dict:
        """
        Generate realistic weather data based on location and seasonal patterns
        """
        return self._generate_realistic_weather_for_time(location_coords, datetime.now())
    
    def _generate_realistic_weather_for_time(self, location_coords: Dict[str, float], target_time: datetime) -> Dict:
        """
        Generate realistic weather data based on location and seasonal patterns
        """
        lat, lon = location_coords['lat'], location_coords['lon']
        current_time = target_time
        
        # Seasonal temperature patterns for India
        month = current_time.month
        if month in [12, 1, 2]:  # Winter
            if lat > 25:  # Northern regions
                base_temp = np.random.uniform(8, 18)
            else:  # Southern regions
                base_temp = np.random.uniform(18, 28)
        elif month in [3, 4, 5]:  # Pre-monsoon/Summer
            if lat > 25:
                base_temp = np.random.uniform(25, 40)
            else:
                base_temp = np.random.uniform(28, 35)
        elif month in [6, 7, 8, 9]:  # Monsoon
            base_temp = np.random.uniform(22, 32)
        else:  # Post-monsoon
            base_temp = np.random.uniform(20, 30)
        
        # Daily temperature variation
        hour = current_time.hour
        daily_variation = 5 * np.sin((hour - 6) * np.pi / 12)  # Peak at 2 PM
        temperature = base_temp + daily_variation
        
        # Humidity patterns
        if month in [6, 7, 8, 9]:  # Monsoon - high humidity
            humidity = np.random.uniform(70, 95)
        elif month in [3, 4, 5]:  # Summer - low humidity
            humidity = np.random.uniform(30, 60)
        else:  # Winter/post-monsoon
            humidity = np.random.uniform(45, 75)
        
        # Pressure (typical sea level pressure with variation)
        pressure = np.random.uniform(1008, 1018)
        
        # Wind patterns
        if month in [6, 7, 8, 9]:  # Monsoon - stronger winds
            wind_speed = np.random.uniform(3, 12)
        else:
            wind_speed = np.random.uniform(1, 8)
        
        wind_direction = np.random.uniform(0, 360)
        
        # Visibility and cloud coverage
        if month in [6, 7, 8, 9]:  # Monsoon - more clouds, less visibility
            cloud_coverage = np.random.uniform(60, 95)
            visibility = np.random.uniform(4, 8)
        elif month in [11, 12, 1, 2]:  # Winter - fog possible, variable clouds
            cloud_coverage = np.random.uniform(20, 70)
            visibility = np.random.uniform(2, 10)
        else:  # Clear seasons
            cloud_coverage = np.random.uniform(10, 50)
            visibility = np.random.uniform(8, 15)
        
        # Weather condition based on cloud coverage and season
        if cloud_coverage > 80:
            if month in [6, 7, 8, 9]:
                weather_condition = np.random.choice(['Rain', 'Thunderstorm', 'Drizzle'])
            else:
                weather_condition = 'Clouds'
        elif cloud_coverage > 50:
            weather_condition = 'Clouds'
        elif visibility < 5:
            weather_condition = 'Mist'
        else:
            weather_condition = 'Clear'
        
        return {
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 0),
            'pressure': round(pressure, 1),
            'wind_speed': round(wind_speed, 1),
            'wind_direction': round(wind_direction, 0),
            'visibility': round(visibility, 1),
            'cloud_coverage': round(cloud_coverage, 0),
            'weather_condition': weather_condition,
            'timestamp': current_time,
            'source': 'Synthetic'
        }
    
    def _generate_historical_weather(self, location_coords: Dict[str, float], hours: int) -> pd.DataFrame:
        """
        Generate realistic historical weather data
        """
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=hours),
            end=datetime.now(),
            freq='H'
        )
        
        weather_data = []
        for date in dates:
            # Generate weather for specific time
            weather_point = self._generate_realistic_weather_for_time(location_coords, date)
            weather_point['timestamp'] = date
            weather_data.append(weather_point)
        
        return pd.DataFrame(weather_data)
    
    def _generate_forecast_weather(self, location_coords: Dict[str, float], hours: int) -> pd.DataFrame:
        """
        Generate realistic weather forecast data
        """
        dates = pd.date_range(
            start=datetime.now(),
            periods=hours,
            freq='H'
        )
        
        # Get current weather as starting point
        current_weather = self.get_current_weather(location_coords)
        
        forecast_data = []
        for i, date in enumerate(dates):
            if i == 0 and current_weather:
                # Use current weather for first point
                weather_point = current_weather.copy()
                weather_point['timestamp'] = date
            else:
                # Generate forecast with some continuity from previous point
                if forecast_data:
                    prev_weather = forecast_data[-1]
                    # Add gradual changes rather than random jumps
                    weather_point = self._generate_realistic_weather_for_time(location_coords, date)
                    
                    # Apply some smoothing from previous values
                    smoothing_factor = 0.7
                    weather_point['temperature'] = (smoothing_factor * prev_weather['temperature'] + 
                                                   (1 - smoothing_factor) * weather_point['temperature'])
                    weather_point['humidity'] = (smoothing_factor * prev_weather['humidity'] + 
                                                (1 - smoothing_factor) * weather_point['humidity'])
                    weather_point['pressure'] = (smoothing_factor * prev_weather['pressure'] + 
                                                (1 - smoothing_factor) * weather_point['pressure'])
                else:
                    weather_point = self._generate_realistic_weather_for_time(location_coords, date)
                
                weather_point['timestamp'] = date
            
            forecast_data.append(weather_point)
        
        return pd.DataFrame(forecast_data)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        age = (datetime.now() - self.cache[cache_key]['timestamp']).total_seconds()
        return age < self.cache_timeout
