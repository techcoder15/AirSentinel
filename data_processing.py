import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Utility class for processing and analyzing air pollution data
    Handles data merging, correlation analysis, and weather impact calculations
    """
    
    def __init__(self):
        self.correlation_cache = {}
        self.cache_timeout = 3600  # 1 hour
    
    def merge_datasets(self, ground_data: pd.DataFrame, weather_data: pd.DataFrame, 
                      satellite_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Merge ground monitoring, weather, and satellite datasets
        
        Args:
            ground_data: DataFrame with ground pollution measurements
            weather_data: DataFrame with weather observations
            satellite_data: DataFrame with satellite AOD data
            
        Returns:
            Merged DataFrame with all data sources
        """
        try:
            # Ensure all datasets have timestamp columns
            for df, name in [(ground_data, 'ground'), (weather_data, 'weather'), (satellite_data, 'satellite')]:
                if df is None or df.empty:
                    print(f"Empty {name} dataset")
                    return None
                if 'timestamp' not in df.columns:
                    print(f"Missing timestamp in {name} data")
                    return None
            
            # Convert timestamps to datetime
            ground_data = ground_data.copy()
            weather_data = weather_data.copy()
            satellite_data = satellite_data.copy()
            
            ground_data['timestamp'] = pd.to_datetime(ground_data['timestamp'])
            weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
            satellite_data['timestamp'] = pd.to_datetime(satellite_data['timestamp'])
            
            # Round timestamps to nearest hour for better alignment
            ground_data['timestamp_hour'] = ground_data['timestamp'].dt.round('H')
            weather_data['timestamp_hour'] = weather_data['timestamp'].dt.round('H')
            satellite_data['timestamp_hour'] = satellite_data['timestamp'].dt.round('H')
            
            # Merge on rounded timestamps
            # Start with ground data (highest frequency)
            merged = ground_data.copy()
            
            # Merge weather data
            merged = pd.merge(merged, weather_data, on='timestamp_hour', 
                            how='left', suffixes=('', '_weather'))
            
            # For satellite data, use nearest time matching (satellites have lower frequency)
            satellite_hourly = self._interpolate_satellite_data(satellite_data)
            merged = pd.merge(merged, satellite_hourly, on='timestamp_hour', 
                            how='left', suffixes=('', '_satellite'))
            
            # Use primary timestamp
            merged['timestamp'] = merged['timestamp'].fillna(merged['timestamp_weather'])
            
            # Drop duplicate timestamp columns
            timestamp_cols = [col for col in merged.columns if 'timestamp' in col and col != 'timestamp']
            merged = merged.drop(columns=timestamp_cols)
            
            # Forward fill missing satellite data (satellites don't have hourly coverage)
            satellite_cols = ['aod_value']
            for col in satellite_cols:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(method='ffill').fillna(method='bfill')
            
            # Drop rows with too many missing values
            merged = merged.dropna(thresh=len(merged.columns) * 0.7)  # At least 70% non-null
            
            # Sort by timestamp
            merged = merged.sort_values('timestamp').reset_index(drop=True)
            
            print(f"Merged dataset created with {len(merged)} records")
            return merged
            
        except Exception as e:
            print(f"Error merging datasets: {e}")
            return None
    
    def get_weather_pollution_correlation(self, location_coords: Dict[str, float], 
                                        days: int = 7) -> Optional[pd.DataFrame]:
        """
        Get correlation data between weather and pollution parameters
        
        Args:
            location_coords: Dictionary with 'lat' and 'lon' keys
            days: Number of days of data to analyze
            
        Returns:
            DataFrame with weather and pollution data for correlation analysis
        """
        cache_key = f"correlation_{location_coords['lat']}_{location_coords['lon']}_{days}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.correlation_cache[cache_key]['data']
        
        try:
            from data_sources.ground_data import GroundDataCollector
            from data_sources.weather_data import WeatherDataCollector
            
            ground_collector = GroundDataCollector()
            weather_collector = WeatherDataCollector()
            
            # Get data for the specified period
            hours = days * 24
            ground_data = ground_collector.get_historical_data(location_coords, hours)
            weather_data = weather_collector.get_historical_weather(location_coords, hours)
            
            if ground_data is None or weather_data is None or ground_data.empty or weather_data.empty:
                return None
            
            # Merge the datasets
            ground_data['timestamp'] = pd.to_datetime(ground_data['timestamp'])
            weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
            
            # Round to nearest hour for merging
            ground_data['timestamp_hour'] = ground_data['timestamp'].dt.round('H')
            weather_data['timestamp_hour'] = weather_data['timestamp'].dt.round('H')
            
            correlation_data = pd.merge(ground_data, weather_data, on='timestamp_hour', how='inner')
            correlation_data['timestamp'] = correlation_data['timestamp_x']
            
            # Select relevant columns for correlation analysis
            correlation_columns = [
                'timestamp', 'pm25', 'pm10', 'no2', 'so2', 'co',
                'temperature', 'humidity', 'pressure', 'wind_speed', 'visibility'
            ]
            
            existing_columns = [col for col in correlation_columns if col in correlation_data.columns]
            correlation_data = correlation_data[existing_columns]
            
            # Cache the result
            self.correlation_cache[cache_key] = {
                'data': correlation_data,
                'timestamp': datetime.now()
            }
            
            return correlation_data
            
        except Exception as e:
            print(f"Error getting weather-pollution correlation: {e}")
            return None
    
    def calculate_weather_impact(self, weather_data: Dict) -> float:
        """
        Calculate weather impact score on pollution (0-100 scale)
        Higher score means weather conditions worsen pollution
        
        Args:
            weather_data: Dictionary with current weather parameters
            
        Returns:
            Weather impact score (0-100)
        """
        try:
            impact_score = 0
            
            # Wind speed impact (lower wind = higher impact)
            wind_speed = weather_data.get('wind_speed', 5)
            if wind_speed < 2:
                impact_score += 30  # Very low wind - high impact
            elif wind_speed < 4:
                impact_score += 20  # Low wind - moderate impact
            elif wind_speed < 8:
                impact_score += 10  # Moderate wind - low impact
            # High wind (>8 m/s) = 0 impact (good for dispersion)
            
            # Temperature impact (temperature inversions)
            temperature = weather_data.get('temperature', 25)
            hour = datetime.now().hour
            
            # Check for temperature inversion conditions
            if 5 <= hour <= 9:  # Morning hours
                if temperature < 15:  # Cold morning
                    impact_score += 15
            elif 18 <= hour <= 23:  # Evening hours
                if temperature < 20:  # Cool evening
                    impact_score += 10
            
            # Humidity impact (high humidity can worsen visibility)
            humidity = weather_data.get('humidity', 60)
            if humidity > 80:
                impact_score += 15
            elif humidity > 70:
                impact_score += 10
            elif humidity < 30:  # Very dry can increase dust
                impact_score += 5
            
            # Pressure impact (high pressure can trap pollutants)
            pressure = weather_data.get('pressure', 1013)
            if pressure > 1020:  # High pressure system
                impact_score += 15
            elif pressure < 1000:  # Low pressure (usually better for dispersion)
                impact_score -= 5
            
            # Visibility impact
            visibility = weather_data.get('visibility', 10)
            if visibility < 5:  # Poor visibility
                impact_score += 20
            elif visibility < 8:
                impact_score += 10
            
            # Cloud coverage impact
            cloud_coverage = weather_data.get('cloud_coverage', 50)
            if cloud_coverage < 20:  # Clear skies during day can increase photochemical reactions
                if 10 <= hour <= 16:
                    impact_score += 5
            
            # Weather condition impact
            weather_condition = weather_data.get('weather_condition', 'Clear')
            if weather_condition in ['Mist', 'Fog', 'Haze']:
                impact_score += 25
            elif weather_condition in ['Rain', 'Drizzle']:
                impact_score -= 20  # Rain helps clean air
            elif weather_condition == 'Thunderstorm':
                impact_score -= 15  # Strong winds and rain
            
            # Ensure score is within bounds
            impact_score = max(0, min(100, impact_score))
            
            return impact_score
            
        except Exception as e:
            print(f"Error calculating weather impact: {e}")
            return 50  # Default moderate impact
    
    def calculate_pollution_trends(self, data: pd.DataFrame, pollutant: str = 'pm25') -> Dict:
        """
        Calculate pollution trends and statistics
        
        Args:
            data: DataFrame with pollution data
            pollutant: Pollutant to analyze
            
        Returns:
            Dictionary with trend statistics
        """
        try:
            if data is None or data.empty or pollutant not in data.columns:
                return {}
            
            values = data[pollutant].dropna()
            
            if len(values) < 2:
                return {}
            
            # Basic statistics
            stats = {
                'mean': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'current': values.iloc[-1] if len(values) > 0 else 0
            }
            
            # Trend analysis
            if len(values) >= 5:
                # Simple linear trend
                x = np.arange(len(values))
                coefficients = np.polyfit(x, values, 1)
                trend_slope = coefficients[0]
                
                if trend_slope > 0.5:
                    trend_direction = 'increasing'
                elif trend_slope < -0.5:
                    trend_direction = 'decreasing'
                else:
                    trend_direction = 'stable'
                
                stats['trend_slope'] = trend_slope
                stats['trend_direction'] = trend_direction
            
            # Recent vs historical comparison
            if len(values) >= 24:  # At least 24 hours of data
                recent_avg = values.tail(6).mean()  # Last 6 hours
                historical_avg = values.head(-6).mean()  # All but last 6 hours
                
                change_percent = ((recent_avg - historical_avg) / historical_avg) * 100
                stats['recent_change_percent'] = change_percent
            
            # Health impact categories
            if pollutant == 'pm25':
                if stats['current'] <= 12:
                    stats['health_category'] = 'Good'
                elif stats['current'] <= 35:
                    stats['health_category'] = 'Moderate'
                elif stats['current'] <= 55:
                    stats['health_category'] = 'Unhealthy for Sensitive Groups'
                elif stats['current'] <= 150:
                    stats['health_category'] = 'Unhealthy'
                else:
                    stats['health_category'] = 'Very Unhealthy'
            
            return stats
            
        except Exception as e:
            print(f"Error calculating pollution trends: {e}")
            return {}
    
    def detect_pollution_episodes(self, data: pd.DataFrame, threshold_multiplier: float = 1.5) -> List[Dict]:
        """
        Detect pollution episodes (periods of elevated pollution)
        
        Args:
            data: DataFrame with pollution data
            threshold_multiplier: Multiplier for average to define episode threshold
            
        Returns:
            List of pollution episodes
        """
        try:
            episodes = []
            
            if data is None or data.empty or 'pm25' not in data.columns:
                return episodes
            
            pm25_data = data[['timestamp', 'pm25']].copy()
            pm25_data = pm25_data.dropna()
            
            if len(pm25_data) < 10:
                return episodes
            
            # Calculate threshold
            mean_pm25 = pm25_data['pm25'].mean()
            threshold = mean_pm25 * threshold_multiplier
            
            # Find periods above threshold
            above_threshold = pm25_data['pm25'] > threshold
            
            # Group consecutive periods
            groups = (above_threshold != above_threshold.shift()).cumsum()
            
            for group_id, group_data in pm25_data.groupby(groups):
                if group_data['pm25'].iloc[0] > threshold and len(group_data) >= 3:  # At least 3 hours
                    episode = {
                        'start_time': group_data['timestamp'].iloc[0],
                        'end_time': group_data['timestamp'].iloc[-1],
                        'duration_hours': len(group_data),
                        'max_pm25': group_data['pm25'].max(),
                        'avg_pm25': group_data['pm25'].mean(),
                        'threshold': threshold
                    }
                    episodes.append(episode)
            
            return episodes
            
        except Exception as e:
            print(f"Error detecting pollution episodes: {e}")
            return []
    
    def calculate_air_quality_index(self, pollutant_data: Dict) -> float:
        """
        Calculate overall Air Quality Index from multiple pollutants
        
        Args:
            pollutant_data: Dictionary with pollutant concentrations
            
        Returns:
            Overall AQI value
        """
        try:
            aqi_values = []
            
            # PM2.5 AQI
            pm25 = pollutant_data.get('pm25', 0)
            if pm25 > 0:
                aqi_values.append(self._pollutant_to_aqi(pm25, 'pm25'))
            
            # PM10 AQI
            pm10 = pollutant_data.get('pm10', 0)
            if pm10 > 0:
                aqi_values.append(self._pollutant_to_aqi(pm10, 'pm10'))
            
            # NO2 AQI (simplified)
            no2 = pollutant_data.get('no2', 0)
            if no2 > 0:
                aqi_values.append(self._pollutant_to_aqi(no2, 'no2'))
            
            # SO2 AQI (simplified)
            so2 = pollutant_data.get('so2', 0)
            if so2 > 0:
                aqi_values.append(self._pollutant_to_aqi(so2, 'so2'))
            
            # CO AQI (simplified)
            co = pollutant_data.get('co', 0)
            if co > 0:
                aqi_values.append(self._pollutant_to_aqi(co * 1000, 'co'))  # Convert mg/m³ to μg/m³
            
            # Return maximum AQI (worst pollutant determines overall AQI)
            return max(aqi_values) if aqi_values else 50
            
        except Exception as e:
            print(f"Error calculating AQI: {e}")
            return 50
    
    def _interpolate_satellite_data(self, satellite_data: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate satellite data to hourly frequency
        """
        try:
            if satellite_data.empty:
                return satellite_data
            
            # Create hourly index covering the satellite data period
            start_time = satellite_data['timestamp_hour'].min()
            end_time = satellite_data['timestamp_hour'].max()
            hourly_index = pd.date_range(start=start_time, end=end_time, freq='H')
            
            # Create hourly DataFrame
            hourly_df = pd.DataFrame({'timestamp_hour': hourly_index})
            
            # Merge with satellite data and interpolate
            merged = pd.merge(hourly_df, satellite_data, on='timestamp_hour', how='left')
            
            # Interpolate AOD values
            if 'aod_value' in merged.columns:
                merged['aod_value'] = merged['aod_value'].interpolate(method='linear')
                merged['aod_value'] = merged['aod_value'].fillna(method='ffill').fillna(method='bfill')
            
            return merged
            
        except Exception as e:
            print(f"Error interpolating satellite data: {e}")
            return satellite_data
    
    def _pollutant_to_aqi(self, concentration: float, pollutant: str) -> float:
        """
        Convert pollutant concentration to AQI using EPA breakpoints
        """
        breakpoints = {
            'pm25': [
                (0, 12, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 500.4, 301, 500)
            ],
            'pm10': [
                (0, 54, 0, 50),
                (55, 154, 51, 100),
                (155, 254, 101, 150),
                (255, 354, 151, 200),
                (355, 424, 201, 300),
                (425, 604, 301, 500)
            ],
            'no2': [
                (0, 53, 0, 50),
                (54, 100, 51, 100),
                (101, 360, 101, 150),
                (361, 649, 151, 200),
                (650, 1249, 201, 300),
                (1250, 2049, 301, 500)
            ],
            'so2': [
                (0, 35, 0, 50),
                (36, 75, 51, 100),
                (76, 185, 101, 150),
                (186, 304, 151, 200),
                (305, 604, 201, 300),
                (605, 1004, 301, 500)
            ],
            'co': [
                (0, 4400, 0, 50),
                (4500, 9400, 51, 100),
                (9500, 12400, 101, 150),
                (12500, 15400, 151, 200),
                (15500, 30400, 201, 300),
                (30500, 50400, 301, 500)
            ]
        }
        
        if pollutant not in breakpoints:
            return 50  # Default moderate AQI
        
        for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints[pollutant]:
            if bp_lo <= concentration <= bp_hi:
                return ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + aqi_lo
        
        return 500  # Hazardous if above all breakpoints
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.correlation_cache:
            return False
        
        age = (datetime.now() - self.correlation_cache[cache_key]['timestamp']).total_seconds()
        return age < self.cache_timeout
