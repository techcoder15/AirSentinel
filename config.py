import os
from typing import Dict, List, Optional
from datetime import datetime

class Config:
    """
    Configuration management for the air pollution monitoring system
    Handles location coordinates, API settings, and system parameters
    """
    
    # Location coordinates for supported cities/regions
    LOCATION_COORDINATES = {
        'Patna, Bihar': {
            'lat': 25.5941,
            'lon': 85.1376,
            'region': 'Northern Plains',
            'population': 'Medium City',
            'primary_sources': ['Vehicles', 'Industry', 'Biomass Burning'],
            'monitoring_stations': ['Patna Collectorate', 'Patna University', 'IGSC Patna'],
            'typical_aod_range': (0.3, 0.8),
            'typical_pm25_range': (40, 120)
        },
        'Varanasi, UP': {
            'lat': 25.3176,
            'lon': 82.9739,
            'region': 'Northern Plains',
            'population': 'Medium City',
            'primary_sources': ['Vehicles', 'Industrial', 'Religious Activities'],
            'monitoring_stations': ['Varanasi Cantonment', 'BHU Varanasi'],
            'typical_aod_range': (0.35, 0.9),
            'typical_pm25_range': (45, 130)
        },
        'Guwahati, Assam': {
            'lat': 26.1445,
            'lon': 91.7362,
            'region': 'Northeastern',
            'population': 'Medium City',
            'primary_sources': ['Vehicles', 'Construction', 'Biomass'],
            'monitoring_stations': ['Guwahati Central', 'IIT Guwahati'],
            'typical_aod_range': (0.2, 0.6),
            'typical_pm25_range': (30, 80)
        },
        'Shimla, HP': {
            'lat': 31.1048,
            'lon': 77.1734,
            'region': 'Himalayan',
            'population': 'Small City',
            'primary_sources': ['Vehicles', 'Heating', 'Tourism'],
            'monitoring_stations': ['Shimla Mall Road', 'Shimla Ridge'],
            'typical_aod_range': (0.1, 0.4),
            'typical_pm25_range': (15, 45)
        },
        'Dehradun, UK': {
            'lat': 30.3165,
            'lon': 78.0322,
            'region': 'Sub-Himalayan',
            'population': 'Medium City',
            'primary_sources': ['Vehicles', 'Construction', 'Industry'],
            'monitoring_stations': ['Dehradun Clock Tower', 'FRI Dehradun'],
            'typical_aod_range': (0.2, 0.5),
            'typical_pm25_range': (25, 70)
        }
    }
    
    # API Configuration
    API_CONFIG = {
        'timeout': 30,
        'max_retries': 3,
        'rate_limit_delay': 1,
        'cache_duration': {
            'current_data': 900,      # 15 minutes
            'historical_data': 3600,  # 1 hour
            'predictions': 1800,      # 30 minutes
            'satellite_data': 1800,   # 30 minutes
            'weather_data': 1800      # 30 minutes
        }
    }
    
    # Data Source Priorities
    DATA_SOURCE_PRIORITY = {
        'satellite': ['MODIS', 'Sentinel-5P', 'INSAT'],
        'ground': ['CPCB', 'OpenAQ', 'AQI.cn'],
        'weather': ['OpenWeatherMap', 'ECMWF', 'CAMS']
    }
    
    # Pollution Thresholds (WHO and Indian Standards)
    POLLUTION_THRESHOLDS = {
        'pm25': {
            'who_annual': 15,
            'who_24h': 45,
            'indian_annual': 40,
            'indian_24h': 60,
            'emergency': 150
        },
        'pm10': {
            'who_annual': 45,
            'who_24h': 120,
            'indian_annual': 60,
            'indian_24h': 100,
            'emergency': 250
        },
        'no2': {
            'who_annual': 25,
            'who_24h': 50,
            'indian_annual': 40,
            'indian_24h': 80,
            'emergency': 200
        },
        'so2': {
            'who_24h': 40,
            'indian_annual': 50,
            'indian_24h': 80,
            'emergency': 150
        },
        'co': {
            'who_24h': 4,  # mg/m³
            'indian_8h': 2,
            'indian_1h': 4,
            'emergency': 10
        }
    }
    
    # AQI Categories
    AQI_CATEGORIES = {
        'good': {'range': (0, 50), 'color': '#00E400', 'description': 'Good'},
        'moderate': {'range': (51, 100), 'color': '#FFFF00', 'description': 'Moderate'},
        'unhealthy_sensitive': {'range': (101, 150), 'color': '#FF7E00', 'description': 'Unhealthy for Sensitive Groups'},
        'unhealthy': {'range': (151, 200), 'color': '#FF0000', 'description': 'Unhealthy'},
        'very_unhealthy': {'range': (201, 300), 'color': '#8F3F97', 'description': 'Very Unhealthy'},
        'hazardous': {'range': (301, 500), 'color': '#7E0023', 'description': 'Hazardous'}
    }
    
    # ML Model Configuration
    ML_CONFIG = {
        'training_data_days': 30,
        'minimum_training_samples': 100,
        'retrain_interval_hours': 24,
        'feature_selection_threshold': 0.01,
        'prediction_confidence_threshold': 0.5,
        'ensemble_size': 100,
        'cross_validation_folds': 5
    }
    
    # System Settings
    SYSTEM_CONFIG = {
        'default_refresh_interval': 30,  # seconds
        'max_historical_days': 90,
        'prediction_horizon_hours': 48,
        'alert_thresholds': {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        },
        'data_quality_threshold': 0.7,
        'backup_data_sources': True
    }
    
    @classmethod
    def get_location_coordinates(cls, location_name: str) -> Optional[Dict]:
        """
        Get coordinates and metadata for a location
        
        Args:
            location_name: Name of the location
            
        Returns:
            Dictionary with location data or None if not found
        """
        return cls.LOCATION_COORDINATES.get(location_name)
    
    @classmethod
    def get_all_locations(cls) -> List[str]:
        """
        Get list of all supported locations
        
        Returns:
            List of location names
        """
        return list(cls.LOCATION_COORDINATES.keys())
    
    @classmethod
    def get_pollution_threshold(cls, pollutant: str, threshold_type: str = 'who_24h') -> Optional[float]:
        """
        Get pollution threshold for a specific pollutant
        
        Args:
            pollutant: Pollutant name (e.g., 'pm25', 'pm10')
            threshold_type: Type of threshold (e.g., 'who_annual', 'indian_24h')
            
        Returns:
            Threshold value or None if not found
        """
        return cls.POLLUTION_THRESHOLDS.get(pollutant, {}).get(threshold_type)
    
    @classmethod
    def get_aqi_category(cls, aqi_value: float) -> Dict:
        """
        Get AQI category information for a given AQI value
        
        Args:
            aqi_value: AQI value
            
        Returns:
            Dictionary with category information
        """
        for category, info in cls.AQI_CATEGORIES.items():
            min_val, max_val = info['range']
            if min_val <= aqi_value <= max_val:
                return {
                    'category': category,
                    'description': info['description'],
                    'color': info['color'],
                    'range': info['range']
                }
        
        # If above all ranges, return hazardous
        return {
            'category': 'hazardous',
            'description': 'Hazardous',
            'color': '#7E0023',
            'range': (301, 500)
        }
    
    @classmethod
    def get_api_key(cls, service: str) -> str:
        """
        Get API key for a service from environment variables
        
        Args:
            service: Service name
            
        Returns:
            API key or empty string if not found
        """
        key_mapping = {
            'openweather': 'OPENWEATHER_API_KEY',
            'nasa': 'NASA_API_KEY',
            'copernicus': 'COPERNICUS_API_KEY',
            'cpcb': 'CPCB_API_KEY',
            'openaq': 'OPENAQ_API_KEY',
            'aqicn': 'AQICN_API_KEY',
            'isro': 'ISRO_API_KEY',
            'ecmwf': 'ECMWF_API_KEY'
        }
        
        env_var = key_mapping.get(service.lower())
        if env_var:
            return os.getenv(env_var, '')
        return ''
    
    @classmethod
    def get_cache_duration(cls, data_type: str) -> int:
        """
        Get cache duration for a data type
        
        Args:
            data_type: Type of data (e.g., 'current_data', 'historical_data')
            
        Returns:
            Cache duration in seconds
        """
        return cls.API_CONFIG['cache_duration'].get(data_type, 1800)
    
    @classmethod
    def is_location_supported(cls, lat: float, lon: float, tolerance: float = 0.5) -> Optional[str]:
        """
        Check if a location is supported (within tolerance of known locations)
        
        Args:
            lat: Latitude
            lon: Longitude
            tolerance: Distance tolerance in degrees
            
        Returns:
            Location name if supported, None otherwise
        """
        for location_name, coords in cls.LOCATION_COORDINATES.items():
            distance = ((lat - coords['lat']) ** 2 + (lon - coords['lon']) ** 2) ** 0.5
            if distance <= tolerance:
                return location_name
        return None
    
    @classmethod
    def get_regional_characteristics(cls, location_name: str) -> Dict:
        """
        Get regional characteristics for a location
        
        Args:
            location_name: Name of the location
            
        Returns:
            Dictionary with regional characteristics
        """
        location_data = cls.get_location_coordinates(location_name)
        if location_data:
            return {
                'region': location_data.get('region', 'Unknown'),
                'population': location_data.get('population', 'Unknown'),
                'primary_sources': location_data.get('primary_sources', []),
                'monitoring_stations': location_data.get('monitoring_stations', []),
                'typical_ranges': {
                    'aod': location_data.get('typical_aod_range', (0.1, 0.5)),
                    'pm25': location_data.get('typical_pm25_range', (20, 80))
                }
            }
        return {}
    
    @classmethod
    def get_data_quality_requirements(cls) -> Dict:
        """
        Get data quality requirements
        
        Returns:
            Dictionary with data quality thresholds
        """
        return {
            'minimum_completeness': cls.SYSTEM_CONFIG['data_quality_threshold'],
            'maximum_gap_hours': 6,
            'outlier_detection_threshold': 3,  # standard deviations
            'validation_rules': {
                'pm25': {'min': 0, 'max': 1000},
                'pm10': {'min': 0, 'max': 2000},
                'no2': {'min': 0, 'max': 500},
                'so2': {'min': 0, 'max': 300},
                'co': {'min': 0, 'max': 50},
                'aod': {'min': 0, 'max': 5},
                'temperature': {'min': -50, 'max': 60},
                'humidity': {'min': 0, 'max': 100},
                'wind_speed': {'min': 0, 'max': 50}
            }
        }
    
    @classmethod
    def get_alert_configuration(cls) -> Dict:
        """
        Get alert system configuration
        
        Returns:
            Dictionary with alert settings
        """
        return {
            'thresholds': cls.SYSTEM_CONFIG['alert_thresholds'],
            'notification_intervals': {
                'high': 1800,  # 30 minutes
                'medium': 3600,  # 1 hour
                'low': 7200   # 2 hours
            },
            'alert_types': {
                'pollution_spike': {'threshold_multiplier': 2.0, 'duration_hours': 3},
                'air_quality_warning': {'aqi_threshold': 150, 'duration_hours': 6},
                'health_advisory': {'pm25_threshold': 55, 'duration_hours': 12}
            }
        }
    
    @classmethod
    def validate_configuration(cls) -> Dict:
        """
        Validate system configuration
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'status': 'valid',
            'warnings': [],
            'errors': []
        }
        
        # Check for required API keys
        critical_apis = ['openweather', 'nasa']
        for api in critical_apis:
            if not cls.get_api_key(api):
                validation_results['warnings'].append(f"Missing API key for {api}")
        
        # Validate location coordinates
        for location, coords in cls.LOCATION_COORDINATES.items():
            if not (-90 <= coords['lat'] <= 90):
                validation_results['errors'].append(f"Invalid latitude for {location}")
            if not (-180 <= coords['lon'] <= 180):
                validation_results['errors'].append(f"Invalid longitude for {location}")
        
        # Validate thresholds
        for pollutant, thresholds in cls.POLLUTION_THRESHOLDS.items():
            for threshold_type, value in thresholds.items():
                if value < 0:
                    validation_results['errors'].append(f"Negative threshold for {pollutant}.{threshold_type}")
        
        if validation_results['errors']:
            validation_results['status'] = 'invalid'
        elif validation_results['warnings']:
            validation_results['status'] = 'warning'
        
        return validation_results
