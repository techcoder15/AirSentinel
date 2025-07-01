import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import joblib
import os
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class PollutionPredictor:
    """
    Machine Learning models for predicting air pollution levels
    Uses Random Forest and Linear Regression with feature engineering
    """
    
    def __init__(self):
        # Initialize models
        self.pm25_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.pm10_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.aqi_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Scalers for feature normalization
        self.feature_scaler = StandardScaler()
        self.target_scalers = {
            'pm25': StandardScaler(),
            'pm10': StandardScaler(),
            'aqi': StandardScaler()
        }
        
        # Model metadata
        self.models_trained = False
        self.feature_names = []
        self.model_metrics = {}
        self.feature_importance = {}
        
        # Training data cache
        self.training_data = None
        self.last_training_time = None
        
    def predict_next_24h(self, location_coords: Dict[str, float]) -> Optional[Dict]:
        """
        Predict pollution levels for the next 24 hours
        
        Args:
            location_coords: Dictionary with 'lat' and 'lon' keys
            
        Returns:
            Dictionary with predicted values and confidence
        """
        try:
            # Ensure models are trained
            self._ensure_models_trained(location_coords)
            
            # Get current features
            current_features = self._get_current_features(location_coords)
            if current_features is None:
                return None
            
            # Make predictions
            features_scaled = self.feature_scaler.transform([current_features])
            
            pm25_pred = self.pm25_model.predict(features_scaled)[0]
            pm10_pred = self.pm10_model.predict(features_scaled)[0]
            aqi_pred = self.aqi_model.predict(features_scaled)[0]
            
            # Calculate confidence based on model uncertainty
            confidence = self._calculate_prediction_confidence(features_scaled)
            
            return {
                'pm25_24h': max(0, pm25_pred),
                'pm10_24h': max(0, pm10_pred),
                'aqi_24h': max(0, aqi_pred),
                'confidence': confidence,
                'prediction_time': datetime.now(),
                'forecast_time': datetime.now() + timedelta(hours=24)
            }
            
        except Exception as e:
            print(f"Error in 24h prediction: {e}")
            return None
    
    def predict_next_hours(self, location_coords: Dict[str, float], hours: int) -> Optional[Dict]:
        """
        Predict pollution levels for a specific number of hours ahead
        
        Args:
            location_coords: Dictionary with 'lat' and 'lon' keys
            hours: Number of hours to predict ahead
            
        Returns:
            Dictionary with predicted values
        """
        try:
            self._ensure_models_trained(location_coords)
            
            # Get features with time adjustment for future prediction
            future_features = self._get_future_features(location_coords, hours)
            if future_features is None:
                return None
            
            features_scaled = self.feature_scaler.transform([future_features])
            
            pm25_pred = self.pm25_model.predict(features_scaled)[0]
            pm10_pred = self.pm10_model.predict(features_scaled)[0]
            aqi_pred = self.aqi_model.predict(features_scaled)[0]
            
            return {
                'pm25': max(0, pm25_pred),
                'pm10': max(0, pm10_pred),
                'aqi': max(0, aqi_pred),
                'hours_ahead': hours,
                'prediction_time': datetime.now()
            }
            
        except Exception as e:
            print(f"Error in {hours}h prediction: {e}")
            return None
    
    def get_hourly_predictions(self, location_coords: Dict[str, float], hours: int = 48) -> Optional[pd.DataFrame]:
        """
        Get detailed hourly predictions for the specified time period
        
        Args:
            location_coords: Dictionary with 'lat' and 'lon' keys
            hours: Number of hours to predict
            
        Returns:
            DataFrame with hourly predictions
        """
        try:
            self._ensure_models_trained(location_coords)
            
            predictions = []
            current_time = datetime.now()
            
            for hour in range(hours):
                future_time = current_time + timedelta(hours=hour)
                
                # Get features for this specific hour
                future_features = self._get_future_features(location_coords, hour)
                if future_features is None:
                    continue
                
                features_scaled = self.feature_scaler.transform([future_features])
                
                pm25_pred = self.pm25_model.predict(features_scaled)[0]
                pm10_pred = self.pm10_model.predict(features_scaled)[0]
                aqi_pred = self.aqi_model.predict(features_scaled)[0]
                
                # Calculate prediction intervals (simple approach using model variance)
                pm25_std = self._get_prediction_std('pm25', features_scaled)
                pm10_std = self._get_prediction_std('pm10', features_scaled)
                
                predictions.append({
                    'timestamp': future_time,
                    'pm25': max(0, pm25_pred),
                    'pm10': max(0, pm10_pred),
                    'aqi': max(0, aqi_pred),
                    'pm25_lower': max(0, pm25_pred - 1.96 * pm25_std),
                    'pm25_upper': pm25_pred + 1.96 * pm25_std,
                    'pm10_lower': max(0, pm10_pred - 1.96 * pm10_std),
                    'pm10_upper': pm10_pred + 1.96 * pm10_std
                })
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            print(f"Error generating hourly predictions: {e}")
            return None
    
    def get_model_metrics(self) -> Optional[Dict]:
        """
        Get model performance metrics
        
        Returns:
            Dictionary with model accuracy metrics
        """
        if not self.models_trained:
            return None
        
        return self.model_metrics.copy()
    
    def get_feature_importance(self) -> Optional[Dict]:
        """
        Get feature importance from trained models
        
        Returns:
            Dictionary with feature importance scores
        """
        if not self.models_trained:
            return None
        
        return self.feature_importance.copy()
    
    def generate_alerts(self, location_coords: Dict[str, float]) -> List[Dict]:
        """
        Generate pollution alerts based on predictions
        
        Args:
            location_coords: Dictionary with 'lat' and 'lon' keys
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        try:
            # Get 48-hour predictions
            hourly_pred = self.get_hourly_predictions(location_coords, 48)
            
            if hourly_pred is not None and not hourly_pred.empty:
                # Check for high PM2.5 levels
                high_pm25_hours = hourly_pred[hourly_pred['pm25'] > 55]
                if not high_pm25_hours.empty:
                    alerts.append({
                        'level': 'high',
                        'message': f"High PM2.5 levels expected in {len(high_pm25_hours)} hours over next 48h. Peak: {high_pm25_hours['pm25'].max():.1f} μg/m³",
                        'pollutant': 'PM2.5',
                        'max_value': high_pm25_hours['pm25'].max(),
                        'hours_affected': len(high_pm25_hours)
                    })
                
                # Check for very high PM10 levels
                very_high_pm10_hours = hourly_pred[hourly_pred['pm10'] > 154]
                if not very_high_pm10_hours.empty:
                    alerts.append({
                        'level': 'high',
                        'message': f"Very high PM10 levels expected in {len(very_high_pm10_hours)} hours. Peak: {very_high_pm10_hours['pm10'].max():.1f} μg/m³",
                        'pollutant': 'PM10',
                        'max_value': very_high_pm10_hours['pm10'].max(),
                        'hours_affected': len(very_high_pm10_hours)
                    })
                
                # Check for unhealthy AQI
                unhealthy_aqi_hours = hourly_pred[hourly_pred['aqi'] > 150]
                if not unhealthy_aqi_hours.empty:
                    alerts.append({
                        'level': 'medium',
                        'message': f"Unhealthy air quality expected in {len(unhealthy_aqi_hours)} hours. Peak AQI: {unhealthy_aqi_hours['aqi'].max():.0f}",
                        'pollutant': 'AQI',
                        'max_value': unhealthy_aqi_hours['aqi'].max(),
                        'hours_affected': len(unhealthy_aqi_hours)
                    })
                
                # Check for improving conditions
                current_pred = hourly_pred.iloc[0]
                future_pred = hourly_pred.iloc[-1] if len(hourly_pred) > 1 else current_pred
                
                if future_pred['aqi'] < current_pred['aqi'] * 0.8:
                    alerts.append({
                        'level': 'info',
                        'message': f"Air quality expected to improve significantly over next 48h. AQI dropping from {current_pred['aqi']:.0f} to {future_pred['aqi']:.0f}",
                        'pollutant': 'AQI',
                        'improvement': True
                    })
            
        except Exception as e:
            print(f"Error generating alerts: {e}")
        
        return alerts
    
    def _ensure_models_trained(self, location_coords: Dict[str, float]):
        """
        Ensure models are trained with recent data
        """
        # Check if models need retraining (every 24 hours or first time)
        needs_training = (
            not self.models_trained or 
            self.last_training_time is None or
            (datetime.now() - self.last_training_time).total_seconds() > 86400
        )
        
        if needs_training:
            self._train_models(location_coords)
    
    def _train_models(self, location_coords: Dict[str, float]):
        """
        Train ML models using historical data
        """
        try:
            # Get training data
            training_data = self._prepare_training_data(location_coords)
            
            if training_data is None or len(training_data) < 10:
                print("Insufficient data for model training")
                return
            
            # Prepare features and targets
            features = training_data[self.feature_names].values
            pm25_target = training_data['pm25'].values
            pm10_target = training_data['pm10'].values
            aqi_target = training_data['aqi'].values
            
            # Split data for validation
            X_train, X_test, y_pm25_train, y_pm25_test = train_test_split(
                features, pm25_target, test_size=0.2, random_state=42
            )
            _, _, y_pm10_train, y_pm10_test = train_test_split(
                features, pm10_target, test_size=0.2, random_state=42
            )
            _, _, y_aqi_train, y_aqi_test = train_test_split(
                features, aqi_target, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train models
            self.pm25_model.fit(X_train_scaled, y_pm25_train)
            self.pm10_model.fit(X_train_scaled, y_pm10_train)
            self.aqi_model.fit(X_train_scaled, y_aqi_train)
            
            # Evaluate models
            pm25_pred = self.pm25_model.predict(X_test_scaled)
            pm10_pred = self.pm10_model.predict(X_test_scaled)
            aqi_pred = self.aqi_model.predict(X_test_scaled)
            
            # Store metrics
            self.model_metrics = {
                'pm25_r2': r2_score(y_pm25_test, pm25_pred),
                'pm10_r2': r2_score(y_pm10_test, pm10_pred),
                'aqi_r2': r2_score(y_aqi_test, aqi_pred),
                'pm25_mae': mean_absolute_error(y_pm25_test, pm25_pred),
                'pm10_mae': mean_absolute_error(y_pm10_test, pm10_pred),
                'aqi_mae': mean_absolute_error(y_aqi_test, aqi_pred),
                'mae': (mean_absolute_error(y_pm25_test, pm25_pred) + 
                       mean_absolute_error(y_pm10_test, pm10_pred)) / 2,
                'training_samples': len(training_data),
                'last_trained': datetime.now()
            }
            
            # Store feature importance
            self.feature_importance = dict(zip(
                self.feature_names,
                self.pm25_model.feature_importances_
            ))
            
            self.models_trained = True
            self.last_training_time = datetime.now()
            
            print(f"Models trained successfully with {len(training_data)} samples")
            print(f"PM2.5 R²: {self.model_metrics['pm25_r2']:.3f}")
            print(f"PM10 R²: {self.model_metrics['pm10_r2']:.3f}")
            
        except Exception as e:
            print(f"Error training models: {e}")
    
    def _prepare_training_data(self, location_coords: Dict[str, float]) -> Optional[pd.DataFrame]:
        """
        Prepare training data by combining historical pollution, weather, and satellite data
        """
        try:
            from data_sources.ground_data import GroundDataCollector
            from data_sources.weather_data import WeatherDataCollector
            from data_sources.satellite_data import SatelliteDataCollector
            from utils.data_processing import DataProcessor
            
            # Initialize data collectors
            ground_collector = GroundDataCollector()
            weather_collector = WeatherDataCollector()
            satellite_collector = SatelliteDataCollector()
            processor = DataProcessor()
            
            # Get historical data (last 30 days)
            ground_data = ground_collector.get_historical_data(location_coords, hours=720)  # 30 days
            weather_data = weather_collector.get_historical_weather(location_coords, hours=720)
            satellite_data = satellite_collector.get_historical_aod(location_coords, days=30)
            
            if any(data is None or data.empty for data in [ground_data, weather_data, satellite_data]):
                print("Insufficient historical data for training")
                return None
            
            # Merge datasets
            merged_data = processor.merge_datasets(ground_data, weather_data, satellite_data)
            
            if merged_data is None or len(merged_data) < 10:
                return None
            
            # Feature engineering
            features_data = self._engineer_features(merged_data)
            
            return features_data
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML models
        """
        try:
            # Ensure timestamp is datetime
            data = data.copy()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Time-based features
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['month'] = data['timestamp'].dt.month
            data['season'] = data['month'] % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
            
            # Lag features (previous hours pollution levels)
            for lag in [1, 3, 6, 12, 24]:
                data[f'pm25_lag_{lag}h'] = data['pm25'].shift(lag)
                data[f'pm10_lag_{lag}h'] = data['pm10'].shift(lag)
            
            # Rolling averages
            for window in [3, 6, 12, 24]:
                data[f'pm25_avg_{window}h'] = data['pm25'].rolling(window=window).mean()
                data[f'pm10_avg_{window}h'] = data['pm10'].rolling(window=window).mean()
                data[f'temp_avg_{window}h'] = data['temperature'].rolling(window=window).mean()
                data[f'humidity_avg_{window}h'] = data['humidity'].rolling(window=window).mean()
            
            # Weather interaction features
            data['temp_humidity_interaction'] = data['temperature'] * data['humidity'] / 100
            data['wind_pressure_interaction'] = data['wind_speed'] * data['pressure'] / 1000
            data['visibility_humidity_ratio'] = data['visibility'] / (data['humidity'] + 1)
            
            # AOD features
            data['aod_lag_1d'] = data['aod_value'].shift(4)  # Assuming 6-hour AOD data
            data['aod_avg_3d'] = data['aod_value'].rolling(window=12).mean()  # 3-day average
            
            # Pollution ratios
            data['pm25_pm10_ratio'] = data['pm25'] / (data['pm10'] + 1)
            data['no2_pm25_ratio'] = data['no2'] / (data['pm25'] + 1)
            
            # Calculate AQI
            data['aqi'] = data.apply(self._calculate_aqi, axis=1)
            
            # Define feature columns
            self.feature_names = [
                # Current weather
                'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 
                'visibility', 'cloud_coverage',
                
                # Time features
                'hour', 'day_of_week', 'month', 'season',
                
                # Lag features
                'pm25_lag_1h', 'pm25_lag_3h', 'pm25_lag_6h', 'pm25_lag_12h', 'pm25_lag_24h',
                'pm10_lag_1h', 'pm10_lag_3h', 'pm10_lag_6h', 'pm10_lag_12h', 'pm10_lag_24h',
                
                # Rolling averages
                'pm25_avg_3h', 'pm25_avg_6h', 'pm25_avg_12h', 'pm25_avg_24h',
                'pm10_avg_3h', 'pm10_avg_6h', 'pm10_avg_12h', 'pm10_avg_24h',
                'temp_avg_3h', 'temp_avg_6h', 'temp_avg_12h', 'temp_avg_24h',
                'humidity_avg_3h', 'humidity_avg_6h', 'humidity_avg_12h', 'humidity_avg_24h',
                
                # Interaction features
                'temp_humidity_interaction', 'wind_pressure_interaction', 'visibility_humidity_ratio',
                
                # AOD features
                'aod_value', 'aod_lag_1d', 'aod_avg_3d',
                
                # Pollution ratios
                'pm25_pm10_ratio', 'no2_pm25_ratio'
            ]
            
            # Keep only feature columns and targets
            target_columns = ['pm25', 'pm10', 'aqi', 'timestamp']
            all_columns = self.feature_names + target_columns
            
            # Filter to existing columns and drop NaN rows
            existing_columns = [col for col in all_columns if col in data.columns]
            filtered_data = data[existing_columns].dropna()
            
            return filtered_data
            
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            return data
    
    def _get_current_features(self, location_coords: Dict[str, float]) -> Optional[List]:
        """
        Get current feature vector for prediction
        """
        try:
            from data_sources.ground_data import GroundDataCollector
            from data_sources.weather_data import WeatherDataCollector
            from data_sources.satellite_data import SatelliteDataCollector
            
            # Get current data
            ground_collector = GroundDataCollector()
            weather_collector = WeatherDataCollector()
            satellite_collector = SatelliteDataCollector()
            
            current_ground = ground_collector.get_current_data(location_coords)
            current_weather = weather_collector.get_current_weather(location_coords)
            current_satellite = satellite_collector.get_current_aod(location_coords)
            
            # Get historical data for lag features
            historical_ground = ground_collector.get_historical_data(location_coords, hours=48)
            historical_weather = weather_collector.get_historical_weather(location_coords, hours=48)
            
            if not all([current_ground, current_weather, current_satellite, 
                       historical_ground is not None, historical_weather is not None]):
                return None
            
            # Calculate features
            current_time = datetime.now()
            features = []
            
            # Current weather features
            features.extend([
                current_weather['temperature'],
                current_weather['humidity'],
                current_weather['pressure'],
                current_weather['wind_speed'],
                current_weather['wind_direction'],
                current_weather['visibility'],
                current_weather['cloud_coverage']
            ])
            
            # Time features
            features.extend([
                current_time.hour,
                current_time.weekday(),
                current_time.month,
                current_time.month % 12 // 3 + 1  # season
            ])
            
            # Lag features (use recent historical data)
            recent_pm25 = historical_ground['pm25'].values
            recent_pm10 = historical_ground['pm10'].values
            
            lag_hours = [1, 3, 6, 12, 24]
            for lag in lag_hours:
                if len(recent_pm25) > lag:
                    features.append(recent_pm25[-lag-1])
                    features.append(recent_pm10[-lag-1])
                else:
                    features.extend([current_ground['pm25'], current_ground['pm10']])
            
            # Rolling averages
            for window in [3, 6, 12, 24]:
                if len(recent_pm25) >= window:
                    features.append(np.mean(recent_pm25[-window:]))
                    features.append(np.mean(recent_pm10[-window:]))
                else:
                    features.extend([current_ground['pm25'], current_ground['pm10']])
                
                # Weather rolling averages
                if len(historical_weather) >= window:
                    features.append(historical_weather['temperature'].tail(window).mean())
                    features.append(historical_weather['humidity'].tail(window).mean())
                else:
                    features.extend([current_weather['temperature'], current_weather['humidity']])
            
            # Interaction features
            features.extend([
                current_weather['temperature'] * current_weather['humidity'] / 100,
                current_weather['wind_speed'] * current_weather['pressure'] / 1000,
                current_weather['visibility'] / (current_weather['humidity'] + 1)
            ])
            
            # AOD features
            features.append(current_satellite['aod_value'])
            # Simplified AOD lag features
            features.extend([current_satellite['aod_value'], current_satellite['aod_value']])
            
            # Pollution ratios
            features.extend([
                current_ground['pm25'] / (current_ground['pm10'] + 1),
                current_ground['no2'] / (current_ground['pm25'] + 1)
            ])
            
            return features
            
        except Exception as e:
            print(f"Error getting current features: {e}")
            return None
    
    def _get_future_features(self, location_coords: Dict[str, float], hours_ahead: int) -> Optional[List]:
        """
        Get feature vector for future prediction
        """
        try:
            # Get current features as base
            current_features = self._get_current_features(location_coords)
            if current_features is None:
                return None
            
            # Adjust time-based features for future prediction
            future_time = datetime.now() + timedelta(hours=hours_ahead)
            
            # Update time features (indices 7-10 in feature vector)
            current_features[7] = future_time.hour
            current_features[8] = future_time.weekday()
            current_features[9] = future_time.month
            current_features[10] = future_time.month % 12 // 3 + 1
            
            # For simplicity, keep other features the same
            # In production, you would want to forecast weather conditions too
            
            return current_features
            
        except Exception as e:
            print(f"Error getting future features: {e}")
            return None
    
    def _calculate_aqi(self, row) -> float:
        """
        Calculate AQI from pollutant concentrations
        """
        try:
            pm25 = row.get('pm25', 0)
            pm10 = row.get('pm10', 0)
            
            # Simplified AQI calculation (US EPA standard)
            pm25_aqi = self._pollutant_to_aqi(pm25, 'pm25')
            pm10_aqi = self._pollutant_to_aqi(pm10, 'pm10')
            
            return max(pm25_aqi, pm10_aqi)
            
        except Exception:
            return 50  # Default moderate AQI
    
    def _pollutant_to_aqi(self, concentration: float, pollutant: str) -> float:
        """
        Convert pollutant concentration to AQI
        """
        if pollutant == 'pm25':
            breakpoints = [
                (0, 12, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 500.4, 301, 500)
            ]
        elif pollutant == 'pm10':
            breakpoints = [
                (0, 54, 0, 50),
                (55, 154, 51, 100),
                (155, 254, 101, 150),
                (255, 354, 151, 200),
                (355, 424, 201, 300),
                (425, 604, 301, 500)
            ]
        else:
            return 50
        
        for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
            if bp_lo <= concentration <= bp_hi:
                return ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + aqi_lo
        
        return 500  # Hazardous
    
    def _calculate_prediction_confidence(self, features_scaled: np.ndarray) -> float:
        """
        Calculate prediction confidence based on model uncertainty
        """
        try:
            # Use ensemble variance as uncertainty measure
            pm25_preds = []
            pm10_preds = []
            
            # Get predictions from individual trees (if available)
            if hasattr(self.pm25_model, 'estimators_'):
                for estimator in self.pm25_model.estimators_[:10]:  # First 10 trees
                    pm25_preds.append(estimator.predict(features_scaled)[0])
            
            if hasattr(self.pm10_model, 'estimators_'):
                for estimator in self.pm10_model.estimators_[:10]:
                    pm10_preds.append(estimator.predict(features_scaled)[0])
            
            # Calculate confidence based on prediction variance
            if pm25_preds and pm10_preds:
                pm25_var = np.var(pm25_preds)
                pm10_var = np.var(pm10_preds)
                avg_var = (pm25_var + pm10_var) / 2
                
                # Convert variance to confidence (0-1)
                confidence = max(0.3, min(0.95, 1 - (avg_var / 100)))
                return confidence
            else:
                return 0.75  # Default confidence
            
        except Exception:
            return 0.75
    
    def _get_prediction_std(self, pollutant: str, features_scaled: np.ndarray) -> float:
        """
        Get prediction standard deviation for uncertainty intervals
        """
        try:
            if pollutant == 'pm25':
                model = self.pm25_model
            elif pollutant == 'pm10':
                model = self.pm10_model
            else:
                return 10.0  # Default std
            
            # Use ensemble variance
            if hasattr(model, 'estimators_'):
                preds = [estimator.predict(features_scaled)[0] for estimator in model.estimators_[:20]]
                return np.std(preds)
            else:
                return 10.0  # Default std
                
        except Exception:
            return 10.0
