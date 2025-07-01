# Air Pollution Monitoring & Prediction System

## Overview

This is a comprehensive air pollution monitoring and prediction system built with Streamlit that integrates multiple data sources to provide real-time monitoring and forecasting capabilities. The system focuses on smaller cities and rural areas where traditional monitoring infrastructure may be limited.

The application combines satellite imagery, ground-based measurements, meteorological data, and machine learning models to deliver accurate pollution assessments and predictions for the next 24 hours.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with interactive dashboard
- **Visualization**: Plotly for charts and graphs, Folium for interactive maps
- **Layout**: Wide layout with expandable sidebar for location selection and controls
- **Deployment**: Configured for autoscale deployment on Replit infrastructure

### Backend Architecture
- **Language**: Python 3.11
- **Structure**: Modular design with separate packages for data sources, ML models, and utilities
- **Caching**: Streamlit resource caching for component initialization and data persistence
- **Data Processing**: Real-time data integration and processing pipeline

### Data Storage Solutions
- **In-Memory Storage**: Pandas DataFrames for data manipulation and analysis
- **Caching Strategy**: Component-level caching with configurable timeout periods
- **Model Persistence**: Joblib for saving and loading trained ML models

## Key Components

### Data Sources Module (`data_sources/`)
1. **Satellite Data Collector** (`satellite_data.py`)
   - Integrates ISRO satellite data (INSAT, MODIS, Sentinel-5P)
   - Focuses on AOD (Aerosol Optical Depth) measurements
   - Supports NASA, Copernicus, and ISRO API endpoints

2. **Ground Data Collector** (`ground_data.py`)
   - Collects data from CPCB and OpenAQ monitoring stations
   - Monitors PM2.5, PM10, NO₂, SO₂, CO levels
   - Supports multiple cities with station mapping

3. **Weather Data Collector** (`weather_data.py`)
   - Integrates meteorological data from CAMS/ERA5
   - Focuses on weather parameters affecting pollution dispersion
   - Supports OpenWeather, Copernicus, and ECMWF APIs

### Machine Learning Module (`ml_models/`)
- **Prediction Engine** (`prediction.py`)
  - Random Forest models for PM2.5, PM10, and AQI prediction
  - Feature engineering and scaling
  - 24-hour pollution forecasting capabilities
  - Model performance tracking and feature importance analysis

### Utilities Module (`utils/`)
1. **Configuration Management** (`config.py`)
   - Location coordinates for supported cities
   - API settings and system parameters
   - Regional pollution source mapping

2. **Data Processing** (`data_processing.py`)
   - Data merging and correlation analysis
   - Weather impact calculations
   - Cache management with timeout functionality

## Data Flow

1. **Data Collection**: Parallel collection from satellite, ground, and weather APIs
2. **Data Integration**: Merging datasets with timestamp alignment and spatial correlation
3. **Feature Engineering**: Creating derived features for ML models
4. **Prediction**: Running trained models for 24-hour forecasts
5. **Visualization**: Real-time dashboard updates with interactive maps and charts

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **folium**: Interactive mapping
- **scikit-learn**: Machine learning algorithms

### API Integrations
- **NASA APIs**: Satellite data access
- **Copernicus Services**: European atmospheric data
- **ISRO APIs**: Indian satellite data
- **OpenWeather**: Meteorological data
- **CPCB**: Indian ground monitoring data
- **OpenAQ**: Global air quality data

### Supported Locations
- Patna, Bihar (Industrial/Agricultural region)
- Varanasi, UP (Urban/Religious activities)
- Guwahati, Assam (Northeastern region)
- Shimla, HP (Himalayan/Tourism region)

## Deployment Strategy

- **Platform**: Replit with autoscale deployment
- **Runtime**: Python 3.11 with Nix package management
- **Port Configuration**: Streamlit server on port 5000
- **Environment Variables**: API keys managed through environment variables
- **Scaling**: Automatic scaling based on demand

## Changelog

- June 24, 2025. Initial setup

## Recent Changes

**June 24, 2025**
- Enhanced user interface with better visual design and explanations
- Added gradient headers, color-coded AQI display, and health recommendations
- Improved sidebar with status indicators and helpful guidance sections
- Added informational boxes explaining technical concepts in simple terms
- Enhanced navigation with clearer tab names and usage tips
- Fixed all datetime assignment errors for stable operation

## User Preferences

Preferred communication style: Simple, everyday language.
Preferred UI style: User-friendly with clear explanations and visual appeal.