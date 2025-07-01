import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import time

# Import custom modules
from data_sources.satellite_data import SatelliteDataCollector
from data_sources.ground_data import GroundDataCollector
from data_sources.weather_data import WeatherDataCollector
from ml_models.prediction import PollutionPredictor
from utils.data_processing import DataProcessor
from utils.config import Config

# Page configuration
st.set_page_config(
    page_title="Air Pollution Monitoring & Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all data collectors and processors"""
    satellite_collector = SatelliteDataCollector()
    ground_collector = GroundDataCollector()
    weather_collector = WeatherDataCollector()
    predictor = PollutionPredictor()
    processor = DataProcessor()
    
    return satellite_collector, ground_collector, weather_collector, predictor, processor

def main():
    st.title("🌍 Air Pollution Monitor")
    st.markdown("""
    <div style='background: linear-gradient(90deg, #4CAF50, #2196F3); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: white; margin: 0;'>Real-time Air Quality Monitoring for Patna</h3>
        <p style='color: white; margin: 5px 0 0 0;'>Live data from satellites, ground sensors, and weather stations with AI predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add clear explanation of what this app does
    with st.expander("📖 What Does This App Do?", expanded=True):
        st.markdown("""
        ### 🎯 Purpose
        This app monitors air pollution in **Patna, Bihar** using advanced technology to help people understand air quality and protect their health.
        
        ### 🛠️ How It Works
        
        **🛰️ Satellite Monitoring**
        - ISRO satellites scan from space to detect pollution particles
        - Measures AOD (Aerosol Optical Depth) - how much pollution blocks sunlight
        - Covers large areas that ground sensors cannot reach
        
        **🌐 Ground Sensors**
        - Local monitoring stations measure harmful particles you breathe
        - Tracks PM2.5, PM10 (tiny particles), NO₂, SO₂, CO (toxic gases)
        - Provides street-level accuracy where people live and work
        
        **🌦️ Weather Analysis**
        - Wind, rain, temperature affect how pollution spreads
        - Strong winds blow pollution away, rain washes it out
        - High pressure can trap pollution near the ground
        
        **🤖 AI Predictions**
        - Machine learning models learn from historical patterns
        - Predicts air quality for the next 24 hours
        - Helps you plan outdoor activities and health precautions
        
        ### 📊 Understanding Air Quality Index (AQI)
        - **Green (0-50)**: Excellent - Safe for everyone
        - **Yellow (51-100)**: Good - Fine for most people
        - **Orange (101-150)**: Moderate - Sensitive people should be careful
        - **Red (151-200)**: Unhealthy - Everyone should limit outdoor time
        - **Purple (201+)**: Very Unhealthy - Stay indoors if possible
        
        ### 💡 Who Can Use This?
        - **Residents** - Check daily air quality before going outside
        - **Parents** - Protect children from harmful pollution
        - **Health-conscious individuals** - Plan exercise and outdoor activities
        - **Researchers** - Study pollution patterns and trends
        - **Government officials** - Monitor environmental conditions
        """)
    
    st.markdown("---")
    
    # Initialize components
    satellite_collector, ground_collector, weather_collector, predictor, processor = initialize_components()
    
    # Sidebar for location selection and controls
    with st.sidebar:
        st.markdown("### 📍 Location")
        
        # Location selection with better description
        location = st.selectbox(
            "Choose monitoring location:",
            ["Patna, Bihar", "Varanasi, UP", "Guwahati, Assam", "Shimla, HP", "Dehradun, UK"],
            index=0,
            help="Select a city to monitor air quality. Patna is currently the main focus."
        )
        
        # Get coordinates for selected location
        location_coords = Config.get_location_coordinates(location)
        
        # Location info in a nice box
        st.info(f"📍 **{location}**\n\n🗺️ {location_coords['lat']:.4f}°N, {location_coords['lon']:.4f}°E\n\n🏭 Region: {location_coords.get('region', 'Unknown')}")
        
        st.markdown("---")
        
        # Data refresh controls with better UX
        st.markdown("### ⚙️ Controls")
        auto_refresh = st.toggle("🔄 Auto-refresh (30s)", value=False, help="Automatically update data every 30 seconds")
        
        if st.button("🔄 Refresh Data Now", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Data source status with colors
        st.markdown("### 📊 Data Sources")
        
        # Create status indicators with colors
        st.markdown("""
        <div style='background-color: #f0f8f0; padding: 10px; border-radius: 5px; margin: 5px 0;'>
            🛰️ <strong>ISRO Satellite</strong><br>
            <span style='color: #4CAF50;'>● Online</span> - AOD Data
        </div>
        
        <div style='background-color: #f0f8f0; padding: 10px; border-radius: 5px; margin: 5px 0;'>
            🌐 <strong>Ground Sensors</strong><br>
            <span style='color: #4CAF50;'>● Online</span> - PM2.5, PM10
        </div>
        
        <div style='background-color: #f0f8f0; padding: 10px; border-radius: 5px; margin: 5px 0;'>
            🌦️ <strong>Weather Data</strong><br>
            <span style='color: #4CAF50;'>● Online</span> - Wind, Humidity
        </div>
        
        <div style='background-color: #f0f8f0; padding: 10px; border-radius: 5px; margin: 5px 0;'>
            🤖 <strong>AI Predictions</strong><br>
            <span style='color: #4CAF50;'>● Ready</span> - 24h Forecast
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Add help section
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            **Dashboard Tabs:**
            - 🏠 **Dashboard** - Overview & current status
            - 🛰️ **Satellite** - ISRO AOD measurements  
            - 🌐 **Ground** - Local air quality sensors
            - 🌦️ **Weather** - Meteorological impact
            - 🤖 **Predictions** - AI forecasting
            
            **Key Indicators:**
            - **PM2.5** - Fine particles (most harmful)
            - **AOD** - Satellite pollution measure
            - **AQI** - Overall air quality index
            """)
        
        # Add about section
        with st.expander("ℹ️ About This System"):
            st.markdown("""
            This system monitors air pollution in Patna using:
            
            🛰️ **ISRO Satellites** - Real-time pollution detection from space
            
            🌐 **Ground Sensors** - Local air quality measurements
            
            🌦️ **Weather Data** - How weather affects pollution
            
            🤖 **AI Models** - Predict pollution 24 hours ahead
            
            Built for smaller cities where monitoring is limited.
            """)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main dashboard content with better tab names
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", "🛰️ Satellite", "🌐 Ground Sensors", 
        "🌦️ Weather Impact", "🤖 AI Predictions"
    ])
    
    with tab1:
        display_main_dashboard(location_coords, satellite_collector, ground_collector, weather_collector, predictor, processor)
    
    with tab2:
        display_satellite_data(location_coords, satellite_collector)
    
    with tab3:
        display_ground_data(location_coords, ground_collector)
    
    with tab4:
        display_weather_impact(location_coords, weather_collector, processor)
    
    with tab5:
        display_ml_predictions(location_coords, predictor, processor)

def display_main_dashboard(location_coords, satellite_collector, ground_collector, weather_collector, predictor, processor):
    """Display the main integrated dashboard"""
    
    # Add current time and last update info
    st.markdown(f"""
    <div style='text-align: center; background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <h4 style='margin: 0; color: #495057;'>Live Air Quality Data</h4>
        <p style='margin: 5px 0 0 0; color: #6c757d;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Location map with better title
        st.markdown("### 📍 Monitoring Location - Patna, Bihar")
        st.markdown("*Real-time air quality monitoring point*")
        
        m = folium.Map(
            location=[location_coords['lat'], location_coords['lon']],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add location marker
        folium.Marker(
            [location_coords['lat'], location_coords['lon']],
            popup=f"Monitoring Station",
            tooltip="Air Quality Monitoring Point",
            icon=folium.Icon(color='red', icon='cloud')
        ).add_to(m)
        
        # Add pollution overlay (AOD visualization)
        try:
            aod_data = satellite_collector.get_current_aod(location_coords)
            if aod_data:
                # Add circle overlay for AOD visualization
                folium.Circle(
                    location=[location_coords['lat'], location_coords['lon']],
                    radius=5000,  # 5km radius
                    popup=f"AOD: {aod_data['aod_value']:.3f}",
                    color='orange',
                    fill=True,
                    opacity=0.6
                ).add_to(m)
        except Exception as e:
            st.warning(f"Satellite data temporarily unavailable: {str(e)}")
        
        st_folium(m, width=700, height=400)
    
    with col2:
        # Current air quality summary with better styling
        st.markdown("### 🎯 Air Quality Now")
        
        try:
            # Get current ground measurements
            ground_data = ground_collector.get_current_data(location_coords)
            
            if ground_data:
                # Create a more visual display of air quality
                pm25_value = ground_data.get('pm25', 0)
                pm25_status = get_aqi_status(pm25_value, 'PM2.5')
                overall_aqi = calculate_overall_aqi(ground_data)
                aqi_status = get_overall_aqi_status(overall_aqi)
                
                # Main AQI display
                aqi_color = get_aqi_color(overall_aqi)
                st.markdown(f"""
                <div style='text-align: center; background: linear-gradient(135deg, {aqi_color}, {aqi_color}99); 
                            padding: 20px; border-radius: 15px; margin: 10px 0; color: white;'>
                    <h2 style='margin: 0; font-size: 3em;'>{overall_aqi:.0f}</h2>
                    <h4 style='margin: 0;'>{aqi_status['category'].replace('_', ' ').title()}</h4>
                    <p style='margin: 5px 0 0 0; opacity: 0.9;'>Air Quality Index</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Key pollutants in a compact format
                col_pm25, col_pm10 = st.columns(2)
                with col_pm25:
                    st.metric("PM2.5", f"{pm25_value:.1f} μg/m³", 
                             delta=pm25_status['status'], delta_color="inverse")
                with col_pm10:
                    pm10_value = ground_data.get('pm10', 0)
                    pm10_status = get_aqi_status(pm10_value, 'PM10')
                    st.metric("PM10", f"{pm10_value:.1f} μg/m³", 
                             delta=pm10_status['status'], delta_color="inverse")
                
                # Additional info
                no2_value = ground_data.get('no2', 0)
                st.metric("NO₂", f"{no2_value:.1f} μg/m³", help="Nitrogen Dioxide from vehicles and industry")
                
                # Health recommendation
                if overall_aqi <= 50:
                    health_msg = "🟢 Great air quality! Perfect for outdoor activities."
                elif overall_aqi <= 100:
                    health_msg = "🟡 Good air quality. Safe for most people."
                elif overall_aqi <= 150:
                    health_msg = "🟠 Moderate. Sensitive people should limit outdoor activities."
                else:
                    health_msg = "🔴 Unhealthy air! Consider staying indoors."
                
                st.info(health_msg)
                
            else:
                st.error("❌ Ground data unavailable")
                
        except Exception as e:
            st.error(f"❌ Error fetching ground data: {str(e)}")
    
    # Add spacing and better section title
    st.markdown("---")
    st.markdown("### 📈 Pollution Trends (Last 24 Hours)")
    st.markdown("*Track how air quality changes throughout the day*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ground measurements trend
        try:
            historical_ground = ground_collector.get_historical_data(location_coords, hours=24)
            if historical_ground is not None and not historical_ground.empty:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('PM2.5 & PM10 Levels', 'Gas Concentrations'),
                    vertical_spacing=0.1
                )
                
                # PM measurements
                fig.add_trace(
                    go.Scatter(x=historical_ground['timestamp'], y=historical_ground['pm25'],
                             mode='lines+markers', name='PM2.5', line=dict(color='red')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=historical_ground['timestamp'], y=historical_ground['pm10'],
                             mode='lines+markers', name='PM10', line=dict(color='orange')),
                    row=1, col=1
                )
                
                # Gas measurements
                fig.add_trace(
                    go.Scatter(x=historical_ground['timestamp'], y=historical_ground['no2'],
                             mode='lines+markers', name='NO₂', line=dict(color='blue')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=historical_ground['timestamp'], y=historical_ground['so2'],
                             mode='lines+markers', name='SO₂', line=dict(color='green')),
                    row=2, col=1
                )
                
                fig.update_layout(height=400, title="Ground Measurements (24h)")
                fig.update_xaxes(title_text="Time")
                fig.update_yaxes(title_text="μg/m³", row=1, col=1)
                fig.update_yaxes(title_text="μg/m³", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Loading historical ground data...")
        except Exception as e:
            st.error(f"❌ Error displaying ground trends: {str(e)}")
    
    with col2:
        # Satellite AOD trend
        try:
            historical_satellite = satellite_collector.get_historical_aod(location_coords, days=7)
            if historical_satellite is not None and not historical_satellite.empty:
                fig = px.line(
                    historical_satellite, 
                    x='timestamp', 
                    y='aod_value',
                    title='Satellite AOD - 7 Days',
                    labels={'aod_value': 'AOD Value', 'timestamp': 'Date'}
                )
                fig.update_traces(line_color='purple', mode='lines+markers')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📡 Loading satellite AOD data...")
        except Exception as e:
            st.error(f"❌ Error displaying satellite trends: {str(e)}")
    
    # Prediction summary with better styling
    st.markdown("---")
    st.markdown("### 🔮 24-Hour Air Quality Forecast")
    st.markdown("*AI predictions based on weather patterns and historical data*")
    try:
        predictions = predictor.predict_next_24h(location_coords)
        if predictions:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pred_pm25 = predictions.get('pm25_24h', 0)
                current_pm25 = ground_data.get('pm25', 0) if 'ground_data' in locals() else 0
                delta_pm25 = pred_pm25 - current_pm25
                st.metric(
                    "Predicted PM2.5", 
                    f"{pred_pm25:.1f} μg/m³",
                    delta=f"{delta_pm25:+.1f}"
                )
            
            with col2:
                pred_pm10 = predictions.get('pm10_24h', 0)
                current_pm10 = ground_data.get('pm10', 0) if 'ground_data' in locals() else 0
                delta_pm10 = pred_pm10 - current_pm10
                st.metric(
                    "Predicted PM10", 
                    f"{pred_pm10:.1f} μg/m³",
                    delta=f"{delta_pm10:+.1f}"
                )
            
            with col3:
                pred_aqi = predictions.get('aqi_24h', 0)
                current_aqi = overall_aqi if 'overall_aqi' in locals() else 0
                delta_aqi = pred_aqi - current_aqi
                st.metric(
                    "Predicted AQI", 
                    f"{pred_aqi:.0f}",
                    delta=f"{delta_aqi:+.0f}"
                )
            
            with col4:
                confidence = predictions.get('confidence', 0) * 100
                st.metric(
                    "Confidence", 
                    f"{confidence:.1f}%"
                )
                
        else:
            st.info("🤖 Generating ML predictions...")
    except Exception as e:
        st.error(f"❌ Error generating predictions: {str(e)}")

def display_satellite_data(location_coords, satellite_collector):
    """Display detailed satellite data analysis"""
    
    # Header with better styling
    st.markdown("### 🛰️ ISRO Satellite Data")
    st.markdown("*Aerosol Optical Depth (AOD) measurements from space*")
    
    # AOD explanation in a more prominent way
    st.info("""
    **What is AOD?** Aerosol Optical Depth measures pollution particles in the atmosphere from satellites.
    
    📊 **Scale**: 0.0 (very clean) → 1.0+ (very polluted)  
    🛰️ **Source**: ISRO satellites (INSAT, MODIS)  
    🔍 **Shows**: Dust, smoke, haze, and fine particles from space
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        try:
            # Current AOD value
            current_aod = satellite_collector.get_current_aod(location_coords)
            if current_aod:
                aod_status = get_aod_status(current_aod['aod_value'])
                st.metric(
                    "Current AOD", 
                    f"{current_aod['aod_value']:.3f}",
                    delta=aod_status['description']
                )
                
                # AOD interpretation gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = current_aod['aod_value'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "AOD Level"},
                    delta = {'reference': 0.2},
                    gauge = {
                        'axis': {'range': [None, 1.0]},
                        'bar': {'color': aod_status['color']},
                        'steps': [
                            {'range': [0, 0.1], 'color': "lightgreen"},
                            {'range': [0.1, 0.3], 'color': "yellow"},
                            {'range': [0.3, 0.6], 'color': "orange"},
                            {'range': [0.6, 1.0], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("❌ Current AOD data unavailable")
                
        except Exception as e:
            st.error(f"❌ Error fetching current AOD: {str(e)}")
    
    with col2:
        try:
            # AOD historical trend
            historical_aod = satellite_collector.get_historical_aod(location_coords, days=30)
            if historical_aod is not None and not historical_aod.empty:
                fig = px.line(
                    historical_aod,
                    x='timestamp',
                    y='aod_value',
                    title='AOD Trend - Last 30 Days',
                    labels={'aod_value': 'AOD Value', 'timestamp': 'Date'}
                )
                
                # Add horizontal reference lines
                fig.add_hline(y=0.1, line_dash="dash", line_color="green", annotation_text="Clean")
                fig.add_hline(y=0.3, line_dash="dash", line_color="orange", annotation_text="Moderate")
                fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="High")
                
                fig.update_traces(line_color='purple', mode='lines+markers')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📡 Loading AOD historical data...")
                
        except Exception as e:
            st.error(f"❌ Error displaying AOD trends: {str(e)}")
    
    # Detailed satellite data table
    st.subheader("📊 Recent Satellite Observations")
    try:
        recent_data = satellite_collector.get_recent_observations(location_coords, days=7)
        if recent_data is not None and not recent_data.empty:
            # Format the data for display
            display_data = recent_data.copy()
            display_data['timestamp'] = pd.to_datetime(display_data['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            display_data['aod_value'] = display_data['aod_value'].round(3)
            display_data['quality'] = display_data['aod_value'].apply(lambda x: get_aod_status(x)['description'])
            
            st.dataframe(
                display_data[['timestamp', 'aod_value', 'quality']],
                column_config={
                    'timestamp': 'Date & Time',
                    'aod_value': 'AOD Value',
                    'quality': 'Air Quality'
                },
                use_container_width=True
            )
        else:
            st.info("📡 Loading recent satellite observations...")
    except Exception as e:
        st.error(f"❌ Error displaying recent observations: {str(e)}")

def display_ground_data(location_coords, ground_collector):
    """Display detailed ground monitoring data"""
    
    # Header with better styling
    st.markdown("### 🌐 Ground Air Quality Sensors")
    st.markdown("*Real-time measurements from local monitoring stations*")
    
    # Key pollutants explanation
    st.info("""
    **Ground Sensors** measure harmful particles and gases at street level where people breathe.
    
    🔴 **PM2.5** - Fine particles (most dangerous to health)  
    🟠 **PM10** - Coarse particles (causes breathing problems)  
    🔵 **NO₂** - Vehicle & industry emissions  
    🟡 **SO₂** - Power plant emissions  
    ⚫ **CO** - Incomplete combustion gases  
    """)
    
    # Pollutant explanations
    with st.expander("ℹ️ About Air Pollutants"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
            **PM2.5**: Particles ≤ 2.5 micrometers
            - Can enter lungs and bloodstream
            - Most dangerous pollutant
            - Sources: vehicles, industry, burning
            
            **PM10**: Particles ≤ 10 micrometers  
            - Can reach lower respiratory tract
            - Sources: dust, construction, roads
            """)
        with col2:
            st.write("""
            **NO₂**: Nitrogen Dioxide
            - Respiratory irritant
            - Sources: vehicles, power plants
            
            **SO₂**: Sulfur Dioxide
            - Causes acid rain
            - Sources: coal burning, industry
            
            **CO**: Carbon Monoxide
            - Reduces oxygen in blood
            - Sources: incomplete combustion
            """)
    
    try:
        # Current measurements
        current_data = ground_collector.get_current_data(location_coords)
        
        if current_data:
            # Real-time measurements grid
            st.subheader("📊 Current Readings")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                pm25_val = current_data.get('pm25', 0)
                pm25_status = get_aqi_status(pm25_val, 'PM2.5')
                st.metric(
                    "PM2.5", 
                    f"{pm25_val:.1f} μg/m³",
                    delta=pm25_status['status'],
                    delta_color="inverse"
                )
            
            with col2:
                pm10_val = current_data.get('pm10', 0)
                pm10_status = get_aqi_status(pm10_val, 'PM10')
                st.metric(
                    "PM10", 
                    f"{pm10_val:.1f} μg/m³",
                    delta=pm10_status['status'],
                    delta_color="inverse"
                )
            
            with col3:
                no2_val = current_data.get('no2', 0)
                st.metric("NO₂", f"{no2_val:.1f} μg/m³")
            
            with col4:
                so2_val = current_data.get('so2', 0)
                st.metric("SO₂", f"{so2_val:.1f} μg/m³")
            
            with col5:
                co_val = current_data.get('co', 0)
                st.metric("CO", f"{co_val:.1f} mg/m³")
            
            # Detailed charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Particulate matter comparison
                pm_data = pd.DataFrame({
                    'Pollutant': ['PM2.5', 'PM10'],
                    'Current Level': [pm25_val, pm10_val],
                    'WHO Guideline': [15, 45]  # WHO annual guidelines
                })
                
                fig = px.bar(
                    pm_data,
                    x='Pollutant',
                    y=['Current Level', 'WHO Guideline'],
                    title='PM Levels vs WHO Guidelines',
                    barmode='group'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Gas pollutants
                gas_data = pd.DataFrame({
                    'Gas': ['NO₂', 'SO₂', 'CO'],
                    'Level': [no2_val, so2_val, co_val * 1000]  # Convert CO to μg/m³
                })
                
                fig = px.bar(
                    gas_data,
                    x='Gas',
                    y='Level',
                    title='Gas Pollutant Levels',
                    color='Level',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("❌ Current ground measurement data unavailable")
    
    except Exception as e:
        st.error(f"❌ Error fetching current ground data: {str(e)}")
    
    # Historical trends
    st.subheader("📈 7-Day Trends")
    try:
        historical_data = ground_collector.get_historical_data(location_coords, hours=168)  # 7 days
        
        if historical_data is not None and not historical_data.empty:
            # Multi-pollutant trend chart
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Particulate Matter (PM)', 'Nitrogen Dioxide (NO₂)', 'Other Gases'),
                vertical_spacing=0.08
            )
            
            # PM trends
            fig.add_trace(
                go.Scatter(x=historical_data['timestamp'], y=historical_data['pm25'],
                         mode='lines', name='PM2.5', line=dict(color='red')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=historical_data['timestamp'], y=historical_data['pm10'],
                         mode='lines', name='PM10', line=dict(color='orange')),
                row=1, col=1
            )
            
            # NO2 trends
            fig.add_trace(
                go.Scatter(x=historical_data['timestamp'], y=historical_data['no2'],
                         mode='lines', name='NO₂', line=dict(color='blue')),
                row=2, col=1
            )
            
            # Other gases
            fig.add_trace(
                go.Scatter(x=historical_data['timestamp'], y=historical_data['so2'],
                         mode='lines', name='SO₂', line=dict(color='green')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=historical_data['timestamp'], y=historical_data['co']*1000,
                         mode='lines', name='CO (×1000)', line=dict(color='purple')),
                row=3, col=1
            )
            
            fig.update_layout(height=600, title="7-Day Pollution Trends")
            fig.update_xaxes(title_text="Date & Time")
            fig.update_yaxes(title_text="μg/m³", row=1, col=1)
            fig.update_yaxes(title_text="μg/m³", row=2, col=1)
            fig.update_yaxes(title_text="μg/m³", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Loading 7-day historical data...")
            
    except Exception as e:
        st.error(f"❌ Error displaying historical trends: {str(e)}")

def display_weather_impact(location_coords, weather_collector, processor):
    """Display weather impact on pollution"""
    st.subheader("🌦️ Weather Impact Analysis")
    
    with st.expander("ℹ️ How Weather Affects Pollution"):
        st.write("""
        **Wind Speed & Direction**: Disperses pollutants or brings them from other areas
        **Temperature**: Affects chemical reactions and pollutant formation
        **Humidity**: Influences particle behavior and visibility
        **Pressure**: Higher pressure can trap pollutants near ground level
        **Precipitation**: Rain washes out particles from the atmosphere
        """)
    
    try:
        # Current weather conditions
        current_weather = weather_collector.get_current_weather(location_coords)
        
        if current_weather:
            st.subheader("🌡️ Current Weather Conditions")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                temp = current_weather.get('temperature', 0)
                st.metric("Temperature", f"{temp:.1f}°C")
            
            with col2:
                humidity = current_weather.get('humidity', 0)
                st.metric("Humidity", f"{humidity:.0f}%")
            
            with col3:
                wind_speed = current_weather.get('wind_speed', 0)
                st.metric("Wind Speed", f"{wind_speed:.1f} m/s")
            
            with col4:
                pressure = current_weather.get('pressure', 0)
                st.metric("Pressure", f"{pressure:.0f} hPa")
            
            # Wind direction visualization
            col1, col2 = st.columns(2)
            
            with col1:
                wind_dir = current_weather.get('wind_direction', 0)
                
                # Create wind rose-like visualization
                fig = go.Figure()
                
                # Add wind direction arrow
                r = 1
                x = r * np.sin(np.radians(wind_dir))
                y = r * np.cos(np.radians(wind_dir))
                
                fig.add_trace(go.Scatter(
                    x=[0, x], y=[0, y],
                    mode='lines+markers',
                    line=dict(color='blue', width=5),
                    marker=dict(size=[5, 15], color='blue'),
                    name=f'Wind: {wind_dir:.0f}°'
                ))
                
                fig.update_layout(
                    title=f"Wind Direction: {wind_dir:.0f}° ({get_wind_direction_name(wind_dir)})",
                    xaxis=dict(range=[-1.5, 1.5], showgrid=True),
                    yaxis=dict(range=[-1.5, 1.5], showgrid=True),
                    height=300,
                    showlegend=False
                )
                
                # Add compass directions
                for angle, label in [(0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')]:
                    x_label = 1.2 * np.sin(np.radians(angle))
                    y_label = 1.2 * np.cos(np.radians(angle))
                    fig.add_annotation(x=x_label, y=y_label, text=label, showarrow=False)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Weather impact score
                impact_score = processor.calculate_weather_impact(current_weather)
                impact_status = get_weather_impact_status(impact_score)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = impact_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Weather Impact on Pollution"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': impact_status['color']},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.write(f"**Impact Level**: {impact_status['description']}")
        
        else:
            st.error("❌ Current weather data unavailable")
    
    except Exception as e:
        st.error(f"❌ Error fetching current weather: {str(e)}")
    
    # Weather-pollution correlation analysis
    st.subheader("📊 Weather-Pollution Correlation (Past 7 Days)")
    try:
        correlation_data = processor.get_weather_pollution_correlation(location_coords, days=7)
        
        if correlation_data is not None and not correlation_data.empty:
            # Multi-axis plot showing weather and pollution together
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Temperature vs PM2.5', 'Wind Speed vs PM2.5', 
                               'Humidity vs PM2.5', 'Pressure vs PM2.5'),
                specs=[[{"secondary_y": True}, {"secondary_y": True}],
                       [{"secondary_y": True}, {"secondary_y": True}]]
            )
            
            # Temperature vs PM2.5
            fig.add_trace(
                go.Scatter(x=correlation_data['timestamp'], y=correlation_data['temperature'],
                         name='Temperature (°C)', line=dict(color='red')),
                row=1, col=1, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=correlation_data['timestamp'], y=correlation_data['pm25'],
                         name='PM2.5 (μg/m³)', line=dict(color='brown')),
                row=1, col=1, secondary_y=True
            )
            
            # Wind Speed vs PM2.5
            fig.add_trace(
                go.Scatter(x=correlation_data['timestamp'], y=correlation_data['wind_speed'],
                         name='Wind Speed (m/s)', line=dict(color='blue')),
                row=1, col=2, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=correlation_data['timestamp'], y=correlation_data['pm25'],
                         name='PM2.5 (μg/m³)', line=dict(color='brown')),
                row=1, col=2, secondary_y=True
            )
            
            # Humidity vs PM2.5
            fig.add_trace(
                go.Scatter(x=correlation_data['timestamp'], y=correlation_data['humidity'],
                         name='Humidity (%)', line=dict(color='green')),
                row=2, col=1, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=correlation_data['timestamp'], y=correlation_data['pm25'],
                         name='PM2.5 (μg/m³)', line=dict(color='brown')),
                row=2, col=1, secondary_y=True
            )
            
            # Pressure vs PM2.5
            fig.add_trace(
                go.Scatter(x=correlation_data['timestamp'], y=correlation_data['pressure'],
                         name='Pressure (hPa)', line=dict(color='purple')),
                row=2, col=2, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=correlation_data['timestamp'], y=correlation_data['pm25'],
                         name='PM2.5 (μg/m³)', line=dict(color='brown')),
                row=2, col=2, secondary_y=True
            )
            
            fig.update_layout(height=500, title="Weather-Pollution Relationships")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Loading weather-pollution correlation data...")
            
    except Exception as e:
        st.error(f"❌ Error displaying weather-pollution correlation: {str(e)}")

def display_ml_predictions(location_coords, predictor, processor):
    """Display ML predictions and model information"""
    
    # Header with better styling
    st.markdown("### 🤖 AI-Powered Predictions")
    st.markdown("*Machine learning forecasts based on satellite, ground, and weather data*")
    
    with st.expander("ℹ️ About Our ML Models"):
        st.write("""
        **Prediction Models Used:**
        - **Random Forest Regressor**: For PM2.5 and PM10 predictions
        - **Time Series Analysis**: For trend-based forecasting
        - **Weather Integration**: Uses meteorological data to improve accuracy
        - **Feature Engineering**: Combines satellite AOD, ground measurements, and weather data
        
        **Model Features:**
        - Historical pollution levels (24h, 48h, 72h averages)
        - Weather conditions (temperature, humidity, wind, pressure)
        - Satellite AOD measurements
        - Time-based features (hour, day of week, season)
        """)
    
    try:
        # Generate predictions for different time horizons
        predictions_6h = predictor.predict_next_hours(location_coords, hours=6)
        predictions_24h = predictor.predict_next_hours(location_coords, hours=24)
        predictions_48h = predictor.predict_next_hours(location_coords, hours=48)
        
        # Display prediction summary
        st.subheader("🔮 Forecast Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**6 Hours**")
            if predictions_6h:
                st.metric("PM2.5", f"{predictions_6h.get('pm25', 0):.1f} μg/m³")
                st.metric("PM10", f"{predictions_6h.get('pm10', 0):.1f} μg/m³")
                st.metric("AQI", f"{predictions_6h.get('aqi', 0):.0f}")
        
        with col2:
            st.write("**24 Hours**")
            if predictions_24h:
                st.metric("PM2.5", f"{predictions_24h.get('pm25', 0):.1f} μg/m³")
                st.metric("PM10", f"{predictions_24h.get('pm10', 0):.1f} μg/m³")
                st.metric("AQI", f"{predictions_24h.get('aqi', 0):.0f}")
        
        with col3:
            st.write("**48 Hours**")
            if predictions_48h:
                st.metric("PM2.5", f"{predictions_48h.get('pm25', 0):.1f} μg/m³")
                st.metric("PM10", f"{predictions_48h.get('pm10', 0):.1f} μg/m³")
                st.metric("AQI", f"{predictions_48h.get('aqi', 0):.0f}")
        
        # Prediction confidence and uncertainty
        col1, col2 = st.columns(2)
        
        with col1:
            # Model confidence indicators
            st.subheader("📊 Model Performance")
            
            model_metrics = predictor.get_model_metrics()
            if model_metrics:
                st.metric("PM2.5 Accuracy", f"{model_metrics.get('pm25_r2', 0)*100:.1f}%")
                st.metric("PM10 Accuracy", f"{model_metrics.get('pm10_r2', 0)*100:.1f}%")
                st.metric("Mean Error", f"{model_metrics.get('mae', 0):.2f} μg/m³")
                
                # Model accuracy visualization
                metrics_data = pd.DataFrame({
                    'Metric': ['PM2.5 R²', 'PM10 R²', 'Overall Accuracy'],
                    'Score': [
                        model_metrics.get('pm25_r2', 0),
                        model_metrics.get('pm10_r2', 0),
                        (model_metrics.get('pm25_r2', 0) + model_metrics.get('pm10_r2', 0)) / 2
                    ]
                })
                
                fig = px.bar(
                    metrics_data,
                    x='Metric',
                    y='Score',
                    title='Model Accuracy Scores',
                    color='Score',
                    color_continuous_scale='Greens'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🤖 Training model metrics...")
        
        with col2:
            # Feature importance
            st.subheader("🎯 Important Factors")
            
            feature_importance = predictor.get_feature_importance()
            if feature_importance:
                importance_df = pd.DataFrame(
                    list(feature_importance.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance for Predictions',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🎯 Calculating feature importance...")
        
        # Detailed hourly predictions
        st.subheader("📈 48-Hour Detailed Forecast")
        
        hourly_predictions = predictor.get_hourly_predictions(location_coords, hours=48)
        if hourly_predictions is not None and not hourly_predictions.empty:
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('PM2.5 & PM10 Forecast', 'AQI Forecast'),
                vertical_spacing=0.1
            )
            
            # PM forecasts
            fig.add_trace(
                go.Scatter(
                    x=hourly_predictions['timestamp'],
                    y=hourly_predictions['pm25'],
                    mode='lines+markers',
                    name='PM2.5 Forecast',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=hourly_predictions['timestamp'],
                    y=hourly_predictions['pm10'],
                    mode='lines+markers',
                    name='PM10 Forecast',
                    line=dict(color='orange')
                ),
                row=1, col=1
            )
            
            # Add confidence intervals if available
            if 'pm25_lower' in hourly_predictions.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hourly_predictions['timestamp'],
                        y=hourly_predictions['pm25_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=hourly_predictions['timestamp'],
                        y=hourly_predictions['pm25_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.2)',
                        name='PM2.5 Confidence',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # AQI forecast
            fig.add_trace(
                go.Scatter(
                    x=hourly_predictions['timestamp'],
                    y=hourly_predictions['aqi'],
                    mode='lines+markers',
                    name='AQI Forecast',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
            
            # Add AQI category lines
            fig.add_hline(y=50, line_dash="dash", line_color="green", 
                         annotation_text="Good", row=2, col=1)
            fig.add_hline(y=100, line_dash="dash", line_color="yellow", 
                         annotation_text="Moderate", row=2, col=1)
            fig.add_hline(y=150, line_dash="dash", line_color="orange", 
                         annotation_text="Unhealthy for Sensitive", row=2, col=1)
            
            fig.update_layout(
                height=500,
                title="48-Hour Air Quality Forecast",
                xaxis_title="Date & Time"
            )
            fig.update_yaxes(title_text="μg/m³", row=1, col=1)
            fig.update_yaxes(title_text="AQI", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📈 Generating detailed hourly forecasts...")
        
        # Alert system
        st.subheader("⚠️ Pollution Alerts")
        alerts = predictor.generate_alerts(location_coords)
        
        if alerts:
            for alert in alerts:
                if alert['level'] == 'high':
                    st.error(f"🚨 {alert['message']}")
                elif alert['level'] == 'medium':
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        else:
            st.success("✅ No pollution alerts for the next 48 hours")
    
    except Exception as e:
        st.error(f"❌ Error generating ML predictions: {str(e)}")

# Helper functions for status calculations
def get_aqi_status(value, pollutant):
    """Get AQI status for a specific pollutant"""
    if pollutant == 'PM2.5':
        if value <= 12: return {'status': 'Good', 'color': 'green'}
        elif value <= 35: return {'status': 'Moderate', 'color': 'yellow'}
        elif value <= 55: return {'status': 'Unhealthy for Sensitive', 'color': 'orange'}
        elif value <= 150: return {'status': 'Unhealthy', 'color': 'red'}
        else: return {'status': 'Very Unhealthy', 'color': 'purple'}
    elif pollutant == 'PM10':
        if value <= 54: return {'status': 'Good', 'color': 'green'}
        elif value <= 154: return {'status': 'Moderate', 'color': 'yellow'}
        elif value <= 254: return {'status': 'Unhealthy for Sensitive', 'color': 'orange'}
        elif value <= 354: return {'status': 'Unhealthy', 'color': 'red'}
        else: return {'status': 'Very Unhealthy', 'color': 'purple'}

def get_aod_status(aod_value):
    """Get AOD status and description"""
    if aod_value < 0.1:
        return {'description': 'Very Clean', 'color': 'green'}
    elif aod_value < 0.3:
        return {'description': 'Clean', 'color': 'lightgreen'}
    elif aod_value < 0.6:
        return {'description': 'Moderate', 'color': 'yellow'}
    elif aod_value < 1.0:
        return {'description': 'Polluted', 'color': 'orange'}
    else:
        return {'description': 'Very Polluted', 'color': 'red'}

def calculate_overall_aqi(data):
    """Calculate overall AQI from pollutant data"""
    pm25_aqi = data.get('pm25', 0) * 2  # Simplified AQI calculation
    pm10_aqi = data.get('pm10', 0) * 1
    return max(pm25_aqi, pm10_aqi)

def get_overall_aqi_status(aqi):
    """Get overall AQI category"""
    if aqi <= 50: return {'category': 'Good', 'color': 'green'}
    elif aqi <= 100: return {'category': 'Moderate', 'color': 'yellow'}
    elif aqi <= 150: return {'category': 'Unhealthy for Sensitive', 'color': 'orange'}
    elif aqi <= 200: return {'category': 'Unhealthy', 'color': 'red'}
    else: return {'category': 'Very Unhealthy', 'color': 'purple'}

def get_wind_direction_name(degrees):
    """Convert wind direction degrees to compass direction"""
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = round(degrees / 22.5) % 16
    return directions[idx]

def get_weather_impact_status(score):
    """Get weather impact status"""
    if score < 30:
        return {'description': 'Low Impact - Weather helps disperse pollution', 'color': 'green'}
    elif score < 60:
        return {'description': 'Moderate Impact - Mixed weather conditions', 'color': 'yellow'}
    else:
        return {'description': 'High Impact - Weather worsens pollution', 'color': 'red'}

def get_aqi_color(aqi_value):
    """Get color for AQI value"""
    if aqi_value <= 50:
        return "#4CAF50"  # Green
    elif aqi_value <= 100:
        return "#FFEB3B"  # Yellow
    elif aqi_value <= 150:
        return "#FF9800"  # Orange
    elif aqi_value <= 200:
        return "#F44336"  # Red
    elif aqi_value <= 300:
        return "#9C27B0"  # Purple
    else:
        return "#800020"  # Maroon

if __name__ == "__main__":
    main()
