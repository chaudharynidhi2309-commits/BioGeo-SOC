import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from src.inference import predict_soc
from geopy.geocoders import Nominatim

# 1. Page Setup (ONLY ONCE!)
st.set_page_config(
    page_title="BioGeo-SOC Advanced", 
    page_icon="🌱", 
    layout="wide"
)

# Persistent Data Storage
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar
with st.sidebar:
    st.header("⚙️ System Specs")
    st.success("✅ Model: Random Forest")
    st.info("🛰️ Sensor: Sentinel-2")
    st.markdown("---")
    st.caption("BioGeo-SOC v1.0")
    st.caption("Gujarat Soil Analysis")

st.title("🌱 Soil Organic Carbon (SOC) Advanced Analytics")
st.markdown("*Predict soil health using satellite data for any location in Gujarat*")

# Search UI
geolocator = Nominatim(user_agent="soc_predictor_amnex", timeout=10)

col1, col2 = st.columns([3, 1])
with col1:
    location_name = st.text_input(
        "Enter Village/Location Name", 
        placeholder="e.g., Chotila, Dholka, Bavla"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze Location", type="primary", use_container_width=True)

if analyze_btn:
    if location_name:
        with st.spinner("🛰️ Fetching Satellite Data & Processing..."):
            try:
                location = geolocator.geocode(f"{location_name}, Gujarat, India")
                
                if location:
                    st.info(f"📍 Found: {location.address}")
                    
                    # Get prediction
                    soc_val, soc_std, indices = predict_soc(
                        location.latitude, 
                        location.longitude,
                        use_grid_average=True,
                        grid_size=3
                    )
                    
                    if soc_val is not None:
                        st.session_state.results = {
                            'soc': soc_val, 
                            'std': soc_std, 
                            'indices': indices,
                            'lat': location.latitude, 
                            'lon': location.longitude, 
                            'addr': location.address
                        }
                        st.success("✅ Analysis Complete!")
                    else:
                        st.error("❌ No satellite data available for this location. Try a different location.")
                else:
                    st.error("❌ Location not found. Please enter a valid Gujarat location.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tip: Try a well-known village or town name in Gujarat")
    else:
        st.warning("⚠️ Please enter a location name")

# --- THE "PROPER" DASHBOARD LAYOUT ---
if st.session_state.results:
    res = st.session_state.results
    
    st.markdown("---")
    st.markdown(f"### 📊 Analysis for: {res['addr']}")
    
    # ROW 1: LARGE METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted SOC", f"{res['soc']:.2f} g/kg", delta=None)
    m2.metric("Uncertainty", f"±{res['std']:.2f} g/kg", delta=None)
    m3.metric("NDVI (Vegetation)", f"{res['indices']['ndvi']:.3f}", delta=None)
    
    # Confidence based on std
    if res['std'] < 1.0:
        confidence = "High ✅"
        conf_color = "green"
    elif res['std'] < 2.0:
        confidence = "Medium ⚠️"
        conf_color = "orange"
    else:
        confidence = "Low ❌"
        conf_color = "red"
    
    m4.metric("Confidence", confidence, delta=None)

    # ROW 2: SIDE-BY-SIDE CHARTS
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("🌡️ Soil Quality Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=res['soc'],
            title={'text': "SOC (g/kg)"},
            delta={'reference': 15},  # Ideal SOC for agriculture
            gauge={
                'axis': {'range': [0, 25]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 10], 'color': "#ff4b4b"},
                    {'range': [10, 20], 'color': "#ffa500"},
                    {'range': [20, 25], 'color': "#2ecc71"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 15
                }
            }
        ))
        fig_gauge.update_layout(
            margin=dict(l=20, r=20, t=50, b=20), 
            height=350
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key="g_final")

    with chart_col2:
        st.subheader("🕸️ Environmental Signature")
        fig_radar = go.Figure(go.Scatterpolar(
            r=[
                abs(res['indices']['ndvi']), 
                abs(res['indices']['evi'])/3,  # Scale EVI to 0-1
                abs(res['indices']['ndwi'])
            ],
            theta=['NDVI', 'EVI', 'NDWI'],
            fill='toself',
            fillcolor='rgba(46, 139, 87, 0.4)',
            line=dict(color='seagreen')
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=350,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="r_final")

    # ROW 3: FULL WIDTH MAP
    st.markdown("---")
    st.subheader("🗺️ Satellite Context & Location")
    
    # Create map with satellite imagery
    m = folium.Map(
        location=[res['lat'], res['lon']], 
        zoom_start=15,
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite'
    )
    
    # Add marker with popup
    folium.Marker(
        [res['lat'], res['lon']],
        popup=f"SOC: {res['soc']:.2f} g/kg<br>NDVI: {res['indices']['ndvi']:.3f}",
        tooltip="Click for details",
        icon=folium.Icon(color='green', icon='leaf', prefix='fa')
    ).add_to(m)
    
    # Add circle to show analysis area
    folium.Circle(
        [res['lat'], res['lon']],
        radius=150,  # 150 meter radius (covers 3x3 grid)
        color='yellow',
        fill=True,
        fillOpacity=0.2,
        popup='Analysis Area (3x3 grid)'
    ).add_to(m)
    
    st_folium(m, width=None, height=450, key="map_final")
    
    # Interpretation
    st.markdown("---")
    st.subheader("📋 Interpretation")
    
    col_int1, col_int2 = st.columns(2)
    
    with col_int1:
        st.markdown("**Soil Quality:**")
        if res['soc'] < 10:
            st.warning("🟡 Low soil carbon - Consider adding organic matter")
        elif res['soc'] < 20:
            st.success("🟢 Good soil carbon - Suitable for agriculture")
        else:
            st.info("🔵 High soil carbon - Excellent soil quality")
    
    with col_int2:
        st.markdown("**Vegetation Health:**")
        if res['indices']['ndvi'] < 0.2:
            st.warning("🟡 Sparse vegetation or bare soil")
        elif res['indices']['ndvi'] < 0.6:
            st.success("🟢 Moderate to healthy vegetation")
        else:
            st.info("🔵 Dense, healthy vegetation")
else:
    # Show example when no results
    st.info("👆 Enter a location name above to start analysis")
    
    st.markdown("---")
    st.markdown("### 📍 Example Locations to Try:")
    
    examples = st.columns(4)
    examples[0].button("📍 Dholka", key="ex1")
    examples[1].button("📍 Bavla", key="ex2")
    examples[2].button("📍 Chotila", key="ex3")
    examples[3].button("📍 Kalol", key="ex4")