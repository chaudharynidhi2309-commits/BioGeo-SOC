import streamlit as st
import rasterio
import os
from pathlib import Path

# Gets the path to the 'data' folder regardless of where the script runs
BASE_DIR = Path(__file__).resolve().parent.parent
TIF_PATH = BASE_DIR / "data" / "gujarat_soc.tif"

@st.cache_resource
def get_validation_soc(lat, lon):
    """
    Fetch ground truth SOC value from SoilGrids GeoTIFF for validation.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        float: SOC value in g/kg, or None if not available
    """
    if not TIF_PATH.exists():
        return None

    try:
        with rasterio.open(str(TIF_PATH)) as src:
            # Note: rasterio uses (longitude, latitude)
            row, col = src.index(lon, lat)
            window = rasterio.windows.Window(col, row, 1, 1)
            data = src.read(1, window=window)
            
            raw_val = data[0, 0]
            
            # Check for 'No Data' values (usually very large negative numbers)
            if raw_val < 0 or raw_val > 1000: 
                return None
                
            # SoilGrids scaling: 152 becomes 15.2 g/kg
            return round(float(raw_val) / 10.0, 2)
    except Exception:
        return None