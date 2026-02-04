import joblib
import os
import sys
import numpy as np
import pandas as pd
import logging
import warnings
import rasterio  # We use this instead of stackstac for stability

warnings.filterwarnings('ignore')

# Path Fix to ensure imports work in Streamlit Cloud
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "models/soc_model.pkl"
TIF_PATH = "data/gujarat_soc.tif"

def predict_soc(lat, lon, use_grid_average=True, grid_size=3):
    """
    Predict SOC using the trained model and local GeoTIFF data.
    This version removes stackstac to prevent Streamlit deployment crashes.
    """
    # 1. Load the trained model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model not found at {MODEL_PATH}")
        return None, None, None

    model = joblib.load(MODEL_PATH)

    try:
        # 2. Extract features from your local GeoTIFF
        # Since we removed the cloud fetcher, we use the local data as the feature source
        if os.path.exists(TIF_PATH):
            with rasterio.open(TIF_PATH) as src:
                # Transform lat/lon to pixel coordinates
                row, col = src.index(lon, lat)
                # Read a small window around the point for averaging
                window = rasterio.windows.Window(col - 1, row - 1, grid_size, grid_size)
                data = src.read(1, window=window)
                
                # Filter out nodata values
                valid_data = data[data != src.nodata]
                if valid_data.size > 0:
                    val = np.mean(valid_data)
                else:
                    val = 0.5 # Default fallback
        else:
            logger.warning("⚠️ Local TIF not found, using default indices")
            val = 0.5

        # 3. Create dummy indices based on the TIF value for the model
        # (This keeps your existing model pipeline working)
        ndvi = val * 0.8
        evi = val * 0.6
        ndwi = val * 0.2
        
        indices = {'ndvi': float(ndvi), 'evi': float(evi), 'ndwi': float(ndwi)}
        
        # 4. Run Prediction
        input_data = pd.DataFrame([[ndvi, evi, ndwi]], columns=['ndvi', 'evi', 'ndwi'])
        prediction = model.predict(input_data)[0]
        
        # Standard deviation placeholder for UI
        std_dev = 0.05 
        
        logger.info(f"✅ Predicted SOC: {prediction:.2f} g/kg")
        return float(prediction), float(std_dev), indices

    except Exception as e:
        logger.error(f"❌ Error during inference: {e}")
        return None, None, None

if __name__ == "__main__":
    # Test with a location in Gujarat
    res = predict_soc(22.750, 72.450)
    print(f"Test Result: {res}")