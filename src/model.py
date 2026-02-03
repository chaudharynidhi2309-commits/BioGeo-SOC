import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import logging
import sys
import stackstac
import warnings

warnings.filterwarnings('ignore')

# Path fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fetch_satellite import (
    generate_augmented_grid, 
    create_bbox_from_point, 
    fetch_satellite_data
)
from src.config import LOCATIONS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = "data/training_data.csv"
AUGMENTED_DATA_PATH = "data/training_data_augmented.csv"
MODEL_PATH = "models/soc_model.pkl"

def augment_training_data(grid_size=3, buffer_meters=50):
    """
    Optimized Data Augmentation:
    - Calculates center from BBox in config
    - Fetches satellite data once per village
    - Extracts 9 samples (3x3) per village
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"🔄 SPATIAL DATA AUGMENTATION")
    logger.info(f"{'='*70}")
    
    augmented_data = []
    
    for idx, village_config in enumerate(LOCATIONS, 1):
        village_name = village_config['name']
        target_soc = village_config['target_soc']
        
        # FIX: Extract center from 'coords': [min_lon, min_lat, max_lon, max_lat]
        coords = village_config['coords']
        center_lon = (coords[0] + coords[2]) / 2
        center_lat = (coords[1] + coords[3]) / 2
        
        logger.info(f"[{idx}/{len(LOCATIONS)}] Processing: {village_name}")
        logger.info(f"   Center: ({center_lat:.4f}, {center_lon:.4f})")

        # Generate grid points
        grid_points = generate_augmented_grid(
            center_lat, center_lon, 
            grid_size=grid_size, 
            buffer_meters=buffer_meters
        )
        
        # Use the bbox from config for satellite fetching
        bbox = coords 
        items = fetch_satellite_data(bbox)
        
        if not items:
            logger.warning(f"   ⚠️ No satellite data found, skipping...")
            continue
        
        try:
            logger.info(f"   📦 Loading satellite stack & computing median...")
            stack = stackstac.stack(
                items,
                assets=["B04", "B08", "B02", "B11"],
                bounds_latlon=bbox,
                epsg=32643,
                resolution=10,
                dtype='float64',
                fill_value=np.nan
            )
            
            median_stack = stack.median(dim="time").compute()
            
            # Extract band data
            red = median_stack.isel(band=0).values
            nir = median_stack.isel(band=1).values
            blue = median_stack.isel(band=2).values
            swir = median_stack.isel(band=3).values
            
            x_coords = median_stack.coords['x'].values
            y_coords = median_stack.coords['y'].values
            
            eps = 1e-8
            success_count = 0

            for point_idx, (lat, lon) in enumerate(grid_points, 1):
                try:
                    # Find nearest pixel indices
                    x_idx = np.abs(x_coords - lon).argmin()
                    y_idx = np.abs(y_coords - lat).argmin()
                    
                    r, n, b, s = red[y_idx, x_idx], nir[y_idx, x_idx], blue[y_idx, x_idx], swir[y_idx, x_idx]
                    
                    if np.isnan([r, n, b, s]).any(): continue
                    
                    # Indices Math
                    ndvi = np.clip((n - r) / (n + r + eps), -1, 1)
                    denom = n + 6.0*r - 7.5*b + 1.0
                    evi = np.clip(2.5 * ((n - r) / denom), -1, 3) if abs(denom) > eps else 0
                    ndwi = np.clip((n - s) / (n + s + eps), -1, 1)
                    
                    augmented_data.append({
                        'village': village_name,
                        'lat': lat, 'lon': lon,
                        'ndvi': float(ndvi), 'evi': float(evi), 'ndwi': float(ndwi),
                        'target_soc': target_soc
                    })
                    success_count += 1
                except Exception: continue
            
            logger.info(f"   ✅ Extracted {success_count} points\n")
            
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            continue
    
    if augmented_data:
        aug_df = pd.DataFrame(augmented_data)
        os.makedirs("data", exist_ok=True)
        aug_df.to_csv(AUGMENTED_DATA_PATH, index=False)
        logger.info(f"✅ Saved {len(aug_df)} samples to {AUGMENTED_DATA_PATH}")
        return aug_df
    return None

def train_model():
    """Trains model with Village-wise split to prevent data leakage."""
    if not os.path.exists(AUGMENTED_DATA_PATH):
        logger.error("❌ Augmented data missing. Run with --augment first!")
        return

    df = pd.read_csv(AUGMENTED_DATA_PATH)
    villages = df['village'].unique()
    
    # Spatial Split: Use one village entirely for testing
    test_vil = villages[-1]
    train_df = df[df['village'] != test_vil]
    test_df = df[df['village'] == test_vil]
    
    X_train, y_train = train_df[['ndvi', 'evi', 'ndwi']], train_df['target_soc']
    X_test, y_test = test_df[['ndvi', 'evi', 'ndwi']], test_df['target_soc']

    logger.info(f"\n🧠 Training on {len(X_train)} samples (Test Village: {test_vil})")
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    logger.info(f"📈 Performance -> MAE: {mean_absolute_error(y_test, y_pred):.2f} | R²: {r2_score(y_test, y_pred):.3f}")

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()

    if args.augment:
        augment_training_data()
    
    train_model()