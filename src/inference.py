import joblib
import os
import sys
import numpy as np
import pandas as pd
import logging
import stackstac
import warnings

warnings.filterwarnings('ignore')

# Path Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fetch_satellite import (
    fetch_satellite_data, 
    create_bbox_from_point, 
    generate_augmented_grid
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "models/soc_model.pkl"


def predict_soc(lat, lon, use_grid_average=True, grid_size=3):
    """
    Predict Soil Organic Carbon for ANY location in Gujarat.
    
    OPTIMIZED: Fetches satellite image ONCE, extracts all grid points.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        use_grid_average (bool): Use 3x3 grid averaging (recommended)
        grid_size (int): Grid dimension (3 or 5)
        
    Returns:
        tuple: (prediction, confidence_interval, indices_dict) or (None, None, None)
    """
    # Load model
    if not os.path.exists(MODEL_PATH):
        logger.error("❌ Model not found! Train first: python src/model.py --augment")
        return None, None, None

    model = joblib.load(MODEL_PATH)

    if use_grid_average and grid_size > 1:
        logger.info(f"🔍 Grid-based prediction ({grid_size}x{grid_size} points)...")
        
        # Generate grid points
        grid_points = generate_augmented_grid(lat, lon, grid_size=grid_size, buffer_meters=50)
        
        # Create bbox that covers all points
        bbox = create_bbox_from_point(lat, lon, buffer_degrees=0.01)
        
        # Fetch satellite data ONCE
        logger.info(f"   📡 Fetching satellite data...")
        items = fetch_satellite_data(bbox)
        
        if not items:
            logger.warning("   ⚠️  No satellite data, trying single point...")
            use_grid_average = False
        else:
            try:
                # Load stack
                logger.info(f"   📦 Loading satellite stack...")
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
                
                # Extract all grid points from this SINGLE image
                logger.info(f"   🎯 Extracting {len(grid_points)} samples...")
                
                red = median_stack.isel(band=0).values
                nir = median_stack.isel(band=1).values
                blue = median_stack.isel(band=2).values
                swir = median_stack.isel(band=3).values
                
                x_coords = median_stack.coords['x'].values
                y_coords = median_stack.coords['y'].values
                
                predictions = []
                all_indices = {'ndvi': [], 'evi': [], 'ndwi': []}
                eps = 1e-8
                
                for point_lat, point_lon in grid_points:
                    try:
                        # Find nearest pixel
                        x_idx = np.abs(x_coords - point_lon).argmin()
                        y_idx = np.abs(y_coords - point_lat).argmin()
                        
                        r = red[y_idx, x_idx]
                        n = nir[y_idx, x_idx]
                        b = blue[y_idx, x_idx]
                        s = swir[y_idx, x_idx]
                        
                        if np.isnan([r, n, b, s]).any():
                            continue
                        
                        # Calculate indices
                        ndvi = np.clip((n - r) / (n + r + eps), -1, 1)
                        
                        denom = n + 6.0*r - 7.5*b + 1.0
                        if abs(denom) > eps:
                            evi = np.clip(2.5 * ((n - r) / denom), -1, 3)
                        else:
                            continue
                        
                        ndwi = np.clip((n - s) / (n + s + eps), -1, 1)
                        
                        # Predict
                        input_data = pd.DataFrame(
                            [[ndvi, evi, ndwi]], 
                            columns=['ndvi', 'evi', 'ndwi']
                        )
                        pred = model.predict(input_data)[0]
                        
                        predictions.append(pred)
                        all_indices['ndvi'].append(float(ndvi))
                        all_indices['evi'].append(float(evi))
                        all_indices['ndwi'].append(float(ndwi))
                        
                    except:
                        continue
                
                if predictions:
                    # Calculate statistics
                    mean_pred = np.mean(predictions)
                    std_pred = np.std(predictions)
                    
                    final_indices = {
                        'ndvi': np.mean(all_indices['ndvi']),
                        'evi': np.mean(all_indices['evi']),
                        'ndwi': np.mean(all_indices['ndwi'])
                    }
                    
                    logger.info(f"   ✅ Success! Used {len(predictions)}/{len(grid_points)} points")
                    logger.info(f"   📊 Prediction: {mean_pred:.2f} ± {std_pred:.2f} g/kg")
                    
                    return float(mean_pred), float(std_pred), final_indices
                else:
                    logger.warning("   ⚠️  No valid predictions, falling back...")
                    use_grid_average = False
                    
            except Exception as e:
                logger.error(f"   ❌ Grid prediction failed: {e}")
                use_grid_average = False
    
    # Fallback: Single point prediction
    if not use_grid_average:
        logger.info(f"🎯 Single-point prediction...")
        
        bbox = create_bbox_from_point(lat, lon, buffer_degrees=0.005)
        items = fetch_satellite_data(bbox)
        
        if not items:
            logger.error("❌ No satellite data available")
            return None, None, None
        
        try:
            stack = stackstac.stack(
                items,
                assets=["B04", "B08", "B02", "B11"],
                bounds_latlon=bbox,
                epsg=32643,
                resolution=10,
                dtype='float64'
            )
            
            median_stack = stack.median(dim="time").compute()
            
            # Get center pixel
            red = median_stack.isel(band=0).values
            nir = median_stack.isel(band=1).values
            blue = median_stack.isel(band=2).values
            swir = median_stack.isel(band=3).values
            
            center_y, center_x = red.shape[0]//2, red.shape[1]//2
            
            r, n, b, s = red[center_y, center_x], nir[center_y, center_x], blue[center_y, center_x], swir[center_y, center_x]
            
            eps = 1e-8
            ndvi = np.clip((n - r) / (n + r + eps), -1, 1)
            evi = np.clip(2.5 * ((n - r) / (n + 6.0*r - 7.5*b + 1.0 + eps)), -1, 3)
            ndwi = np.clip((n - s) / (n + s + eps), -1, 1)
            
            input_data = pd.DataFrame([[ndvi, evi, ndwi]], columns=['ndvi', 'evi', 'ndwi'])
            prediction = model.predict(input_data)[0]
            
            indices = {'ndvi': float(ndvi), 'evi': float(evi), 'ndwi': float(ndwi)}
            
            logger.info(f"   ✅ Prediction: {prediction:.2f} g/kg")
            
            return float(prediction), 0.0, indices
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return None, None, None


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🧪 TESTING INFERENCE ENGINE")
    logger.info("="*70)
    
    # Test location (Dholka area)
    test_lat, test_lon = 22.750, 72.450
    
    logger.info(f"\n📍 Test Location: ({test_lat}, {test_lon})")
    logger.info(f"{'='*70}\n")
    
    pred, std, indices = predict_soc(test_lat, test_lon, use_grid_average=True, grid_size=3)
    
    if pred:
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ PREDICTION RESULT")
        logger.info(f"{'='*70}")
        logger.info(f"Soil Organic Carbon: {pred:.2f} ± {std:.2f} g/kg")
        logger.info(f"\nVegetation Indices:")
        logger.info(f"   NDVI: {indices['ndvi']:.3f}")
        logger.info(f"   EVI:  {indices['evi']:.3f}")
        logger.info(f"   NDWI: {indices['ndwi']:.3f}")
        logger.info(f"{'='*70}\n")