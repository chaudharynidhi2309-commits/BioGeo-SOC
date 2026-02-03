import numpy as np
import pandas as pd
import stackstac
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fetch_satellite import fetch_satellite_data, generate_augmented_grid
from src.config import LOCATIONS, FEATURES

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def calculate_indices(red, nir, blue, swir):
    """
    Calculate vegetation and moisture indices from satellite bands.
    
    Args:
        red (np.array): Red band (B04)
        nir (np.array): Near-infrared band (B08)
        blue (np.array): Blue band (B02)
        swir (np.array): Short-wave infrared band (B11)
        
    Returns:
        dict: Dictionary containing NDVI, EVI, and NDWI values
    """
    eps = 1e-8  # Small value to avoid division by zero
    
    # NDVI: Normalized Difference Vegetation Index
    ndvi = (nir - red) / (nir + red + eps)
    ndvi = np.clip(ndvi, -1, 1)
    
    # EVI: Enhanced Vegetation Index
    # Formula: 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
    denominator = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = 2.5 * ((nir - red) / (denominator + eps))
    evi = np.clip(evi, -1, 3)  # EVI can range from -1 to 3
    
    # NDWI: Normalized Difference Water Index
    ndwi = (nir - swir) / (nir + swir + eps)
    ndwi = np.clip(ndwi, -1, 1)
    
    return {
        'ndvi': float(ndvi),
        'evi': float(evi),
        'ndwi': float(ndwi)
    }


def create_dataset(augment=False, grid_size=3):
    """
    Creates training dataset from configured locations.
    
    Args:
        augment (bool): If True, use grid-based data augmentation
        grid_size (int): Size of augmentation grid (3 or 5)
        
    Returns:
        pd.DataFrame: Training dataset with features and target SOC values
    """
    logger.info("="*70)
    logger.info("🌍 CREATING TRAINING DATASET")
    logger.info("="*70)
    logger.info(f"Augmentation: {'ENABLED' if augment else 'DISABLED'}")
    logger.info(f"Grid Size: {grid_size}x{grid_size} points per location" if augment else "Single point per location")
    logger.info("="*70 + "\n")
    
    all_data = []
    
    for loc_idx, location in enumerate(LOCATIONS, 1):
        logger.info(f"[{loc_idx}/{len(LOCATIONS)}] Processing: {location['name']}")
        
        bbox = location["coords"]
        target_soc = location["target_soc"]
        
        # Fetch satellite data
        items = fetch_satellite_data(bbox)
        
        if not items:
            logger.warning(f"   ⚠️  No satellite data available, skipping...")
            continue
        
        try:
            # Stack satellite imagery
            logger.info(f"   📡 Loading satellite stack...")
            stack = stackstac.stack(
                items,
                assets=["B04", "B08", "B02", "B11"],  # Red, NIR, Blue, SWIR
                bounds_latlon=bbox,
                epsg=32643,  # UTM Zone 43N (Gujarat)
                resolution=10,  # 10m resolution
                dtype='float64',
                fill_value=np.nan
            )
            
            # Compute median composite (reduces cloud/shadow effects)
            median_stack = stack.median(dim="time").compute()
            
            if augment:
                # Generate grid of sample points
                center_lat = (bbox[1] + bbox[3]) / 2
                center_lon = (bbox[0] + bbox[2]) / 2
                grid_points = generate_augmented_grid(center_lat, center_lon, grid_size=grid_size)
                
                logger.info(f"   🎯 Extracting {len(grid_points)} augmented samples...")
                
                # Extract bands
                red = median_stack.isel(band=0).values
                nir = median_stack.isel(band=1).values
                blue = median_stack.isel(band=2).values
                swir = median_stack.isel(band=3).values
                
                x_coords = median_stack.coords['x'].values
                y_coords = median_stack.coords['y'].values
                
                valid_samples = 0
                for lat, lon in grid_points:
                    # Find nearest pixel
                    x_idx = np.abs(x_coords - lon).argmin()
                    y_idx = np.abs(y_coords - lat).argmin()
                    
                    r = red[y_idx, x_idx]
                    n = nir[y_idx, x_idx]
                    b = blue[y_idx, x_idx]
                    s = swir[y_idx, x_idx]
                    
                    # Skip if any band has invalid data
                    if np.isnan([r, n, b, s]).any():
                        continue
                    
                    # Calculate indices
                    indices = calculate_indices(r, n, b, s)
                    
                    # Add to dataset
                    all_data.append({
                        **indices,
                        'soc': target_soc,
                        'location': location['name']
                    })
                    valid_samples += 1
                
                logger.info(f"   ✅ Extracted {valid_samples} valid samples")
                
            else:
                # Single point extraction (center of bbox)
                logger.info(f"   🎯 Extracting center point...")
                
                red = median_stack.isel(band=0).values
                nir = median_stack.isel(band=1).values
                blue = median_stack.isel(band=2).values
                swir = median_stack.isel(band=3).values
                
                # Get center pixel
                center_y, center_x = red.shape[0] // 2, red.shape[1] // 2
                
                r = red[center_y, center_x]
                n = nir[center_y, center_x]
                b = blue[center_y, center_x]
                s = swir[center_y, center_x]
                
                if not np.isnan([r, n, b, s]).any():
                    indices = calculate_indices(r, n, b, s)
                    
                    all_data.append({
                        **indices,
                        'soc': target_soc,
                        'location': location['name']
                    })
                    logger.info(f"   ✅ Sample extracted")
                else:
                    logger.warning(f"   ⚠️  Invalid data at center point")
        
        except Exception as e:
            logger.error(f"   ❌ Error processing location: {e}")
            continue
        
        logger.info("")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    logger.info("="*70)
    logger.info("📊 DATASET SUMMARY")
    logger.info("="*70)
    logger.info(f"Total Samples: {len(df)}")
    logger.info(f"Features: {FEATURES}")
    logger.info(f"Target: SOC (Soil Organic Carbon)")
    logger.info(f"\nSOC Statistics:")
    logger.info(f"   Mean: {df['soc'].mean():.2f} g/kg")
    logger.info(f"   Std:  {df['soc'].std():.2f} g/kg")
    logger.info(f"   Min:  {df['soc'].min():.2f} g/kg")
    logger.info(f"   Max:  {df['soc'].max():.2f} g/kg")
    logger.info("="*70 + "\n")
    
    return df


if __name__ == "__main__":
    # Test without augmentation
    logger.info("Testing Dataset Creation (No Augmentation)...\n")
    df_basic = create_dataset(augment=False)
    
    if not df_basic.empty:
        logger.info(f"✅ Basic dataset created: {len(df_basic)} samples\n")
        print(df_basic.head())
    
    # Test with augmentation
    logger.info("\n" + "="*70)
    logger.info("Testing Dataset Creation (With Augmentation)...\n")
    df_augmented = create_dataset(augment=True, grid_size=3)
    
    if not df_augmented.empty:
        logger.info(f"✅ Augmented dataset created: {len(df_augmented)} samples\n")
        print(df_augmented.head())