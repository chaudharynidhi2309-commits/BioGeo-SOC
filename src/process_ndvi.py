import stackstac
import numpy as np
import os
import sys
import logging
import warnings

# Suppress GDAL warnings BEFORE any imports that use GDAL
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tif'
warnings.filterwarnings('ignore')

# 1. Path Fix: Ensure script can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fetch_satellite import fetch_satellite_data
from src.config import LOCATIONS
from src.dataset import save_to_csv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def calculate_advanced_indices(items, bbox, village_name):
    """
    Calculate vegetation indices (NDVI, EVI, NDWI) for a village.
    
    Args:
        items (list): List of STAC items (satellite images)
        bbox (list): Bounding box [min_lon, min_lat, max_lon, max_lat]
        village_name (str): Name of the village
        
    Returns:
        dict: Mean values of NDVI, EVI, and NDWI, or None on error
    """
    logger.info(f"📊 Processing Village: {village_name}...")
    
    try:
        # Load required bands: Red(B04), NIR(B08), Blue(B02), SWIR(B11)
        stack = stackstac.stack(
            items, 
            assets=["B04", "B08", "B02", "B11"],
            bounds_latlon=bbox,
            epsg=32643,
            resolution=10,
            dtype='float64',
            fill_value=np.nan
        )
        
        # Temporal Median: Averages the whole year to get the 'True' soil state
        logger.info(f"   ⏳ Computing temporal median across {stack.shape[0]} images...")
        median_stack = stack.median(dim="time").compute()

        # Extract bands by index
        red = median_stack.isel(band=0).values   # B04
        nir = median_stack.isel(band=1).values   # B08
        blue = median_stack.isel(band=2).values  # B02
        swir = median_stack.isel(band=3).values  # B11

        # Add epsilon to prevent division by zero
        eps = 1e-8

        # --- 1. NDVI (Normalized Difference Vegetation Index) ---
        # Range: -1 to 1, higher = more vegetation
        ndvi = (nir - red) / (nir + red + eps)
        ndvi = np.clip(ndvi, -1, 1)
        
        # --- 2. EVI (Enhanced Vegetation Index) ---
        # Formula: 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
        # CRITICAL FIX: Added safety checks for denominator
        denominator = nir + 6.0 * red - 7.5 * blue + 1.0
        
        # Replace invalid denominators (near zero or negative) with NaN
        denominator = np.where(np.abs(denominator) < eps, np.nan, denominator)
        
        evi = 2.5 * ((nir - red) / denominator)
        
        # EVI should be between -1 and 3 (anything else is invalid)
        evi = np.clip(evi, -1, 3)
        
        # --- 3. NDWI (Normalized Difference Water Index) ---
        # Higher values = more moisture/water content
        ndwi = (nir - swir) / (nir + swir + eps)
        ndwi = np.clip(ndwi, -1, 1)

        # Calculate means, ignoring NaN values
        ndvi_mean = float(np.nanmean(ndvi))
        evi_mean = float(np.nanmean(evi))
        ndwi_mean = float(np.nanmean(ndwi))

        logger.info(f"   ✅ Successfully processed.")
        
        return {
            "ndvi": ndvi_mean,
            "evi": evi_mean,
            "ndwi": ndwi_mean
        }
        
    except Exception as e:
        logger.error(f"   ❌ Error processing {village_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_feature_extraction():
    """Main loop to process all villages listed in config."""
    all_village_data = []
    
    for i, loc in enumerate(LOCATIONS, 1):
        # Pass the specific village coordinates to the fetcher
        items = fetch_satellite_data(loc['coords'])
        
        if not items:
            logger.warning(f"⚠️  No satellite data found for {loc['name']}")
            continue
        
        indices = calculate_advanced_indices(items, loc['coords'], loc['name'])
        
        if indices:
            # Link the Ground Truth (SOC) from config to the satellite data
            indices['target_soc'] = loc['target_soc']
            indices['village'] = loc['name']
            all_village_data.append(indices)
        else:
            logger.warning(f"⚠️  Failed to calculate indices for {loc['name']}")
            
    return all_village_data

if __name__ == "__main__":
    try:
        # Run the feature extraction
        results = run_feature_extraction()
        
        if results:
            # Print results to screen
            print("\n" + "="*65)
            print("VILLAGE                   | NDVI   | EVI    | NDWI   | SOC")
            print("-"*65)
            
            for r in results:
                print(f"{r['village']:<25} | {r['ndvi']:<6.3f} | {r['evi']:<6.3f} | {r['ndwi']:<6.3f} | {r['target_soc']:<4.1f}")
            
            print("="*65)
            logger.info(f"\n✅ Successfully processed {len(results)}/{len(LOCATIONS)} villages")
            
            # Save results to CSV
            save_to_csv(results)
            
        else:
            logger.error("\n❌ No results generated.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
       