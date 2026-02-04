import pystac_client
import planetary_computer
import sys
import os
import numpy as np
import stackstac
import logging

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DATE_RANGE, MAX_CLOUD

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def fetch_satellite_data(bbox):
    """
    Fetches Sentinel-2 data for a SPECIFIC bounding box.
    Returns only the clearest image to optimize speed.
    
    Args:
        bbox (list): Bounding box [min_lon, min_lat, max_lon, max_lat]
        
    Returns:
        list: STAC items collection (single clearest image) or None
    """
    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=DATE_RANGE,
            query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}]
        )

        items = search.item_collection()
        
        if len(items) == 0:
            return None
        
        # Return only the clearest image (avoid processing all 118!)
        return [items[0]]
        
    except Exception as e:
        logger.error(f"Error fetching satellite data: {e}")
        return None


def generate_augmented_grid(center_lat, center_lon, grid_size=3, buffer_meters=50):
    """
    Generate a grid of points around a center coordinate for data augmentation.
    
    Args:
        center_lat (float): Center latitude
        center_lon (float): Center longitude
        grid_size (int): Grid dimension (3 = 9 points, 5 = 25 points)
        buffer_meters (int): Distance between grid points in meters
        
    Returns:
        list: List of (lat, lon) tuples forming a grid
    """
    # Approximate conversion at Gujarat latitude (~23°N)
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 km × cos(23°) ≈ 102 km
    lat_meters_to_degrees = buffer_meters / 111000.0
    lon_meters_to_degrees = buffer_meters / 102000.0
    
    # Generate grid offsets
    half_grid = grid_size // 2
    lat_offsets = np.linspace(-half_grid, half_grid, grid_size) * lat_meters_to_degrees
    lon_offsets = np.linspace(-half_grid, half_grid, grid_size) * lon_meters_to_degrees
    
    grid_points = []
    for lat_offset in lat_offsets:
        for lon_offset in lon_offsets:
            grid_points.append((
                center_lat + lat_offset,
                center_lon + lon_offset
            ))
    
    return grid_points


def create_bbox_from_point(lat, lon, buffer_degrees=0.005):
    """
    Create a bounding box around a single point.
    Buffer of 0.005° ≈ 550 meters (covers 3x3 grid with 50m spacing)
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        buffer_degrees (float): Buffer size in degrees
        
    Returns:
        list: [min_lon, min_lat, max_lon, max_lat]
    """
    return [
        lon - buffer_degrees,
        lat - buffer_degrees,
        lon + buffer_degrees,
        lat + buffer_degrees
    ]


def extract_pixels_at_points(stack, grid_points, bbox):
    """
    Extract pixel values at specific lat/lon points from a satellite image stack.
    This is the KEY optimization - fetch image ONCE, extract multiple points.
    
    Args:
        stack (xarray.DataArray): Satellite data stack from stackstac
        grid_points (list): List of (lat, lon) tuples
        bbox (list): Original bounding box [min_lon, min_lat, max_lon, max_lat]
        
    Returns:
        list: List of pixel value arrays for each point
    """
    pixel_values = []
    
    # Get spatial coordinates from stack
    x_coords = stack.coords['x'].values
    y_coords = stack.coords['y'].values
    
    for lat, lon in grid_points:
        # Find nearest pixel to this lat/lon
        # Note: x=longitude, y=latitude in the stack
        x_idx = np.abs(x_coords - lon).argmin()
        y_idx = np.abs(y_coords - lat).argmin()
        
        # Extract pixel values for all bands at this location
        pixel = stack.isel(x=x_idx, y=y_idx).values
        pixel_values.append(pixel)
    
    return pixel_values


if __name__ == "__main__":
    print("🧪 Testing fetch_satellite.py with optimized augmentation...")
    print("="*70)
    
    # Test single fetch
    test_bbox = [72.42, 22.75, 72.48, 22.78]
    data = fetch_satellite_data(test_bbox)
    if data:
        print(f"✅ Successfully fetched {len(data)} scene(s)")
        print(f"   Image ID: {data[0].id}")
        print(f"   Date: {data[0].datetime.date()}")
    
    # Test grid generation
    center_lat, center_lon = 22.75, 72.45
    grid_points = generate_augmented_grid(center_lat, center_lon, grid_size=3)
    print(f"\n📍 Generated {len(grid_points)} augmented points (3x3 grid):")
    for i, (lat, lon) in enumerate(grid_points):
        print(f"   Point {i+1}: ({lat:.6f}, {lon:.6f})")
    
    print("\n" + "="*70)
    
