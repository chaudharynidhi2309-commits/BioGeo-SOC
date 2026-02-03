#!/usr/bin/env python3
"""
BioGeo-SOC: Main Entry Point
Alternative entry point for running predictions from command line
"""

import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import predict_soc
from src.report import generate_report, save_report

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='BioGeo-SOC: Soil Organic Carbon Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --lat 22.75 --lon 72.45
  python main.py --lat 22.75 --lon 72.45 --grid-size 5
  python main.py --lat 22.75 --lon 72.45 --no-grid --report
        """
    )
    
    parser.add_argument('--lat', type=float, required=True,
                        help='Latitude (e.g., 22.75)')
    parser.add_argument('--lon', type=float, required=True,
                        help='Longitude (e.g., 72.45)')
    parser.add_argument('--grid-size', type=int, default=3, choices=[3, 5],
                        help='Grid size for averaging (default: 3)')
    parser.add_argument('--no-grid', action='store_true',
                        help='Disable grid averaging (single point prediction)')
    parser.add_argument('--report', action='store_true',
                        help='Generate and save detailed report')
    parser.add_argument('--output', type=str, default='reports',
                        help='Output directory for reports (default: reports)')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("🌱 BioGeo-SOC: Soil Organic Carbon Prediction")
    logger.info("="*70)
    logger.info(f"Location: ({args.lat:.6f}, {args.lon:.6f})")
    logger.info(f"Mode: {'Grid-based' if not args.no_grid else 'Single-point'}")
    if not args.no_grid:
        logger.info(f"Grid Size: {args.grid_size}x{args.grid_size}")
    logger.info("="*70 + "\n")
    
    # Perform prediction
    use_grid = not args.no_grid
    soc, std, indices = predict_soc(
        args.lat, 
        args.lon, 
        use_grid_average=use_grid,
        grid_size=args.grid_size
    )
    
    if soc is None:
        logger.error("\n❌ Prediction failed. Please check the coordinates and try again.")
        sys.exit(1)
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("📊 PREDICTION RESULTS")
    logger.info("="*70)
    logger.info(f"Soil Organic Carbon: {soc:.2f} ± {std:.2f} g/kg")
    logger.info("")
    logger.info("Vegetation Indices:")
    logger.info(f"  NDVI (Vegetation):    {indices['ndvi']:.3f}")
    logger.info(f"  EVI (Enhanced Veg):   {indices['evi']:.3f}")
    logger.info(f"  NDWI (Water Index):   {indices['ndwi']:.3f}")
    logger.info("="*70 + "\n")
    
    # Generate report if requested
    if args.report:
        location_info = {
            'address': f'Coordinates: {args.lat:.6f}, {args.lon:.6f}',
            'lat': args.lat,
            'lon': args.lon
        }
        
        report_text = generate_report(soc, std, indices, location_info)
        filepath = save_report(report_text, output_dir=args.output)
        logger.info(f"\n✅ Report saved to: {filepath}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())