import pandas as pd
import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def generate_report(soc_value, std_value, indices, location_info):
    """
    Generate a detailed analysis report for SOC prediction.
    
    Args:
        soc_value (float): Predicted SOC value in g/kg
        std_value (float): Standard deviation/uncertainty
        indices (dict): Dictionary containing NDVI, EVI, NDWI values
        location_info (dict): Location details (lat, lon, address)
        
    Returns:
        str: Formatted report text
    """
    report_lines = []
    
    report_lines.append("="*70)
    report_lines.append("🌱 SOIL ORGANIC CARBON (SOC) ANALYSIS REPORT")
    report_lines.append("="*70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Location Information
    report_lines.append("📍 LOCATION DETAILS")
    report_lines.append("-"*70)
    report_lines.append(f"Address: {location_info.get('address', 'N/A')}")
    report_lines.append(f"Latitude: {location_info.get('lat', 0):.6f}°")
    report_lines.append(f"Longitude: {location_info.get('lon', 0):.6f}°")
    report_lines.append("")
    
    # SOC Prediction
    report_lines.append("🔬 SOC PREDICTION")
    report_lines.append("-"*70)
    report_lines.append(f"Predicted SOC: {soc_value:.2f} ± {std_value:.2f} g/kg")
    
    # Interpret SOC level
    if soc_value < 5:
        soc_category = "Very Low"
        soc_interpretation = "Critical - Soil degradation evident"
    elif soc_value < 10:
        soc_category = "Low"
        soc_interpretation = "Below optimal - Organic matter addition recommended"
    elif soc_value < 15:
        soc_category = "Moderate"
        soc_interpretation = "Acceptable - Maintain current practices"
    elif soc_value < 20:
        soc_category = "Good"
        soc_interpretation = "Suitable for agriculture - Good soil health"
    else:
        soc_category = "Excellent"
        soc_interpretation = "High soil carbon - Excellent soil quality"
    
    report_lines.append(f"Category: {soc_category}")
    report_lines.append(f"Interpretation: {soc_interpretation}")
    report_lines.append("")
    
    # Confidence Assessment
    if std_value < 1.0:
        confidence = "High (±{:.2f} g/kg)".format(std_value)
        conf_interpretation = "Prediction is highly reliable"
    elif std_value < 2.0:
        confidence = "Medium (±{:.2f} g/kg)".format(std_value)
        conf_interpretation = "Prediction has moderate uncertainty"
    else:
        confidence = "Low (±{:.2f} g/kg)".format(std_value)
        conf_interpretation = "Prediction has high uncertainty - use with caution"
    
    report_lines.append(f"Confidence Level: {confidence}")
    report_lines.append(f"Note: {conf_interpretation}")
    report_lines.append("")
    
    # Vegetation Indices
    report_lines.append("🌿 VEGETATION & ENVIRONMENTAL INDICES")
    report_lines.append("-"*70)
    
    ndvi = indices.get('ndvi', 0)
    evi = indices.get('evi', 0)
    ndwi = indices.get('ndwi', 0)
    
    report_lines.append(f"NDVI (Vegetation): {ndvi:.3f}")
    if ndvi < 0.2:
        report_lines.append("   → Sparse vegetation or bare soil")
    elif ndvi < 0.4:
        report_lines.append("   → Moderate vegetation cover")
    elif ndvi < 0.6:
        report_lines.append("   → Healthy vegetation")
    else:
        report_lines.append("   → Dense, vigorous vegetation")
    
    report_lines.append("")
    report_lines.append(f"EVI (Enhanced Vegetation): {evi:.3f}")
    if evi < 0.2:
        report_lines.append("   → Low vegetation activity")
    elif evi < 0.4:
        report_lines.append("   → Moderate vegetation activity")
    else:
        report_lines.append("   → High vegetation activity")
    
    report_lines.append("")
    report_lines.append(f"NDWI (Water Index): {ndwi:.3f}")
    if ndwi < 0:
        report_lines.append("   → Low moisture / Dry conditions")
    elif ndwi < 0.3:
        report_lines.append("   → Moderate moisture")
    else:
        report_lines.append("   → High moisture / Water presence")
    
    report_lines.append("")
    
    # Recommendations
    report_lines.append("💡 RECOMMENDATIONS")
    report_lines.append("-"*70)
    
    recommendations = []
    
    if soc_value < 10:
        recommendations.append("• Add organic matter (compost, farmyard manure)")
        recommendations.append("• Consider cover cropping to increase soil carbon")
        recommendations.append("• Minimize soil disturbance (reduced tillage)")
    
    if ndvi < 0.3:
        recommendations.append("• Improve vegetation cover to enhance carbon sequestration")
        recommendations.append("• Consider crop rotation or intercropping")
    
    if ndwi < 0:
        recommendations.append("• Monitor soil moisture levels")
        recommendations.append("• Consider irrigation if applicable")
    
    if soc_value >= 15:
        recommendations.append("• Maintain current soil management practices")
        recommendations.append("• Continue monitoring to preserve soil health")
    
    if not recommendations:
        recommendations.append("• Continue current management practices")
        recommendations.append("• Regular monitoring recommended")
    
    for rec in recommendations:
        report_lines.append(rec)
    
    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("Note: This analysis is based on satellite-derived vegetation indices")
    report_lines.append("and should be supplemented with field measurements for validation.")
    report_lines.append("="*70)
    
    return "\n".join(report_lines)


def save_report(report_text, output_dir="reports", filename=None):
    """
    Save report to a text file.
    
    Args:
        report_text (str): Report content
        output_dir (str): Directory to save reports
        filename (str): Optional custom filename
        
    Returns:
        str: Path to saved report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"SOC_Report_{timestamp}.txt"
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(report_text)
    
    logger.info(f"📄 Report saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Example usage
    logger.info("Testing Report Generation...\n")
    
    # Sample data
    sample_soc = 14.5
    sample_std = 1.2
    sample_indices = {
        'ndvi': 0.65,
        'evi': 0.42,
        'ndwi': 0.15
    }
    sample_location = {
        'address': 'Dholka, Gujarat, India',
        'lat': 22.7500,
        'lon': 72.4500
    }
    
    # Generate report
    report = generate_report(sample_soc, sample_std, sample_indices, sample_location)
    print(report)
    
    # Save report
    save_report(report)