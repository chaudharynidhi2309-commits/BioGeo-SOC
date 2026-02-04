# 🌱 BioGeo-SOC: Soil Organic Carbon Inference System

URL : https://biogeo-soc-final-irnjlrap5otqdzbgc8niam.streamlit.app/


> Advanced satellite-based soil health prediction for Gujarat, India

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

BioGeo-SOC is an advanced machine learning system that predicts **Soil Organic Carbon (SOC)** levels using satellite imagery from Sentinel-2. The system analyzes vegetation indices (NDVI, EVI, NDWI) to provide real-time soil health assessments for any location in Gujarat.

### Key Features

- 🛰️ **Real-time Satellite Analysis** - Uses Sentinel-2 L2A imagery
- 🎯 **Grid-based Prediction** - 3x3 point averaging for robust estimates
- 📊 **Interactive Dashboard** - Beautiful Streamlit interface with maps
- 🌍 **Location Search** - Search any village/town in Gujarat
- 📈 **Visual Analytics** - Gauge charts, radar plots, satellite maps
- 🔬 **Scientific Accuracy** - Multiple vegetation indices for precision

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- Internet connection (for satellite data)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/BioGeo_SOC_Inference.git
cd BioGeo_SOC_Inference
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model** (First time only)
```bash
python src/train.py --augment
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
BioGeo_SOC_Inference/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── packages.txt           # System dependencies (for Streamlit Cloud)
├── README.md              # This file
├── src/
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration and settings
│   ├── dataset.py         # Dataset creation and processing
│   ├── fetch_satellite.py # Satellite data fetching
│   ├── inference.py       # SOC prediction engine
│   ├── train.py          # Model training script
│   └── report.py         # Report generation
├── models/
│   └── soc_model.pkl     # Trained Random Forest model
└── data/
    └── training_data.csv  # Training dataset
```

## 🔧 Usage

### Web Application

1. Enter a location name (e.g., "Dholka", "Bavla", "Chotila")
2. Click "Analyze Location"
3. View results:
   - SOC prediction with uncertainty
   - Vegetation health metrics
   - Interactive satellite map
   - Soil quality interpretation

### Command Line Prediction

```python
from src.inference import predict_soc

# Predict SOC for a location
lat, lon = 22.7500, 72.4500  # Dholka coordinates
soc_value, std, indices = predict_soc(lat, lon, use_grid_average=True)

print(f"SOC: {soc_value:.2f} ± {std:.2f} g/kg")
print(f"NDVI: {indices['ndvi']:.3f}")
```

### Training Custom Models

```bash
# Train with data augmentation (recommended)
python src/train.py --augment --grid-size 3

# Train without augmentation
python src/train.py
```

## 🛰️ How It Works

### 1. Satellite Data Acquisition
- Fetches Sentinel-2 L2A imagery from Microsoft Planetary Computer
- Uses cloud cover < 5% for optimal quality
- Temporal range: October - December 2024

### 2. Feature Extraction
Calculates three key vegetation indices:

**NDVI (Normalized Difference Vegetation Index)**
```
NDVI = (NIR - Red) / (NIR + Red)
```

**EVI (Enhanced Vegetation Index)**
```
EVI = 2.5 × [(NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)]
```

**NDWI (Normalized Difference Water Index)**
```
NDWI = (NIR - SWIR) / (NIR + SWIR)
```

### 3. Machine Learning Prediction
- **Algorithm**: Random Forest Regressor
- **Features**: NDVI, EVI, NDWI
- **Target**: Soil Organic Carbon (g/kg)
- **Validation**: 5-fold cross-validation

### 4. Grid-Based Averaging
- Creates 3×3 grid around target point
- Extracts satellite data for all 9 points
- Averages predictions for robustness
- Provides uncertainty estimate (std dev)

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| RMSE   | ~1.5 g/kg |
| MAE    | ~1.2 g/kg |
| R²     | ~0.85 |

*Performance varies based on training data and augmentation settings*

## 🌍 Supported Locations

Currently optimized for:
- **Gujarat, India**
- Agricultural zones around:
  - Dholka
  - Bavla
  - Sanand
  - Viramgam
  - Kalol

Can work for any location with Sentinel-2 coverage!

## 🎨 Features in Detail

### Dashboard Components

1. **Metrics Panel**
   - Predicted SOC value
   - Uncertainty estimate
   - NDVI vegetation index
   - Confidence level

2. **Soil Quality Gauge**
   - Visual SOC indicator
   - Color-coded ranges (low/medium/high)
   - Target reference line at 15 g/kg

3. **Environmental Radar**
   - Multi-index visualization
   - NDVI, EVI, NDWI comparison
   - Normalized 0-1 scale

4. **Satellite Map**
   - Google Satellite basemap
   - Analysis area overlay (150m radius)
   - Interactive marker with details

5. **Interpretation Panel**
   - Soil quality assessment
   - Vegetation health status
   - Actionable recommendations

## 🔬 Technical Details

### Dependencies

**Core ML & Data Science:**
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- joblib >= 1.3.0

**Geospatial:**
- stackstac >= 0.5.0
- pystac-client >= 0.7.0
- planetary-computer >= 1.0.0
- rioxarray >= 0.15.0
- xarray >= 2023.0.0

**Visualization:**
- streamlit >= 1.28.0
- plotly >= 5.14.0
- folium >= 0.14.0
- streamlit-folium >= 0.15.0
- geopy >= 2.4.0

### Configuration

Edit `src/config.py` to customize:
- Training locations
- Date ranges
- Cloud cover thresholds
- Model parameters
- Features to extract

### Model Training

```python
# In src/config.py
MODEL_PARAMS = {
    'n_estimators': 100,      # Number of trees
    'max_depth': 10,          # Maximum tree depth
    'min_samples_split': 5,   # Min samples to split
    'min_samples_leaf': 2,    # Min samples per leaf
    'random_state': 42        # Reproducibility
}
```

## 🚢 Deployment

URL : https://biogeo-soc-final-irnjlrap5otqdzbgc8niam.streamlit.app/

### Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with:
   - **Main file**: `app.py`
   - **Python version**: 3.11

### Local Server

```bash
streamlit run app.py --server.port 8501
```

## 📈 Future Enhancements

- [ ] Multi-temporal analysis (seasonal trends)
- [ ] Additional soil parameters (pH, nitrogen, phosphorus)
- [ ] Deep learning models (CNN for imagery)
- [ ] Field validation data integration
- [ ] Mobile app version
- [ ] Export reports as PDF
- [ ] Batch processing for multiple locations
- [ ] Historical SOC trend analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Microsoft Planetary Computer** for Sentinel-2 data access
- **ESA Copernicus** for Sentinel-2 satellite program
- **Streamlit** for the amazing web framework
- Gujarat agricultural research community

## 📧 Contact

For questions or feedback:
- Create an issue on GitHub
- Email: chaudharynidhi2309@gmail.com

## 📚 References

1. Sentinel-2 User Handbook - ESA
2. "Remote Sensing of Soil Organic Carbon" - Journal of Environmental Quality
3. Microsoft Planetary Computer Documentation
4. Streamlit Documentation

---

**Made with ❤️ for sustainable agriculture in Gujarat**

*Version 1.0.0 - February 2026*


BioGeo_SOC_Inference/
├── app.py                      # ✅ Web application
├── main.py                     # ✅ CLI tool
├── setup.sh                    # ✅ Setup script
├── requirements.txt            # ✅ Dependencies (CLEAN)
├── packages.txt                # ✅ System packages
├── README.md                   # ✅ Documentation
├── PROJECT_GUIDE.md           # ✅ Detailed guide
├── .gitignore                 # ✅ Git config
│
├── .streamlit/
│   └── config.toml            # ✅ Streamlit config
│
├── src/
│   ├── __init__.py           # ✅ Package init
│   ├── config.py             # ✅ Configuration
│   ├── dataset.py            # ✅ Data processing
│   ├── fetch_satellite.py    # ✅ Satellite fetching
│   ├── inference.py          # ✅ Prediction engine
│   ├── train.py              # ✅ Model training
│   └── report.py             # ✅ Report generation
│
├── models/
│   ├── .gitkeep              # ✅ Placeholder
│   └── soc_model.pkl         # ✅ Trained model
│
├── data/
│   └── .gitkeep              # ✅ Placeholder
│
└── reports/
    └── .gitkeep              # ✅ Placeholder
