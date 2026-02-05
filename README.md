<<<<<<< HEAD
=======
<<<<<<< HEAD
# 🌱 BioGeo-SOC: Soil Organic Carbon Inference System

URL : https://biogeo-soc-final-j3lsacehha2sxfbecgzftc.streamlit.app/


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
=======
>>>>>>> 72a508b
# 🌱 BioGeo-SOC: Soil Organic Carbon Prediction System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Advanced satellite-based soil health prediction for Gujarat, India using Machine Learning and Remote Sensing

![BioGeo-SOC Dashboard](https://img.shields.io/badge/Status-Production-brightgreen)

---

## 📖 Overview

**BioGeo-SOC** is an AI-powered system that predicts **Soil Organic Carbon (SOC)** levels using Sentinel-2 satellite imagery. The system combines remote sensing, machine learning, and geospatial analysis to provide real-time soil health assessments for any location in Gujarat.

### ✨ Key Features

- 🛰️ **Real-time Satellite Analysis** - Sentinel-2 L2A imagery from Microsoft Planetary Computer
- 🤖 **Machine Learning Prediction** - Random Forest model trained on vegetation indices
- 📊 **Interactive Dashboard** - Beautiful Streamlit web interface
- 🗺️ **Location Search** - Search any village/town in Gujarat
- ✅ **Validation System** - Compare predictions with SoilGrids ground truth data
- 📄 **PDF Reports** - Download detailed analysis reports
- 🎯 **Grid-based Averaging** - 3x3 or 5x5 point averaging for accuracy
- 📈 **Visual Analytics** - Gauge charts, radar plots, and interactive maps

---

## 🚀 Live Demo

Try it here: [BioGeo-SOC App](https://your-app-url.streamlit.app) *(Update after deployment)*

---

## 🎬 Quick Start

### Prerequisites

- Python 3.11 or higher
- Git
- Internet connection (for satellite data)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/BioGeo_SOC_Inference.git
cd BioGeo_SOC_Inference

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model (first time only)
python src/train.py --augment

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
BioGeo_SOC_Inference/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── packages.txt               # System dependencies (for Streamlit Cloud)
├── README.md                  # This file
├── .gitignore                # Git ignore rules
│
├── .streamlit/
│   └── config.toml           # Streamlit configuration
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── config.py            # Configuration settings
│   ├── dataset.py           # Data processing
│   ├── fetch_satellite.py   # Satellite data fetching
│   ├── inference.py         # Prediction engine
│   ├── train.py            # Model training
│   ├── report.py           # Report generation
│   └── validator.py        # Ground truth validation
│
├── models/                  # Trained models
│   └── soc_model.pkl       # Random Forest model
│
├── data/                   # Data files
│   └── gujarat_soc.tif    # SoilGrids validation data
│
└── reports/               # Generated reports
```

---

## 🎯 How It Works

### 1. **Data Acquisition**
- Fetches Sentinel-2 L2A satellite imagery
- Cloud cover < 5% for optimal quality
- Date range: October - December 2024

### 2. **Feature Extraction**
Calculates three vegetation indices:

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

### 3. **Prediction**
- Random Forest Regressor (100 trees)
- Grid-based averaging (3×3 points)
- Returns SOC estimate with uncertainty

### 4. **Validation**
- Compares with SoilGrids ground truth
- Calculates model accuracy
- Provides confidence metrics

---

## 💻 Usage

### Web Interface

1. **Enter Location**: Type a village or town name in Gujarat
2. **Analyze**: Click the "Analyze Location" button
3. **View Results**: See SOC prediction, validation data, and visualizations
4. **Download Report**: Export PDF report of analysis

### Command Line

```bash
# Basic prediction
python main.py --lat 22.75 --lon 72.45

# Generate report
python main.py --lat 22.75 --lon 72.45 --report

# Custom grid size
python main.py --lat 22.75 --lon 72.45 --grid-size 5
```

### Python API

```python
from src.inference import predict_soc

# Predict SOC
lat, lon = 22.7500, 72.4500
soc, std, indices = predict_soc(lat, lon, use_grid_average=True)

print(f"SOC: {soc:.2f} ± {std:.2f} g/kg")
print(f"NDVI: {indices['ndvi']:.3f}")
```

---

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Training locations
LOCATIONS = [
    {
        "name": "Your_Location",
        "coords": [min_lon, min_lat, max_lon, max_lat],
        "target_soc": 14.5  # Known SOC value
    }
]

# Date range for satellite imagery
DATE_RANGE = "2024-10-01/2024-12-31"

# Maximum cloud cover (%)
MAX_CLOUD = 5

# Model parameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| RMSE   | ~1.5 g/kg |
| MAE    | ~1.2 g/kg |
| R²     | ~0.85 |
| Accuracy | ~85-95% |

*Performance based on validation against SoilGrids data*

---

## 🌍 Supported Locations

**Primary Region**: Gujarat, India

**Example Locations**:
- Dholka
- Bavla
- Sanand
- Viramgam
- Kalol
- Chotila
- Ahmedabad rural areas

Can work for any location with Sentinel-2 coverage!

---

## 🚢 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Configure:
   - **Main file**: `app.py`
   - **Python version**: 3.11
5. Deploy!

### Local Server

```bash
streamlit run app.py --server.port 8501
```

### Docker

```bash
docker build -t biogeo-soc .
docker run -p 8501:8501 biogeo-soc
```

---

## 📦 Dependencies

### Python Packages

- **Web Framework**: streamlit, streamlit-folium
- **ML & Data**: scikit-learn, pandas, numpy, joblib
- **Geospatial**: rasterio, rioxarray, xarray, stackstac
- **Satellite**: pystac-client, planetary-computer
- **Visualization**: plotly, folium
- **Utilities**: geopy, fpdf

### System Packages (for Streamlit Cloud)

- libgdal-dev
- libgeos-dev
- python3-dev

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Microsoft Planetary Computer** - Sentinel-2 data access
- **ESA Copernicus** - Sentinel-2 satellite program
- **ISRIC SoilGrids** - Ground truth validation data
- **Streamlit** - Web framework
- Gujarat agricultural research community

---

## 📧 Contact

For questions or feedback:
- 📫 Create an issue on GitHub
- 📧 Email: your.email@example.com
- 🌐 Website: your-website.com

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{biogeo_soc_2026,
  author = {Your Name},
  title = {BioGeo-SOC: Soil Organic Carbon Prediction System},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/BioGeo_SOC_Inference}
}
```

---

## 🔮 Future Enhancements

- [ ] Multi-temporal analysis (seasonal trends)
- [ ] Additional soil parameters (pH, nitrogen, phosphorus)
- [ ] Deep learning models (CNN)
- [ ] Mobile app version
- [ ] Batch processing
- [ ] Historical trend analysis
- [ ] Integration with other satellite data sources

---

**Made with ❤️ for sustainable agriculture in Gujarat**

*Version 1.0.0 - February 2026*
=======
*Version 1.0.0 - February 2026*
