import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 1. New Agricultural Locations (Villages around Dholka/Bavla)
# Format: [min_lon, min_lat, max_lon, max_lat]
LOCATIONS = [
    {
        "name": "Dholka_Village_North",
        "coords": [72.42, 22.75, 72.45, 22.78],
        "target_soc": 14.20  # Estimated g/kg
    },
    {
        "name": "Bavla_Agricultural_East",
        "coords": [72.35, 22.82, 72.38, 22.85],
        "target_soc": 12.80
    },
    {
        "name": "Koth_Village_Fields",
        "coords": [72.28, 22.68, 72.31, 22.71],
        "target_soc": 16.10
    },
    {
        "name": "Rupal_Village_South",
        "coords": [72.40, 22.60, 72.43, 22.63],
        "target_soc": 13.50
    },
    {
        "name": "Sanand_Industrial_Area",
        "coords": [72.37, 22.99, 72.40, 23.02],
        "target_soc": 11.20
    },
    {
        "name": "Viramgam_Agricultural_Zone",
        "coords": [72.03, 23.11, 72.06, 23.14],
        "target_soc": 15.80
    }
]

# 2. Spectral Indices to Calculate
# We are adding EVI (Vegetation) and NDWI (Water/Moisture)
FEATURES = ['ndvi', 'evi', 'ndwi']

# 3. Time and Quality Settings
DATE_RANGE = "2024-10-01/2024-12-31"
MAX_CLOUD = 5  # Dropped to 5% for higher quality

# 4. Model Settings
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

def validate():
    for loc in LOCATIONS:
        c = loc["coords"]
        if not (len(c) == 4 and c[0] < c[2] and c[1] < c[3]):
            raise ValueError(f"Invalid coordinates for {loc['name']}")
    logger.info(f"✅ Config Loaded: {len(LOCATIONS)} Villages ready for processing.")

validate()