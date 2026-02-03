import pandas as pd
import numpy as np
import joblib
import os
import sys
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import create_dataset
from src.config import FEATURES, MODEL_PARAMS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "soc_model.pkl")


def train_model(augment=False, grid_size=3):
    """
    Train a Random Forest model to predict SOC from satellite indices.
    
    Args:
        augment (bool): Use data augmentation
        grid_size (int): Grid size for augmentation (3 or 5)
        
    Returns:
        RandomForestRegressor: Trained model
    """
    logger.info("="*70)
    logger.info("🚀 TRAINING SOC PREDICTION MODEL")
    logger.info("="*70)
    logger.info(f"Algorithm: Random Forest Regressor")
    logger.info(f"Features: {FEATURES}")
    logger.info(f"Target: SOC (g/kg)")
    logger.info(f"Augmentation: {'ENABLED' if augment else 'DISABLED'}")
    logger.info("="*70 + "\n")
    
    # Create dataset
    df = create_dataset(augment=augment, grid_size=grid_size)
    
    if df.empty or len(df) < 5:
        logger.error("❌ Insufficient data for training. Need at least 5 samples.")
        return None
    
    # Prepare features and target
    X = df[FEATURES]
    y = df['soc']
    
    logger.info(f"📊 Training Dataset:")
    logger.info(f"   Samples: {len(df)}")
    logger.info(f"   Features: {list(X.columns)}")
    logger.info(f"   Target Range: {y.min():.2f} - {y.max():.2f} g/kg\n")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"📈 Train/Test Split:")
    logger.info(f"   Training Set: {len(X_train)} samples")
    logger.info(f"   Test Set: {len(X_test)} samples\n")
    
    # Initialize and train model
    logger.info(f"🔧 Model Configuration:")
    for param, value in MODEL_PARAMS.items():
        logger.info(f"   {param}: {value}")
    logger.info("")
    
    model = RandomForestRegressor(**MODEL_PARAMS)
    
    logger.info("⏳ Training model...")
    model.fit(X_train, y_train)
    logger.info("✅ Training complete!\n")
    
    # Cross-validation
    logger.info("🔄 Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                 scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    logger.info(f"   CV RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f} g/kg\n")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info("="*70)
    logger.info("📊 MODEL PERFORMANCE")
    logger.info("="*70)
    logger.info(f"Test Set Metrics:")
    logger.info(f"   RMSE:  {rmse:.3f} g/kg")
    logger.info(f"   MAE:   {mae:.3f} g/kg")
    logger.info(f"   R²:    {r2:.3f}")
    logger.info("="*70 + "\n")
    
    # Feature importance
    logger.info("🔍 Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        logger.info(f"   {row['feature'].upper()}: {row['importance']:.3f}")
    logger.info("")
    
    # Show predictions vs actual
    logger.info("📋 Sample Predictions:")
    logger.info(f"{'Actual':<10} {'Predicted':<12} {'Error':<10}")
    logger.info("-" * 35)
    for actual, pred in zip(y_test.values[:5], y_pred[:5]):
        error = abs(actual - pred)
        logger.info(f"{actual:<10.2f} {pred:<12.2f} {error:<10.2f}")
    logger.info("")
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"💾 Model saved to: {MODEL_PATH}\n")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SOC prediction model')
    parser.add_argument('--augment', action='store_true', 
                        help='Enable data augmentation')
    parser.add_argument('--grid-size', type=int, default=3, 
                        choices=[3, 5],
                        help='Grid size for augmentation (3 or 5)')
    
    args = parser.parse_args()
    
    model = train_model(augment=args.augment, grid_size=args.grid_size)
    
    if model:
        logger.info("="*70)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("Next steps:")
        logger.info("   1. Test the model: python src/inference.py")
        logger.info("   2. Run the app: streamlit run app.py")
        logger.info("="*70)