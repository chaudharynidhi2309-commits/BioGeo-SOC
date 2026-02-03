import torch
import sys
import os
import logging

# Setup paths and logging
sys.path.append(os.getcwd())
from src.model import get_model

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def predict_soc(ndvi_input: float):
    """
    Uses the trained Neural Network to predict SOC with a Physics-based constraint.
    """
    model_path = "models/trained_soc_model.pth"
    
    # 1. Safety Check: Does the model exist?
    if not os.path.exists(model_path):
        logger.error(f"❌ Error: {model_path} not found. Did you run train.py?")
        return

    # 2. Load the brain and move to correct device
    model, device = get_model()
    try:
        # map_location ensures it works even if trained on a different computer
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() 
    except Exception as e:
        logger.error(f"❌ Failed to load model weights: {e}")
        return

    # 3. Convert input to tensor
    input_tensor = torch.tensor([[ndvi_input]], dtype=torch.float32).to(device)

    # 4. Get the AI's raw guess
    with torch.no_grad():
        raw_prediction = model(input_tensor).item()

    # 5. Physics-Informed Constraint (PINN logic)
    # Soil Carbon cannot be below zero; 0.1 g/kg is a realistic floor for bare soil.
    physics_corrected = max(0.1, raw_prediction)

    # 6. Final Report
    print("\n" + "="*40)
    print(f"🔮 PINN PREDICTION REPORT")
    print("="*40)
    print(f"📡 Input NDVI        : {ndvi_input:.4f}")
    print(f"🤖 Raw AI Guess      : {raw_prediction:.2f} g/kg")
    print(f"⚖️  Physics-Corrected : {physics_corrected:.2f} g/kg")
    print("-" * 40)

    # Assessment logic
    if physics_corrected > 20:
        logger.info("🌳 Result: Extremely Rich Soil Carbon (Forest-like)")
    elif physics_corrected > 10:
        logger.info("🌾 Result: High Carbon Sequestration (Healthy Farmland)")
    else:
        logger.info("🏜️ Result: Low/Moderate Carbon Area (Arid/Bare)")
    print("="*40)

if __name__ == "__main__":
    # Test with a high NDVI value (Lush vegetation)
    predict_soc(0.8)