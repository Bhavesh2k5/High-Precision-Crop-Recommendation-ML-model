"""
predict.py
----------
Standalone inference script for the Genetic-Fuzzy Ensemble Crop Recommendation system.

Usage (after running the notebooks to train models):
    python src/predict.py --N 90 --P 42 --K 43 --pH 6.5 --rainfall 202 --temperature 24

Or import as a module:
    from src.predict import predict_crop
    result = predict_crop(N=90, P=42, K=43, pH=6.5, rainfall=202, temperature=24)
    print(result)
"""

import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering  (must match 01_data_cleaning.ipynb exactly)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(N: float, P: float, K: float,
                      pH: float, rainfall: float, temperature: float) -> pd.DataFrame:
    """
    Create a single-row DataFrame with the 9 features expected by the trained model.

    Parameters
    ----------
    N           : Nitrogen content in soil (kg/ha)
    P           : Phosphorus content in soil (kg/ha)
    K           : Potassium content in soil (kg/ha)
    pH          : Soil pH (0–14)
    rainfall    : Annual rainfall (mm)
    temperature : Average temperature (°C)

    Returns
    -------
    pd.DataFrame with columns: [n, p, k, ph, rainfall, temperature,
                                 n_p_ratio, n_k_ratio, temp_rainfall]
    """
    n_p_ratio    = N / (P + 1)
    n_k_ratio    = N / (K + 1)
    temp_rainfall = temperature * rainfall

    return pd.DataFrame([{
        "n":            N,
        "p":            P,
        "k":            K,
        "ph":           pH,
        "rainfall":     rainfall,
        "temperature":  temperature,
        "n_p_ratio":    n_p_ratio,
        "n_k_ratio":    n_k_ratio,
        "temp_rainfall": temp_rainfall,
    }])


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy membership helper  (lightweight, no full fuzzy system required)
# ─────────────────────────────────────────────────────────────────────────────
def _fuzzy_label(value: float, low_thresh: float, high_thresh: float) -> str:
    """Return a simple linguistic label for a numeric value."""
    if value <= low_thresh:
        return "Low"
    elif value >= high_thresh:
        return "High"
    return "Medium"


def fuzzy_advice(N: float, P: float, K: float,
                 pH: float, rainfall: float, temperature: float) -> list:
    """
    Generate human-readable agronomic advice based on fuzzy thresholds.

    Returns a list of advisory strings.
    """
    advice = []
    if _fuzzy_label(N, 40, 100) == "High":
        advice.append("Nitrogen is HIGH — reduce nitrogenous fertiliser application.")
    elif _fuzzy_label(N, 40, 100) == "Low":
        advice.append("Nitrogen is LOW — consider urea or ammonium nitrate application.")

    if _fuzzy_label(pH, 5.5, 7.5) == "Low":
        advice.append("Soil is ACIDIC (pH < 5.5) — apply agricultural lime to raise pH.")
    elif _fuzzy_label(pH, 5.5, 7.5) == "High":
        advice.append("Soil is ALKALINE (pH > 7.5) — consider sulphur treatment.")

    if _fuzzy_label(rainfall, 50, 200) == "Low":
        advice.append("Rainfall is LOW — ensure supplemental irrigation.")
    elif _fuzzy_label(rainfall, 50, 200) == "High":
        advice.append("Rainfall is HIGH — ensure good drainage to prevent waterlogging.")

    if not advice:
        advice.append("Soil and climate conditions appear optimal for the recommended crop.")

    return advice


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction function
# ─────────────────────────────────────────────────────────────────────────────
def predict_crop(N: float, P: float, K: float,
                 pH: float, rainfall: float, temperature: float,
                 model=None, label_encoder=None) -> dict:
    """
    Predict the recommended crop for given soil and climate inputs.

    Parameters
    ----------
    N, P, K       : Soil nutrient values (kg/ha)
    pH            : Soil pH
    rainfall      : Annual rainfall (mm)
    temperature   : Average temperature (°C)
    model         : Trained sklearn-compatible classifier. If None, returns
                    feature engineering output only (useful for testing).
    label_encoder : Fitted LabelEncoder used during training.

    Returns
    -------
    dict with keys:
        'crop'         : Predicted crop name (str) or None if no model given
        'confidence'   : Prediction confidence % (float) or None
        'top3'         : Top-3 crop predictions with probabilities (list of tuples)
        'fuzzy_advice' : List of agronomic advisory strings
        'features'     : Engineered feature dict
    """
    features_df = engineer_features(N, P, K, pH, rainfall, temperature)
    advice      = fuzzy_advice(N, P, K, pH, rainfall, temperature)

    result = {
        "crop":         None,
        "confidence":   None,
        "top3":         [],
        "fuzzy_advice": advice,
        "features":     features_df.iloc[0].to_dict(),
    }

    if model is not None and label_encoder is not None:
        probs       = model.predict_proba(features_df)[0]
        pred_idx    = int(np.argmax(probs))
        top3_idx    = np.argsort(probs)[::-1][:3]

        result["crop"]       = label_encoder.classes_[pred_idx]
        result["confidence"] = round(float(probs[pred_idx]) * 100, 2)
        result["top3"]       = [
            (label_encoder.classes_[i], round(float(probs[i]) * 100, 2))
            for i in top3_idx
        ]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Crop recommendation inference — Genetic-Fuzzy Ensemble"
    )
    parser.add_argument("--N",           type=float, required=True, help="Nitrogen (kg/ha)")
    parser.add_argument("--P",           type=float, required=True, help="Phosphorus (kg/ha)")
    parser.add_argument("--K",           type=float, required=True, help="Potassium (kg/ha)")
    parser.add_argument("--pH",          type=float, required=True, help="Soil pH (0-14)")
    parser.add_argument("--rainfall",    type=float, required=True, help="Annual rainfall (mm)")
    parser.add_argument("--temperature", type=float, required=True, help="Avg temperature (°C)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print("\n=== Crop Recommendation System ===")
    print(f"  Input — N:{args.N}  P:{args.P}  K:{args.K}  "
          f"pH:{args.pH}  Rainfall:{args.rainfall}mm  Temp:{args.temperature}°C\n")

    # Without a saved model we can only show engineered features + fuzzy advice
    result = predict_crop(
        N=args.N, P=args.P, K=args.K,
        pH=args.pH, rainfall=args.rainfall, temperature=args.temperature
    )

    print("Engineered Features:")
    for k, v in result["features"].items():
        print(f"  {k:20s}: {v:.4f}")

    print("\nFuzzy Agronomic Advice:")
    for tip in result["fuzzy_advice"]:
        print(f"  • {tip}")

    print("\nNote: To get a crop prediction, load a trained model and label encoder")
    print("from the notebooks and pass them as `model` and `label_encoder` arguments.")
