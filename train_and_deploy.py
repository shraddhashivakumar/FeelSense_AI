# train_and_deploy.py
# This script executes the training process defined in model_utils.py.

import os
import sys
# Add the current directory to the path to ensure model_utils can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_utils import load_dataset, build_and_train

# Define the path to your dataset and where the models should be saved
DATA_PATH = "data/emotion.csv"
MODEL_SAVE_PATH = "models/mood_model.joblib"
LE_SAVE_PATH = "models/label_encoder.joblib"

def train_new_model():
    """
    Loads data, trains the model, and saves the resulting pipeline and encoder.
    """
    print("--- Starting Model Training Process ---")
    print(f"Loading data from: {DATA_PATH}")

    # 1. Load the data (which now includes your new negative sentences)
    texts, labels = load_dataset(path=DATA_PATH)
    
    if texts is None or labels is None:
        print("ERROR: Could not load data. Check if emotion.csv exists and has data.")
        return

    print(f"Successfully loaded {len(texts)} samples across {len(set(labels))} unique moods.")
    
    # 2. Build and train the new model
    pipeline, le = build_and_train(
        texts, 
        labels, 
        save_to=MODEL_SAVE_PATH,
        le_save_to=LE_SAVE_PATH
    )
    
    if pipeline:
        print("\n--- Training Complete ---")
        print("The model has been saved and is ready for deployment.")
        print(f"Next step: **RESTART YOUR FLASK SERVER** to load the new model files.")
    else:
        print("Training failed.")


if __name__ == "__main__":
    train_new_model()
