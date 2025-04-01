import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import roc_auc_score

# ==========================
# Load the Trained Model
# ==========================
model = keras.models.load_model("fire_detection_vgg16_finetuned.keras")

# ==========================
# Load the Validation Dataset
# ==========================
dataset_path = "fire_data/fire_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64  

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_ds = val_datagen.flow_from_directory(
    dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    subset="validation", class_mode="sparse", shuffle=False
)

# ==========================
# Generate Predictions
# ==========================
y_true = val_ds.classes  # Actual labels (0 = Non-Fire, 1 = Fire)
y_pred_probs = model.predict(val_ds, verbose=1)  # Predicted probabilities

# ==========================
# Compute AUC Score
# ==========================
auc_score = roc_auc_score(y_true, y_pred_probs[:, 1])

print(f"📈 AUC Score: {auc_score:.4f}")

# ==========================
# Plot AUC Graph
# ==========================
plt.figure(figsize=(6, 4))
plt.bar(["AUC Score"], [auc_score], color="red", alpha=0.7)
plt.ylim(0, 1)
plt.ylabel("AUC Score")
plt.title("🔥 AUC Score for Fire Detection")
plt.text(0, auc_score + 0.02, f"{auc_score:.4f}", ha="center", fontsize=12, fontweight="bold", color="black")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
