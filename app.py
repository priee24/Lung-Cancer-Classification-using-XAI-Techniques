# ================================
# Imports
# ================================
import os
import cv2
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_from_directory

from classify import (
    preprocess_image,
    classify_image,
    save_image_with_label,
    generate_heatmap,
    overlay_heatmap,
    lime_explanation
)

# Base directory for this module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================
# Flask App Initialization
# ================================
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# ================================
# Predict and Visualize
# ================================
def predict_and_visualize(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Load image
    original_image = cv2.imread(image_path)
    image_tensor = preprocess_image(image_path)

    # Classify
    try:
        predicted_class_label, predicted_class_idx, predicted_prob = classify_image(image_path)
    except Exception as e:
        print("Error in classification:", e)
        predicted_class_label, predicted_class_idx, predicted_prob = "Unknown", 0, 0.0

    # Save original image with label and probability
    label_text = f"{predicted_class_label} ({predicted_prob:.1f}%)"
    original_output_path = os.path.join(output_folder, "original_with_label.png")
    save_image_with_label(original_image, label_text, original_output_path)

    # Generate Grad-CAM
    try:
        heatmap = generate_heatmap(image_tensor, predicted_class_idx)
        gradcam_image = overlay_heatmap(heatmap, original_image) if heatmap is not None else original_image.copy()
    except Exception as e:
        print("Grad-CAM error:", e)
        gradcam_image = original_image.copy()
    gradcam_output_path = os.path.join(output_folder, "gradcam.png")
    cv2.imwrite(gradcam_output_path, gradcam_image)

    # Generate LIME explanation
    try:
        lime_original, lime_mask, lime_expl, top_label = lime_explanation(image_path, predicted_class_idx)
        lime_mask = lime_mask if lime_mask is not None else original_image.copy()
    except Exception as e:
        print("LIME error:", e)
        lime_mask = original_image.copy()
        lime_expl = None
        top_label = 0
    lime_output_path = os.path.join(output_folder, "lime.png")
    cv2.imwrite(lime_output_path, lime_mask)

    # Generate LIME bar chart
    try:
        scores = lime_expl.local_exp.get(top_label, []) if lime_expl else []
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]

        feature_labels = [f"Feature {x[0]}" for x in scores]
        feature_contributions = [x[1] for x in scores]

        plt.figure(figsize=(8, 5))
        plt.barh(feature_labels, feature_contributions)
        plt.xlabel("Contribution")
        plt.ylabel("Features")
        plt.title("LIME Feature Contributions")
        plt.gca().invert_yaxis()
        lime_chart_output_path = os.path.join(output_folder, "lime_bar_chart.png")
        plt.savefig(lime_chart_output_path)
        plt.close()
    except Exception as e:
        print("LIME chart error:", e)
        lime_chart_output_path = os.path.join(output_folder, "lime_bar_chart.png")
        cv2.imwrite(lime_chart_output_path, original_image)

    # Return all results with predicted class, probability, and filenames
    return {
        "original": "original_with_label.png",
        "gradcam": "gradcam.png",
        "lime": "lime.png",
        "lime_bar_chart": "lime_bar_chart.png",
        "predicted_class": predicted_class_label,
        "predicted_prob": f"{predicted_prob:.1f}%",  # percentage
        "original_name": os.path.basename(image_path)
    }

# ================================
# Flask Routes
# ================================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {"error": "No file part"}, 400
    file = request.files['file']
    if file.filename == '':
        return {"error": "No selected file"}, 400

    upload_folder = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    try:
        results = predict_and_visualize(file_path, upload_folder)
    except Exception as e:
        print("Error in prediction:", e)
        return {"error": "Processing failed"}, 500

    return results

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'uploads'), filename)

# ================================
# Run Flask App
# ================================
if __name__ == "__main__":
    app.run(debug=True)
