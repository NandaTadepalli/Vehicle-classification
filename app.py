from flask import Flask, request, render_template, url_for
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image
import os

app = Flask(__name__)

# Configure upload folder and allowed file types
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists

# Load model and feature extractor
model = ViTForImageClassification.from_pretrained("results/checkpoint-294")
feature_extractor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224")

# Label mapping
label_map = {
    0: "Ambulance",
    1: "Barge",
    2: "Bicycle",
    3: "Boat",
    4: "Bus",
    5: "Car",
    6: "Cart",
    7: "Caterpillar",
    8: "Helicopter",
    9: "Limousine",
    10: "Motorcycle",
    11: "Segway",
    12: "Snowmobile",
    13: "Tank",
    14: "Taxi",
    15: "Truck",
    16: "Van"
}


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html', label_map=label_map)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part in the request.", label_map=label_map)

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected for uploading.", label_map=label_map)

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Process the image for prediction
            image = Image.open(filepath)
            image = image.convert("RGB")  # Ensure image is in RGB mode
            inputs = feature_extractor(images=image, return_tensors="pt")

            # Perform prediction
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

            # Map prediction to label
            predicted_label = label_map.get(predicted_class, "Unknown")

            # Provide file path for preview
            file_url = url_for('static', filename=f'uploads/{filename}')
            return render_template(
                'index.html',
                prediction=predicted_label,
                file_url=file_url,
                label_map=label_map
            )

        except Exception as e:
            return render_template('index.html', error=f"Error processing the image: {str(e)}", label_map=label_map)
    else:
        return render_template('index.html', error="File type not allowed.", label_map=label_map)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/model-info')
def model_info():
    return render_template('model-info.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
