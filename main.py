import os
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

# ─────── CONFIG ───────────────────────────────────────────────────────────────
# Project root is one level up from this file (i.e. the folder that holds 'templates/' etc.)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Tell Flask where to find templates and static uploads
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, 'templates'),
    static_folder=None  # we’ll serve uploads explicitly
)

# uploads folder path
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your Keras model
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'model.h5')
model = load_model(MODEL_PATH)

# The four classes your model predicts
CLASS_LABELS = ['pituitary', 'notumor', 'glioma', 'meningioma']


# ─────── HELPERS ──────────────────────────────────────────────────────────────
def predict_tumor(image_path):
    """
    Loads an image, runs model.predict(), and returns a human-readable result &
    confidence score.
    """
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    confidence = float(preds[0][idx])

    label = CLASS_LABELS[idx]
    if label == 'notumor':
        return "No Tumor Detected", confidence
    else:
        return f"Tumor: {label.capitalize()}", confidence


# ─────── ROUTES ───────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded = request.files.get('file')
        if uploaded:
            # save to uploads/
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded.filename)
            uploaded.save(save_path)

            # run prediction
            result_text, conf = predict_tumor(save_path)
            conf_pct = f"{conf * 100:.2f}%"

            # tell Jinja how to find the image
            img_url = f"/uploads/{uploaded.filename}"
            return render_template('index.html',
                                   result=result_text,
                                   confidence=conf_pct,
                                   file_path=img_url)

    # GET or no file: just show blank
    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve back the raw uploaded image so that <img src="/uploads/…"> works.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ─────── APP LAUNCH ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    # listen on all interfaces if you want, or leave as default localhost
    app.run(debug=True)
