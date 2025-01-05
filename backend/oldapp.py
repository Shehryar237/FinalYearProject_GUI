from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import keras as keras
from tensorflow.python.keras import backend as K
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
CORS(app)  

TEMP_IMGS = 'uploads'
os.makedirs(TEMP_IMGS, exist_ok=True)
app.config['TEMP_IMGS'] = TEMP_IMGS

model = None

@keras.utils.register_keras_serializable()
def focal_loss_fixed(y_true, y_pred):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    where p = sigmoid(x), p_t = p if y = 1 else 1 - p

    Parameters:
    - y_true: true labels, one-hot encoded
    - y_pred: predicted labels, probabilities after softmax
    - alpha: balancing factor
    - gamma: focusing parameter

    Returns:
    - Loss value
    """
    gamma = 2.0
    alpha = 0.25
    num_classes = 9

    # Clip the predictions to prevent log(0) error
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # Convert labels to one-hot encoding if they aren't already
    if len(y_true.shape) == 2 and y_true.shape[-1] == num_classes:
        y_true_one_hot = y_true
    else:
        y_true_one_hot = K.one_hot(K.cast(y_true, 'int32'), num_classes)

    # Compute cross-entropy loss
    cross_entropy_loss = -y_true_one_hot * K.log(y_pred)

    # Compute focal loss
    focal_loss = alpha * K.pow((1 - y_pred), gamma) * cross_entropy_loss

    # Sum the losses over the classes and take the mean over the batch
    return K.mean(K.sum(focal_loss, axis=1))

def load_model():
    global model
    try:
        model_path = 'model/best_model_EfficientNetB3.keras'
        print(f"Loading model from {os.path.abspath(model_path)}")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss_fixed},
            compile=False  # Set to True if you need to compile the model
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

load_model()

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')  
        image = image.resize((300, 300))  
        image_array = np.array(image) / 255.0 
        image_array = np.expand_dims(image_array, axis=0)  
        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("No image uploaded in the request")
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        print("No file selected in the upload")
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(image_file.filename):
        print("Uploaded file has an invalid extension")
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(app.config['TEMP_IMGS'], filename)
        image_file.save(file_path)

        image_array = preprocess_image(file_path)
        if image_array is None:
            print("Error processing the image during preprocessing")
            return jsonify({'error': 'Error processing image'}), 500

        print("Making predictions...")
        predictions = model.predict(image_array).tolist() 
        predictions = predictions[0]
        print(f"Predictions: {predictions}")

        # Delete the uploaded file
        os.remove(file_path)
        print(f"Temporary file {file_path} removed")

        return jsonify({'predictions': predictions})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error during prediction'}), 500

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    if model is not None:
        return jsonify({'status': 'Model loaded and service is healthy'}), 200
    else:
        return jsonify({'status': 'Model not loaded'}), 500

if __name__ == '__main__':
    if model is not None:
        app.run(port=3001, debug=True)
    else:
        print("Failed to load the model. Exiting.")
