from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
import json


app = Flask(__name__)
CORS(app)  

TEMP_IMGS = 'uploads'
os.makedirs(TEMP_IMGS, exist_ok=True)
app.config['TEMP_IMGS'] = TEMP_IMGS

CLASS_LABELS = ["AK", "Acne", "BCC", "DF", "MEL", "NV", "SCC", "SEK", "VASC"]

@tf.keras.utils.register_keras_serializable()
def focal_loss_fixed(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    num_classes = 9
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    if len(y_true.shape) == 2 and y_true.shape[-1] == num_classes:
        y_true_one_hot = y_true
    else:
        y_true_one_hot = K.one_hot(K.cast(y_true, 'int32'), num_classes)
    cross_entropy_loss = -y_true_one_hot * K.log(y_pred)
    focal_loss = alpha * K.pow((1 - y_pred), gamma) * cross_entropy_loss
    return K.mean(K.sum(focal_loss, axis=1))

def load_model():
    global model
    model_path = 'model/best_model_EfficientNetB3.keras'  # Ensure this path is correct
    try:
        if not os.path.exists(model_path):
            print(f"Model file does not exist at the specified path: {model_path}")
            model = None
            return
        print(f"Loading model from {os.path.abspath(model_path)}")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'focal_loss_fixed': focal_loss_fixed},
            compile=False
        )
        print("Model loaded successfully.")
        model.summary()
        
        dummy_input = np.zeros((1, 300, 300, 3))
        model.predict(dummy_input)
        print("Model warm-up prediction done.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

load_model()

def preprocess_image(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Image file does not exist at the specified path: {image_path}")
            return None

        image = Image.open(image_path).convert('RGB')  
        image = image.resize((300, 300))  
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)  
        image_array = preprocess_input(image_array)  # Apply EfficientNet preprocessing
        print(f"Preprocessed image shape: {image_array.shape}")
        print(f"Preprocessed image stats - min: {image_array.min()}, max: {image_array.max()}, mean: {image_array.mean():.4f}")
        return image_array
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        print(f"Image saved to {file_path}")

        image_array = preprocess_image(file_path)
        if image_array is None:
            print("Error processing the image during preprocessing")
            return jsonify({'error': 'Error processing image'}), 500

        print("Making predictions...")
        predictions = model.predict(image_array)
        print(f"Raw Predictions: {predictions}")

        # Check if the model's last layer uses softmax
        last_layer = model.layers[-1]
        if not hasattr(last_layer, 'activation') or last_layer.activation.__name__ != 'softmax':
            print("Applying softmax to the model's output...")
            predictions = tf.nn.softmax(predictions, axis=-1).numpy()
            print(f"Softmax Applied Predictions: {predictions}")
        else:
            print("Model output already uses softmax activation.")

        predictions_list = predictions[0].tolist()
        print(f"Predictions list: {predictions_list}")

        predictions_dict = {disease: prob for disease, prob in zip(CLASS_LABELS, predictions_list)}
        print(f"Predictions as dictionary: {predictions_dict}")

        #  class with the highest probability
        predicted_class = CLASS_LABELS[np.argmax(predictions_list)]
        predicted_probability = np.max(predictions_list)
        print(f"Predicted Class: {predicted_class} with probability {predicted_probability:.4f}")

        # Del the uploaded file
        os.remove(file_path)
        print(f"Temporary file {file_path} removed")

        return jsonify({'predictions': predictions_dict, 'predicted_class': predicted_class, 'probability': predicted_probability})

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


@app.route('/predict/info', methods=['GET'])
def get_disease_info():
    ## disease name was sent by frontend as a url parameter, we will extract it
    disease_name = request.args.get('disease') 
    try:
        with open('disease-data.json', 'r') as file:
            disease_data = json.load(file)
        
        if disease_name in disease_data:
            return jsonify({disease_name: disease_data[disease_name]}), 200 
        else:
            return jsonify({'error': f'Disease "{disease_name}" not found'}), 404  # handle unknown diseases
    except Exception as e:
        print(f"Error loading disease data: {e}")
        return jsonify({'error': 'Could not load disease data'}), 500

if __name__ == '__main__':
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    if model is not None:
        app.run(port=3001, debug=True)
    else:
        print("Failed to load the model. Exiting.")
