from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load the Keras model
model = tf.keras.models.load_model('linux_distribution_predictor.h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    user_input = np.array(data['input'])  # Assume data['input'] is a list of inputs
    prediction = model.predict(user_input.reshape(1, -1))
    top_3 = np.argsort(prediction[0])[-3:][::-1]  # Get top 3 predictions
    return jsonify({'top_3': top_3.tolist()})

if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0")
