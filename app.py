from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import predict
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return "Welcome to the Currency Recogniser!"

@app.route('/predict', methods=['POST'])
def predict_output():
    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request.", "status": "error"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"message": "No file selected.", "status": "error"}), 400

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"message": "Unsupported file type. Please upload a PNG or JPG image.", "status": "error"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = predict.load_and_predict(filepath)
        os.remove(filepath)  # Clean up file

        if result['status'] == 'error':
            return jsonify({"message": result['message'], "status": "error"}), 500

        return jsonify({
            "message": "Prediction successful.",
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "status": "success"
        }), 200

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"message": str(e), "status": "error"}), 500

@app.route('/predict', methods=['GET', 'PUT', 'DELETE', 'PATCH'])
def method_not_allowed():
    return jsonify({"message": "This method is not supported.", "status": "error"}), 405

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"message": "Endpoint not found.", "status": "error"}), 404

@app.errorhandler(500)
def handle_500(e):
    return jsonify({"message": "Internal server error.", "status": "error"}), 500

@app.errorhandler(400)
def handle_400(e):
    return jsonify({"message": "Bad request.", "status": "error"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
