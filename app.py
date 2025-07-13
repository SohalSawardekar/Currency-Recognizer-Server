from flask import Flask, request, jsonify, make_response
import predict
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Currency Recogniser!"

@app.route('/predict', methods=['POST'])
def predict_output():
    if 'file' not in request.files:
        return jsonify({
            "message": "No file part in the request.",
            "status": "error"
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            "message": "No file selected.",
            "status": "error"
        }), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({
            "message": "Unsupported file type. Please upload a PNG or JPG image.",
            "status": "error"
        }), 400
    
    os.makedirs('uploads', exist_ok=True)
    file.save(f"./uploads/{file.filename}")

    # Call the prediction function
    try:
        result = predict.load_and_predict()

        if result['status'] == 'error':
            return jsonify({
                "message": result['message'],
                "status": "error"
            }), 500

        if result['status'] == 'success':
            return jsonify({
                "message": "Prediction successful.",
                "prediction": result['prediction'],
                "confidence": result['confidence'],
                "probabilities": result['probabilities'],
                "status": "success"
            }), 200
        
    except Exception as e:
        # Handle any exceptions that occur during prediction
        return jsonify({
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/predict', methods=['GET', 'PUT', 'DELETE', 'PATCH'])
def method_not_allowed():
    return jsonify({
        "message": "This method is not supported.",
        "status": "error"
    }), 405

@app.errorhandler(404)
def handle_404(e):
    return jsonify({
        "message": "Endpoint not found.",
        "status": "error"
    }), 404

@app.errorhandler(500)
def handle_500(e):
    return jsonify({
        "message": "Internal server error.",
        "status": "error"
    }), 500

@app.errorhandler(400)
def handle_400(e):
    return jsonify({
        "message": "Bad request.",
        "status": "error"
    }), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT
    app.run(host='0.0.0.0', port=port, debug=True)

