# Third-party imports
from flask import Flask, request, jsonify
from PIL import Image
import io

# Local imports
import detection

app = Flask(__name__)

# Initialize the object detector
detector = detection.ObjectDetector()

@app.route('/classify', methods=['POST'])
def classify_image():
    
    try:
        # Get the PIL Image from the request
        frame_data = request.files['image'].read()
        pil_image = Image.open(io.BytesIO(frame_data))

        # Run object detection
        results = detector.detect(pil_image)

        # Return the result as JSON
        return jsonify(results), 200
    
    except Exception as e:
        # Handle any exceptions (e.g., invalid image format, errors in classify_stuff)
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

