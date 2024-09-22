from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

# Initialize Flask app
app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = './static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="UrWuY0Sj3WhHv673XvWb"
)

# Mapping of class IDs to class names for `digits_segmentation`
class_id_to_name = {
    0: '.',
    1: '0',
    2: '1',
    3: '2',
    4: '3',
    5: '4',
    6: '5',
    7: '6',
    8: '7',
    9: '8',
    10: '9'
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Perform inference using `counter_detection` model
    result = CLIENT.infer(image_path, model_id="counter_detection/1")

    if 'predictions' in result:
        predictions = result['predictions']
        response_data = {}

        for i, prediction in enumerate(predictions):
            x = prediction.get('x', 0)
            y = prediction.get('y', 0)
            width = prediction.get('width', 0)
            height = prediction.get('height', 0)

            # Calculate bounding box coordinates
            left = x - width / 2
            top = y - height / 2
            right = x + width / 2
            bottom = y + height / 2

            # Crop the detected region (ROI)
            cropped_image = Image.open(image_path).crop((left, top, right, bottom))
            cropped_image_path = f"{UPLOAD_FOLDER}/cropped_image_{i}.jpg"
            cropped_image.save(cropped_image_path)

            # Perform inference on cropped image with `digits_segmentation`
            digit_result = CLIENT.infer(cropped_image_path, model_id="digits_segmentation/1")

            if 'predictions' in digit_result:
                digit_predictions = sorted(digit_result['predictions'], key=lambda d: d.get('x', 0))

                # Group digits and generate a reading
                grouped_digits = []
                current_group = []
                last_x = None
                threshold = 20

                for digit in digit_predictions:
                    x_coord = digit.get('x', 0)
                    class_id = digit.get('class_id', 'unknown')
                    class_name = class_id_to_name.get(int(class_id), 'unknown')

                    if last_x is not None and abs(x_coord - last_x) > threshold:
                        grouped_digits.append(''.join(current_group))
                        current_group = []

                    current_group.append(class_name)
                    last_x = x_coord

                if current_group:
                    grouped_digits.append(''.join(current_group))

                top_left_reading = grouped_digits[0] if len(grouped_digits) > 0 else None
                main_reading = ''.join(grouped_digits[1:])

                response_data[f"Reading_{i}"] = {
                    "top_left_reading": top_left_reading,
                    "main_reading": main_reading
                }

        return jsonify(response_data)

    return jsonify({"error": "No predictions found"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
