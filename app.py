import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template_string
from flasgger import Swagger
from paddleocr import PaddleOCR
import easyocr
import pytesseract
import ollama

app = Flask(__name__)
Swagger(app)

class OCRBenchmark:
    def __init__(self, model_name='llama3.2'):
        print("Initializing OCR Engines for OCR Benchmarking.")
        self.model_name = model_name

        # OCR Model initialization
        self.easy_reader = easyocr.Reader(['en'], gpu=False)
        self.paddle_reader = PaddleOCR(use_textline_orientation=True, lang='en')

    def preprocess(self, image_bytes):

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file provided.")
        height, width = img.shape[:2]
        target_height = 1000
        aspect_ratio = width / float(height)
        target_width = int(target_height * aspect_ratio)
        resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_img)

        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr

    def run_easyocr(self, img):
        print("----- Running EasyOCR Engine -----")
        import time
        start_time = time.perf_counter()
        results = self.easy_reader.readtext(img)
        lines = [text for box, text, conf in results]

        elapsed = time.perf_counter() - start_time
        return lines, elapsed

    def run_paddleocr(self, img):
        print("----- Running PaddleOCR Engine -----")
        import time
        start_time = time.perf_counter()

        try:
            if hasattr(self.paddle_reader, 'predict'):
                results = self.paddle_reader.predict(img)
            else:
                results = self.paddle_reader.ocr(img)
        except Exception as e:
            print(f"PaddleOCR Engine Error: {e}")
            return [], time.perf_counter() - start_time

        lines = []
        if results:
            if hasattr(results, '__iter__') and not isinstance(results, (list, tuple)):
                results = list(results)

            if len(results) > 0:
                first_page = results[0]

                # Case 1: Classic PaddleOCR format (List of Lists)
                if isinstance(first_page, (list, tuple)) and len(first_page) > 0 and isinstance(first_page[0], (list, tuple)):
                    extracted = []
                    for item in first_page:
                        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (tuple, list)):
                            try:
                                y_coord = item[0][0][1]
                                text = item[1][0]
                                extracted.append((float(y_coord), str(text)))
                            except (IndexError, TypeError):
                                pass
                    extracted.sort(key=lambda x: x[0])
                    lines = [txt for _, txt in extracted]


                if not lines:
                    def get_field(obj, keys):
                        for k in keys:
                            try:
                                if hasattr(obj, 'get'):
                                    v = obj.get(k)
                                    if v is not None: return v
                                v = obj[k]
                                if v is not None: return v
                            except Exception:
                                pass
                        for k in keys:
                            if hasattr(obj, k):
                                return getattr(obj, k)
                        return None

                    texts = get_field(first_page, ['rec_texts', 'rec_text', 'text', 'texts'])
                    boxes = get_field(first_page, ['dt_polys', 'polys', 'boxes'])

                    if texts is not None:
                        # Convert generator or array elements safely to lists
                        if hasattr(texts, '__iter__') and not isinstance(texts, (list, tuple)):
                            texts = list(texts)
                        if boxes is not None:
                            if hasattr(boxes, '__iter__') and not isinstance(boxes, (list, tuple)):
                                boxes = list(boxes)

                        if boxes is not None and len(texts) == len(boxes):
                            extracted = []
                            for i in range(len(texts)):
                                box = boxes[i]
                                txt = str(texts[i])
                                try:
                                    box_arr = np.array(box)
                                    if box_arr.ndim >= 2:
                                        y_coord = float(np.min(box_arr[:, 1]))
                                    else:
                                        y_coord = float(box_arr[1])
                                    extracted.append((y_coord, txt))
                                except Exception:
                                    extracted.append((0.0, txt))
                            extracted.sort(key=lambda x: x[0])
                            lines = [txt for _, txt in extracted]
                        else:
                            lines = [str(t) for t in texts]

        elapsed = time.perf_counter() - start_time
        return lines, elapsed

    def run_tesseract(self, img):
        print("----- Running Tesseract Engine -----")

        import time
        start_time = time.perf_counter()
        raw_string = pytesseract.image_to_string(img)
        lines = [line.strip() for line in raw_string.split('\n') if line.strip()]
        elapsed = time.perf_counter() - start_time
        return lines, elapsed

    def parse_with_llm(self, raw_lines):

        sys_prompt = """
                You are an AI data extraction agent. Analyze the given OCR text array from a business
                card and return a strict JSON object with the following keys
                - "name": Full name of the person
                - "designation": Job title, specialty or role in company 
                - "email": Email address of the person
                - "phone": Phone number of the person or business
                - "address": Address or location of the person or the company
                If a field is missing, set to null. Return ONLY valid JSON.
                """

        try:
            response = ollama.chat(
                model = self.model_name,
                messages = [
                    {'role':'system', 'content':sys_prompt},
                    {'role':'user', 'content': json.dumps(raw_lines, indent=2)}
                ],
                format = 'json',
                options = {'temperature':0.1}
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            print(f"LLM Parsing ErrorL {e}")
            return {'name': None, 'designation': None, 'email': None, 'phone': None, 'address': None}

    def benchmark_card_bytes(self, image_bytes):
        preprocessed_img = self.preprocess(image_bytes)
        engines = {}

        lines, t = self.run_easyocr(preprocessed_img)
        parsed = self.parse_with_llm(lines)
        engines['EasyOCR'] = {'time_seconds': t, 'parsed_data': parsed, 'raw_text': lines}


        lines, t = self.run_paddleocr(preprocessed_img)
        parsed = self.parse_with_llm(lines)
        engines['PaddleOCR'] = {'time_seconds': t, 'parsed_data': parsed, 'raw_text': lines}


        lines, t = self.run_tesseract(preprocessed_img)
        parsed = self.parse_with_llm(lines)
        engines['Tesseract'] = {'time_seconds': t, 'parsed_data': parsed, 'raw_text': lines}

        return engines

benchmarker = OCRBenchmark()

@app.route('/')
def home():
    return render_template_string("""
    <!doctype html>
    <title> OCR Benchmark API</title>
    <h2>Upload Business Card IMage for Benchmarking</h2>
    <form action="/benchmark" method="post" enctype="multipart/form-data">        <input type="file" name="image" accept="image/*" required>
        <input type="submit" value="Run Benchmark">
    </form>
    <p><a href="/apidocs">Click here for Interactive Swagger API Documentation</a></p> 
    """)

@app.route('/benchmark', methods=['POST'])
def benchmark_endpoint():
    """
        Benchmark OCR engines with an uploaded business card image.
        ---
        parameters:
          - name: image
            in: formData
            type: file
            required: true
            description: The business card image file (JPEG, PNG)
        responses:
          200:
            description: Successful benchmark payload...
        """
    if 'image' not in request.files:
        return jsonify({"error": "no image file provided in the request payload."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error":"No selected file"}), 400
    try:
        image_bytes = file.read()
        report = benchmarker.benchmark_card_bytes(image_bytes)
        return jsonify(report), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Run locally on localhost port 5000
    app.run(host='0.0.0.0', port=5001, debug=False)


