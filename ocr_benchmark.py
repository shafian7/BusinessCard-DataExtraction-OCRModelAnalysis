import os
import time
import cv2
import numpy as np
import json
import ollama
import easyocr
import pytesseract
from paddleocr import PaddleOCR


class OCRBenchmark:
    def __init__(self, model_name='llama3.2'):
        print("Initialising OCR Engines for Benchmarking")
        self.model_name = model_name

        # Initialise all OCR models
        self.easy_reader = easyocr.Reader(['en'], gpu=False)
        self.paddle_reader = PaddleOCR(use_textline_orientation=True, lang='en')

    def preprocess(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"File not found: {path} ")

        height, width = img.shape[:2]
        target_height = 1000
        aspect_ratio = width / float(height)
        new_width = int(target_height * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Convert back to 3-channel BGR to satisfy PaddleOCR channel requirements
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr

    def run_easyocr(self, img):
        start_time = time.perf_counter()
        results = self.easy_reader.readtext(img, detail=1)
        sorted_results = sorted(results, key=lambda x: x[0][0][1])
        lines = [text for bbox, text, conf in sorted_results]
        elapsed = time.perf_counter() - start_time
        return lines, elapsed

    def run_paddleocr(self, img):
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

                if isinstance(first_page, dict) or hasattr(first_page, 'keys'):
                    if 'rec_text' in first_page:
                        lines=  list(first_page['rec_text'])
                    elif hasattr(first_page, 'get') and first_page.get('rec_text'):
                        lines = list(first_page.get('rec_text'))

                elif isinstance(first_page, (list, tuple)):
                    extracted = []
                    for item in first_page:
                        if isinstance(item, list) and len(item) ==2 and isinstance(item[1], (tuple, list)):
                            try:
                                y_coord = item[0][0][1]
                                text = item[1][0]
                                extracted.append((y_coord, text))
                            except (IndexError, TypeError):
                                pass
                    extracted.sort(key=lambda x: x[0])
                    lines = [txt for _, txt in extracted]

        elapsed = time.perf_counter() - start_time
        return lines, elapsed

    def run_tesseract(self, img):
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
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': json.dumps(raw_lines, indent=2)}
                ],
                format='json',
                options={'temperature': 0.1}
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            print(f"LLM Parsing Error: {e}")
            return {'name': None, 'designation': None, 'email': None, 'phone': None, 'address': None}

    def benchmark_card(self, path):
        preprocessed_img = self.preprocess(path)

        engines = {}

        print("----- Running EasyOCR -----")
        lines, t = self.run_easyocr(preprocessed_img)
        parsed = self.parse_with_llm(lines)
        engines["EasyOCR"] = {"time_seconds": round(t, 4), 'parsed_data': parsed, "raw_text": lines}

        print("----- Running PaddleOCR -----")
        lines, t = self.run_paddleocr(preprocessed_img)
        parsed = self.parse_with_llm(lines)
        engines["PaddleOCR"] = {"time_seconds": round(t, 4), "parsed_data": parsed, "raw_text": lines}

        print("----- Running Tesseract -----")
        lines, t = self.run_tesseract(preprocessed_img)
        parsed = self.parse_with_llm(lines)
        engines["Tesseract"] = {"time_seconds": round(t, 4), "parsed_data": parsed, "raw_text": lines}

        return engines


if __name__ == "__main__":
    import sys

    card_image_path = 'Businesscard.jpeg'
    if len(sys.argv) > 1:
        card_image_path = sys.argv[1]

    if not os.path.exists(card_image_path):
        print(f"File not found: {card_image_path}")
    else:
        benchmarker = OCRBenchmark()
        report = benchmarker.benchmark_card(card_image_path)

        print("\n ----- Benchmark Results Summary -----")
        for engine, data in report.items():
            print(f"\nEngine: {engine}")
            print(f" Time Taken: {data['time_seconds']}")
            print(f"Extracted Fields: {data['parsed_data']}")
            print(f"Raw Text Line Count: {len(data['raw_text'])}")

        output_filename = 'ocr_benchmark_report.json'
        with open(output_filename, 'w', encoding='utf-8') as file:
            json.dump(report, file, indent=4, ensure_ascii=False)
        print(f"\n Success: Full report saved to {output_filename}")