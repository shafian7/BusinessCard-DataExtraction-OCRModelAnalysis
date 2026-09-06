import os
import cv2
import numpy as np
import json
import ollama
import easyocr
import pytesseract
from paddleocr import PaddleOCR
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Literal

app = FastAPI(
    title="OCR Benchmark & Extraction API",
    description="API for running multi-engine benchmarks and single-engine extraction from business cards.",
    version="2.1"
)


class OCRBenchmark:
    def __init__(self, model_name='llama3.2'):
        print("Initialising OCR Engines for Benchmarking...")
        self.model_name = model_name

        # Initialise all OCR models once on startup
        self.easy_reader = easyocr.Reader(['en'], gpu=False)
        self.paddle_reader = PaddleOCR(use_textline_orientation=True, lang='en')

    def preprocess_image_bytes(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file provided.")

        height, width = img.shape[:2]
        target_height = 1000
        aspect_ratio = width / float(height)
        new_width = int(target_height * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Convert back to 3-channel BGR to satisfy PaddleOCR requirements
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr

    def run_easyocr(self, img):
        import time
        start_time = time.perf_counter()
        results = self.easy_reader.readtext(img, detail=1)
        sorted_results = sorted(results, key=lambda x: x[0][0][1])
        lines = [text for bbox, text, conf in sorted_results]
        elapsed = time.perf_counter() - start_time
        return lines, elapsed

    def run_paddleocr(self, img):
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
                if isinstance(first_page, (list, tuple)) and len(first_page) > 0 and isinstance(first_page[0],
                                                                                                (list, tuple)):
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
        import time
        start_time = time.perf_counter()
        raw_string = pytesseract.image_to_string(img)
        lines = [line.strip() for line in raw_string.split('\n') if line.strip()]
        elapsed = time.perf_counter() - start_time
        return lines, elapsed

    def parse_with_llm(self, raw_lines):
        sys_prompt = """
        You are an AI data extraction agent. Analyze the given OCR text array from a business
        card and return a strict JSON object with the following keys:
        - "name": Full name of the person
        - "designation": Job title, specialty or role in company 
        - "email": Email address of the person
        - "phone": Phone number of the person or business
        - "address": Address or location of the person or the company
        If a field is missing, set to null. Return ONLY valid JSON.
        """
        try:
            user_content_str = str(json.dumps(raw_lines, indent=2))
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': str(sys_prompt)},
                    {'role': 'user', 'content': user_content_str}
                ],
                format='json',
                options={'temperature': 0.1}
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            print(f"LLM Parsing Error: {e}")
            return {'name': None, 'designation': None, 'email': None, 'phone': None, 'address': None}


# Global instance
benchmarker = OCRBenchmark(model_name='llama3.2')


@app.get("/", response_class=HTMLResponse)
def home():
    """Web dashboard providing forms for both Multi-Engine Benchmarking and Single-Engine Extraction."""
    return """
    <!doctype html>
    <title>OCR API Dashboard</title>
    <body style="font-family: Arial; padding: 30px; max-width: 700px; margin: auto; background: #f9f9f9;">
      <h2 style="color: #333;">Business Card OCR & Extraction API</h2>

      <!-- Section 1: Multi-Engine Benchmark -->
      <div style="background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="margin-top:0; color: #007BFF;">1. Run Full Multi-Engine Benchmark</h3>
        <p style="font-size: 14px; color: #666;">Compares EasyOCR, PaddleOCR, and Tesseract simultaneously against a business card.</p>
        <form action="/benchmark" method="post" enctype="multipart/form-data">
          <input type="file" name="image" accept="image/*" required><br><br>
          <button type="submit" style="padding: 8px 16px; background: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">Run Full Benchmark</button>
        </form>
      </div>

      <!-- Section 2: Single Engine Extraction & Download -->
      <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="margin-top:0; color: #28a745;">2. Single Engine Extraction & Download</h3>
        <p style="font-size: 14px; color: #666;">Choose a specific OCR engine, process the card, and download the resulting structured JSON file.</p>
        <form action="/extract" method="post" enctype="multipart/form-data">
          <label style="font-size: 14px;"><b>Choose Engine:</b></label><br>
          <select name="engine_name" style="padding: 6px; width: 220px; margin: 8px 0;">
            <option value="EasyOCR">EasyOCR</option>
            <option value="PaddleOCR">PaddleOCR</option>
            <option value="Tesseract">Tesseract</option>
          </select><br><br>
          <input type="file" name="image" accept="image/*" required><br><br>
          <button type="submit" style="padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">Extract & Download JSON</button>
        </form>
      </div>

      <p style="margin-top: 25px; text-align: center;"><a href="/docs" target="_blank" style="color: #555;">View Interactive FastAPI Swagger Documentation (/docs)</a></p>
    </body>
    """


@app.post("/benchmark")
async def benchmark_endpoint(image: UploadFile = File(...)):
    """
    Runs all three engines (EasyOCR, PaddleOCR, Tesseract) and returns a comparative JSON report payload.
    """
    try:
        image_bytes = await image.read()
        preprocessed_img = benchmarker.preprocess_image_bytes(image_bytes)

        engines = {}
        for name, runner in [
            ("EasyOCR", benchmarker.run_easyocr),
            ("PaddleOCR", benchmarker.run_paddleocr),
            ("Tesseract", benchmarker.run_tesseract)
        ]:
            lines, t = runner(preprocessed_img)
            parsed = benchmarker.parse_with_llm(lines)
            engines[name] = {
                "time_seconds": round(t, 4),
                "parsed_data": parsed,
                "raw_text": lines
            }

        return JSONResponse(content=engines)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
async def extract_business_card(
        image: UploadFile = File(...),
        engine_name: Literal["EasyOCR", "PaddleOCR", "Tesseract"] = Form(...)
):
    """
    Upload a business card, choose a single engine, and receive a downloadable JSON report containing
    the parsed fields, raw text lines, time taken, and engine name.
    """
    try:
        image_bytes = await image.read()
        preprocessed_img = benchmarker.preprocess_image_bytes(image_bytes)

        if engine_name == "EasyOCR":
            lines, t = benchmarker.run_easyocr(preprocessed_img)
        elif engine_name == "PaddleOCR":
            lines, t = benchmarker.run_paddleocr(preprocessed_img)
        elif engine_name == "Tesseract":
            lines, t = benchmarker.run_tesseract(preprocessed_img)
        else:
            raise HTTPException(status_code=400, detail="Invalid engine selected.")

        parsed_data = benchmarker.parse_with_llm(lines)

        report = {
            "engine_name": engine_name,
            "time_seconds": round(t, 4),
            "parsed_data": parsed_data,
            "raw_text": lines
        }

        return JSONResponse(
            content=report,
            headers={"Content-Disposition": f"attachment; filename=ocr_report_{engine_name.lower()}.json"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5002, reload=True)