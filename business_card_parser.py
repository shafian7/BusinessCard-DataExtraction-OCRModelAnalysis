import os
import re
import cv2
import numpy as np
import easyocr
import json
import ollama

class BusinessCardParser:

    def __init__(self, languages=['en'], model_name='llama3.2'):
        #Initialise right at start, avoids reinitialisation every time an image begins to get processed.
        print("OCR Engine Initialising")
        self.reader = easyocr.Reader(languages, gpu=False)
        self.model_name = model_name


        # Keywords for obtaining/parsing designation from the business card
        # as regex won't be accurate enough.
        self.designation_keywords = [
            'manager', 'director', 'executive', 'officer', 'engineer', 'developer',
            'lead', 'head', 'specialist', 'consultant', 'president', 'ceo', 'cto',
            'cfo', 'coo', 'founder', 'co-founder', 'architect', 'designer', 'analyst',
            'administrator', 'coordinator', 'associate', 'vp', 'vice president',
            'partner', 'principal', 'representative', 'professor', 'specialist', 'professor',
            'surgeon', 'assistant professor', 'associate professor'
        ]

        # Keywords to identify addresses as well because
        # it has the same situation as for designation.
        self.address_keywords = [
            'street', 'st', 'avenue', 'ave', 'road', 'rd', 'boulevard', 'blvd',
            'lane', 'ln', 'drive', 'dr', 'suite', 'ste', 'floor', 'fl', 'building',
            'bldg', 'p.o. box', 'po box', 'box', 'zip', 'city', 'state', 'country',
            'park', 'way', 'plaza', 'highway', 'hwy'
        ]

    def image_preprocessor(self, path):
        '''Used for image preprocessing in order to increase the accuracy of the to-be used
        easyOCR  model'''
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"File could not be found in the specified path: {path}")

        # Image resizing
        height, width = img.shape[:2]
        target_height = 1000
        aspect_ratio = width / float(height)
        new_width = int(target_height * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

        #Grayscaling
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # Remove noise - Gaussian Blur
        blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

        # Adaptive contrast - basically convert to black and white (binarization) using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(blurred_img)

        return enhanced_img

    def extract_text_positions(self, path):

        preprocessed_img = self.image_preprocessor(path)
        results = self.reader.readtext(preprocessed_img, detail=1)

        # SO in case processing hte preprocessed image yields too few results
        # make the OCR engine read the raw image instead to
        # increase results for better postprocessing (regex and keyword checking)
        if len(results) < 3:
            results = self.reader.readtext(path, detail=1)

        sorted_results = sorted(results, key=lambda item: item[0][0][1])

        text_lines = [text for bbox, text, conf in sorted_results]
        return text_lines

    def parse_with_llama(self, raw_lines):

        # Parsing now done with local LLM with semantic logic in to intelligently extract and map fields.
        print(f"Parsing extracted raw text using local LLM: {self.model_name}")

        sys_prompt = """
        You are an AI data extraction agent. You will be given a list of text strings extracted from a business card via OCR.
        You task is to analyse the text and map it into a strict JSON object with following keys:
        -"name": The full name of the person, excluding honorifics like Mr. Mrs. Ms. or Dr. if separate or include them properly
        -"designation": The job title, role and/or rank of the person
        -"email": The email address of the person
        -"phone": The phone number (with country code if present)
        -"address": The physical location, chamber or company address provided
        
        If a field cannot be found, set its value to null. Return ONLY valid JSON.
        """

        user_prompt = f"Here is the raw OCR text array from the business card: \n {json.dumps(raw_lines, indent=2)}"

        try:
            response = ollama.chat(
                model = self.model_name,
                messages = [
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                format='json', #Additional safety measure to ensure ollama generates only valid JSON responses
                options={'temperature': 0.1}
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
            return {
                "name": None, "designation": None, "email": None,
                "phone": None, "address": None
            }


    def process_card(self, path):

        raw_lines = self.extract_text_positions(path)

        extracted_data = self.parse_with_llama(raw_lines)

        return {
        **extracted_data,
        "raw_text": raw_lines
        }

if __name__ == "__main__":
    import sys

    card_image_path = 'BusinessCard.jpeg'

    if len(sys.argv) > 1:
        card_image_path = sys.argv[1]

    if not os.path.exists(card_image_path):
        print(f"Please provide a valid path for finding the image. File Not Found: {card_image_path}")
    else:
        parser = BusinessCardParser()
        extracted_info = parser.process_card(card_image_path)
        print("---- Extracted Info ----")
        print(json.dumps(extracted_info, indent=4))

        output_filename = "extracted_card_data.json"
        with open(output_filename, 'w') as file:
            json.dump(extracted_info, file, indent=4, ensure_ascii=False)

        print(f"Success: Data successfully saved to {output_filename}`")










