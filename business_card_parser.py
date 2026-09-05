import os
import re
import cv2
import numpy as np
import easyocr
import json

class BusinessCardParser:

    def __init__(self, languages=['en']):
        #Initialise right at start, avoids reinitialisation every time an image begins to get processed.
        print("OCR Engine Initialising")
        self.reader = easyocr.Reader(languages, gpu=False)


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

    def parse_email(self, lines):
        email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'
        full_text = " ".join(lines)
        matches = re.findall(email_regex, full_text)
        return matches[0] if matches else None

    def parse_phone(self, lines):
        phone_regex = r'(?:\+\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}'
        full_text = ' '.join(lines)
        matches = re.findall(phone_regex, full_text)
        for m in matches:
            cleaned = re.sub(r'\D', '', m)
            if 7 <= len(cleaned) <= 15:
                return m.strip()
        return None

    def parse_designation(self, lines):
        found_designations = []
        for line in lines:
            line = line.lower()
            if any (kw in line for kw in self.designation_keywords):
                found_designations.append(line)
        return ", ".join(found_designations) if found_designations else None
    def parse_address(self, lines):
        address_kw = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'hospital', 'college', 'chamber']
        address_parts = []
        zip_pattern = r'\b\d{5}(?:-\d{4})?\b|\b\d{6}\b'

        for line in lines:
            line = line.lower()
            if any (kw in line for kw in address_kw) or bool(re.search(zip_pattern, line)):
                # Exclusion of possible emails, phones and designations
                if '@' not in line and not re.search(r'\++?\d{10,}', line):
                    address_parts.append(line.strip())

        return ", ".join(address_parts) if address_parts else None

    def parse_name(self, lines, email, designation, phone, address):

        # Only look at top 50% because that is where person's name normally resides
        top_lines = lines[:max(2, len(lines) // 2)]
        exclusions = ['dr', 'prof', 'md', 'mbbs', 'fcps', 'mr', 'mrs', 'ms', 'chamber', 'time', 'appointment']
        if email:exclusions.append(email.lower())
        if designation: exclusions.append(designation.lower())
        if phone: exclusions.append(re.sub(r'\D', '', phone))
        if address: exclusions.extend([part.lower() for part in address.split(" ")])

        candidate_lines = []

        for line in top_lines:
            line = line.strip().lower()

            if '@' in line or re.search(r'\d', line):
                continue

            if any(ex in line for ex in exclusions):
                continue

            words = line.split()

            if 2 <= len(words) <= 4:
                alphabetic_words = [w for w in words if w.isalpha()]
                if alphabetic_words and all(w[0].isupper() for w in alphabetic_words):
                    return line

        return None

    def process_card(self, path):

        raw_lines = self.extract_text_positions(path)

        email = self.parse_email(raw_lines)
        phone = self.parse_phone(raw_lines)
        designation = self.parse_designation(raw_lines)
        address = self.parse_address(raw_lines)
        name = self.parse_name(raw_lines, email, designation, phone, address)

        return {
            "name": name, 'designation': designation, 'email': email,
            'phone': phone, 'address': address, 'raw_text':raw_lines
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











