import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image
import cv2
import numpy as np
import os

def extract_korean_text(image_path, preprocess=None):
    """
    Extract Korean text from an image using OCR.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    if preprocess == 'thresh':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    elif preprocess == 'blur':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.GaussianBlur(gray, (5, 5), 0)
    elif preprocess == 'adaptive':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        processed_image = image
    
    config = '--psm 6'
    pil_image = Image.fromarray(processed_image) if preprocess else processed_image
    return pytesseract.image_to_string(pil_image, lang='kor', config=config)

def extract_korean_text_with_layout(image_path):
    """Extract Korean text with layout information."""
    image = cv2.imread(image_path)
    data = pytesseract.image_to_data(image, lang='kor', output_type=pytesseract.Output.DICT)
    return [{
        'text': data['text'][i], 'confidence': data['conf'][i],
        'left': data['left'][i], 'top': data['top'][i]
    } for i in range(len(data['text'])) if int(data['conf'][i]) > 0]

def process_batch_of_korean_images(directory_path, output_file='extracted_korean_text.txt', preprocess=None):
    """Process batch images sorted in ascending order by filename."""
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    image_files = sorted(
        [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(image_extensions)]
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_file in image_files:
            try:
                text = extract_korean_text(image_file, preprocess)
                f.write(f"=== {os.path.basename(image_file)} ===\n{text}\n\n")
                print(f"Processed: {image_file}")
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

def main():
    print("Pilih fitur yang ingin digunakan:")
    print("1. Ekstrak teks dari satu gambar")
    print("2. Ekstrak teks dengan layout dari satu gambar")
    print("3. Proses batch gambar dalam satu folder")
    choice = input("Masukkan nomor fitur: ")
    
    if choice == '1':
        image_path = input("Masukkan path gambar: ")
        preprocess = input("Pilih preprocessing (thresh/blur/adaptive/tidak): ")
        preprocess = preprocess if preprocess in ['thresh', 'blur', 'adaptive'] else None
        print("Hasil OCR:")
        print(extract_korean_text(image_path, preprocess))
    
    elif choice == '2':
        image_path = input("Masukkan path gambar: ")
        print("Hasil dengan layout:")
        print(extract_korean_text_with_layout(image_path))
    
    elif choice == '3':
        directory_path = input("Masukkan path folder: ")
        preprocess = input("Pilih preprocessing (thresh/blur/adaptive/tidak): ")
        preprocess = preprocess if preprocess in ['thresh', 'blur', 'adaptive'] else None
        output_file = input("Masukkan nama file output (default: extracted_korean_text.txt): ") or 'extracted_korean_text.txt'
        process_batch_of_korean_images(directory_path, output_file, preprocess)
    
    else:
        print("Pilihan tidak valid.")

if __name__ == "__main__":
    main()
