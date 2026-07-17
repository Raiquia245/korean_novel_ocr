import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image
import cv2
import os

def extract_korean_text(image_path):
    """
    Extract Korean text from an image using OCR without preprocessing.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    config = '--psm 6'
    return pytesseract.image_to_string(image, lang='kor', config=config)

def process_batch_of_korean_images(directory_path='extract', output_file='extracted_korean_text.txt'):
    """
    Process batch images sorted in ascending order by filename.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    image_files = sorted(
        [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(image_extensions)]
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_file in image_files:
            try:
                text = extract_korean_text(image_file)
                f.write(f"=== {os.path.basename(image_file)} ===\n{text}\n\n")
                print(f"Processed: {image_file}")
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

if __name__ == "__main__":
    process_batch_of_korean_images()
