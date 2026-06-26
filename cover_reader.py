import cv2
import pytesseract
import os

# Change this path if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def read_cover(image_path):
    """
    Read text from a book cover image using OCR.
    """

    if not os.path.exists(image_path):
        print("Error: Image not found!")
        return None

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to open image.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to improve OCR
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    # Extract text
    text = pytesseract.image_to_string(gray)

    return text.strip()


# Test
if __name__ == "__main__":
    result = read_cover("data/images.jpg")

    print("\nDetected Text\n")
    print("------------------------")
    print(result)