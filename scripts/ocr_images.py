import os
import easyocr

def batch_ocr(input_folder="images", output_file="extracted_text.txt"):
    if not os.path.exists(input_folder):
        print(f"Error: Directory '{input_folder}' not found.")
        return

    # Initialize the reader for English. 
    # Set gpu=True if your VM has a dedicated GPU and CUDA installed.
    print("Initializing EasyOCR models (this may take a moment to download on first run)...")
    reader = easyocr.Reader(['en'], gpu=False)

    with open(output_file, 'a', encoding='utf-8') as f:
        for filename in sorted(os.listdir(input_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')):
                img_path = os.path.join(input_folder, filename)
                
                try:
                    # detail=0 returns only the extracted text strings, stripping out bounding boxes and confidence scores
                    text_list = reader.readtext(img_path, detail=0)
                    
                    # Join the detected text blocks and write to file
                    # raw_text = "\n".join(text_list) + "\n"
                    f.write("".join(text_list))
                    print(f"Processed: {filename}")
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    batch_ocr()