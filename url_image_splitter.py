import urllib.parse

def separate_urls(input_filepath, images_filepath, non_images_filepath):
    # A comprehensive tuple of common image extensions
    image_extensions = (
        '.jpg', '.jpeg', '.png', '.gif', '.svg', 
        '.webp', '.bmp', '.ico', '.tiff', '.tif'
    )
    
    img_count = 0
    web_count = 0

    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(images_filepath, 'w', encoding='utf-8') as img_out, \
         open(non_images_filepath, 'w', encoding='utf-8') as web_out:
         
        for line in infile:
            url = line.strip()
            if not url:
                continue # Skip empty lines
            
            # Parse the URL to isolate the path (ignoring ?queries or #fragments)
            parsed_url = urllib.parse.urlsplit(url)
            
            # Convert path to lowercase to catch extensions like .JPG or .PnG
            path = parsed_url.path.lower()
            
            # Check if the path ends with any of the defined image extensions
            if path.endswith(image_extensions):
                img_out.write(url + '\n')
                img_count += 1
            else:
                web_out.write(url + '\n')
                web_count += 1

    print("Separation complete.")
    print(f"🖼️ Found {img_count} Image URLs -> Saved to '{images_filepath}'")
    print(f"📄 Found {web_count} Webpage URLs -> Saved to '{non_images_filepath}'")

# --- Usage ---
if __name__ == "__main__":
    # Point this to your cleaned, deduplicated text file from the previous step
    INPUT_FILE = 'urls_unique_clean.txt'  
    
    # Define your output files
    IMAGES_FILE = 'urls_image.txt'
    NON_IMAGES_FILE = 'urls_non_image.txt'
    
    separate_urls(INPUT_FILE, IMAGES_FILE, NON_IMAGES_FILE)