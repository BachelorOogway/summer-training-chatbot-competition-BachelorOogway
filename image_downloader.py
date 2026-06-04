import os
import urllib.request
import urllib.parse

def download_images(input_filepath, download_dir):
    # 1. Create the target directory if it doesn't already exist
    os.makedirs(download_dir, exist_ok=True)
    
    success_count = 0
    fail_count = 0

    # 2. Set up a User-Agent so websites don't block our script for acting like a bot
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)

    # 3. Read the URLs into a list
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        urls = [line.strip() for line in infile if line.strip()]

    print(f"Starting download of {len(urls)} images to '{download_dir}/'...\n")

    # 4. Iterate through the URLs and download them
    for idx, url in enumerate(urls, 1):
        try:
            # Extract the actual filename from the URL (e.g., 'logo.png' from 'example.com/logo.png')
            parsed_url = urllib.parse.urlsplit(url)
            filename = os.path.basename(parsed_url.path)
            
            # Fallback just in case the URL path is weird and lacks a standard filename
            if not filename:
                filename = f"downloaded_img_{idx}.jpg"
                
            filepath = os.path.join(download_dir, filename)
            
            # Download and save the image
            urllib.request.urlretrieve(url, filepath)
            print(f"[{idx}/{len(urls)}] ✅ Saved: {filename}")
            success_count += 1
            
        except Exception as e:
            # If a link is dead or blocks us, skip it and keep going without crashing
            print(f"[{idx}/{len(urls)}] ❌ Failed: {url} | Error: {e}")
            fail_count += 1

    # 5. Print final stats
    print("\n--- Download Complete ---")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed to download: {fail_count}")

# --- Usage ---
if __name__ == "__main__":
    # Point this to the text file containing ONLY image URLs
    INPUT_FILE = 'urls_image.txt'  
    
    # The folder where all images will be saved
    DOWNLOAD_FOLDER = 'images'
    
    download_images(INPUT_FILE, DOWNLOAD_FOLDER)