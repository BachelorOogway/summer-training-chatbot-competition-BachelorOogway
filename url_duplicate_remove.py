import urllib.parse
import re

def clean_and_normalize_url(url):
    # 1. Strip leading/trailing whitespace and line breaks
    url = url.strip()
    
    # 2. Strip accidental trailing punctuation (like the colon in your example)
    url = re.sub(r'[.,:]+$', '', url)
    
    # 3. Parse the URL into its components
    parsed = urllib.parse.urlsplit(url)
    
    # 4. Remove trailing slashes from the path (so example.com/ == example.com)
    path = parsed.path.rstrip('/')
    
    # 5. Rebuild the URL *without* the fragment (removes #content, etc.)
    # urlunsplit format: (scheme, netloc, path, query, fragment)
    canonical_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ''))
    
    return canonical_url

def extract_unique_urls(input_filepath, output_filepath):
    unique_urls = set()
    total_lines = 0

    # Read the raw URLs
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue # Skip empty lines
            
            total_lines += 1
            clean_url = clean_and_normalize_url(line)
            unique_urls.add(clean_url)

    # Write the unique URLs to the new file
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        # Sorting is optional, but helps keep the output file organized
        for url in sorted(unique_urls):
            outfile.write(url + '\n')

    print(f"Processed {total_lines} total URLs.")
    print(f"Found {len(unique_urls)} UNIQUE URLs.")
    print(f"Saved to: '{output_filepath}'")

# --- Usage ---
if __name__ == "__main__":
    # Point this to your massive text file of URLs
    INPUT_FILE = 'urls_clean.txt'  
    
    # This is where the clean, unique list will be saved
    OUTPUT_FILE = 'urls_unique_clean.txt'  
    
    extract_unique_urls(INPUT_FILE, OUTPUT_FILE)