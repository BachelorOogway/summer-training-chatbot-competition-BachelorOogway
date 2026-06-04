import urllib.request
import urllib.error
import ssl
from html.parser import HTMLParser

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.extracted_text = []
        self.ignore_tags = {'script', 'style', 'head', 'meta', 'noscript'}
        self.is_ignoring = False

    def handle_starttag(self, tag, attrs):
        if tag in self.ignore_tags:
            self.is_ignoring = True

    def handle_endtag(self, tag):
        if tag in self.ignore_tags:
            self.is_ignoring = False

    def handle_data(self, data):
        if not self.is_ignoring:
            clean_text = data.strip()
            if clean_text:
                self.extracted_text.append(clean_text)

    def get_text(self):
        return ' '.join(self.extracted_text)

def scrape_text_from_urls(input_filepath, output_filepath):
    req_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with open(input_filepath, 'r', encoding='utf-8') as infile:
        urls = [line.strip() for line in infile if line.strip()]

    print(f"Scraping {len(urls)} URLs for pure text...\n")

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for idx, url in enumerate(urls, 1):
            try:
                req = urllib.request.Request(url, headers=req_headers)
                with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                    html_content = response.read().decode('utf-8', errors='ignore')
                
                parser = TextExtractor()
                parser.feed(html_content)
                page_text = parser.get_text()
                
                # ONLY write the text content. No source URL or formatting.
                if page_text:
                    outfile.write(page_text)
                
                print(f"[{idx}/{len(urls)}] ✅ Scraped: {url}")
                
            except Exception as e:
                # Print errors to terminal only. Keeps your text file perfectly clean.
                print(f"[{idx}/{len(urls)}] ❌ Failed: {url} | Error: {e}")

if __name__ == "__main__":
    INPUT_FILE = 'urls_non_image.txt'  
    OUTPUT_FILE = 'all_scraped_text.txt'  
    
    scrape_text_from_urls(INPUT_FILE, OUTPUT_FILE)