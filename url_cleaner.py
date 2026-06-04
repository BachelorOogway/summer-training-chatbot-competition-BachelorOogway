import re

def extract_urls(input_file, output_file):
    # Regex pattern to match any string starting with http:// or https:// 
    # and continuing until a space or the end of the line
    url_pattern = re.compile(r'https?://\S+')

    extracted_count = 0

    # Open the input and output files (using utf-8 to safely handle emojis)
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            # Search for a URL in the current line
            match = url_pattern.search(line)
            
            if match:
                # If a URL is found, write it to the output file
                outfile.write(match.group(0) + '\n')
                extracted_count += 1

    print(f"Successfully extracted {extracted_count} URLs and saved them to '{output_file}'.")

# --- Usage ---
if __name__ == "__main__":
    # Replace 'input.txt' with the path to your current text file
    INPUT_FILE = 'website-list.txt'  
    
    # Replace 'urls.txt' with whatever you want the new file to be named
    OUTPUT_FILE = 'urls_clean.txt'  
    
    extract_urls(INPUT_FILE, OUTPUT_FILE)