import os
import orjson
import argparse
import weasyprint
import urllib.parse
import os
import time
import concurrent.futures
import pandas as pd
from pathlib import Path
import threading
from tqdm import tqdm

def download_wiki_pdf(title, output_dir="./wiki_pdfs", max_retries=3):
    """
    Download a Wikipedia page as PDF, trying multiple language versions if needed
    
    Args:
        title (str): Wikipedia article title
        output_dir (str): Directory to save PDFs
        max_retries (int): Maximum number of retry attempts per language
        
    Returns:
        dict: Results including original_title, file_path, page_count, size_mb, language
    """
    # Create safe filename
    safe_title = title.replace("/", "___")
    output_file = os.path.join(output_dir, f"{safe_title}.pdf")

    if os.path.exists(output_file):
        # Fetch the page count from the local PDF file
        pdf_document = weasyprint.PDF(output_file)
        page_count = len(pdf_document.pages)
        size_mb = round(os.path.getsize(output_file) / (1024 * 1024), 2)
        return {
            "original_title": title,
            "file_path": output_file,
            "page_count": page_count,
            "size_mb": size_mb,
            "language": "en"  # Assume English for existing files
        }
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    result = {
        "original_title": title,
        "file_path": output_file,
        "page_count": 0,
        "size_mb": 0.0,
        "language": None
    }
    
    # Try different language Wikipedia versions
    languages = ["en", "fr", "ru", "es"]
    
    for lang in languages:
        # Create URL for current language
        encoded_title = urllib.parse.quote(title)
        url = f"https://{lang}.wikipedia.org/wiki/{encoded_title}"
        
        # Try to download with retries
        for attempt in range(max_retries):
            try:
                # Download and convert to PDF
                pdf_document = weasyprint.HTML(url).render()
                pdf_document.write_pdf(output_file)
                
                # Get file metadata
                result["page_count"] = len(pdf_document.pages)
                result["size_mb"] = round(os.path.getsize(output_file) / (1024 * 1024), 2)
                result["language"] = lang
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: wait longer between each retry
                    time.sleep(2 ** attempt)
                else:
                    # Log the error on the final attempt for this language
                    print(f"Failed to generate PDF for '{title}' from {lang} Wikipedia after {max_retries} attempts: {e}")
                    # Continue to the next language
    
    # If we reach here, all languages and attempts failed
    return result

def process_wikipedia_titles(titles, max_workers=4, output_dir="./wiki_pdfs"):
    """
    Parallel process a list of Wikipedia titles to PDFs
    
    Args:
        titles (list): List of Wikipedia article titles
        max_workers (int): Number of parallel workers
        output_dir (str): Directory to save PDFs
        
    Returns:
        tuple: (DataFrame with results, List of failed titles)
    """
    results = []
    failed_titles = []
    lock = threading.Lock()  # For thread-safe access to shared lists
    
    # Progress bar setup
    pbar = tqdm(total=len(titles), desc="Downloading Wikipedia PDFs")
    
    def download_and_update(title):
        result = download_wiki_pdf(title, output_dir)
        with lock:
            results.append(result)
            # Check if download completely failed across all languages
            if result["size_mb"] == 0:
                failed_titles.append(title)
            pbar.update(1)
        return result
    
    # Create and run the thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_title = {executor.submit(download_and_update, title): title for title in titles}
        
        # Wait for all tasks to complete
        concurrent.futures.wait(future_to_title)
    
    pbar.close()
    
    # Create a DataFrame from results
    df = pd.DataFrame(results)
    
    # Print success statistics
    success_count = df[df['size_mb'] > 0].shape[0]
    print(f"Successfully downloaded {success_count} of {len(titles)} Wikipedia articles as PDFs")
    print(f"Failed to download {len(failed_titles)} articles from any language version")
    
    # Save failed titles to a file
    with open(os.path.join(output_dir, "failed_titles.txt"), "w") as f:
        for title in failed_titles:
            f.write(f"{title}\n")
    
    return df, failed_titles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./data/FlashRAG_datasets/2wiki/test.json")
    parser.add_argument("--output_dir", type=str, default="./wiki_pdfs")
    args = parser.parse_args()

    with open(args.input_file, "r") as file:
        data = orjson.loads(file.read())

    titles = set()

    for i in data:
        context = i['context']
        t = [j[0] for j in context]
        for j in t:
            titles.add(j)


    titles = list(titles)
    print("Number of titles: ", len(titles))

    df, failed_titles = process_wikipedia_titles(titles, max_workers=12, output_dir=args.output_dir)

    # Save the results DataFrame to CSV
    df.to_csv(os.path.join(args.output_dir, "download_results.csv"), index=False)

    # Print some statistics about languages used
    if 'language' in df.columns:
        lang_counts = df['language'].value_counts()
        print("\nArticles by language:")
        for lang, count in lang_counts.items():
            if pd.notna(lang):
                print(f"  {lang}: {count}")