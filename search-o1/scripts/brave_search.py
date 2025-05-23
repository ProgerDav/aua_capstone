import concurrent
import json
import os
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Optional, Tuple

import pdfplumber
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from requests.exceptions import Timeout
from tqdm import tqdm

# ----------------------- Custom Headers -----------------------
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/58.0.3029.110 Safari/537.36",
    "Referer": "https://www.google.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Initialize session
session = requests.Session()
session.headers.update(headers)


def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)


def extract_snippet_with_context(
    full_text: str, snippet: str, context_chars: int = 2500
) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        sentences = sent_tokenize(full_text)

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            return False, full_text[: context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"


def extract_text_from_url(
    url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None
):
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): URL of a webpage or PDF.
        use_jina (bool): Whether to use Jina for extraction.
        snippet (Optional[str]): The snippet to search for.

    Returns:
        str: Extracted text or context.
    """
    try:
        if use_jina:
            jina_headers = {
                "Authorization": f"Bearer {jina_api_key}",
                "X-Return-Format": "markdown",
            }
            response = requests.get(
                f"https://r.jina.ai/{url}", headers=jina_headers
            ).text
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = (
                re.sub(pattern, "", response)
                .replace("---", "-")
                .replace("===", "=")
                .replace("   ", " ")
                .replace("   ", " ")
            )
        else:
            response = session.get(url, timeout=20)  # Set timeout to 20 seconds
            response.raise_for_status()  # Raise HTTPError if the request failed
            # Determine the content type
            content_type = response.headers.get("Content-Type", "")
            if "pdf" in content_type:
                # If it's a PDF file, extract PDF text
                return extract_pdf_text(url)
            # Try using lxml parser, fallback to html.parser if unavailable
            try:
                soup = BeautifulSoup(response.text, "lxml")
            except Exception:
                print("lxml parser not found or failed, falling back to html.parser")
                soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            # If no snippet is provided, return directly
            return text[:8000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def fetch_page_content(
    urls,
    max_workers=32,
    use_jina=False,
    jina_api_key=None,
    snippets: Optional[dict] = None,
):
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to display a progress bar
        futures = {
            executor.submit(
                extract_text_from_url,
                url,
                use_jina,
                jina_api_key,
                snippets.get(url) if snippets else None,
            ): url
            for url in urls
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            desc="Fetching URLs",
            total=len(urls),
        ):
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            time.sleep(0.2)  # Simple rate limiting
    return results


def brave_web_search(query, api_key, num_results=10, timeout=20):
    """
    Perform a search using the Brave Search API with a set timeout.

    Args:
        query (str): Search query.
        api_key (str): Brave Search API key.
        num_results (int): Number of search results to return.
        timeout (int or float or tuple): Request timeout in seconds.

    Returns:
        dict: JSON response of the search results. Returns empty dict if request fails.
    """
    base_url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
    params = {"q": query, "count": num_results}

    try:
        response = requests.get(
            base_url, headers=headers, params=params, timeout=timeout
        )
        response.raise_for_status()  # Raise exception if the request failed
        search_results = response.json()
        return search_results
    except Timeout:
        print(f"Brave Search request timed out ({timeout} seconds) for query: {query}")
        return {}
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during Brave Search request: {e}")
        return {}


def extract_relevant_info(search_results):
    """
    Extract relevant information from Brave search results.

    Args:
        search_results (dict): JSON response from the Brave Search API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []

    if "web" in search_results and "results" in search_results["web"]:
        for id, result in enumerate(search_results["web"]["results"]):
            info = {
                "id": id + 1,  # Increment id for easier subsequent operations
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get(
                    "description", ""
                ),  # Brave uses 'description' instead of 'snippet'
                "context": "",  # Reserved field to be filled later
            }
            useful_info.append(info)

    return useful_info


def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=20)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"

        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text

        # Limit the text length
        cleaned_text = " ".join(full_text.split()[:600])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


# ------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    # Define the query to search
    query = "Structure of dimethyl fumarate"

    # Brave Search API key
    BRAVE_API_KEY = "BSAoVzacG6Ojl8OM6Q_Ryx-JG-5KMag"

    if not BRAVE_API_KEY:
        raise ValueError("Please set the BRAVE_API_KEY environment variable.")

    # Perform the search
    print("Performing Brave Web Search...")
    search_results = brave_web_search(query, BRAVE_API_KEY)

    print("Extracting relevant information from search results...")
    extracted_info = extract_relevant_info(search_results)

    print("Fetching and extracting context for each snippet...")
    for info in tqdm(extracted_info, desc="Processing Snippets"):
        full_text = extract_text_from_url(
            info["url"], use_jina=True
        )  # Get full webpage text
        if full_text and not full_text.startswith("Error"):
            success, context = extract_snippet_with_context(
                full_text, info["description"]
            )
            if success:
                info["context"] = context
            else:
                info["context"] = (
                    f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
                )
        else:
            info["context"] = f"Failed to fetch full text: {full_text}"

    # print("Info")
