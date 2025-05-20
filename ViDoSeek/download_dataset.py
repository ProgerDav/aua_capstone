import requests
import os
import zipfile


def download_file(url, output_file):
    """Download a file from a URL and save it locally."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully and saved as '{output_file}'")
    else:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")
        return False
    return True


def extract_and_cleanup(zip_file, extract_to):
    """Extract a zip file and remove it after extraction."""
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"File extracted successfully to '{extract_to}'")
    os.remove(zip_file)
    print(f"Zip file '{zip_file}' removed successfully")


if __name__ == "__main__":
    url_pdfs = "https://huggingface.co/datasets/autumncc/ViDoSeek/resolve/main/vidoseek_pdf_document.zip?download=true"
    url_slidevqa = "https://huggingface.co/datasets/autumncc/ViDoSeek/resolve/main/slidevqa_pdf_document.zip?download=true"

    # Paths for the first file
    output_file_pdfs = "./ViDoRAG/ViDoSeek/vidoseek_pdf_document.zip"
    extract_to_pdfs = "./ViDoRAG/ViDoSeek"

    # Paths for the second file
    output_file_slidevqa = "./ViDoRAG/SlideVQA/slidevqa_pdf_document.zip"
    extract_to_slidevqa = "./ViDoRAG/SlideVQA"

    # Ensure directories exist
    os.makedirs(extract_to_pdfs, exist_ok=True)
    os.makedirs(extract_to_slidevqa, exist_ok=True)

    # Download and extract the first file
    if download_file(url_pdfs, output_file_pdfs):
        extract_and_cleanup(output_file_pdfs, extract_to_pdfs)

    # Download and extract the second file
    if download_file(url_slidevqa, output_file_slidevqa):
        extract_and_cleanup(output_file_slidevqa, extract_to_slidevqa)

    def count_files_in_directory(directory):
        """Count the number of files in a directory."""
        return sum([len(files) for _, _, files in os.walk(directory)])

    pdfs_file_count = count_files_in_directory(extract_to_pdfs + "/pdf")
    slidevqa_file_count = count_files_in_directory(extract_to_slidevqa + "/pdf")

    print(f"Number of files in '{extract_to_pdfs}': {pdfs_file_count}")
    print(f"Number of files in '{extract_to_slidevqa}': {slidevqa_file_count}")
