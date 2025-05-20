import json
import requests

# URL of the file to download
url = "https://huggingface.co/datasets/autumncc/ViDoSeek/resolve/main/vidoseek.json?download=true"

# Path to save the downloaded file
output_file = "vidoseek.json"

# Download the file
response = requests.get(url)
if response.status_code == 200:
    with open(output_file, "wb") as file:
        file.write(response.content)
    print(f"File downloaded successfully and saved as '{output_file}'.")
else:
    print(f"Failed to download file. HTTP status code: {response.status_code}")


with open("vidoseek.json", "r") as file:
    data = json.load(file)
data = data["examples"]

query = data[0]["query"]
GT_answer = data[0]["reference_answer"]
