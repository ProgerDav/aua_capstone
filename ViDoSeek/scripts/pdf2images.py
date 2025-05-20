import os
from tqdm import tqdm
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor

datasets = ["ViDoSeek", "SlideVQA"]


def process_pdf(args):
    dataset, filename = args
    root_path = f"./ViDoRAG/data/{dataset}"
    pdf_path = os.path.join(root_path, "pdf")
    filepath = os.path.join(pdf_path, filename)
    imgname = filename.split(".pdf")[0]
    images = convert_from_path(filepath)
    for i, image in enumerate(images):
        idx = i + 1
        img_dir = os.path.join(root_path, "img")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        image.save(os.path.join(root_path, "img", f"{imgname}_{idx}.jpg"), "JPEG")


tasks = []
for dataset in datasets:
    root_path = f"./ViDoRAG/data/{dataset}"
    pdf_path = os.path.join(root_path, "pdf")
    pdf_files = [file for file in os.listdir(pdf_path) if file.endswith("pdf")]
    tasks.extend([(dataset, filename) for filename in pdf_files])

with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_pdf, tasks), total=len(tasks)))
