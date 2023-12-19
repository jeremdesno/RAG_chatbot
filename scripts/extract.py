
import pandas as pd
import yaml
import os
from pathlib import Path
from llama_index import download_loader


# Load hugging Face token
with open('../config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
hugging_face_api_key = config['huggingface']['token_api']


def load_docs(file_path : str)->str:
    doc = ""
    # Initialise loader and load documents from specified documents
    MarkdownReader = download_loader("MarkdownReader")
    loader = MarkdownReader()
    documents = loader.load_data(file=Path(file_path))

    for i in range(len(documents)):
        doc =  doc + documents[i].get_text()
    return doc


def explore_folders(root_dir):
    docs = []
    titles = []
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.md') and filename != 'index.md':
                file_path = os.path.join(foldername, filename)
                doc = load_docs(file_path=file_path)
                docs.append(doc)
                titles.append(filename[:-3])
    return docs, titles

if __name__ == '__main__':

    folders = os.listdir('../content')
    docs = []
    titles = []
    root_directory = os.path.dirname(os.getcwd()) + "\content\\" 
    for folder in folders:
        print(f'Extracting files from folder: {folder}')
        if folder.endswith(".md") != True:
            docs__in_folder, titles_in_folder = explore_folders(root_directory + folder)
            docs = docs + docs__in_folder
            titles = titles + titles_in_folder
    
    df = pd.DataFrame({'content': docs, 'title': titles})
    df.to_csv("../data/documents.csv", index=False)