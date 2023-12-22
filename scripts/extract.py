import pandas as pd
import os
from pathlib import Path
from llama_index import download_loader


class DocumentProcessor:
    def __init__(self, root_dir='../content', output_path='../data/documents.csv'):
        self.root_dir = root_dir
        self.output_path = output_path

    def load_docs(self, file_path):
        doc = ""
        MarkdownReader = download_loader("MarkdownReader")
        loader = MarkdownReader()
        documents = loader.load_data(file=Path(file_path))

        for i in range(len(documents)):
            doc = doc + documents[i].get_text()
        return doc

    def explore_folders(self, root_dir):
        docs = []
        titles = []
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.md') and filename != 'index.md':
                    file_path = os.path.join(foldername, filename)
                    doc = self.load_docs(file_path=file_path)
                    docs.append(doc)
                    titles.append(filename[:-3])
        return docs, titles

    def process_documents(self):
        folders = os.listdir(self.root_dir)
        docs = []
        titles = []
        for folder in folders:
            print(f'Extracting files from folder: {folder}')
            if not folder.endswith(".md"):
                docs_in_folder, titles_in_folder = self.explore_folders(os.path.join(self.root_dir, folder))
                docs = docs + docs_in_folder
                titles = titles + titles_in_folder

        df = pd.DataFrame({'content': docs, 'title': titles})
        df.to_csv(self.output_path, index=False)
