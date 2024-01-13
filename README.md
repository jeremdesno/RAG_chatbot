# RAG_chatbot
Application using retrieval augmented generation (RAG) for Q&amp;A based on a knowledge base which was created by extracting the markdown files from the official github documentation content.
https://github.com/github/docs

The aim is to to answer queries by providing context to an LLM obtained using semantique search to retreive the most relevant documents in our knowledge base using vector similarity and FAISS indexes that are built for fast search capabilities.

We've built a simple web interface using streamlit which offers the functionnality of trying different open source llms and embedding models from HuggingFace. 

## How to install and launch the streamlit app?
### Creating virtual environnement
python -m venv env_name
(python version used 3.10.11)
### Activating the virtual environnement 
env_name\Scripts\activate
### Installing the requirements
pip install -r requirements.txt
### Launching streamlit app
streamlit run app.py

NB: For the app to work you need to add to the root folder a config.yml file that contains your HuggingFace token in the following format: 
huggingface:
  token_api: "key"
You can go to this link https://huggingface.co/ to create an account and get a token key.

## How to navigate this repo ?
This repo contains the following folders:
- scripts : Contains the extract, qa_dataset_manager & rag scripts which contains the classes responsible of:
    - extracting the raw data (this doesn't have to be re-done)
    - Creating QA Dataset. The process of creating the questions and answers has been seperated into 2 differet worklows and because of how long text generation can take through the HuggingFace inference endpoint, we've built a loggic that allows to create questions & answers and store them in intermediate files before integrating in the main QA dataset file.
    - Creating a rag pipeline in a single step. Aswell as evaluating it through different metrics (MRR, Hit score and a semantic similarity score)
    Finally this folder contains the rag_evaluation notebook we used to create the QA_dataset, and compare the search performance of different FAISS indexes aswell as the text generation performance of different HuggingFace llms. 
- data :  This folder contains the extracted documents in the documents.csv file aswell as the QA dataset, nodes(obtained after chunking the documents) and indexes
- jeremy-dev & toufik-dev : folders to try out ideas. 
- content : The raw content folder downloaded from the github documentation.
