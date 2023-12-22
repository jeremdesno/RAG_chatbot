# RAG_chatbot
Application using retrieval augmented generation (RAG) for Q&amp;A based on a knowledge base which was created by extracting the markdown files from the official github documentation content.
https://github.com/github/docs

The aim is to to answer queries by providing context to an LLM obtained using semantique search to retreive the most relevant documents in our knowledge base using vector similarity and indexes that are built for fast search capabilities.

## How to install
### Creating virtual environnement
python -m venv env_name
(python version used 3.10.11)
### Activating the virtual environnement 
env_name\Scripts\activate
### Installing the requirements
pip install -r requirements.txt

