# This file contains the functions that can be used to generate answers for a given query

import os
import pandas as pd
import numpy as np
import faiss
import base64
import yaml
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from huggingface_hub import InferenceClient
import re
import time
from requests.exceptions import HTTPError

model ="HuggingFaceH4/zephyr-7b-beta"
with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
hugging_face_api_key = config['huggingface']['token_api']

GENERATION_PROMPT = """\
Context information:

{context_str}

Given the context information and not prior knowledge, answer the following question. 

{query}

If the context doesn't give enough information to give an answer, 
then say that you weren't given enough information to give an answer.

Answer:
"""

# Assuming all necessary models are already downloaded and accessible
# Initialize models
model_L6_v2 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
inference_client = InferenceClient(token=hugging_face_api_key)

# Load the index and DataFrame outside of function definitions if they are constant and don't need to be reloaded each time
index_path = 'data/index_flatIP'  # Update this to the correct path
indexIVFFlat = faiss.read_index(index_path)
df = pd.read_csv("data/documents.csv")  # Update this to the correct path

def query_search(index, model, query, k=10):
    query_vector = model.encode([query]).astype(np.float32)
    faiss.normalize_L2(query_vector)
    D, I = index.search(query_vector, k)
    return D, I

def rerank_and_sort(search_results, reranker, query):
    D, I = search_results
    reranked_results = []
    for i, idx in enumerate(I[0]):
        doc = df.iloc[idx]['content']
        reranked_results.append((idx, D[0][i], doc))
    
    # Here, we will compute the rerank scores for each document-query pair
    rerank_scores = reranker.compute_score([[text, query] for _, _, text in reranked_results])

    # Now, we combine the original scores with the rerank scores
    reranked_results_with_scores = [
        (idx, original_score, rerank_score, text)
        for (idx, original_score, text), rerank_score in zip(reranked_results, rerank_scores)
    ]

    # Sort by rerank score, not the original score
    reranked_results_with_scores.sort(key=lambda x: x[2], reverse=True)  # Sort by rerank score
    return reranked_results_with_scores


def get_response(reranked_results, top_k=5):
    responses = []
    for i in range(top_k):
        idx, score, rerank_score, text = reranked_results[i]
        response = {
            'id': idx,
            'text': text,
            'similarity_score': score,
            'rerank_score': rerank_score
        }
        responses.append(response)
    return responses

# Use these functions in your Streamlit app or any other application
query = "What are codespaces?"
search_results = query_search(indexIVFFlat, model_L6_v2, query)
reranked_results = rerank_and_sort(search_results, reranker, query)
responses = get_response(reranked_results)

# The responses can now be formatted and displayed as needed
"""for response in responses:
    print(f"Document ID: {response['id']}")
    print(f"Document Text: {response['text']}")
    print(f"Similarity Score: {response['similarity_score']}")
    print(f"Rerank Score: {response['rerank_score']}\n")"""

def get_context_for_query(query):
    search_results = query_search(indexIVFFlat, model_L6_v2, query)
    reranked_results = rerank_and_sort(search_results, reranker, query)
    responses = get_response(reranked_results)

    if responses:
        # Sort the responses by rerank score to get the highest scoring document
        top_response = sorted(responses, key=lambda x: x['rerank_score'], reverse=True)[0]
        context = top_response['text']
        return context
    else:
        print("No responses found for the query.")
        return None

def generate_answer_from_context(context, query, client, model, max_new_tokens=500, prompt_template=GENERATION_PROMPT):
    retries = 3
    for attempt in range(retries):
        try:
            prompt = prompt_template.format(context_str=context, query=query)
            answer = client.text_generation(prompt=prompt, model=model, max_new_tokens=max_new_tokens)
            answer_parsed = re.sub(r"^(-{1,})","", re.sub(r"^\s*\d+\.\s*|\n", " ", answer))
            answer_parsed = answer_parsed.strip()
            return answer_parsed
            
        except HTTPError as e:
            print(f"Error encountered: {e}")
            if e.response.status_code == 429:
                print("Too Many Requests. Retrying after 5 minutes...")
                if attempt < retries - 1:
                    time.sleep(300)  # Sleep for 5 minutes before retrying
                else:
                    print("Maximum retries reached. Skipping this query.")
                    return None
            else:
                if attempt < retries - 1:
                    print(f"Retrying after 60 seconds ({attempt + 1}/{retries})...")
                    time.sleep(60)  # Sleep for 1 minute before retrying
                else:
                    print("Maximum retries reached. Skipping this query.")
                    return None
        except Exception as e:
            print(f"Error encountered: {e}")
            if attempt < retries - 1:
                print(f"Retrying after 60 seconds ({attempt + 1}/{retries})...")
                time.sleep(60)  # Sleep for 1 minute before retrying
            else:
                print("Maximum retries reached. Skipping this query.")
                return None

# Example usage:
# Assuming you have a function to get context for a given query
context = get_context_for_query(query)
answer = generate_answer_from_context(context, query, inference_client, model)
if answer:
    print(f"Answer: {answer}")
else:
    print("Failed to generate an answer.")