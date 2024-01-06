import numpy as np
import re
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker

import faiss
from typing import List, Tuple, Dict

import time
from requests.exceptions import HTTPError

GENERATION_PROMPT = """\
Context information:

{context_str}

Given the context information and not prior knowledge, answer the following question. 

{query}

If the context doesn't give enough information to give an answer, 
then say that you weren't given enough information to give an answer.

Answer:
"""


class rag_manager():
    def __init__(self, 
                 nodes : dict[str,str], 
                 client : InferenceClient, 
                 llm_model : str,
                 index : str=None, 
                 prompt : str = GENERATION_PROMPT) -> None:
        self.index = index
        self.client = client
        self.llm_model = llm_model
        self.prompt = prompt
        self.texts = np.array(list(nodes.values()))
        self.ids = np.array(list(nodes.keys()))

    def create_index(self, model : SentenceTransformer) -> Tuple[faiss.Index, np.array]:
        embedding = model.encode(self.texts)
        print("Embeddings created")
        # dimension
        dimension = embedding.shape[1]
        
        # create the vector/embeddings and their IDs                                                                                                                                                                                                                                                embedding vectors and ids:
        db_vectors = embedding.copy().astype(np.float32)
        db_ids = list(range(len(db_vectors)))

        faiss.normalize_L2(db_vectors)  
        index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap(index)
        index.add_with_ids(db_vectors, db_ids)
    
        return index, embedding

    def save_index(self, 
                   index : faiss.Index, 
                   saving_path : str) -> None:
        faiss.write_index(index, saving_path)
        print(f"Successfuly saved index to {saving_path}")

    def load_index(self,
                   loading_path : str, 
                   verbose : bool=True) ->  faiss.Index:
        index = faiss.read_index(loading_path)
        if verbose == True:
            print(f"Successfuly loaded index from {loading_path}")
        return index

    def search(self,
               index : faiss.Index, 
               model : SentenceTransformer, 
               queries : List[str], 
               k : int=5) -> Tuple[Dict[str, int | float], float]:
    
        t=time.time()
        query_vector = model.encode(queries).astype(np.float32)
        faiss.normalize_L2(query_vector)
        
        similarities, similarities_ids = index.search(query_vector, k)
        search_time = round(time.time()-t,2)
        #print('total search time: {}\n'.format(search_time))
        #print('average search time per query: {}\n'.format((time.time()-t)/len(queries)))
        
        similarities = np.clip(similarities, 0, 1)
        search_results = {'indexes': similarities_ids,
                          'scores': similarities}
        
        return search_results, search_time

    def reranking(self,
                  query : str, 
                  results: Dict[str, str], 
                  reranker : str = 'BAAI/bge-reranker-base', 
                  k_final : str = 3) -> Dict[str, str]:
        reranker = FlagReranker(reranker, use_fp16=True) 
        pairs = [[result['text'], query] for result in results]
        scores = reranker.compute_score(pairs)
        for i, score in enumerate(scores):
            results[i]['rerank_score'] = score
        return sorted(results, key=lambda x: x['rerank_score'], reverse=True)[:k_final]
    
    def retrieve_context(self, 
                         index : faiss.Index, 
                         embedding_model : SentenceTransformer, 
                         queries : List[str], 
                         k : int=5) -> Tuple[List[str], List[str], float]:
        search_results, search_time = self.search(index=index, model=embedding_model, queries=queries, k=k)
        contexts = self.texts[search_results['indexes']]
        contexts_ids = self.ids[search_results['indexes']]

        return contexts.tolist(), contexts_ids.tolist(), search_time
        
    def generate_answer(self,
                        queries :  List[str],
                        contexts : List[str],
                        max_new_tokens : int=500) -> List[str]:
        
        answers = []
        for idx, query in enumerate(queries):
            print(f"Answering query: {query}")
            query_prompt = self.prompt.format(context_str=' Next Context: '.join(contexts[idx]), query= query)
            retries = 3
            for attempt in range(retries):
                try:
                    answer = self.client.text_generation(prompt=query_prompt, model=self.llm_model, max_new_tokens=max_new_tokens)
                    answer_parsed = re.sub(r"^(-{1,})","",re.sub(r"^\s*\d+\.\s*|\n", " ",answer))
                    answer_parsed = answer_parsed.strip()
                    if answer_parsed:
                        answers.append(answer_parsed)
                    else:
                        answer.append('Could not answer this query.')
                    break
                
                except HTTPError as e:
                    print(f"Error encountered: {e}")
                    if e.response.status_code == 429:
                        print("Too Many Requests. Retrying after 30 minutes...")
                        if attempt < retries - 1:
                            time.sleep(1800) 
                        else:
                            print("Maximum retries reached. Could not answer this query.")
                    else:
                        if attempt < retries - 1:
                            print(f"Retrying after 30 secondes ({attempt + 1}/{retries})...")
                            time.sleep(30)  
                        else:
                            print("Maximum retries reached. Could not answer this query.") 
                except Exception as e:
                    print(f"Error encountered: {e}")
                    if attempt < retries - 1:
                        print(f"Retrying after 30 secondes ({attempt + 1}/{retries})...")
                        time.sleep(30)  
                    else:
                        print("Maximum retries reached. Could not answer this query.")
        return answers
    
    def augmented_retrieval_generation(self,
                                       queries : List[str],
                                       index : faiss.Index,
                                       embedding_model : SentenceTransformer, 
                                       k : int = 5
                                       ) -> Tuple[List[str], List[str], List[str]]:
        
        contexts, contexts_ids, _ = self.retrieve_context(index=index, embedding_model=embedding_model, queries=queries)
        answers= self.generate_answer(queries=queries, contexts=contexts)
        return answers, contexts, contexts_ids
    
    def calculate_cross_encoder_similarity(self,
                                           model: CrossEncoder, 
                                           answers: List[str],
                                           ground_truths: List[str]
                                           ) -> Tuple[List[float], List[float]]:
        pairs = [(answer, ground_truths[index]) for index,answer in enumerate(answers)]
        scores = model.predict(pairs)
        scores = list(map(lambda x: x if x >= 0 else 0, scores))
        return np.mean(scores)
    
    def calculate_mrr(self, 
                      contexts_ids: List[List[str]], 
                      relevant_docs: List[str]) -> float:
        
        reciprocal_ranks = np.zeros(len(contexts_ids))
        for i,query_contexts in enumerate(contexts_ids):
            try:
                rank = query_contexts.index(relevant_docs[i]) + 1  # Get the rank of the relevant document
                reciprocal_ranks[i] = 1/rank
            except ValueError:
                reciprocal_ranks[i] = 0

        mrr = np.mean(reciprocal_ranks)
        return mrr

    def calculate_hit_rate(self, 
                           contexts_ids: List[List[str]], 
                           relevant_docs: List[str]) -> float:
        hit_count = 0
        total_queries = len(contexts_ids)
        for query_contexts, relevant_doc in zip(contexts_ids, relevant_docs):
            for context in query_contexts:
                if context in relevant_doc:
                    hit_count += 1  

        hit_rate = hit_count / total_queries if total_queries else 0
        return hit_rate
    
    
    def evaluate_rag(self,
                     metrics : List[str], 
                     data : dict[str, str]) -> Dict[str, float]:
        scores = {}
        for metric in metrics:
            if metric == 'mrr':
                mrr = self.calculate_mrr(contexts_ids=data['contexts_ids'], relevant_docs=data['relevant_docs'])
                scores['mrr'] = mrr
            elif metric == 'hit':
                hit_score = self.calculate_hit_rate(contexts_ids = data['contexts_ids'], relevant_docs= data['relevant_docs'])
                scores['hit_score'] = hit_score
            elif metric == 'semantic':
                crossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
                similarity_score = self.calculate_cross_encoder_similarity(model=crossEncoder, answers= data['answers'], ground_truths= data['ground_truths'])
                scores['semantic_similarity_avg'] = similarity_score
        return scores