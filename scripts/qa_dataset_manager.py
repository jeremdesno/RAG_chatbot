import base64
import json
import random
import numpy as np
import re
import time
import os
from datetime import datetime 
import uuid
from typing import List


QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
EXACTLY {num_questions_per_chunk} questions for an upcoming \
quiz/examination. 
The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided. 
Questions should be understandable without having access to the context.
Don't ask for examples and keep the questions focused on once aspect at a time.
questions:
"""

class QADatasetManager:
    def __init__(self):
        self.temp_path = '../data/qa_dataset_intermed.json'
        self.qa_path = '../data/qa_dataset.json'  
        self.metadata_path = '../data/metadata.json'
        self.qa_intermed = None

        files = os.listdir('../data')
        if 'metadata.json' in files:
            try :
                with open(self.metadata_path, 'r') as json_file:
                    self.metadata = json.load(json_file)
                    json_file.close()
            except : 
                print(f"Could not load metadata from {self.metadata_path}")
        else :
            # Create metadata file
            with open(self.metadata_path, 'w') as json_file:
                    metadata = {'max_index': -1,
                                'last_updated': datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                                'creation_date': datetime.utcnow().strftime("%Y-%m-%d %H:%M")}
                    self.metadata = metadata
                    json.dump(metadata, json_file)
                    print('Initialised metadata file')
                    json_file.close()

    @staticmethod
    def encode_chunk_idx(chunk_num, idx):
        # Encode chunk_num and idx into a dictionary and then base64 encode it
        encoded_id = base64.b64encode(f"chunk_{chunk_num}_index_{idx}".encode()).decode()
        return encoded_id

    @staticmethod
    def decode_chunk_idx(encoded_id):
        # Decode the base64 string and extract chunk_num and idx
        decoded_id = base64.b64decode(encoded_id).decode()
        parts = decoded_id.split('_')
        chunk_num = int(parts[1])
        idx = int(parts[3])
        return idx, chunk_num

    def create_qa_pairs(self, 
                        texts: List[str], 
                        client,
                        model, 
                        qa_generate_prompt_tmpl=QA_GENERATE_PROMPT_TMPL,
                        max_new_tokens=100, 
                        num_questions_per_chunk=2, 
                        chunk_size=1024):
        
        random.seed(12345)
        indexes = list(range(len(texts)))
        random.shuffle(indexes)
        map = {shuffled_idx: original_idx for shuffled_idx, original_idx in enumerate(indexes)}
        shuffled_text = np.array(texts)[indexes]

        self.qa_intermed = {
                'queries': {},
                'corpus': {},
                'relevant_docs': {}
            }

        node_dict = {}
        queries = {}
        relevant_docs = {}
        max_index = self.metadata['max_index']

        for idx, text in enumerate(shuffled_text[max_index+1:]):
            idx_df = map[idx + max_index + 1]
            num_chunks = (len(text) + chunk_size - 1) // chunk_size
            print(f"Number of chunks in text {idx_df}: {num_chunks}")
            for chunk_num in range(num_chunks):
                start_idx = chunk_num * chunk_size
                end_idx = min((chunk_num + 1) * chunk_size, len(text))
                chunk_text = text[start_idx:end_idx]

                node_id = self.encode_chunk_idx(chunk_num, idx_df)
                node_dict[node_id] = chunk_text

                query = qa_generate_prompt_tmpl.format(
                    context_str=chunk_text, num_questions_per_chunk=num_questions_per_chunk
                )

                retries = 3
                for attempt in range(retries):
                    try:
                        response = client.text_generation(prompt=query, model=model, max_new_tokens=max_new_tokens)
                        response = response.split("?\n")
                        questions = [
                            re.sub(r"^(-{1,})","",re.sub(r"^\s*\d+\.\s*|\n", "",question)) for question in response
                        ]
                        questions = [question for question in questions if len(question) > 10]
                        for question in list(set(questions)):
                            question_id = str(uuid.uuid4())
                            queries[question_id] = question
                            relevant_docs[question_id] = [node_id]
                        break
                    except Exception as e:
                        print(f"Error encountered: {e}")
                        if attempt < retries - 1:
                            print(f"Retrying after 2 minutes ({attempt+1}/{retries})...")
                            time.sleep(90)
                        else:
                            print("Maximum retries reached. Skipping this text chunk.")

            if idx_df % 5 == 0:
                print(f"Updating QA dataset")
                self.qa_intermed['queries'].update(queries)
                self.qa_intermed['corpus'].update(node_dict)
                self.qa_intermed['relevant_docs'].update(relevant_docs)

                # Save questions to tempory file
                with open(self.temp_path, 'w') as json_file:
                    json.dump(self.qa_intermed, json_file)
                    print(f"Saved questions to --> {self.temp_path}")
                    json_file.close()

                # Update metadata file
                with open(self.metadata_path, 'w') as json_file:
                    self.metadata.update({'max_index': idx_df,
                                      'last_updated': datetime.utcnow().strftime("%Y-%m-%d %H:%M")})
                    json.dump(self.metadata, json_file)
                    print('Updated metadata')
                    json_file.close()

                # Re-initialise dicts
                node_dict = {}
                queries = {}
                relevant_docs = {}
                time.sleep(30)

    def add_qa_to_dataset(self):
        # Load intermediate file if not loaded
        if self.qa_intermed is None:
            with open(self.temp_path, 'r') as temp_file:
                    self.qa_intermed = json.load(temp_file)
                    temp_file.close()

        # Load QA dataset and perform the update 
        with open(self.qa_path, 'r') as qa_file:
            try:
                with open(self.qa_path, 'r') as qa_file:
                    file_content = qa_file.read()
                    if not file_content:  
                        qa_dataset = {
                            'queries': {},
                            'corpus': {},
                            'relevant_docs': {}
                        }
                    else:
                        qa_dataset = json.loads(file_content)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading or decoding JSON: {e}")

            nb_before = len(qa_dataset['queries'])
            qa_dataset['queries'].update(self.qa_intermed['queries'])
            qa_dataset['corpus'].update(self.qa_intermed['corpus'])
            qa_dataset['relevant_docs'].update(self.qa_intermed['relevant_docs'])
            nb_after = len(qa_dataset['queries'])
            qa_file.close()

        if nb_after >= nb_before:
            # Save the updates made to the main QA dataset
            with open(self.qa_path, 'w') as qa_file:
                json.dump(qa_dataset, qa_file)
                print(f"Added questions from {self.temp_path} to --> {self.qa_path}")
                qa_file.close()

        # Delete temporary file