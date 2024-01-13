import yaml 

def load_config(file):
    with open(file, 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_file.close()
    return config

map_llm_models = {"zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
              "falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
              "openchat-3.5-1210": "openchat/openchat-3.5-1210"}

map_emb_models = {'all-MiniLM-L6-v2' : 'sentence-transformers/all-MiniLM-L6-v2',
                  'all-mpnet-base-v2' : 'sentence-transformers/all-mpnet-base-v2'}

indexes = ['Flat IP', 'IVF Flat 50', 'IVF Flat 100', 'IVF Flat 200', 'IVFPQ 8', 'IVFPQ 32', 'IVFPQ 128']

indexes_map ={'all-MiniLM-L6-v2': {'Flat IP': 'L6_V2_index_flatIP', 
                                    'IVF Flat 50': 'L6_V2_index_IVFFlat_50', 
                                    'IVF Flat 100': 'L6_V2_index_IVFFlat_100', 
                                    'IVF Flat 200': 'L6_V2_index_IVFFlat_200', 
                                    'IVFPQ 8': 'L6_V2_index_IVFPQ_8', 
                                    'IVFPQ 32': 'L6_V2_index_IVFPQ_32', 
                                    'IVFPQ 128': 'L6_V2_index_IVFPQ_128'},
               'all-mpnet-base-v2': {'Flat IP': 'mpnet_index_flatIP', 
                                    'IVF Flat 50': 'mpnet_index_IVFFlat_50', 
                                    'IVF Flat 100': 'mpnet_index_IVFFlat_100', 
                                    'IVF Flat 200': 'mpnet_index_IVFFlat_200', 
                                    'IVFPQ 8': 'mpnet_index_IVFPQ_8', 
                                    'IVFPQ 32': 'mpnet_index_IVFPQ_32', 
                                    'IVFPQ 128': 'mpnet_index_IVFPQ_128'}
               }