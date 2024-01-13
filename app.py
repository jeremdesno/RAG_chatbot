# This is the file containing the app that is run by Streamlit to generate response from the chatbot.
import streamlit as st
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from scripts.qa_dataset_manager import QADatasetManager
from scripts.rag import rag_manager
from utils import load_config, map_llm_models, map_emb_models, indexes, indexes_map

# Load hugging Face token
config = load_config('./config.yaml')
hugging_face_api_key = config['huggingface']['token_api']

# load nodes
dataset_manager = QADatasetManager()
nodes = dataset_manager.load_nodes('./data/nodes.json')
# create rag manager
inference = InferenceClient(token=hugging_face_api_key)
rag_chain = rag_manager(nodes, inference)


# Set sidebar options
model_llm = st.sidebar.selectbox('Choose LLM model :', options=["zephyr-7b-beta","falcon-7b-instruct","openchat-3.5-1210"])
embedding_model = st.sidebar.selectbox('Choose embedding model :', options=['all-MiniLM-L6-v2','all-mpnet-base-v2'])
index = st.sidebar.selectbox('Choose an index :', options=indexes)
k = st.sidebar.slider('Number of contexts to retrieve :', min_value=1, max_value=20)

# load index and embedding model 
emb_model = SentenceTransformer(map_emb_models[embedding_model])
index = rag_chain.load_index('./data/indexes/' + indexes_map[embedding_model][index])

st.title("Chatbot based on Github's documentation")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle the chat input
if prompt := st.chat_input("Ask a question about Github:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Let me look that up for you...▌")
        print(prompt)
        try: 
            answer,_,_ = rag_chain.augmented_retrieval_generation(queries=[prompt], 
                                                          index=index,
                                                          embedding_model=emb_model,
                                                          llm_model=map_llm_models[model_llm], 
                                                          k=k)
            answer = answer[0]
        except:
            answer = "I couldn't find enough information in our database to answer that question."
        
        # Update the placeholder with the final answer
        message_placeholder.markdown(answer)
        
        # Add the response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
