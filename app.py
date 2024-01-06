# This is the file containing the app that is run by Streamlit to generate response from the chatbot.
import streamlit as st
from chatbot_utils import get_context_for_query, generate_answer_from_context, inference_client, model


st.title("ChatGPT-like clone using our data")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle the chat input
if prompt := st.chat_input("Ask a question about our database:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Let me look that up for you...▌")
        
        # Use our answer module to get the answer
        context = get_context_for_query(prompt)
        if context:
            answer = generate_answer_from_context(context, prompt, inference_client, model)
        else:
            answer = "I couldn't find enough information in our database to answer that question."
        
        # Update the placeholder with the final answer
        message_placeholder.markdown(answer)
        
        # Add the response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
