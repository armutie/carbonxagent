import streamlit as st
import requests
import time
from initiatives.rag import update_vector_store
import os
from datetime import datetime

# Streamed response emulator
# def response_generator(prompt, history):
#     try:
#         payload = {
#             "query": prompt,
#             "history": history  # List of {"role": "user/assistant", "content": "text"}
#         }
#         response = requests.post("http://127.0.0.1:8000/chat", json=payload)
#         response.raise_for_status()  # Raises exception for non-200 codes
#         answer = response.json()["response_content"]
#     except requests.RequestException as e:
#         answer = f"Error: {str(e)}"  # Catches 422, 500, or network issues
    
#     for word in answer.split():
#         yield word + " "
#         time.sleep(0.05)

def response_generator(prompt, history, file_content):
    try:
        payload = {
            "query": prompt,
            "history": history,
            "user_id": st.session_state.user_id,
            "file_text": file_content
        }
        response = requests.post("http://127.0.0.1:8000/chat", json=payload)
        response.raise_for_status()
        answer = response.json()["response_content"]
    except requests.RequestException as e:
        answer = f"Error: {str(e)}"
    
    # Split by newlines to maintain markdown formatting
    for line in answer.split("\n"):
        yield line + "\n"
        time.sleep(0.1)  # You can adjust the delay as needed

st.title("CarbonXAgent")

if "user_id" not in st.session_state:
    st.session_state.user_id = "123"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm excited to figure out your company operations in detail! What information do you have for me? "}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Ask about carbon emissions reduction", accept_file=True, file_type=["txt"]):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query.text})
    file_text = ""
    if query.files:
        os.makedirs("uploads", exist_ok=True)
        file = query.files[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uploads/user_{st.session_state.user_id}_{timestamp}.txt"
        with open(filename, "wb") as f:
            f.write(file.getvalue())
        file_text = file.getvalue().decode("utf-8")
        update_vector_store(st.session_state.user_id, file_text)  # Build index here
        st.write(f"Saved {file.name} for analysis.")
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query.text)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        streamed_response = st.write_stream(response_generator(query.text, st.session_state.messages, file_text))
    
    # Add assistant response to chat history (join the streamed list into a string)
    full_response = "".join(streamed_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})