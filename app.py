import streamlit as st
import requests
import time

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

def response_generator(prompt, history):
    try:
        payload = {
            "query": prompt,
            "history": history
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm excited to figure out your company operations in detail! What information do you have for me? "}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Ask about carbon emissions reduction", accept_file=True, file_type=["pdf"]):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query.text})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query.text)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        streamed_response = st.write_stream(response_generator(query.text, st.session_state.messages))
    
    # Add assistant response to chat history (join the streamed list into a string)
    full_response = "".join(streamed_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})