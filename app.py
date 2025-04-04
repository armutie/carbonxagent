import streamlit as st
import requests
import time

# Streamed response emulator
def response_generator(prompt):
    try:
        response = requests.post("http://127.0.0.1:8000/chat", json={"query": prompt})
        response.raise_for_status()  # Raises exception for non-200 codes
        answer = response.json()["response_content"]
    except requests.RequestException as e:
        answer = f"Error: {str(e)}"  # Catches 422, 500, or network issues
    
    for word in answer.split():
        yield word + " "
        time.sleep(0.05)

st.title("CarbonXAgent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Ask about carbon emissions reduction:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        streamed_response = st.write_stream(response_generator(query))
    
    # Add assistant response to chat history (join the streamed list into a string)
    full_response = "".join(streamed_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})