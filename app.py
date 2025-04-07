import streamlit as st
import requests
import time

#     for word in answer.split():
#         yield word + " "
#         time.sleep(0.05)

def response_generator(prompt, history):
    try:
        payload = {
            "query": prompt,
            "history": history,
            "user_id": st.session_state.user_id,
        }
        response = requests.post("http://127.0.0.1:8000/chat", json=payload)
        response.raise_for_status()
        answer = response.json().get("response_content", "No response received")
    except requests.RequestException as e:
        answer = f"Error: {str(e)}"
    
    # Split by newlines to maintain markdown formatting
    for line in answer.split("\n"):
        yield line + "\n"
        time.sleep(0.1)  # You can adjust the delay as needed

st.title("CarbonXAgent")

if "user_id" not in st.session_state:
    st.session_state.user_id = "789"

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
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query.text)

    file_text = ""
    if query.files:
        file = query.files[0]
        files = {"file": (file.name, file.getvalue(), "text/plain")}
        data = {"user_id": st.session_state.user_id}
        response = requests.post("http://127.0.0.1:8000/update_vector", files=files, data=data)
        if response.status_code == 200:
            st.write(f"Saved {file.name} for analysis.")
            st.session_state.messages.append({"role": "system", "content": f"Saved {file.name} for analysis."})
        else:
            st.write(f"Error saving {file.name}: {response.json().get('response_content', 'Unknown error')}")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        streamed_response = st.write_stream(response_generator(query.text, st.session_state.messages))
    
    # Add assistant response to chat history (join the streamed list into a string)
    full_response = "".join(streamed_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})