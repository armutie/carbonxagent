import streamlit as st
import requests
import time

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
    
    if "Here’s your company’s carbon footprint breakdown:" in answer:
        for line in answer.split("\n"):
            yield line + "\n"
            time.sleep(0.1)
    else:
        for word in answer.split():
            yield word + " "
            time.sleep(0.05)

st.title("CarbonXAgent")

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "user_id" not in st.session_state:
    st.session_state.user_id = "789"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm excited to figure out your company operations in detail! What information do you have for me? "}]

st.sidebar.markdown(
    f"<h3 style='margin-top: -30px;'>Your User ID: {st.session_state.user_id}</h3>",
    unsafe_allow_html=True
)

user_id = st.sidebar.header("Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Add files to your knowledge base",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_files:
            file_text = file.read().decode("utf-8") 
            files = {"file": (file.name, file_text.encode(), "text/plain")}
            data = {"user_id": st.session_state.user_id}
            response = requests.post("http://127.0.0.1:8000/update_vector", files=files, data=data)
            if response.status_code == 200:
                st.session_state.uploaded_files.append(file.name)
                # st.sidebar.write(f"Saved {file.name} for analysis.")
                st.session_state.messages.append({"role": "system", "content": f"{file.name} was added to the knowledge base."})
            else:
                st.sidebar.write(f"Error saving {file.name}: {response.json().get('response_content', 'Unknown error')}")

# st.sidebar.markdown(
#     f"<small>Your User ID: {st.session_state.user_id}</small>",
#     unsafe_allow_html=True
# )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = "⚙️" if message["role"] == "system" else message["role"]  # Default for user/assistant
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Ask about carbon emissions reduction"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        streamed_response = st.write_stream(response_generator(query, st.session_state.messages))
    
    # Add assistant response to chat history (join the streamed list into a string)
    full_response = "".join(streamed_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})