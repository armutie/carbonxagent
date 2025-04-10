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

if "admin" not in st.session_state:
    st.session_state.admin = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm excited to figure out your company operations in detail! What information do you have for me? "}]

def check_admin_password():
    if st.session_state.get("password_input") == "admin123":
        st.session_state.admin = True

if not st.session_state.admin:
    st.sidebar.header("Upload Files")
    
    uploaded_files = st.sidebar.file_uploader(
        "Add files to your knowledge base",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )
    
    st.sidebar.text_input("Enter Admin Password", type="password",
                            key="password_input", on_change=check_admin_password)
    
    password = "admin123"
    if "password_input" in st.session_state and st.session_state.password_input:
        if st.session_state.password_input == password:
            st.sidebar.success("Logged in as admin!")
        else:
            st.sidebar.error("Incorrect password")
else:
    st.sidebar.header(":red[ADMIN PRIVILEGES]")
    # TODO: Some lines do not resize, such as long file names
    try:
        response = requests.get("http://127.0.0.1:8000/list_files", params={"collection_name": "core_db"})
        response.raise_for_status()
        core_files = response.json().get("response_content", [])
        if core_files:
            st.sidebar.subheader("Current Files in Core Knowledge Base")
            for file in core_files:
                st.sidebar.write(f"- {file}")
        else:
            st.sidebar.write("No files in core knowledge base yet.")
    except requests.RequestException as e:
        st.sidebar.error(f"Error fetching files: {str(e)}")

    uploaded_files = st.sidebar.file_uploader(
    "Add files to the CORE knowledge base",
    type=["txt", "pdf"],
    accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_files:
            content_type = "application/pdf" if file.name.lower().endswith(".pdf") else "text/plain"
            files = {"file": (file.name, file.getvalue(), content_type)}
            data = {
                "user_id": st.session_state.user_id,
                "is_core": "true" if st.session_state.admin else "false"
            }
            response = requests.post("http://127.0.0.1:8000/update_vector", files=files, data=data)
            if response.status_code == 200:
                st.session_state.uploaded_files.append(file.name)
                # st.sidebar.write(f"Saved {file.name} for analysis.")
                st.session_state.messages.append({"role": "system", "content": f"{file.name} was added to the knowledge base."})
            else:
                st.sidebar.write(f"Error saving {file.name}: {response.json().get('response_content', 'Unknown error')}")

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