import streamlit as st
import requests
import time

def response_generator(prompt, history):
    try:
        if "user_id" not in st.session_state or not st.session_state.user_id:
            yield "Error: Please log in again."
            return
        payload = {
            "query": prompt,
            "history": history,
            "user_id": st.session_state.user_id,
        }
        placeholder = st.empty()
        with placeholder:
            with st.spinner("", show_time=True):
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                response = requests.post("http://127.0.0.1:8000/chat", json=payload, headers=headers)
                if response.status_code != 200:
                    yield f"Error: {response.json().get('detail', 'Unknown error')}"
                    return
                answer = response.json().get("response_content", "No response received")

            if "Here’s your company’s carbon footprint breakdown:" in answer:
                for line in answer.split("\n"):
                    yield line + "\n"
                    time.sleep(0.1)
            else:
                for word in answer.split():
                    yield word + " "
                    time.sleep(0.05)
    except requests.RequestException as e:
        answer = f"Error: {str(e)}"

st.title("CarbonXAgent")

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm excited to figure out your company operations in detail! What information do you have for me?"}
    ]
if "admin" not in st.session_state:
    st.session_state.admin = False


def check_admin_password():
    password = st.session_state.get("password_input")
    if password:
        try:
            response = requests.post("http://127.0.0.1:8000/authenticate", data={"password": password})
            if response.status_code == 200 and response.json().get("admin"):
                st.session_state.admin = True
                st.sidebar.success("Logged in as admin!")
            else:
                st.session_state.admin = False
                st.sidebar.error("Incorrect password")
        except requests.RequestException:
            st.sidebar.error("Error connecting to authentication service")


if "access_token" not in st.session_state:
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        signup_button = st.form_submit_button("Sign Up")
        if login_button:
            try:
                response = requests.post("http://127.0.0.1:8000/login", data={"email": email, "password": password})
                if response.status_code == 200:
                    st.session_state.access_token = response.json()["access_token"]
                    st.session_state.user_id = response.json()["user_id"]
                    # Fetch history
                    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                    history_response = requests.get("http://127.0.0.1:8000/history", headers=headers)
                    if history_response.status_code == 200:
                        st.session_state.messages = history_response.json()["response_content"]
                    else:
                        st.session_state.messages = [
                            {"role": "assistant", "content": "Hi! I'm excited to figure out your company operations in detail! What information do you have for me?"}
                        ]
                    st.rerun()
                else:
                    st.error("Login failed: Invalid email or password")
            except requests.RequestException as e:
                st.error(f"Login error: {str(e)}")
        if signup_button:
            try:
                response = requests.post("http://127.0.0.1:8000/signup", data={"email": email, "password": password})
                if response.status_code == 200:
                    st.success("Signed up! Please log in.")
                else:
                    st.error("Signup failed: Email may already exist")
            except requests.RequestException as e:
                st.error(f"Signup error: {str(e)}")
else:
    if not st.session_state.admin:
        st.sidebar.header("Upload Files")
        uploaded_files = st.sidebar.file_uploader(
            "Add files to your knowledge base",
            type=["txt", "pdf"],
            accept_multiple_files=True
        )
        st.sidebar.text_input("Enter Admin Password", type="password",
                              key="password_input", on_change=check_admin_password)
    else:
        st.sidebar.header(":red[ADMIN PRIVILEGES]")
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
            accept_multiple_files=True
        )

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
                    st.session_state.messages.append({"role": "system", "content": f"{file.name} was added to the knowledge base."})
                else:
                    st.sidebar.write(f"Error saving {file.name}: {response.json().get('response_content', 'Unknown error')}")

    with st.container():
        for message in st.session_state.messages:
            avatar = "⚙️" if message["role"] == "system" else message["role"]
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    if query := st.chat_input("Ask about carbon emissions reduction"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.container():
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                streamed_response = st.write_stream(response_generator(query, st.session_state.messages))
                full_response = "".join(streamed_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})