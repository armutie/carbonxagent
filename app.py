import streamlit as st
import requests
import time
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def response_generator(prompt, history):
    try:
        if "user_id" not in st.session_state or not st.session_state.user_id:
            yield "Error: Please log in again."
            return
        payload = {
            "query": prompt,
            "history": history
        }
        placeholder = st.empty()
        with placeholder:
            with st.spinner("", show_time=True):
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                response = requests.post(f"{BACKEND_URL}/chat", json=payload, headers=headers)
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
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

if "access_token" not in st.session_state:
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        signup_button = st.form_submit_button("Sign Up")
        if login_button:
            try:
                response = requests.post(f"{BACKEND_URL}/login", data={"email": email, "password": password})
                if response.status_code == 200:
                    st.session_state.access_token = response.json()["access_token"]
                    st.session_state.user_id = response.json()["user_id"]
                    st.session_state.user_email = response.json()["user_email"]
                    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}

                    try:
                         role_response = requests.get(f"{BACKEND_URL}/my_role", headers=headers)
                         if role_response.status_code == 200:
                             st.session_state.user_role = role_response.json().get("role", "user") # Default to 'user'
                         else:
                             st.error("Could not determine user role.")
                             st.session_state.user_role = "user" # Fallback
                    except Exception as role_err:
                         st.error(f"Error fetching user role: {role_err}")
                         st.session_state.user_role = "user" # Fallback

                    # --- Fetch history --- (haven't wrapped it in a try block)
                    history_response = requests.get(f"{BACKEND_URL}/history", headers=headers)
                    if history_response.status_code == 200:
                        st.session_state.messages = history_response.json()["response_content"]
                        if not st.session_state.messages: # Handle empty history
                                    st.session_state.messages = [{"role": "assistant", "content": "Welcome! How can I help?"}]
                    else:
                        st.error("Could not load chat history.")
                        st.session_state.messages = [{"role": "assistant", "content": "Hi! Could not load history."}]

                    st.rerun()
                elif response.status_code == 403: # Approval failed during login
                    st.error(response.json().get("detail", "Account requires admin approval."))
                elif response.status_code == 401: # Invalid credentials
                    st.error(response.json().get("detail", "Invalid email or password."))
                else:
                    st.error("Login failed: Cannot access database")
            except requests.RequestException as e:
                st.error(f"Login error: {str(e)}")
        if signup_button:
            try:
                response = requests.post(f"{BACKEND_URL}/signup", data={"email": email, "password": password})
                if response.status_code == 200:
                    st.success("Signed up successfully! Please log in.")
                elif response.status_code == 409: # Check for Conflict
                    st.warning(response.json().get("detail", "Email already registered."))
                else:
                # Handle other errors reported by the backend
                    st.error(f"Signup failed: {response.json().get('detail', 'Unknown error')}")
            except requests.RequestException as e:
                st.error(f"Signup error: {str(e)}")
else:
    access_token = st.session_state.access_token
    headers = {"Authorization": f"Bearer {access_token}"}
    is_user_admin = st.session_state.get('user_role') == 'admin'

    if not is_user_admin:
        st.sidebar.header("Upload Files")
        uploaded_files = st.sidebar.file_uploader(
            "Add files to your knowledge base",
            type=["txt", "pdf"],
            accept_multiple_files=True
        )

        is_core_flag = "false"
    else:
        st.sidebar.header(":red[ADMIN PRIVILEGES]")
        # Removing listing feature for now because it's fairly complicated through Pinecone
        # try:
        #     response = requests.get(f"{BACKEND_URL}/list_files", params={"collection_name": "core_db"}, headers=headers)
        #     response.raise_for_status()
        #     core_files = response.json().get("response_content", [])
        #     if core_files:
        #         st.sidebar.subheader("Current Files in Core Knowledge Base")
        #         for file in core_files:
        #             st.sidebar.write(f"- {file}")
        #     else:
        #         st.sidebar.write("No files in core knowledge base yet.")
        # except requests.RequestException as e:
        #     st.sidebar.error(f"Error fetching files: {str(e)}")

        uploaded_files = st.sidebar.file_uploader(
            "Add files to the CORE knowledge base",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="admin_uploader"
        )

        is_core_flag = "true"

    if uploaded_files:
        if not access_token:
            st.sidebar.error("Authentication error. Please log in again.")
        else:
            for file in uploaded_files:
                if file.file_id not in st.session_state.uploaded_files:
                    content_type = "application/pdf" if file.name.lower().endswith(".pdf") else "text/plain"
                    files = {"file": (file.name, file.getvalue(), content_type)}
                    data = {
                        "is_core": is_core_flag
                    }
                    response = requests.post(f"{BACKEND_URL}/update_vector", files=files, data=data, headers=headers)
                    if response.status_code == 200:
                        st.session_state.uploaded_files.append(file.file_id)
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

    st.sidebar.write(f"Logged in as: {st.session_state.get('user_email', 'Unknown User')}") # Optional: display email

    if st.sidebar.button("Logout"):
        st.session_state.pop('access_token', None)
        st.session_state.pop('user_id', None)
        st.session_state.pop('user_email', None) # Clear email if stored
        st.session_state.pop('messages', None)
        st.session_state.pop('admin', None)
        st.success("Logged out successfully.")
        time.sleep(1) # Brief pause to show message
        st.rerun() 