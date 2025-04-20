# CarbonXAgent

CarbonXAgent is an AI-powered chatbot designed to help companies understand and calculate their carbon footprint. It engages users in a conversation to gather details about their operations, leverages Retrieval-Augmented Generation (RAG) with both core and user-specific knowledge bases, and utilizes an agentic workflow (CrewAI) for detailed analysis, emission calculations, and reduction suggestions.

## Features

*   **Conversational Data Gathering:** Engages users naturally to collect operational details and emission sources.
*   **User Authentication & Approval:** Secure signup/login powered by Supabase Auth, with an admin approval workflow for new users.
*   **Role-Based Access:** Differentiates between regular users and admins.
*   **Knowledge Base Management:**
    *   Admins can upload documents (PDF, TXT) to a shared "core" knowledge base.
    *   Regular users can upload documents to their private, user-specific knowledge base.
*   **Retrieval-Augmented Generation (RAG):** Uses LangChain and Pinecone to retrieve relevant context from both core and user knowledge bases to inform responses.
*   **Agentic Workflow (CrewAI):** Triggers a multi-agent system (Operations Analyst, Emissions Expert, Sustainability Advisor) upon sufficient data collection ("FINAL DESCRIPTION:") to:
    *   Parse operational summaries.
    *   Calculate emissions based on provided data and context (potentially using tools for lookups/calculations).
    *   Suggest tailored emission reduction initiatives with trackable metrics.
*   **Persistent Chat History:** Stores conversation history per user in Supabase.
*   **File Upload Tracking:** Records uploaded file metadata (filename, uploader, namespace) in Supabase.

## Architecture Overview

The application consists of several key components:

1.  **Frontend (Streamlit - `app.py`):** Provides the user interface for login/signup, chat interactions, and file uploads. Manages session state and communicates with the backend API.
2.  **Backend (FastAPI - `main.py`):** Exposes API endpoints for:
    *   Authentication (`/login`, `/signup`) via Supabase.
    *   User role checking (`/my_role`) and history retrieval (`/history`).
    *   Handling chat requests (`/chat`), incorporating LangChain RAG with history awareness.
    *   Triggering the CrewAI workflow based on chat content.
    *   Handling file uploads (`/update_vector`), processing files, generating embeddings, adding data to Pinecone, and tracking uploads in Supabase.
    *   Listing uploaded files for admins (`/list_files`).
3.  **RAG & Vector Store (`rag.py`):**
    *   Uses `langchain-openai` with `text-embedding-3-small` for embeddings.
    *   Uses `langchain-pinecone` to interact with a Pinecone serverless index.
    *   Implements Pinecone namespaces (`core_db`, `user_{user_id}`) to separate core and user knowledge.
4.  **Agentic Workflow (`initiatives/` directory):**
    *   **CrewAI:** Orchestrates the agent workflow.
    *   **Agents (`agents.py`):** Defines specialized agents (Operations Analyst, Emissions Expert, Sustainability Advisor) potentially using different LLMs (Groq, OpenAI).
    *   **Tasks (`tasks.py`):** Defines the specific goals and instructions for each agent.
    *   **Tools (`tools.py`):** Provides custom tools for agents (e.g., `CoreKnowledgeLookupTool`, `CustomCalculatorTool`).
5.  **Databases & Services:**
    *   **Pinecone:** Serverless vector database storing document embeddings for RAG.
    *   **Supabase (PostgreSQL):**
        *   Handles user authentication (`auth.users`).
        *   Stores chat history (`chat_messages` table).
        *   Stores user roles and approval status (`user_roles` table).
        *   Tracks uploaded file metadata (`uploaded_files` table).
    *   **LLMs:** Uses Groq and/or OpenAI via LangChain integrations.

## Technology Stack

*   **Python:** 3.10+
*   **Frontend:** Streamlit
*   **Backend:** FastAPI, Uvicorn/Gunicorn
*   **LLM Orchestration:** LangChain, CrewAI
*   **Vector Database:** Pinecone (Serverless)
*   **Embeddings:** OpenAI Embedding Model (`text-embedding-3-small`)
*   **Database & Auth:** Supabase (PostgreSQL)
*   **LLM APIs:** Groq, OpenAI (Optional)

## Setup and Installation (Local)

### Prerequisites

*   Python 3.10 or later and Pip
*   Git
*   Supabase Account & Project
*   Pinecone Account & Serverless Index

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate.bat

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Use the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in the project root directory and add the following variables with your actual credentials:

```dotenv
# Supabase
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_role_key

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name

# LLMs (Add others if used)
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=sk-your_openai_api_key 
```

### 5. External Services Setup

**a) Supabase:**

*   Create a new project on Supabase.
*   Enable Authentication. Note your Project URL, Anon Key, and Service Role Key.
*   Go to the SQL Editor and run the necessary SQL commands to create tables:
    *   `user_roles` (Columns: `user_id` (UUID, FK to auth.users), `role` (TEXT), `is_approved` (BOOLEAN, default false)). Remember to set up RLS policies allowing admins to update `is_approved`.
    *   `chat_messages` (Columns: `id`, `user_id` (FK to auth.users), `role` (TEXT), `content` (TEXT), `created_at` (TIMESTAMPTZ, default now())). Set up RLS allowing users to access only their own messages.
    *   `uploaded_files` (Columns: `id`, `user_id`, `filename`, `namespace`, `uploaded_at`, optional `file_size`, `chunk_count`).

**b) Pinecone:**

*   Create a Serverless index.
*   Set **Dimensions** to `1536`.
*   Set **Metric** to `cosine`.
*   Note your Index Name and API Key for the `.env` file.

### 6. Running Locally

*   **Start Backend (Terminal 1):**
    ```bash
    # Make sure venv is active
    uvicorn main:app --reload --port 8000
    ```
*   **Start Frontend (Terminal 2):**
    ```bash
    # Make sure venv is active
    streamlit run app.py
    ```
*   Access the Streamlit app, usually at `http://localhost:8501`.

## Deployment (Render Example)

This application can be deployed using two Render Web Services (one for backend, one for frontend).

1.  **Backend Service (`carbonx-backend`):**
    *   Connect your Git repository.
    *   Set **Runtime** to Python 3.
    *   Set **Build Command:** `pip install -r requirements.txt`
    *   Set **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
    *   **Environment Variables:** Add **ALL** variables from your local `.env` file (`SUPABASE_*`, `PINECONE_*`, `GROQ_*`, etc.) to the Render Environment settings.
    *   **Disk:** Ensure **NO Persistent Disk** is attached.
    *   Choose instance type (Free tier available).
2.  **Frontend Service (`carbonx-frontend`):**
    *   Connect the same Git repository.
    *   Set **Runtime** to Python 3.
    *   Set **Build Command:** `pip install -r requirements.txt`
    *   Set **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false`
    *   **Environment Variables:**
        *   Add `BACKEND_URL` with the public URL of your deployed `carbonx-backend` service (e.g., `https://carbonx-backend.onrender.com`).
    *   Choose instance type.
    *   Deploy both services in the **same region**.

## Usage

1.  Access the Streamlit application URL.
2.  Sign up for a new account. **Note:** New accounts require manual approval by an admin directly in the Supabase `user_roles` table (set `is_approved` to `true`).
3.  Log in with approved credentials.
4.  Interact with the chatbot to provide information about your company.
5.  **Admins:** Can upload documents to the core knowledge base via the sidebar uploader. The sidebar also lists files currently tracked in the `core_db` namespace.
6.  **Users:** Can upload documents to their personal knowledge base via the sidebar uploader.
7.  When enough information is gathered, the bot might output a "FINAL DESCRIPTION:", triggering the CrewAI workflow in the backend to provide emission calculations and suggestions.
