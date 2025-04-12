from dotenv import load_dotenv

load_dotenv()

from initiatives.agents import CarbonAgents
from initiatives.tasks import CarbonTasks

from crewai import Task, Crew
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

TEST_SUMMARY = "A logistics company operating a fleet of 30 diesel trucks, with each truck consuming 800 gallons of diesel per month."
TEST_USER_ID = "789"
TEST_FILE_CONTEXT = "" # Start with no extra file context for simplicity

print("Simulating RAG lookup...")
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    user_db = Chroma(client=client, collection_name=f"user_{TEST_USER_ID}", embedding_function=embeddings)
    user_retriever = user_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = user_retriever.get_relevant_documents(TEST_SUMMARY)
    TEST_FILE_CONTEXT = "\n---\n".join([doc.page_content for doc in docs]) if docs else "No relevant user context found."
    print(f"Simulated RAG Context Length: {len(TEST_FILE_CONTEXT)}")
except Exception as rag_e:
    print(f"Warning: Failed to simulate RAG context: {rag_e}")
    TEST_FILE_CONTEXT = "Failed to retrieve user context." 

# --- Configuration ---


print("--- Starting CrewAI Local Test ---")
print(f"Input Summary: {TEST_SUMMARY}")
print(f"File Context: {'Provided' if TEST_FILE_CONTEXT else 'None'}")
print("-" * 30)

try:
    # 1. Initialize Agents and Tasks
    agents = CarbonAgents()
    tasks = CarbonTasks()

    # 2. Create Agents
    operations_analyst = agents.operations_analyst()
    emissions_expert = agents.emissions_expert()
    sustainability_advisor = agents.sustainability_advisor()

    print("Agents Initialized...")

    # 3. Define Tasks
    # Task 1: Parse Description
    parse_task = Task(
        description=tasks.parse_description(TEST_SUMMARY, TEST_FILE_CONTEXT),
        expected_output="JSON structured data containing company type and emission sources with quantities.",
        agent=operations_analyst,
        # output_file="parse_task_output.json" # Optionally save output
    )

    # Task 2: Calculate Emissions
    # The description for this task should guide the agent to use tools if needed
    calc_task = Task(
        description=tasks.calculate_emissions_description(),
        expected_output="JSON structured data with total emissions, unit, breakdown by source, and any unhandled sources.",
        agent=emissions_expert,
        context=[parse_task], # Depends on the output of the previous task
        # output_file="calc_task_output.json" # Optionally save output
    )

    # Task 3: Suggest Initiatives
    suggest_task = Task(
        description=tasks.suggest_initiatives_description(),
        expected_output="JSON list of 3 tailored sustainability initiatives with descriptions, impacts, and metrics.",
        agent=sustainability_advisor,
        context=[parse_task, calc_task], # Depends on the output of the previous tasks
        # output_file="suggest_task_output.json" # Optionally save output
    )

    print("Tasks Defined...")

    # 4. Create and Run the Crew
    # Ensure verbose=True in your agent definitions if you want detailed logs
    crew = Crew(
        agents=[operations_analyst, emissions_expert, sustainability_advisor],
        tasks=[parse_task, calc_task, suggest_task],
        verbose=True # Use verbose=2 for detailed agent step-by-step logging
    )

    print("\n--- Kicking off Crew ---")
    result = crew.kickoff()

    print("\n--- Crew Finished ---")
    print("\nFinal Result from Crew:")
    print(result) # Note: Crew.kickoff() often returns the result of the LAST task

    print("\n--- Individual Task Outputs (Raw) ---")
    print("\n[Parse Task Output]:")
    print(parse_task.output.raw if parse_task.output else "No output")
    print("-" * 20)

    print("\n[Calculate Task Output]:")
    print(calc_task.output.raw if calc_task.output else "No output")
    print("-" * 20)

    print("\n[Suggest Task Output]:")
    print(suggest_task.output.raw if suggest_task.output else "No output")
    print("-" * 20)

except Exception as e:
    print("\n---!!! An Error Occurred During Crew Execution !!!---")
    import traceback
    traceback.print_exc() # Print detailed traceback
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")

print("\n--- Test Script Finished ---")