from crewai.tools import BaseTool #pip install crewai_tools
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import re
from dotenv import load_dotenv

load_dotenv()

# --- Re-use or initialize ChromaDB components ---
client = chromadb.PersistentClient(path="./chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
core_db = Chroma(client=client, collection_name="core_db", embedding_function=embeddings)
# Retrieve slightly more context if it's general knowledge
core_retriever = core_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

class CoreKnowledgeLookupTool(BaseTool):
    name: str = "Core Knowledge Lookup"
    description: str = (
        "Use this tool to look up supplementary information, definitions, context, "
        "best practices, benchmarks, or specific data points potentially relevant "
        "to the task from the core knowledge base. Input should be a clear question "
        "about the specific information needed."
    )

    def _run(self, query: str) -> str:
        """Queries the core vector database for general information."""
        try:
            # Maybe add a prefix to the query for better retrieval focus
            # query = f"Regarding carbon emissions and sustainability: {query}"
            docs = core_retriever.get_relevant_documents(query)

            if not docs:
                return "No specific information found in the core knowledge base for that query."

            # Combine the content of the retrieved documents
            context = "\n---\n".join([doc.page_content for doc in docs])
            return f"Retrieved context from core knowledge base related to '{query}':\n{context}"

        except Exception as e:
            return f"Error using CoreKnowledgeLookupTool for query '{query}': {str(e)}"
        
class CustomCalculatorTool(BaseTool):
    name: str = "Custom Calculator"
    description: str = (
        "Performs basic mathematical calculations like addition (+), subtraction (-), multiplication (*), and division (/). "
        "Input MUST be a string representing the mathematical expression (e.g., '50 * 10.21', '30 * 800', '(1500 + 450) / 2'). "
        "Returns the numerical result as a string, or an error message if the calculation fails or the input is invalid."
    )

    def _run(self, expression: str) -> str:
        """Evaluates a mathematical expression string safely."""
        # 1. Validate input characters to allow only safe ones
        # Allows numbers, decimal points, +, -, *, /, parentheses, and spaces
        allowed_chars_pattern = r"^[0-9\.\s\+\-\*\/\(\)]+$"
        if not re.match(allowed_chars_pattern, expression):
            return f"Error: Expression '{expression}' contains invalid characters."

        # 2. Attempt to evaluate the expression safely
        try:
            # Use a restricted eval environment for safety
            # It prevents access to builtins and global/local variables
            result = eval(expression, {"__builtins__": {}}, {})

            # 3. Check if the result is a number (int or float)
            if isinstance(result, (int, float)):
                return str(result) # Return the result as a string
            else:
                # Should not happen with basic math ops, but handles weird edge cases
                return f"Error: Calculation for '{expression}' did not produce a number."

        except ZeroDivisionError:
            return f"Error: Division by zero in expression '{expression}'."
        except SyntaxError:
            return f"Error: Invalid mathematical syntax in expression '{expression}'."
        except NameError:
             # Catch if the restricted eval still tries to access something disallowed
             return f"Error: Invalid operation or disallowed name in expression '{expression}'."
        except Exception as e:
            # Catch any other unexpected errors during evaluation
            print(f"Unexpected calculator error for expression '{expression}': {e}") # Log it
            return f"Error: Could not evaluate expression '{expression}'. Reason: {type(e).__name__}"
