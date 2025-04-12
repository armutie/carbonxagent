from crewai import Agent, LLM
from textwrap import dedent
from groq import Groq
from dotenv import load_dotenv
from .tools import CoreKnowledgeLookupTool, CustomCalculatorTool
from langchain_openai import ChatOpenAI

load_dotenv()
groq_client = Groq()

llm = LLM(model="groq/deepseek-r1-distill-llama-70b")
llama_llm = LLM(model="groq/llama-3.3-70b-versatile")
llama_fast_llm = LLM(model="groq/llama-3.3-70b-specdec")

openai_llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.7 
        )


class CarbonAgents:

    def operations_analyst(self):
        return Agent(
            role="Operations Analyst",
            backstory=dedent("""Expert in analyzing company operations and extracting key data points."""),
            goal=dedent("""Parse the company description and any uploaded file data to extract structured data about emission sources."""),
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def emissions_expert(self):
        knowledge_tool_instance = CoreKnowledgeLookupTool()
        calculator_tool_instance = CustomCalculatorTool()


        return Agent(
            role="Emissions Expert",
            goal=dedent("""Calculate total carbon emissions based on structured data.
                Before calculating, critically assess if you need specific information (like emission factors, calculation methodologies for unusual sources, or conversion constants) that isn't common knowledge or provided in the context.
                If specific information is needed, formulate a precise question and use the 'Core Knowledge Lookup' tool ONCE for that piece of information.
                Evaluate the retrieved information: Is it directly relevant and useful for *this calculation*?
                If relevant, use it. If not relevant, or if the tool finds nothing, proceed using standard assumptions or clearly state the limitation/unhandled source in your output.
                Do NOT use the tool speculatively or for general knowledge. Focus only on necessary data for the calculation. Avoid getting sidetracked by irrelevant details.
                Output the results in the specified JSON format."""),
            backstory=dedent("""\
                You are a precise and efficient emissions calculation specialist. 
                You rely on provided data and standard factors, but you know when to seek specific, necessary information using the knowledge lookup tool. 
                You don't waste time on irrelevant lookups and focus solely on accurate calculation based on available, relevant data."""),
            tools=[knowledge_tool_instance, calculator_tool_instance],  
            allow_delegation=False, 
            verbose=True,
            llm=openai_llm
        )

    def sustainability_advisor(self):
        return Agent(
            role="Sustainability Advisor",
            backstory=dedent("""Expert in sustainable practices with actionable ideas to cut emissions."""),
            goal=dedent("""Provide tailored suggestions to reduce the company’s carbon footprint, including metrics to track."""),
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    # def tracking_system_designer(self):
    #     return Agent(
    #         role="Tracking System Designer",
    #         backstory=dedent("""Systems architect who designs efficient data tracking solutions."""),
    #         goal=dedent("""Create a JSON schema and API structure to monitor emissions over time for each initiative."""),
    #         verbose=True,
    #         allow_delegation=False,
    #         llm=self.DeepSeek,
    #     )