from crewai import Agent, LLM
from textwrap import dedent
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq()

llm = LLM(model="groq/deepseek-r1-distill-llama-70b")

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
        return Agent(
            role="Emissions Expert",
            goal="Calculate total carbon emissions from provided data",
            backstory="You’re a lone wolf who calculates emissions with the data you’re given—no help needed.",
            tools=[],  
            allow_delegation=False,  # Explicitly disable coworker delegation
            verbose=True,
            llm=llm
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