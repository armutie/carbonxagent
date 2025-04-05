from crewai import Crew, Agent, Task, LLM
from textwrap import dedent
from groq import Groq
from dotenv import load_dotenv
import re

load_dotenv()
groq_client = Groq()

llm = LLM(model="groq/deepseek-r1-distill-llama-70b")

class CarbonAgents:

    def operations_analyst(self):
        return Agent(
            role="Operations Analyst",
            backstory=dedent("""Expert in analyzing company operations and extracting key data points."""),
            goal=dedent("""Parse the company description to extract structured data about emission sources."""),
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
    

class CarbonTasks:
    def parse_description(self, company_description):
        return dedent(f"""
            You are given the following company description: '{company_description}'.

            Your task is to parse this description and extract the company type and all emission sources with their quantities into a structured JSON format.

            For example, if the description is: 'A logistics company using 50 diesel trucks at 200 gallons each monthly.',
            your output should be:
            {{
            "company_type": "logistics",
            "emission_sources": [
                {{"type": "diesel trucks", "quantity": 50, "fuel_per_truck_monthly": 200, "unit": "gallons"}}
            ]
            }}

            Note: You do not need to take any actions to obtain the company description; it is already provided above. Simply parse the given description into the required JSON format. Do NOT add backticks like ` to indicate that it is JSON text, only the content is necessary.
        """)

    def calculate_emissions_description(self):
        return dedent("""
            Use context emission sources to calculate total carbon emissions. Use these factors:
- Diesel: 10.21 kg CO2e/gallon
- Gasoline: 8.78 kg CO2e/gallon
- Natural gas: 5.31 kg CO2e/therm
- Propane: 5.31 kg CO2e/gallon
- Coal: 2.86 kg CO2e/kg (convert tons to kg: tons * 1000 * 2.86)
- Electricity: 0.4 kg CO2e/kWh (average; adjust per region if specified)
For each source:
1. If "fuel_total_monthly" exists, use: total_fuel * emission_factor.
2. If "quantity" and "fuel_per_unit_monthly" exist, use: quantity * fuel_per_unit * emission_factor.
3. If "quantity" and daily usage (e.g., "fuel_per_unit_daily") exist, convert to monthly: quantity * fuel_per_unit * 30 * emission_factor.
4. Add to breakdown: {"source": "<type>", "emissions": <value>}.
Return:
{
  "total_emissions": <sum>,
  "unit": "kg CO2e monthly",
  "breakdown": [{"source": "<type>", "emissions": <value>}, ...]
}
If a source’s factor isn’t listed, note it in "unhandled_sources". Do NOT add backticks like ` to indicate that it is JSON text, only the content is necessary.
        """)

    def suggest_initiatives_description(self):
        return dedent("""
        Use context (e.g., {"total_emissions": <value>, "unit": "kg CO2e monthly", "breakdown": [{"source": "<source1>", "emissions": <value1>}, ...]}). Suggest 3 initiatives targeting the largest emission sources in the breakdown. Output a JSON list with 'initiative', 'description', 'impact', and 'metrics'.

Steps:
1. Read the breakdown and rank sources by emissions value.
2. For the top 1-3 sources (or fewer if less than 3), propose a unique initiative directly addressing that source’s emissions (e.g., switch fuel, improve efficiency, offset emissions).
3. Write a description explaining how the initiative reduces emissions for that specific source.
4. Calculate the impact as a percentage range and absolute reduction (e.g., "20-30%, <X>-<Y> kg CO2e/month") using the source’s emissions value.
5. Define 2-3 concise, lowercase metrics tied to the initiative and source.
6. Return:
[
  {
    "initiative": "<short name>",
    "description": "<how it reduces emissions for this source>",
    "impact": "<X%-Y%, <A>-<B> kg CO2e/month>",
    "metrics": ["<metric1>", "<metric2>", ...]
  },
  ...
]

If context lacks a breakdown:
[
  {"initiative": "Error", "description": "No emission sources provided", "impact": "None", "metrics": []}
]

Note: Base each initiative on the exact source name and emissions from the breakdown (e.g., "propane dragons", not generic terms). Use the emissions value for impact calculations. Avoid generic or unrelated suggestions—stay specific to the data. Do NOT add backticks like ` to indicate that it is JSON text, only the content is necessary.
    """)

def remove_code_fences(text):
    # Removes all code block markers like ```json or ```
    text = re.sub(r"```(?:\w+)?\n?|```", "", text)
    text = text.replace("**", "")
    return text

def process_summary(summary: str):
    agents = CarbonAgents()
    tasks = CarbonTasks()

    parse_task = Task(
        description=tasks.parse_description(summary),
        expected_output="JSON structured data",
        agent=agents.operations_analyst(),
    )

    calc_task = Task(
        description=tasks.calculate_emissions_description(),
        expected_output="JSON emissions data",
        agent=agents.emissions_expert(),
        context=[parse_task],
    )

    suggest_task = Task(
        description=tasks.suggest_initiatives_description(),
        expected_output="JSON list of initiatives",
        agent=agents.sustainability_advisor(),
        context=[parse_task, calc_task],
    )

    crew = Crew(
    agents=[agents.operations_analyst(), agents.emissions_expert(), 
            agents.sustainability_advisor()],
    tasks=[parse_task, calc_task, suggest_task],
    verbose=True
    )

    crew.kickoff()
    outputs = [
        parse_task.output.raw,
        calc_task.output.raw,
        suggest_task.output.raw
    ]
    for output in outputs: 
        output = remove_code_fences(output)
        print(output + "----------")
    return outputs

# process_summary("A manufacturing plant operating 20 coal-fired furnaces that consume 75 tons of coal per month each, and a fleet of 20 diesel delivery vans using 20 gallons of diesel per month each.")
