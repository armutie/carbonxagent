from textwrap import dedent

class CarbonTasks:
    def parse_description(self, company_description, file_context=""):
        return dedent(f"""
            You are given the following company description: '{company_description}'.
            Additional context from uploaded files: '{file_context}'.

            Your task is to parse the description and file context to extract the company type and all emission sources with their quantities into a structured JSON format.

            For example, if the description is: 'A logistics company using 50 diesel trucks at 200 gallons each monthly.',
            your output should be:
            {{
            "company_type": "logistics",
            "emission_sources": [
                {{"type": "diesel trucks", "quantity": 50, "fuel_per_truck_monthly": 200, "unit": "gallons"}}
            ]
            }}

            Note: You do not need to take any actions to obtain the company description; it is already provided above. Simply parse the given description into the required JSON format. Do NOT add backticks like ` to indicate that it is JSON text, only the content is necessary.
            Combine data from both sources, prioritizing the description if conflicts arise. If no file context is provided, use only the description.
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