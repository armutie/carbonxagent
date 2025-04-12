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
        return dedent("""\
            You are provided with structured data about a company's emission sources in the context.
            Your task is to calculate the total carbon emissions.

            Follow these steps:
            1. Review each emission source provided in the context.
            2. For each source, determine the substance (e.g., 'diesel', 'natural gas', 'coal', 'electricity', 'refrigerant R-410a').
            3. **Assess Information Need:** Do you have a reliable emission factor (kg CO2e per unit) for this specific substance readily available or from common knowledge? Is the calculation method clear?
            4. **Conditional Tool Use:** If, AND ONLY IF, you lack a specific factor or methodology necessary for the calculation:
                a. Formulate a *targeted question* (e.g., "What is the CO2e emission factor per kg for refrigerant R-410a?", "Standard method for calculating emissions from industrial waste incineration?").
                b. Use the 'Core Knowledge Lookup' tool with your specific question. Use it *sparingly* - only when essential information is missing.
                c. **Evaluate Relevance:** Examine the tool's response. Does it directly answer your question and provide the necessary factor/method?
                d. **Integrate or Discard:** If relevant and useful, use the information in your calculation. If the response is irrelevant, unhelpful, or nothing is found, *ignore it* and proceed. Note the source as potentially unhandled or use a documented standard assumption if appropriate. Do NOT include irrelevant retrieved text in your final output.
            5. If you have the factor (either from knowledge or the tool):
               - Calculate the monthly emissions for that source using the provided quantities and the factor.
               - Ensure units match (convert daily to monthly * 30, tons to kg * 1000, etc.).
               - Add to the breakdown: {"source": "<source type>", "emissions": <calculated value>}.
            6. If a factor/method couldn't be reliably determined (even after checking the tool):
               - Add the source type to an 'unhandled_sources' list in the final JSON.
            7. Sum the emissions from all successfully calculated sources.
            8. Format the final output as a JSON object:
            {
              "total_emissions": <sum>,
              "unit": "kg CO2e monthly",
              "breakdown": [{"source": "<type>", "emissions": <value>}, ...],
              "unhandled_sources": ["<type1>", ...] // Only if applicable
            }
            Prioritize accuracy and transparency. Only use information you trust, whether from context, common knowledge, or *relevant* tool lookups. Avoid speculation. Ensure valid JSON output without backticks.
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