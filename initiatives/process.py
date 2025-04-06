import re
from initiatives.agents import CarbonAgents
from initiatives.tasks import CarbonTasks
# from agents import CarbonAgents
# from tasks import CarbonTasks
from crewai import Task, Crew
from initiatives.rag import get_rag_context

def remove_code_fences(text):
    # Removes all code block markers like ```json or ```
    text = re.sub(r"```(?:\w+)?\n?|```", "", text)
    text = text.replace("**", "")
    text = re.sub(r"\n", "", text)
    # text = text.replace(" ", "")
    return text

def clean_outputs(full_list):
    cleaned_outputs = []  
    for output in full_list:
        cleaned_output = remove_code_fences(output) 
        cleaned_outputs.append(cleaned_output) 
    return cleaned_outputs

def process_summary(summary: str, user_id: str, file_text: str = ""):
    agents = CarbonAgents()
    tasks = CarbonTasks()

    file_context = get_rag_context(summary, user_id)

    parse_task = Task(
        description=tasks.parse_description(summary, file_context),
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
    final_output = clean_outputs(outputs)
    for output in final_output:
        print(output + "-------")
    return final_output

# process_summary("A manufacturing plant with 70 diesel vans, each consuming 30 gallons of diesel a month.")