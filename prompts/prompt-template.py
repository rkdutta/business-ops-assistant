import json
from langchain_core.load import dumpd
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["prompt_text"],
    template="{prompt_text}",
    validate_template=True,
)

# with open("prompt.json", "w") as f:
#     json.dump(dumpd(template), f)

print(dumpd(template))
prompt_text = "india"
prompt = template.invoke({"prompt_text": prompt_text}).to_string()