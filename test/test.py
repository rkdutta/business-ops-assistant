from models.llm import llm as LLM

llm =  LLM(local=True).get_llm()
response = llm.invoke("hi")
print(response.content)