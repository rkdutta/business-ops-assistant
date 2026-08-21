from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from models.base import LLMProvider
import os


DEFAULT_LOCAL_MODEL = "north-mini-code-1.0"
DEFAULT_REMOTE_MODEL = "openai.gpt-oss-120b"

os.environ.setdefault("AWS_REGION", "eu-west-1")

class llm(LLMProvider):
    def __init__(self, local: bool = True):
        self.model = DEFAULT_LOCAL_MODEL
        self.local = local
        if not self.local:
            self.model = DEFAULT_REMOTE_MODEL


    def get_llm(self):
        print("chosen model = ", self.model)
        if self.local:
            return ChatOllama(model=self.model)
        return ChatOpenAI(model=self.model)
