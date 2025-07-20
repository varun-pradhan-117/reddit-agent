from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Any, List
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.outputs import ChatResult
from pydantic import PrivateAttr

class DeepSeekWrapper(BaseChatModel):
    _model_name: str = PrivateAttr()
    _model: ChatOllama = PrivateAttr()
    def __init__(self, model_name: str = "deepseek-r1:7b"):
        #super().__init__()
        self._model_name = model_name
        self._model = ChatOllama(model=model_name)

    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def invoke(self,input:Any,config:RunnableConfig|None=None,**kwargs):
        return self._model.invoke(input)
    
    def stream(self,input:Any,config:RunnableConfig|None=None,**kwargs):
        return self._model.stream(input)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: RunnableConfig | None = None,
        **kwargs: Any
    ) -> ChatResult:
        return self._model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)