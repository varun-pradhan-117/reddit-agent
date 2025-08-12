from langchain_ollama import ChatOllama
from typing import Any, List
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.outputs import ChatResult

class DeepSeekChat(ChatOllama):
    def __init__(self, model_name: str = "deepseek-r1:8b", **kwargs):
        super().__init__(model=model_name, **kwargs)
        # Your custom init here

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    # Override invoke if you want to add pre/post processing, otherwise no need
    def invoke(self, input: Any, verbose=False, config: RunnableConfig | None = None, **kwargs):
        # Example: add custom logging or input modifications here
        op=super().invoke(input, config=config, **kwargs)
        if verbose:
            return op
        else:
            key_content=op.content.split("</think>",1)[1]
            op.content=key_content.strip()
            return op

    # Same for stream
    def stream(self, input: Any, config: RunnableConfig | None = None, **kwargs):
        return super().stream(input, config=config, **kwargs)

    # Override _generate only if you need to customize generation logic
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: RunnableConfig | None = None,
        **kwargs: Any
    ) -> ChatResult:
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)