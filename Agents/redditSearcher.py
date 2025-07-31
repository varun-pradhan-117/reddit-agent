from typing import Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.outputs import ChatResult
from pydantic import PrivateAttr, BaseModel

class UserInput(BaseModel):
    user_input: str
    
class OutputState(BaseModel):
    graph_output: str
    
    
class RedditSearcher(StateGraph):
    def __init__(self, model: BaseChatModel):
        super().__init__()