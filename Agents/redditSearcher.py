from typing import Any, List, Optional, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.outputs import ChatResult
from pydantic import BaseModel, Field
from Scraper.reddit_analyzer import fetch_post_comments, analyze_query
from llm_wrappers import DeepSeekChat

class SentimentState(BaseModel):
    """State for the Reddit sentiment analysis workflow"""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    topic: Optional[str] = None
    subreddits: List[str] = Field(default_factory=lambda: ["all"])
    time_filter: str = "week"
    post_limit: int = 30
    comment_limit: int = 20
    query_id: Optional[str] = None
    analysis_complete: bool = False
    results_summary: Optional[str] = None
    error_message: Optional[str] = None
    input_required: bool = False
    clarification_request: Optional[str] = None
    partial_parameters: Optional[Dict] = Field(default_factory=dict)


class ParameterExtraction(BaseModel):
    """Structured output for parameter extraction"""
    topic: Optional[str] = Field(None, description="The main topic to search for")
    subreddits: List[str] = Field(default=["all"], description="List of subreddits to search in")
    time_filter: str = Field(default="week", description="Time filter: hour, day, week, month, year, all")
    post_limit: int = Field(default=30, description="Number of posts to fetch per subreddit")
    comment_limit: int = Field(default=20, description="Number of comments to fetch per post")
    needs_clarification: bool = Field(default=False, description="Whether more information is needed")
    clarification_request: Optional[str] = Field(None, description="What information is missing")

class RedditSearcher:
    def __init__(self, model_name = "deepseek-r1:8b", **kwargs):
        self._model=DeepSeekChat(model_name=model_name)
        self.graph=self._build_graph()
        
        self._structured_model=self._model.with_structured_output(
            ParameterExtraction, method='json_schema'
        )
        
        
        
    def _build_graph(self)-> StateGraph:
        """Build the state graph for the Reddit sentiment analysis workflow"""
        workflow=StateGraph(SentimentState)
        workflow.add_node("clarify_input",self._clarify_input)
        workflow.add_node("fetch_data", self._fetch_data)
        workflow.add_node("analyze_data", self._analyze_data)
        workflow.add_node("summarize_results", self._summarize_results)
        workflow.add_node("handle_error", self._handle_error)
        
        workflow.add_edge(START, "clarify_input")
        workflow.add_conditional_edges(
            "clarify_input",
            self._should_continue_after_input,
            {
                "fetch_data":"fetch_data",
                "clarify_input":"clarify_input",
                "error":"handle_error"
            }
        )
        workflow.add_conditional_edges(
            "fetch_data",
            self._should_continue_after_fetch,
            {
                "analyze_sentiment": "analyze_sentiment",
                "error": "handle_error"
            }
        )
        workflow.add_conditional_edges(
            "analyze_sentiment",
            self._should_continue_after_analysis,
            {
                "generate_summary": "generate_summary",
                "error": "handle_error"
            }
        )
        workflow.add_edge("generate_summary", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _clarify_input(self,state:SentimentState) -> SentimentState:
        """Clarify input parameters with the user"""
        try:
            user_messages=[]
            for msg in state.messages:
                if isinstance(msg, HumanMessage):
                    user_messages.append(msg.content)
            if not user_messages:
                state.error_message = "No user input provided."
                return state
            
            combined_context= "\n".join(user_messages)
            if state.partial_parameters:
                combined_context += "\n Previously extracted info:" + str(state.partial_parameters)
        
            clarification_prompt = ([
                ("system",
                """
                You are a parameter extraction assistant for Reddit sentiment analysis.
                Extract the following information from the user's request(s):
                - topic: The main topic to analyze (REQUIRED - cannot proceed without this)
                - subreddits: List of subreddits (default: ["all"])
                - time_filter: Time period (hour/day/week/month/year/all, default: "week")
                - post_limit: Number of posts per subreddit (default: 30)
                - comment_limit: Number of comments per post (default: 20)
                
                If you have partial information from previous messages, merge it with new information.
                
                CRITICAL: Set needs_clarification=True if:
                - The topic is missing, unclear, or too vague
                - You need more specific information to proceed
                
                If needs_clarification=True, provide a helpful clarification_request asking for the missing information.
                """),
                ("user", "{input}")
            ])
            
            prompt=clarification_prompt.invoke({
                "input": combined_context
            })
            result=self._structured_model.invoke(prompt)
            if result.needs_clarification:
                # Store partial parameters for next iteration and request more info
                state.input_required = True
                state.clarification_request = result.clarification_request 
                partial_params={}
                
                if result.topic:
                    partial_params['topic'] = result.topic
                if result.subreddits != ["all"]:
                    partial_params['subreddits'] = result.subreddits
                if result.time_filter != "week":
                    partial_params['time_filter'] = result.time_filter
                if result.post_limit != 30:
                    partial_params['post_limit'] = result.post_limit
                if result.comment_limit != 20:
                    partial_params['comment_limit'] = result.comment_limit
                
            
        except Exception as e:
            state.error_message = f"Error preparing clarification prompt: {str(e)}"
            return state