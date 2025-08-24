from langgraph.graph.message import add_messages
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from reddit_workers import (
    queries_collection, 
    posts_collection, 
    fetch_comments, 
    analyze_query
    )
import reddit_workers.db as db
from langgraph.checkpoint.memory import InMemorySaver
from types import MethodType
from typing import Optional, List, Dict, Any, Union, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
import statistics

from llm_wrappers import DeepSeekChat

class SentimentState(BaseModel):
    """State for the Reddit sentiment analysis workflow"""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    topic: Optional[str] = None
    subreddits: List[str] = Field(default_factory=lambda: ["all"])
    time_filter: Literal["day", "week", "month", "year", "all"] = "week"
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
    time_filter: Literal["hour", "day", "week", "month", "year", "all"] = Field(
        default="week",
        description="Time filter: hour, day, week, month, year, all"
    )
    needs_clarification: bool = Field(default=False, description="Whether more information is needed")
    clarification_request: Optional[str] = Field(None, description="What information is missing")

class RedditSearcher:
    def __init__(self, model_name = "deepseek-r1:8b", **kwargs):
        self._model=DeepSeekChat(model_name=model_name)
        self.checkpointer = InMemorySaver()
        self.graph=self._build_graph()
        
        self._structured_model=self._model.with_structured_output(
            ParameterExtraction, method='json_schema'
        )
        
        
        
    def _build_graph(self)-> StateGraph:
        """Build the state graph for the Reddit sentiment analysis workflow"""
        workflow=StateGraph(SentimentState)
        workflow.add_node("clarify_input",self._clarify_input)
        workflow.add_node("fetch_data", self._fetch_data)
        workflow.add_node("analyze_sentiment", self._analyze_sentiment)
        workflow.add_node("generate_summary", self._generate_summary)
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
        # Attach checkpointer
        self.checkpointer=MongoDBSaver(db.mongo, "reddit_sentiment", "workflow_checkpoints")
        return workflow.compile(checkpointer=self.checkpointer)
    
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
        
            clarification_prompt = ChatPromptTemplate.from_messages([
                ("system",
                """
                You are a parameter extraction assistant for Reddit sentiment analysis.
                Extract the following information from the user's request(s):
                - topic: The main topic to analyze (REQUIRED - cannot proceed without this)
                - subreddits: List of subreddits (default: ["all"])
                - time_filter: Time period (hour/day/week/month/year/all, default: "week")
                
                If you have partial information from previous messages, merge it with new information.
                
                CRITICAL: Set needs_clarification=True if:
                - The topic is missing, unclear, or too vague
                - You need more specific information to proceed
                - One word topics are sufficient. Don't ask for more if the topic can be deciphered.
                - Time filter is always relative to the current time. It will always be the last day/week/month/year/all
                
                If needs_clarification=True, provide a helpful clarification_request asking for the missing information.
                """),
                ("user", "{input}")
            ])
            
            prompt=clarification_prompt.invoke({
                "input": combined_context
            })
            result=self._structured_model.invoke(prompt)
            print("Parameter extraction result:", result)
            if result.needs_clarification:
                # Store partial parameters for next iteration and request more info
                state.input_required = True
                state.clarification_request = result.clarification_request 
                partial_params={}
                
                # Update partial parameters
                if result.topic:
                    partial_params['topic'] = result.topic
                if result.subreddits != ["all"]:
                    partial_params['subreddits'] = result.subreddits
                if result.time_filter != "week":
                    partial_params['time_filter'] = result.time_filter
                state.partial_parameters = partial_params

                clarification_msg=result.clarification_request
                if state.partial_parameters:
                    clarification_msg += "\n\nSo far we have:\n" + str(state.partial_parameters)
                    
                state.messages.append(HumanMessage(content=clarification_msg))
            else:
                # We have enough information to proceed
                state.input_required = False
                state.topic = result.topic
                state.subreddits = result.subreddits
                state.time_filter = result.time_filter
                
                # Clear partial parameters since we're done
                state.partial_parameters = {}
                
                # Add confirmation message
                confirmation_msg = f"""Perfect! I'll analyze Reddit sentiment for: "{result.topic}"
                
                Parameters:
                - Subreddits: {', '.join(result.subreddits)}
                - Time filter: {result.time_filter}

                Starting data collection..."""
                
                state.messages.append(AIMessage(content=confirmation_msg))
            
                
            
        except Exception as e:
            state.error_message = f"Error preparing clarification prompt: {str(e)}"
        return state
        
    
    def _fetch_data(self,state:SentimentState) -> SentimentState:
        """Fetch Reddit posts and commments based on the topic and parameters"""
        try:
            if not state.topic:
                state.error_message="No topic specified."
                return state
            
            query_id=fetch_comments(
                topic=state.topic,
                subreddits=state.subreddits,
                time_filter=state.time_filter,
            )
            state.query_id = query_id
            state.messages.append(AIMessage(content=f"✅ Data collection complete. Query ID: {query_id}"))
            
        except Exception as e:
            state.error_message = f"Error during data fetching: {str(e)}"
        
        return state
    
    def _analyze_sentiment(self, state: SentimentState) -> SentimentState:
        """Analyze sentiment of collected data"""
        try:
            if not state.query_id:
                state.error_message = "No query ID available for sentiment analysis"
                return state
            
            # Call the analyze function from your analyzer
            analyze_query(state.query_id)
            
            state.analysis_complete = True
            state.messages.append(AIMessage(content="✅ Sentiment analysis complete. Generating summary..."))
            
        except Exception as e:
            state.error_message = f"Error during sentiment analysis: {str(e)}"
        
        return state
    
    def _generate_summary(self, state: SentimentState) -> SentimentState:
        """Generate a final formatted summary of results"""
        try:
            
            # Get the analysis results
            query_data = queries_collection.find_one({"_id": state.query_id})
            posts_data = list(posts_collection.find({"query_id": state.query_id}))
            
            if not query_data or not posts_data:
                state.error_message = "No analysis results found"
                return state
            
            # Calculate statistics
            total_comments = sum(len(post.get("comments", [])) for post in posts_data)
            subreddits_analyzed = list(set(post.get("subreddit", "") for post in posts_data))
            overall_score = query_data.get("overall_avg_sentiment_score", 0)
            
            # Format sentiment score with interpretation
            def format_sentiment_score(score):
                if score is None:
                    return "No data"
                
                if score >= 1.5:
                    return f"**{score:.2f}** (Strongly Positive 😊)"
                elif score >= 0.5:
                    return f"**{score:.2f}** (Positive 🙂)"
                elif score >= -0.5:
                    return f"**{score:.2f}** (Neutral 😐)"
                elif score >= -1.5:
                    return f"**{score:.2f}** (Negative 😕)"
                else:
                    return f"**{score:.2f}** (Strongly Negative 😞)"
            
            # Create formatted summary
            summary_lines = [
                f"# 📊 Reddit Sentiment Analysis: {state.topic}",
                "",
                f"**Overall Sentiment:** {format_sentiment_score(overall_score)}",
                "",
                "## 📈 Analysis Overview",
                f"• **Posts Analyzed:** {len(posts_data)}",
                f"• **Comments Processed:** {total_comments:,}",
                f"• **Subreddits:** {', '.join(f'r/{sub}' for sub in subreddits_analyzed if sub)}",
                f"• **Time Period:** {state.time_filter}",
                "",
                "## 🎯 Key Insights",
                query_data.get("overall_summary", "No summary available"),
                ""
            ]
            
            # Add top positive and negative posts if we have enough data
            if len(posts_data) > 1:
                # Sort posts by sentiment score
                posts_with_scores = [(post, post.get("aggregate_sentiment_score", 0)) for post in posts_data if post.get("aggregate_sentiment_score") is not None]
                
                if posts_with_scores:
                    posts_with_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    summary_lines.extend([
                        "## 📝 Notable Posts",
                        ""
                    ])
                    
                    # Most positive post
                    if posts_with_scores[0][1] > 0:
                        most_positive = posts_with_scores[0][0]
                        summary_lines.extend([
                            f"**Most Positive Post** ({format_sentiment_score(posts_with_scores[0][1])}):",
                            f"*\"{most_positive.get('title', 'No title')}\"* in r/{most_positive.get('subreddit', 'unknown')}",
                            f"> {most_positive.get('summary', 'No summary available')[:200]}{'...' if len(most_positive.get('summary', '')) > 200 else ''}",
                            ""
                        ])
                    
                    # Most negative post
                    if posts_with_scores[-1][1] < 0:
                        most_negative = posts_with_scores[-1][0]
                        summary_lines.extend([
                            f"**Most Critical Post** ({format_sentiment_score(posts_with_scores[-1][1])}):",
                            f"*\"{most_negative.get('title', 'No title')}\"* in r/{most_negative.get('subreddit', 'unknown')}",
                            f"> {most_negative.get('summary', 'No summary available')[:200]}{'...' if len(most_negative.get('summary', '')) > 200 else ''}",
                            ""
                        ])
            
            # Add subreddit breakdown if multiple subreddits
            if len(subreddits_analyzed) > 1:
                summary_lines.extend([
                    "## 🏢 Subreddit Breakdown",
                    ""
                ])
                
                subreddit_stats = {}
                for post in posts_data:
                    sub = post.get("subreddit", "unknown")
                    score = post.get("aggregate_sentiment_score")
                    if score is not None:
                        if sub not in subreddit_stats:
                            subreddit_stats[sub] = []
                        subreddit_stats[sub].append(score)
                
                for sub, scores in sorted(subreddit_stats.items()):
                    avg_score = statistics.mean(scores)
                    post_count = len(scores)
                    summary_lines.append(f"• **r/{sub}**: {format_sentiment_score(avg_score)} ({post_count} posts)")
                
                summary_lines.append("")
            
            # Add methodology note
            summary_lines.extend([
                "---",
                "*Analysis based on sentiment scoring from -2 (strongly negative) to +2 (strongly positive). Results reflect public opinion on Reddit and may not represent broader population views.*"
            ])
            
            formatted_summary = "\n".join(summary_lines)
            state.results_summary = formatted_summary
            state.messages.append(AIMessage(content=formatted_summary))
            
        except Exception as e:
            state.error_message = f"Error generating summary: {str(e)}"
        
        return state
    
    def _handle_error(self, state: SentimentState) -> SentimentState:
        """Handle errors in the workflow"""
        error_msg = f"❌ An error occurred: {state.error_message}"
        state.messages.append(AIMessage(content=error_msg))
        return state
    
    # Conditional edge functions
    def _should_continue_after_input(self, state: SentimentState) -> str:
        if state.error_message:
            return "error"
        elif state.input_required:
            return "clarify_input"  # Loop back for more clarification
        else:
            return "fetch_data"
    
    def _should_continue_after_fetch(self, state: SentimentState) -> str:
        return "error" if state.error_message else "analyze_sentiment"
    
    def _should_continue_after_analysis(self, state: SentimentState) -> str:
        return "error" if state.error_message else "generate_summary"
    
    
    def run(self, user_input: str, config:Dict, existing_state: Optional[SentimentState] = None) -> Dict[str, Any]:
        """Run the sentiment analysis workflow"""
        if existing_state:
            # Continue from existing state with new input
            existing_state.messages.append(HumanMessage(content=user_input))
            initial_state = existing_state
        else:
            # Start fresh
            initial_state = SentimentState(
                messages=[HumanMessage(content=user_input)]
            )
        
        final_state = self.graph.invoke(initial_state,config=config)
        
        return {
            "state": final_state  # Return state for continuation
        }
        
    async def run_streaming(self,user_input:str,config:Dict, existing_state:Optional[SentimentState]=None)->Any:
        """Run the sentiment analysis workflow in streaming mode"""
        if existing_state:
            # Continue from existing state with new input
            existing_state.messages.append(HumanMessage(content=user_input))
            initial_state = existing_state
        else:
            # Start fresh
            initial_state = SentimentState(
                messages=[HumanMessage(content=user_input)]
            )
        
        for partial_state in self.graph.stream(initial_state,config=config):
            yield {
                "state": partial_state  # Yield intermediate states for streaming
            }
            if partial_state.input_required:
                return  # Stop streaming if more input is needed