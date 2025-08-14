import pandas as pd
import praw 
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
import os
from bson import ObjectId
import statistics
from datetime import datetime
from llm_wrappers import DeepSeekChat
load_dotenv()

mongo=MongoClient(os.getenv("MONGO_URI"))
db = mongo["reddit_sentiment"]
queries_collection = db["queries"]
posts_collection = db["posts"]


def fmt(ts):
    """Format timestamps to HH:MM DD/MM/YYYY"""
    if isinstance(ts, (int, float)):  # Reddit's created_utc is in epoch seconds
        return datetime.fromtimestamp(ts).strftime("%H:%M %d/%m/%Y")
    elif isinstance(ts, datetime):
        return ts.strftime("%H:%M %d/%m/%Y")
    return ts  # fallback

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent="Sentibot/0.1 by u/BraiseTheSun",
)

def fetch_post_comments(topic:str,subreddits:list[str]=["all"],
                time_filter:str="week",
                post_limit:int=30,
                comment_limit:int=20):
    """
    Fetches posts and comments from Reddit based on a topic and subreddits.
    
    Args:
        topic (str): The topic to search for in posts.
        subreddits (list[str], optional): List of subreddits to search. Defaults to ["all"].
        time_filter (str, optional): Time period of searches. Defaults to "week".
        post_limit (int, optional): Number of posts to search for in each subreddit. Defaults to 30.
        comment_limit (int, optional): Number of comments to fetch from each post. Defaults to 20.

    Returns:
        int: The query ID for the operation.
    """
    query_id=str(ObjectId())
    
    query_data={
        "_id":query_id,
        "topic":topic,
        "subreddits":subreddits,
        "time_filter":time_filter,
        "post_limit":post_limit,
        "comment_limit":comment_limit,
        "created_at":fmt(datetime.now()),
    }

    queries_collection.insert_one(query_data)
    
    ops=[]
    for sub in subreddits:
        subreddit=reddit.subreddit(sub)
        print(f"Searching r/{subreddit}...")
        posts=subreddit.search(topic,time_filter=time_filter,limit=post_limit)
        for post in posts:
            content=(post.title.rstrip()+ " " + post.selftext.rstrip())
            if topic.lower() not in content.lower():
                continue
            
            post_comments=[]
            post.comment_sort="top"
            post.comments.replace_more(limit=0)
            for c in post.comments[:comment_limit]:
                post_comments.append({
                    "comment_id":c.id,
                    "text":c.body,
                    "score":c.score,
                    "created_at":fmt(c.created_utc),
                    "sort_type":"top",
                })
                
            post.comment_sort="controversial"
            post.comments.replace_more(limit=0)
            for c in post.comments[:comment_limit]:
                post_comments.append({
                    "comment_id":c.id,
                    "text":c.body,
                    "score":c.score,
                    "created_at":fmt(c.created_utc),
                    "sort_type":"controversial",
                })
            
            
            seen=set()
            unique_comments=[]
            for comment in post_comments:
                cid=comment["comment_id"]
                if cid not in seen:
                    seen.add(cid)
                    unique_comments.append(comment)
                    
            doc = {
                "_id":post.id,
                "query_id":query_id,
                "title":post.title,
                "subreddit":sub,
                "topic":topic,
                "post_id":post.id,
                "post_content":content,
                "comments":unique_comments,
                "created_at":fmt(post.created_utc),
            }
            
            ops.append(UpdateOne({"_id": post.id}, {"$set": doc}, upsert=True))

        if ops:
            posts_collection.bulk_write(ops)
            print(f"[✔] Saved {len(ops)} posts with comments for query_id: {query_id}")
        else:
            print("[!] No matching posts/comments found.")

        return query_id
    
    
# Model for per-comment sentiment output
class SentimentOutput(BaseModel):
    score: int = Field(..., description="Score from 1 to 5 based on sentiment toward the topic")
    explanation: str = Field(..., description="Explanation for the score")    
    
def analyze_query(query_id:str):
    posts=list(posts_collection.find({"query_id":query_id}))
    
    if not posts:
        print(f"[!] No posts found for query_id: {query_id}")
        return
    
    # Sentiment scoring prompt
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are an insightful and helpful rater. You give a score from -2 to 2 based on the sentiment of the statement towards the topic along with an explanation for the score.
         -2 - Strongly Negative
         -1 - Negative
         0 - Neutral
         1 - Positive
         2 - Strongly Positive
         NEVER comment on the enthusiasm of the comments or about howw they are excited to discuss the topic.
         The only thing that matters is the sentiment towards the topic itself.
         The score should be based on the overall sentiment of the statement towards the topic.
         If the statement is not related to the topic, return a score of 0 with an explanation that it is neutral.
         """),
        ("user", "Topic: {topic}\nStatement: {statement}")
    ])
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are a post summarizer. You summarize the sentiment of all the comments in a post towards the topic.
         You will receive a list of comments with the average sentiment score. 
         It is fine to use the average score as a reference, but feel free to adjust the summary if the comments suggest a different sentiment.
         Provide a concise summary of the overall sentiment towards the topic.
         NEVER comment on the enthusiasm of the comments or about howw they are excited to discuss the topic.
         The only thing that matters is the sentiment towards the topic itself.
         The summary should be 1-2 sentences long.
         """),
        ("user", "Topic: {topic}\nAverage Sentiment Score: {avg_score}\nComments: {comments}")
    ])
    
    topic_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are a topic summarizer. You summarize the sentiment of all the comments towards a specific topic.
         You will receive a topic and a list of comments with the average sentiment score.
         It is fine to use the average score as a reference, but feel free to adjust the summary if the comments suggest a different sentiment.
         Provide a concise summary of the overall sentiment towards the topic.
         NEVER comment on the enthusiasm of the comments or about howw they are excited to discuss the topic.
         The only thing that matters is the sentiment towards the topic itself.
         The summary should be thoughtful and insightful trying to capture the general sentiment of people towards the topic.
         """),
        ("user", "Topic: {topic}\nAverage Sentiment Score: {avg_score}\nComments: {comments}")
    ])
    ds = DeepSeekChat()
    ds_sentiment=ds.with_structured_output(SentimentOutput,method='json_schema')
    # For overall topic summary later
    all_comments_text = []
    all_scores = []
    
    for post in posts:
        print(f"Analyzing post: {post['title']} in r/{post['subreddit']}")
        scores = []
        analyzed_comments=[]
        for comment in post["comments"]:
            prompt=sentiment_prompt.invoke({
                "topic": post["topic"],
                "statement": comment["text"]
            })
            result=ds_sentiment.invoke(prompt)
            
            analyzed_comments.append({
                **comment,
                "sentiment_score": result.score,
                "sentiment_explanation": result.explanation
            })
            scores.append(result.score)
            all_scores.append(result.score)
            all_comments_text.append(comment["text"])
            
        avg_score=statistics.mean(scores) if scores else None
        
        prompt=summary_prompt.invoke({
            "topic": post["topic"],
            "avg_score": avg_score,
            "comments": "\n".join([f"{c['text']}" for c in analyzed_comments])
        })
        sentiment_summary=ds.invoke(prompt).content
        posts_collection.update_one(
            {"_id": post["_id"]},
            {
                "$set": {
                    "comments": analyzed_comments,
                    "aggregate_sentiment_score": avg_score,
                    "summary": sentiment_summary
                }
            }
        )
        print(f"[✔] Analysis completed and saved for query_id: {query_id}")
        
    # Generate overall topic summary
    overall_avg_score = statistics.mean(all_scores) if all_scores else None
    topic_prompt = topic_summary_prompt.invoke({
        "topic": posts[0]["topic"],
        "avg_score": overall_avg_score,
        "comments": "\n".join(all_comments_text)
    })
    overall_summary = ds.invoke(topic_prompt).content
    
    queries_collection.update_one(
        {"_id": query_id},
        {
            "$set": {
                "overall_avg_sentiment_score": overall_avg_score,
                "overall_summary": overall_summary
            }
        }
    )
    print(f"[✔] Overall topic summary saved for query_id: {query_id}")
        
        