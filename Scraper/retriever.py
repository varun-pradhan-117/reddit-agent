import pandas as pd
import praw 
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
import os
from bson import ObjectId
from datetime import datetime

load_dotenv()

mongo=MongoClient(os.getenv("MONGO_URI"))
db = mongo["reddit_sentiment"]
queries_collection = db["queries"]
posts_collection = db["posts"]


reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent="Sentibot/0.1 by u/BraiseTheSun",
)

def fetch_post_comments(topic:str,subreddits:list[str]=["all"],
                time_filter:str="week",
                post_limit:int=30,
                comment_limit:int=20):
    query_id=str(ObjectId())
    
    query_data={
        "_id":query_id,
        "topic":topic,
        "subreddits":subreddits,
        "time_filter":time_filter,
        "post_limit":post_limit,
        "comment_limit":comment_limit,
        "created_at":datetime.now(),
    }

    queries_collection.insert_one(query_data)
    
    ops=[]
    for sub in subreddits:
        subreddit=reddit.subreddit(sub)
        print(f"Searching r/{subreddit}...")
        posts=subreddit.search(topic,time_filter=time_filter,limit=post_limit)
        for post in posts:
            content=(post.title+)

