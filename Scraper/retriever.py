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
                    "created_at":c.created_utc,
                    "sort_type":"top",
                })
                
            post.comment_sort="controversial"
            post.comments.replace_more(limit=0)
            for c in post.comments[:comment_limit]:
                post_comments.append({
                    "comment_id":c.id,
                    "text":c.body,
                    "score":c.score,
                    "created_at":c.created_utc,
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
                "created_at":post.created_utc,
            }
            
            ops.append(UpdateOne({"_id": post.id}, {"$set": doc}, upsert=True))

        if ops:
            posts_collection.bulk_write(ops)
            print(f"[✔] Saved {len(ops)} posts with comments for query_id: {query_id}")
        else:
            print("[!] No matching posts/comments found.")

        return query_id
