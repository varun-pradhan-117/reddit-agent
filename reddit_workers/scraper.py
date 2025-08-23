import os
import praw
from dotenv import load_dotenv
from bson import ObjectId
from pymongo import UpdateOne
from datetime import datetime

from reddit_workers.db import queries_collection, posts_collection

load_dotenv()

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

def fetch_comments_for_post(post, comment_limit: int, sort_type: str):
    submission = reddit.submission(id=post.id)
    submission.comment_sort = sort_type
    submission.comments.replace_more(limit=0)
    return [
        {
            "comment_id": c.id,
            "text": c.body,
            "score": c.score,
            "created_at": fmt(c.created_utc),
            "sort_type": sort_type,
        }
        for c in submission.comments[:comment_limit]
    ]

def fetch_comments(
    topic: str,
    subreddits: list[str] = ["all"],
    time_filter: str = "week",
    post_limit: int = 30,
    comment_limit: int = 20,
):
    """
    Fetches posts and comments from Reddit based on a topic and subreddits.
    Saves results to MongoDB.
    """
    query_id = str(ObjectId())

    query_data = {
        "_id": query_id,
        "topic": topic,
        "subreddits": subreddits,
        "time_filter": time_filter,
        "post_limit": post_limit,
        "comment_limit": comment_limit,
        "created_at": fmt(datetime.now()),
    }

    queries_collection.insert_one(query_data)

    ops = []
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        print(f"Searching r/{subreddit}...")
        posts = subreddit.search(topic, time_filter=time_filter, limit=post_limit)

        for post in posts:
            content = (post.title.rstrip() + " " + post.selftext.rstrip())
            if topic.lower() not in content.lower():
                continue

            post_comments = []
            post_comments.extend(fetch_comments_for_post(post, comment_limit, "top"))
            post_comments.extend(fetch_comments_for_post(post, comment_limit, "controversial"))

            # Deduplicate comments
            seen = set()
            unique_comments = []
            for comment in post_comments:
                cid = comment["comment_id"]
                if cid not in seen:
                    seen.add(cid)
                    unique_comments.append(comment)

            doc = {
                "_id": post.id,
                "query_id": query_id,
                "title": post.title,
                "subreddit": sub,
                "topic": topic,
                "post_id": post.id,
                "post_content": content,
                "comments": unique_comments,
                "created_at": fmt(post.created_utc),
            }

            ops.append(UpdateOne({"_id": post.id}, {"$set": doc}, upsert=True))

        if ops:
            posts_collection.bulk_write(ops)
            print(f"[✔] Saved {len(ops)} posts with comments for query_id: {query_id}")
        else:
            print("[!] No matching posts/comments found.")

    return query_id