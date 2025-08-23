import statistics
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List
from langchain.prompts import ChatPromptTemplate

from reddit_workers.db import posts_collection, queries_collection
from llm_wrappers import DeepSeekChat


# Model for per-comment sentiment output
class SentimentOutput(BaseModel):
    score: int = Field(..., description="Score from -2 to 2 based on sentiment toward the topic")
    explanation: str = Field(..., description="Explanation for the score")


def analyze_query(query_id: str):
    posts = list(posts_collection.find({"query_id": query_id}))

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
         If the statement is not related to the topic, return a score of 0 with an explanation that it is neutral.
         """),
        ("user", "Topic: {topic}\nStatement: {statement}")
    ])

    # Per-post summary prompt
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are a post summarizer. You summarize the sentiment of all the comments in a post towards the topic.
         You will receive a list of comments with the average sentiment score.
         Provide a concise summary of the overall sentiment towards the topic.
         """),
        ("user", "Topic: {topic}\nAverage Sentiment Score: {avg_score}\nComments: {comments}")
    ])

    # Overall topic summary prompt
    topic_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are a topic summarizer. You summarize the sentiment of all the comments towards a specific topic.
         Provide a thoughtful and insightful summary of the overall sentiment.
         """),
        ("user", "Topic: {topic}\nAverage Sentiment Score: {avg_score}\nComments: {comments}")
    ])

    ds = DeepSeekChat()
    ds_sentiment = ds.with_structured_output(SentimentOutput, method='json_schema')

    all_comments_text = []
    all_scores = []

    for post in posts:
        print(f"Analyzing post: {post['title']} in r/{post['subreddit']}")
        scores = []
        analyzed_comments = []

        for comment in post["comments"]:
            prompt = sentiment_prompt.invoke({
                "topic": post["topic"],
                "statement": comment["text"]
            })
            result = ds_sentiment.invoke(prompt)

            analyzed_comments.append({
                **comment,
                "sentiment_score": result.score,
                "sentiment_explanation": result.explanation
            })
            scores.append(result.score)
            all_scores.append(result.score)
            all_comments_text.append(comment["text"])

        avg_score = statistics.mean(scores) if scores else None

        prompt = summary_prompt.invoke({
            "topic": post["topic"],
            "avg_score": avg_score,
            "comments": "\n".join([f"{c['text']}" for c in analyzed_comments])
        })
        sentiment_summary = ds.invoke(prompt).content

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


