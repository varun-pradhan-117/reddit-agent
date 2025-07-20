import requests
import requests.auth
from dotenv import load_dotenv
import os
load_dotenv(override=True)
reddit_app_id = os.getenv('REDDIT_APP_ID')
reddit_app_secret = os.getenv('REDDIT_APP_SECRET')
reddit_username = os.getenv('REDDIT_USERNAME')
reddit_password = os.getenv('REDDIT_PASSWORD')
client_auth=requests.auth.HTTPBasicAuth(reddit_app_id, reddit_app_secret)
post_data = {"grant_type": "password", "username": reddit_username, "password": reddit_password}
headers = {"User-Agent": "RedditAgent/0.1 by BraiseTheSun"}
response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers)


