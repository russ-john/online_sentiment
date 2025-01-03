import requests
import random
import re
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser
from newspaper import Article
from newspaper import Config
import datetime as dt
import nltk
from googlenewsdecoder import new_decoderv1
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os


# Set dates for today and yesterday
now = dt.date.today()
now = now.strftime('%m-%d-%Y')
yesterday = dt.date.today() - dt.timedelta(days=1)
yesterday = yesterday.strftime('%m-%d-%Y')

# Setup requests configurations
nltk.download('punkt')

# header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)\
#                          Version/17.3.1 Safari/605.1.15'}

# Create a list of random user agents
user_agent_list = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)\
                           Version/17.3.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko)\
                           Chrome/83.0.4103.97 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)\
                           Chrome/83.0.4103.97 Safari/537.36'
]

config = Config()

for u_agent in user_agent_list:
    # Pick a random user agent
    user_agent = random.choice(user_agent_list)
# user_agent = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)\
#                            Version/17.3.1 Safari/605.1.15')

    config.browser_user_agent = user_agent
    config.request_timeout = 20

    header = {'User-Agent': user_agent}

# Read in Alerts file and create empty lists for storing values
# read_file = pd.read_csv('EmergingRiskAlerts.csv')
read_file = pd.read_csv('EmergingRisksList.csv', encoding='utf-8')
read_file['EMERGING_RISK_ID'] = pd.to_numeric(read_file['EMERGING_RISK_ID'], downcast='integer', errors='coerce')
read_file.columns = read_file.columns.str.strip()

search_terms = []
title = []
published = []
link = []
domain = []
source = []
summary = []
keywords = []
sentiments = []
polarity = []

# Create or open dataframe file and create source df
# try:
#     alerts_df = pd.read_csv("EmergingRisks.csv", engine='python')
#     alerts_df.PUBLISHED_DATE = pd.to_datetime(alerts_df.PUBLISHED_DATE)
# except FileNotFoundError:
#     alerts_df = pd.DataFrame(columns=['EMERGING_RISK_ID', 'alert_name', 'TITLE', 'SUMMARY', 'KEYWORDS', 'PUBLISHED_DATE', 'LINK',
#                                       'SOURCE', 'SOURCE_URL', 'SENTIMENT', 'POLARITY'])

# alerts_new_df = pd.DataFrame(columns=['EMERGING_RISK_ID', 'TITLE', 'SUMMARY', 'KEYWORDS', 'PUBLISHED_DATE'
#                                       , 'LINK', 'SOURCE', 'SOURCE_URL', 'SENTIMENT', 'POLARITY'])


source_df = pd.read_csv('sources.csv')
source_list = source_df.source_name.tolist()

print('Created dataframes')

url_start = 'https://news.google.com/rss/search?q={'
url_end = '}%20when%3A7d'  # Grabs search results over the days of the week


# Grab Google links
for i, term in enumerate(read_file.SEARCH_TERMS):
    req = requests.get(url=url_start + term + url_end, headers=header)
    soup = BeautifulSoup(req.text, 'xml')
    for item in soup.find_all("item"):
        source_text = item.source.text
        title_text = item.title.text
        if source_text in source_list:
            encoded_url = item.link.text
            interval_time = 5
            try:
                decoded_url = new_decoderv1(encoded_url, interval=interval_time)
                if decoded_url.get("status"):
                    title.extend([title_text])
                    search_terms.extend([term])
                    # emerging_risk_id.extend(read_file.EMERGING_RISK_ID[i])
                    regex_pattern = re.compile('(https?):((|(\\\\))+[\w\d:#@%;$()~_?\+-=\\\.&]*)')
                    source_search = regex_pattern.search(str(item.source))
                    domain.extend([source_search.group(0)])
                    source.extend([source_text])
                    pub_text = parser.parse(item.pubDate.text)
                    published.extend([pub_text.date()])
                    decoded_url = decoded_url['decoded_url']
                    link.append(decoded_url)
                else:
                    print("Error:", decoded_url['message'])
            except Exception as e:
                print(f"Error occurred: {e}")

print('Created lists')

# Find article information
for article_link in link:
    article = Article(article_link, config=config)  # providing the link
    try:
        article.download()  # downloading the article
        article.parse()  # parsing the article
        article.nlp()  # performing natural language processing (nlp)
    except:
        pass
    summary.extend([article.summary])
    keywords.extend([article.keywords])
    analyzer = SentimentIntensityAnalyzer().polarity_scores(article.summary)
    neg = analyzer['neg']
    neu = analyzer['neu']
    pos = analyzer['pos']
    comp = analyzer['compound']
    if neg > pos or neg == -1:
        sentiments.extend(['negative'])
        polarity.extend(['-' + str(neg)])  # appending the news that satisfies this condition
    elif neg < pos:
        sentiments.extend(['positive'])
        polarity.extend(['+' + str(pos)])
    else:
        sentiments.extend(['neutral'])
        polarity.extend([str(neu)])

print('Length alert name: ', len(search_terms), ' Length Title: ', len(title), ' Length Link: ', len(link),
      ' Length KW: ', len(keywords))

alerts = pd.DataFrame(
    {'SEARCH_TERMS': search_terms,
     'TITLE': title,
     'SUMMARY': summary,
     'KEYWORDS': keywords,
     'PUBLISHED_DATE': published,
     'LINK': link,
     'SOURCE': source,
     'SOURCE_URL': domain,
     'SENTIMENT': sentiments,
     'POLARITY': polarity
     })

print('Created sentiments')

joined_df = alerts.merge(read_file, on='SEARCH_TERMS', how='left')
final_df = joined_df[['EMERGING_RISK_ID', 'ALERT_NAME', 'TITLE', 'SUMMARY', 'KEYWORDS', 'PUBLISHED_DATE', 'LINK',
                      'SOURCE', 'SOURCE_URL', 'SENTIMENT', 'POLARITY']]
final_df = final_df.sort_values(by='PUBLISHED_DATE', ascending=False)

# add timestamp of the last run
final_df['LAST_RUN_TIMESTAMP'] = pd.Timestamp.now()

# define output directory and file
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'online_sentiment/output')
os.makedirs(output_dir, exist_ok=True)

# save inside the repo
output_path = os.path.join(output_dir, 'emerging_risks_online_sentiment.csv')
final_df.to_csv(output_path, index=False)

print(f"DataFrame saved to {output_path}")

