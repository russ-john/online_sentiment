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
now_str = now.strftime('%m-%d-%Y')  # Changed variable name to avoid confusion
yesterday = now - dt.timedelta(days=1)
yesterday_str = yesterday.strftime('%m-%d-%Y')

# Setup requests configurations
nltk.download('punkt')

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

# Select a random user agent for the session
user_agent = random.choice(user_agent_list)
config.browser_user_agent = user_agent
config.request_timeout = 20

header = {'User-Agent': user_agent}

# Read in encoded alerts and create empty lists for storing values
read_file = pd.read_csv('EmergingRisksListEncoded.csv', encoding='utf-8')
read_file['EMERGING_RISK_ID'] = pd.to_numeric(read_file['EMERGING_RISK_ID'], downcast='integer', errors='coerce')

def process_encoded_search_terms(term):
    try:
        term_str = str(term)
        byte_rep = term_str.encode('utf-8')
        decoded_rep = byte_rep.decode('utf-8')
        return decoded_rep
    except Exception as e:
        print(f"Error processing term {term}: {e}")
        return None

# Apply the function to each element
read_file['SEARCH_TERMS'] = read_file['ENCODED_TERMS'].apply(process_encoded_search_terms)

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

print('Created dataframes')

url_start = 'https://news.google.com/rss/search?q={'
url_end = '}%20when%3A1d'  # Grabs search results during the day

# Grab Google links
for i, term in enumerate(read_file.SEARCH_TERMS):
    req = requests.get(url=url_start + term + url_end, headers=header)
    soup = BeautifulSoup(req.text, 'xml')
    for item in soup.find_all("item"):
        source_text = item.source.text
        title_text = item.title.text
        encoded_url = item.link.text
        interval_time = 5
        try:
            decoded_url = new_decoderv1(encoded_url, interval=interval_time)
            if decoded_url.get("status"):
                title.append(title_text)
                search_terms.append(term)
                regex_pattern = re.compile('(https?):((|(\\\\))+[\w\d:#@%;$()~_?\+-=\\\.&]*)')
                source_search = regex_pattern.search(str(item.source))
                if source_search:
                    domain.append(source_search.group(0))
                else:
                    domain.append(None)
                source.append(source_text)
                pub_text = parser.parse(item.pubDate.text)
                published.append(pub_text.date())
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
        summary.append(None)
        keywords.append(None)
        sentiments.append(None)
        polarity.append(None)
        continue
    summary.append(article.summary)
    keywords.append(article.keywords)
    analyzer = SentimentIntensityAnalyzer().polarity_scores(article.summary)
    neg = analyzer['neg']
    neu = analyzer['neu']
    pos = analyzer['pos']
    comp = analyzer['compound']
    if neg > pos or neg == -1:
        sentiments.append('negative')
        polarity.append('-' + str(neg))  # appending the news that satisfies this condition
    elif neg < pos:
        sentiments.append('positive')
        polarity.append('+' + str(pos))
    else:
        sentiments.append('neutral')
        polarity.append(str(neu))

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
     'POLARITY': polarity}
)

print('Created sentiments')

joined_df = alerts.merge(read_file, on='SEARCH_TERMS', how='left')
final_df = joined_df[['EMERGING_RISK_ID', 'TITLE', 'SUMMARY', 'KEYWORDS', 'PUBLISHED_DATE', 'LINK',
                      'SOURCE', 'SOURCE_URL', 'SENTIMENT', 'POLARITY']]
final_df = final_df.sort_values(by='PUBLISHED_DATE', ascending=False)

# add timestamp of the last run
final_df['LAST_RUN_TIMESTAMP'] = dt.datetime.now()
final_df['LAST_RUN_TIMESTAMP'] = pd.to_datetime(final_df['LAST_RUN_TIMESTAMP'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# define output directory and file
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'online_sentiment/output')
os.makedirs(output_dir, exist_ok=True)

# add paths for csv files
main_csv_path = os.path.join(output_dir, 'emerging_risks_online_sentiment.csv')
archive_csv_path = os.path.join(output_dir, 'emerging_risks_sentiment_archive.csv')

# read existing main CSV if it exists
if os.path.exists(main_csv_path):
    existing_main_df = pd.read_csv(main_csv_path, parse_dates=['PUBLISHED_DATE'], infer_datetime_format=True)
else:
    existing_main_df = pd.DataFrame()

# combine existing data with new data
combined_df = pd.concat([existing_main_df, final_df], ignore_index=True)

# rolling 6 mos
six_months_ago = dt.datetime.now() - dt.timedelta(days=6*30)  # Approximation of 6 months only

# Split into recent and old data
combined_df['PUBLISHED_DATE'] = pd.to_datetime(combined_df['PUBLISHED_DATE'], errors='coerce')
recent_df = combined_df[combined_df['PUBLISHED_DATE'] >= six_months_ago].copy()
old_df = combined_df[combined_df['PUBLISHED_DATE'] < six_months_ago].copy()

# Save recent data back to the main CSV
# Sort by 'PUBLISHED_DATE' descending before saving
recent_df_sorted = recent_df.sort_values(by='PUBLISHED_DATE', ascending=False)
recent_df_sorted.to_csv(main_csv_path, index=False)

print(f"Main CSV updated with data from the last 6 months: {main_csv_path}")

# Prepare old data for archiving with specified columns
archive_columns = ['EMERGING_RISK_ID', 'TITLE', 'PUBLISHED_DATE', 'LINK', 'SENTIMENT', 'POLARITY', 'LAST_RUN_TIMESTAMP']
archive_data = old_df[archive_columns].copy()

# Append old data to the archive CSV
if os.path.exists(archive_csv_path):
    archive_data.to_csv(archive_csv_path, mode='a', index=False, header=False)
else:
    archive_data.to_csv(archive_csv_path, mode='w', index=False, header=True)

print(f"Archived data older than 6 months to: {archive_csv_path}")
