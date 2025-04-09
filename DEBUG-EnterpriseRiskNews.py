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
import chardet
from urllib.parse import urlparse

# Set dates for today and yesterday
now = dt.date.today()
yesterday = now - dt.timedelta(days=1)

# Setup requests configurations
nltk.download('punkt')

# Create a list of random user agents
user_agent_list = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'
]

config = Config()
user_agent = random.choice(user_agent_list)
config.browser_user_agent = user_agent
config.request_timeout = 20
header = {'User-Agent': user_agent}

# # load existing dataset to avoid duplicate fetching
# script_dir = os.path.dirname(os.path.abspath(__file__))
# output_dir = os.path.join(script_dir, 'online_sentiment/output')
# main_csv_path = os.path.join(output_dir, 'enterprise_risks_online_sentiment.csv')

# if os.path.exists(main_csv_path):
#     existing_df = pd.read_csv(main_csv_path, usecols=lambda x: 'LINK' in x, encoding="utf-8")
#     existing_links = set(existing_df["LINK"].dropna().str.lower().str.strip())  # normalize existing links for efficient processing
# else:
#     existing_links = set()

# debug prefix
# encode-decode search terms
read_file = pd.read_csv('EnterpriseRisksListEncoded.csv', encoding='utf-8')
read_file['ENTERPRISE_RISK_ID'] = pd.to_numeric(read_file['ENTERPRISE_RISK_ID'], downcast='integer', errors='coerce')

def process_encoded_search_terms(term):
    try:
        encoded_number = int(term)
        byte_length = (encoded_number.bit_length() + 7) // 8
        byte_rep = encoded_number.to_bytes(byte_length, byteorder='little')
        decoded_text = byte_rep.decode('utf-8')
        return decoded_text
    except (ValueError, UnicodeDecodeError, OverflowError):
        return None

read_file['SEARCH_TERMS'] = read_file['ENCODED_TERMS'].apply(process_encoded_search_terms)

# prep lists to store new entries
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

# load filter_out_sources.csv file
filter_out_path = 'filter_out_sources.csv'
if os.path.exists(filter_out_path):
    filter_out_df = pd.read_csv(filter_out_path, encoding='utf-8')
    filtered_sources = set(filter_out_df.iloc[:, 0].dropna().str.lower().str.strip())  #only 1 column, use it.
else:
    filtered_sources = set()

# Grab Google links
url_start = 'https://news.google.com/rss/search?q={'
url_end = '}%20when%3A1d'

for term in read_file.SEARCH_TERMS.dropna():
    try:
        req = requests.get(url=url_start + term + url_end, headers=header)
        soup = BeautifulSoup(req.text, 'xml')
        for item in soup.find_all("item"):
            title_text = item.title.text.strip()
            encoded_url = item.link.text.strip()
            source_text = item.source.text.strip().lower()
            
            interval_time = 5
            decoded_url = new_decoderv1(encoded_url, interval=interval_time)

            if decoded_url.get("status"):
                decoded_url = decoded_url['decoded_url'].strip().lower()  # normalize link to check duplicates
                
                parsed_url = urlparse(decoded_url)
                domain_name = parsed_url.netloc.lower()

                # FILTER LOGIC SEQUENCE
                # 1. Valid domain extension only
                valid_extensions = ('.com', '.edu', '.org', '.net')
                if not any(domain_name.endswith(ext) for ext in valid_extensions):
                    print(f"Skipping {decoded_url} (Invalid domain extension)")
                    continue  # skip where domain extension is not valid

                # 2. Check if the source name is in filter-out list
                if source_text in filtered_sources:
                    print(f"Skipping article from {source_text} (Filtered source)")
                    continue  # skip if true

                # 3. Skip articles if the URL contains '/en/' (translated articles)
                if "/en/" in decoded_url:
                    print(f"Skipping {decoded_url} (Detected translated article)")
                    continue  # skip if true

                # if decoded_url in existing_links:
                #     continue  # skip if article was previously collected
                
                title.append(title_text)
                search_terms.append(term)
                source.append(source_text)
                link.append(decoded_url)
                
                #date has to work for deduping
                try:
                    published.append(parser.parse(item.pubDate.text).date())
                except (ValueError, TypeError):
                    published.append(None)
                    print(f"WARNING! Date Error: {item.pubDate.text}")

                regex_pattern = re.compile('(https?):((|(\\\\))+[\w\d:#@%;$()~_?\+-=\\\.&]*)')
                domain_search = regex_pattern.search(str(item.source))
                domain.append(domain_search.group(0) if domain_search else None) # prevent AttributeError: 'NoneType'
            else:
                print("Error:", decoded_url['message'])
    except requests.exceptions.RequestException as e:
        print(f"Request error for term {term}: {e}")

print('Created lists')

# Find article information
for article_link in link:
    article = Article(article_link, config=config)
    try:
        article.download()
        article.parse()
        article.nlp()
    except:
        pass
    summary.append(article.summary)
    keywords.append(article.keywords)
    analyzer = SentimentIntensityAnalyzer().polarity_scores(article.summary)
    comp = analyzer['compound']
    if comp <= -0.05:
        sentiments.append('negative')
        polarity.append(f'{comp}')
    elif -0.05 < comp < 0.05:
        sentiments.append('neutral')
        polarity.append(f'{comp}')
    elif comp >= 0.05:
        sentiments.append('positive')
        polarity.append(f'{comp}')

alerts = pd.DataFrame({
    'SEARCH_TERMS': search_terms,
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

# write
alerts['LAST_RUN_TIMESTAMP'] = dt.datetime.now().isoformat()
alerts.to_csv('DEBUG-EnterpriseRiskSample.csv', index=False, encoding='utf-8')
