import requests
import random
from bs4 import BeautifulSoup
import pandas as pd
import re
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
    config.browser_user_agent = user_agent
    config.request_timeout = 20
    header = {'User-Agent': user_agent}

# Read in Alerts file and create empty lists for storing values
# read_file = pd.read_csv('EnterpriseRisksList.csv', encoding='utf-8')
# read_file['ENTERPRISE_RISK_ID'] = pd.to_numeric(read_file['ENTERPRISE_RISK_ID'], downcast='integer', errors='coerce')
# read_file.columns = read_file.columns.str.strip()

# read in encoded alerts and create empty lists for storing values
read_file = pd.read_csv('EnterpriseRisksListEncoded.csv', encoding='utf-8')
read_file['ENTERPRISE_RISK_ID'] = pd.to_numeric(read_file['ENTERPRISE_RISK_ID'], downcast='integer', errors='coerce')

def process_encoded_search_terms(term):
    try:
        term_str = str(term)
        byte_rep = term_str.encode('utf-8')
        decoded_rep = byte_rep.decode('utf-8')
        return decoded_rep
    except Exception as e:
        print(f"Error processing term {term}: {e}")
        return None

# apply the function to each element
read_file['SEARCH_TERMS'] = read_file['ENCODED_TERMS'].apply(process_encoded_search_terms)

search_terms = []
title = []
published = []
link = []
domain = []
source = []
summary = []
keywords = []
sentiment = []
polarity = []

# source_df = pd.read_csv('sources.csv')
# source_list = source_df.source_name.tolist()

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
                # enterprise_risk.extend(read_file.ENTERPRISE_RISK[i])
                title.extend([title_text])
                search_terms.extend([term])
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
        sentiment.extend(['negative'])
        polarity.extend(['-' + str(neg)])  # appending the news that satisfies this condition
    elif neg < pos:
        sentiment.extend(['positive'])
        polarity.extend(['+' + str(pos)])
    else:
        sentiment.extend(['neutral'])
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
     'SENTIMENT': sentiment,
     'POLARITY': polarity
     })

print('Created sentiments')

joined_df = pd.merge(alerts, read_file, on='SEARCH_TERMS', how='left')
final_df = joined_df[['ENTERPRISE_RISK_ID', 'TITLE', 'SUMMARY', 'KEYWORDS', 'PUBLISHED_DATE', 'LINK',
                      'SOURCE', 'SOURCE_URL', 'SENTIMENT', 'POLARITY']]
final_df = final_df.sort_values(by='PUBLISHED_DATE', ascending=False)

# Add timestamp of the last run
final_df['LAST_RUN_TIMESTAMP'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Define output directory and file
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'online_sentiment/output')
os.makedirs(output_dir, exist_ok=True)

# Paths for CSV files
main_csv_path = os.path.join(output_dir, 'enterprise_risks_online_sentiment.csv')
archive_csv_path = os.path.join(output_dir, 'enterprise_risks_sentiment_archive.csv')

# Step 5: Detect file encoding and convert to UTF-8
if os.path.exists(main_csv_path):
    with open(main_csv_path, 'rb') as f:
        detected_encoding = chardet.detect(f.read())['encoding']
    
    if detected_encoding and detected_encoding.lower() != 'utf-8':
        print(f"Converting {main_csv_path} from {detected_encoding} to UTF-8...")
        temp_df = pd.read_csv(main_csv_path, encoding=detected_encoding)
        temp_df.to_csv(main_csv_path, index=False, encoding='utf-8')

# Read existing main CSV if it exists
if os.path.exists(main_csv_path):
    existing_main_df = pd.read_csv(main_csv_path, parse_dates=['PUBLISHED_DATE'], encoding='utf-8')
else:
    existing_main_df = pd.DataFrame()

# Combine existing data with new data
combined_df = pd.concat([existing_main_df, final_df], ignore_index=True)

# Rolling 6-months filter
six_months_ago = dt.datetime.now() - dt.timedelta(days=6*30)

# Convert PUBLISHED_DATE to datetime
combined_df['PUBLISHED_DATE'] = pd.to_datetime(combined_df['PUBLISHED_DATE'], errors='coerce')

# Split into recent and old data
recent_df = combined_df[combined_df['PUBLISHED_DATE'] >= six_months_ago].copy()
old_df = combined_df[combined_df['PUBLISHED_DATE'] < six_months_ago].copy()

# Save recent data back to the main CSV
recent_df_sorted = recent_df.sort_values(by='PUBLISHED_DATE', ascending=False)
recent_df_sorted.to_csv(main_csv_path, index=False, encoding='utf-8')

print(f"Main CSV updated with data from the last 6 months: {main_csv_path}")

# Prepare old data for archiving with specified columns
archive_columns = ['ENTERPRISE_RISK_ID', 'TITLE', 'PUBLISHED_DATE', 'LINK', 'SENTIMENT', 'POLARITY', 'LAST_RUN_TIMESTAMP']
archive_data = old_df[archive_columns].copy()

# Append old data to the archive CSV
if os.path.exists(archive_csv_path):
    archive_data.to_csv(archive_csv_path, mode='a', index=False, header=False, encoding='utf-8')
else:
    archive_data.to_csv(archive_csv_path, mode='w', index=False, header=True, encoding='utf-8')

print(f"Archived data older than 6 months to: {archive_csv_path}")


# joined_df = alerts.merge(read_file, on='SEARCH_TERMS', how='left')
# final_df = joined_df[['ENTERPRISE_RISK_ID', 'TITLE', 'SUMMARY', 'KEYWORDS', 'PUBLISHED_DATE', 'LINK',
#                       'SOURCE', 'SOURCE_URL', 'SENTIMENT', 'POLARITY']]
# final_df = final_df.sort_values(by='PUBLISHED_DATE', ascending=False)

# # add timestamp of the last run
# final_df['LAST_RUN_TIMESTAMP'] = dt.datetime.now()
# final_df['LAST_RUN_TIMESTAMP'] = pd.to_datetime(final_df['LAST_RUN_TIMESTAMP'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

# # define output directory and file
# script_dir = os.path.dirname(os.path.abspath(__file__))
# output_dir = os.path.join(script_dir, 'online_sentiment/output')
# os.makedirs(output_dir, exist_ok=True)

# # add paths for csv files
# main_csv_path = os.path.join(output_dir, 'enterprise_risks_online_sentiment.csv')
# archive_csv_path = os.path.join(output_dir, 'enterprise_risks_sentiment_archive.csv')

# # read existing main CSV if it exists
# if os.path.exists(main_csv_path):
#     existing_main_df = pd.read_csv(main_csv_path, parse_dates=['PUBLISHED_DATE'], infer_datetime_format=True, encoding='utf-8', errors='replace')
# else:
#     existing_main_df = pd.DataFrame()

# # combine existing data with new data
# combined_df = pd.concat([existing_main_df, final_df], ignore_index=True)

# # rolling 6 mos
# six_months_ago = dt.datetime.now() - dt.timedelta(days=6*30)  # Approximation of 6 months only

# # split into recent and old data
# combined_df['PUBLISHED_DATE'] = pd.to_datetime(combined_df['PUBLISHED_DATE'], errors='coerce')
# recent_df = combined_df[combined_df['PUBLISHED_DATE'] >= six_months_ago].copy()
# old_df = combined_df[combined_df['PUBLISHED_DATE'] < six_months_ago].copy()

# # save recent data back to the main CSV, sort before write
# recent_df_sorted = recent_df.sort_values(by='PUBLISHED_DATE', ascending=False)
# recent_df_sorted.to_csv(main_csv_path, index=False, encoding='utf-8')

# print(f"Main CSV updated with data from the last 6 months: {main_csv_path}")

# # prepare old data for archiving with specified columns
# archive_columns = ['ENTERPRISE_RISK_ID', 'TITLE', 'PUBLISHED_DATE', 'LINK', 'SENTIMENT', 'POLARITY', 'LAST_RUN_TIMESTAMP']
# archive_data = old_df[archive_columns].copy()

# # append old data to the archive CSV
# if os.path.exists(archive_csv_path):
#     archive_data.to_csv(archive_csv_path, mode='a', index=False, header=False)
# else:
#     archive_data.to_csv(archive_csv_path, mode='w', index=False, header=True)

# print(f"Archived data older than 6 months to: {archive_csv_path}")



