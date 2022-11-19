import logging, time, os, re, json
import requests
import feedparser
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class FeedScraper:
    def __init__(self, db_path, feeds_path, keywords_path):
        self.db_path = db_path
        self.urls_path = feeds_path
        self.keywords_path = keywords_path
        self._initLogging()
        self._initDataFrame(self.db_path)
        self._initFeeds(self.urls_path)
        self._initKeywords(self.keywords_path)
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
        self.update_db = False

    def _initLogging(self):
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(format=format, level=logging.INFO)

    def _initDataFrame(self, path):
        if os.path.isfile(path):
            self.df = pd.read_pickle(path)
            logging.info("Read database from '{}' with {} entries".format(path, len(self.df.index)))
        else:
            logging.info("Create empty database '{}'".format(path))
            columns = []
            columns.append('source')
            columns.append('id')
            columns.append('title')
            columns.append('published')
            columns.append('author')
            columns.append('link')
            self.df = pd.DataFrame(columns=columns)

    def _initFeeds(self, path):
        self.feeds = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    url = line.rstrip('\n')
                    if url[-1] != '/':
                        url += "/"
                    self.feeds.append(self.Feed(url))
                logging.info("Read feeds from '{}' with {} entries".format(path, len(self.feeds)))
        except EnvironmentError as e:
            logging.error(e)

    def _initKeywords(self, path):
        self.keywords = {}
        try:
            with open(path, 'r') as f:
                self.keywords = json.load(f)
                logging.info("Read keywords from '{}' with {} entries".format(path, len(self.keywords)))
        except json.JSONDecodeError as e:
            logging.error("JSON decoder errer when parsing file '{}': {}".format(path, e))
        except EnvironmentError as e:
            logging.error(e)

    def _isKeywordInSenctence(self, sentence, keyword):
        match = re.search(fr"\b{keyword}\b", sentence, re.IGNORECASE | re.MULTILINE)
        if match != None:
            return True
        else:
            return False

    def _fetchAndProcessFeed(self, feed):
        try:
            headers = {}
            if len(feed.etag):
                headers["ETag"] = feed.etag
            if len(feed.modified):
                headers["If-Modified-Since"] = feed.modified
            logging.debug("Request headers: '{}'".format(headers))

            r = requests.get(feed.url, headers=headers, timeout=10)

            if "etag" in r.headers:
                feed.etag = r.headers["etag"]
                logging.debug("Server returned etag: '{}'".format(feed.etag))
            if "last-modified" in r.headers:
                feed.modified = r.headers["last-modified"]
                logging.debug("Server returned last-modified: '{}'".format(feed.modified))

            if r.status_code == 304:
                logging.info(f"Request to {feed.url} returned status code {r.status_code} Not Modified")
            elif r.status_code != 200:
                logging.error(f"Request to {feed.url} returned status code {r.status_code}")
            else:
                d = feedparser.parse(r.content)
                logging.info("Fetched feed from url '{}' with {} entries.".format(feed.url, len(d.entries)))
                if 'entries' in d:
                    for entry in d.entries:
                        self._processFeedEntry(feed, entry)
        except Exception as e:
            logging.error("Failed to get the feeds from '{}': {}".format(feed.url, e))

    def _processFeedEntry(self, feed, entry):
        if 'title' not in entry:
            logging.error("Failed to add entry due to missing key 'title'")
        elif 'published_parsed' not in entry:
            logging.error("Failed to add entry due to missing key 'published_parsed'")
        elif 'id' not in entry:
            logging.error("Failed to add entry due to missing key 'id'")
        else:
            # Check if the entry is already in the database, add entry if not
            if ((self.df['source'] == feed.url) & (self.df['id'] == entry.id)).any() == False:
                data = {}
                data['source'] = [feed.url]
                data['id'] = [entry.id]
                data['title'] = [entry.title]
                data['published'] = [pd.Timestamp(time.mktime(entry.published_parsed), unit='s')]
                if 'author' in entry:
                    data['author'] = [entry.author]
                if 'link' in entry:
                    data['link'] = [entry.link]

                sentiment = self.sia.polarity_scores(entry.title)['compound']

                keywords_found = self._findCoinsInSentence(entry.title)
                for keyword in keywords_found:
                    data[keyword] = [sentiment]

                df_new = pd.DataFrame(data)
                self.df = pd.concat([self.df, df_new], ignore_index=True)

                self.update_db = True

                logging.info("New entry added to database: '{}', '{}', '{}', '{}', '{}'".format(feed.url, entry.title, entry.published, keywords_found, sentiment))
            else:
                logging.debug("Entry already in database: '{}', '{}', '{}'".format(feed.url, entry.title, entry.published))

    def _findCoinsInSentence(self, sentence):
        keywords_found = []
        for keyword in self.keywords:
            found = False
            if self._isKeywordInSenctence(sentence, keyword):
                found = True
            else:
                for tag in self.keywords[keyword]:
                    if self._isKeywordInSenctence(sentence, tag):
                        found = True
                        break
            if found:
                keywords_found.append(keyword)
        return keywords_found

    def _calc_interval(self, interval):
        if interval is None:
            return None
        else:
            if type(interval) is int:
                logging.info(f"Interval given as integer: '{interval}'")
                return interval
            elif type(interval) is str:
                logging.info(f"Interval given as string: '{interval}'")
                valid_units = ['s', 'm', 'h', 'd']
                conversions = [1, 60, 3600, 86400]
                unit = interval[-1]
                if unit not in valid_units:
                    raise ValueError(f"Invalid unit for interval given: '{unit}', valid units are: {valid_units}")
                value_str = interval[0:-1]
                try:
                    value = int(value_str)
                    value *= conversions[valid_units.index(unit)]
                except ValueError:
                    raise ValueError(f"Invalid integer value for interval given: '{value_str}'")
                logging.info(f"Interval of {value}s will be used for scraping")
                return value

    def scrape(self, interval=None):
        interval = self._calc_interval(interval)

        while True:
            last_scape_start = time.time()
            for feed in self.feeds:
                self._fetchAndProcessFeed(feed)
                if self.update_db:
                    self.df.to_pickle(self.db_path)
                    self.df.to_csv('db.csv')
                    logging.debug("Database file '{}' updated".format(self.db_path))
                    self.update_db = False

            if interval is None:
                break
            else:
                # Calculate start of next scrape
                time_delay = max(0, interval - (time.time() - last_scape_start))
                next_update = time.localtime(time.time() + time_delay)
                logging.info("Sleep until {:02d}:{:02d}:{:02d} for next update of feeds".format(next_update.tm_hour, next_update.tm_min, next_update.tm_sec))
                time.sleep(time_delay)

    class Feed:
        def __init__(self, url):
            self.url = url
            self.etag = ""
            self.modified = ""

if __name__ == '__main__':
    scraper = FeedScraper('db.pkl', 'feeds.txt', 'keywords.json')
    scraper.scrape('10m')
