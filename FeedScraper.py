import logging, time, os, re, json
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
        d = feedparser.parse(feed.url, etag=feed.etag, modified=feed.modified)

        if 'etag' in d:
            feed.etag = d.etag
        if 'modified' in d:
            feed.modified = d.modified
        if 'entries' in d:
            logging.info("Fetched feed from url '{}' with {} entries.".format(feed.url, len(d.entries)))
            logging.debug("Server returned ETag: '{}', Modified: '{}'".format(feed.etag, feed.modified))

            for entry in d.entries:
                self._processFeedEntry(feed, entry)
    
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
    
    def scrape(self, interval):
        while True:
            for feed in self.feeds:
                self._fetchAndProcessFeed(feed)
                if self.update_db:
                    self.df.to_pickle(self.db_path)
                    self.df.to_csv('db.csv')
                    logging.debug("Database file '{}' updated".format(self.db_path))
                    self.update_db = False

            if interval >= 0:
                time_delay = 60 * interval
                next_update = time.localtime(time.time() + time_delay)
                logging.info("Sleep until {:02d}:{:02d} for next update of feeds".format(next_update.tm_hour, next_update.tm_min))
                time.sleep(time_delay)
            else:
                break
    
    class Feed:
        def __init__(self, url):
            self.url = url
            self.etag = ""
            self.modified = ""

if __name__ == '__main__':
    scraper = FeedScraper('db.pkl', 'feeds.txt', 'keywords.json')
    scraper.scrape(10)