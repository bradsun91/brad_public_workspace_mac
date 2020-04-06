import time
import csv
from datetime import datetime, timezone
from urllib import parse as urlparse

from selenium import webdriver
import pandas as pd


from .base import RobotBase
from . import config as CONFIG


class sp500_news(RobotBase):

    def __init__(self):
        super().__init__(CONFIG.BROWSER_DRIVER_PATH)

    def run(self):
        self._maximize_window()
        self._fetch_news()

    def _fetch_news(self):
        output_path = CONFIG.OUTPUT_PATH.format(
            str(datetime.today().strftime('%Y-%m-%d')))

        with open(output_path, 'w', newline='', encoding='utf-8') as output:
            fieldnames = ['ticker', 'title', 'public_time', 'url', 'content']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            output.flush()

            # get search url for each ticker
            for ticker, search_url in self._search_page_url_generator():
                # fetch latest news for each ticker
                try:
                    results = self._fetch_ticker_news(ticker, search_url)
                    writer.writerows(results)
                    output.flush()
                    print('fetch {count} news for {ticker}'.format(
                        count=len(results),
                        ticker=ticker
                    ))
                except Exception as e:
                    print(str(e))
                    continue

    def _search_page_url_generator(self):
        search_url_pattern = urlparse.urljoin(CONFIG.BASE_URL,
                                              'market-activity/stocks/{ticker}/news-headlines')
        sp500 = pd.read_csv(CONFIG.SP500_TICKERS_PATH)
        sp500_list = sp500['symbol'].tolist()
        for ticker in sp500_list:
            search_url = search_url_pattern.format(ticker=ticker)
            yield ticker, search_url

    def _fetch_ticker_news(self, ticker, search_url):
        results = []
        self._load_url(search_url)
        self._sleep()
        search_bs = self._get_bs()
        # obtain all links of news on the page
        news_links = search_bs.select('.quote-news-headlines__link')
        for news_link in news_links:
            try:
                news_url = urlparse.urljoin(CONFIG.BASE_URL, news_link['href'])
                self._load_url(news_url)
                self._sleep()

                news_bs = self._get_bs()
                title = self._text_filter(news_bs.select_one(
                    '.article-header__headline').text)

                public_time_str = news_bs.select_one(
                    '.timestamp__date')['datetime']
                public_time = datetime.strptime(
                    public_time_str, '%Y-%m-%dT%H:%M:%S%z')

                content_paragraphs = news_bs.select(
                    '.body__content>p:not(.body__disclaimer)')
                content = [
                    content_paragraph.text for content_paragraph in content_paragraphs
                ]
                content = self._text_filter(' '.join(content))

                now = datetime.now(timezone.utc)
                time_diff = now - public_time
                if (time_diff.days < CONFIG.NEWS_RECENT_DAYS):
                    results.append({
                        'ticker': ticker,
                        'title': title,
                        'public_time': public_time.timestamp(),
                        'url': news_url,
                        'content': content
                    })
                else:
                    break
            except Exception as e:
                print(str(e))
                continue

        return results

    def _text_filter(self, text):
        return text.replace(",", " ").replace("\n", " ").replace('\r', ' ').replace('\\', '').replace('\t', ' ').strip()

    def _sleep(self):
        time.sleep(CONFIG.SLEEP_SECONDS)
