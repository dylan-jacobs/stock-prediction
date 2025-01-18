# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:54:03 2022

@author: dylan
"""

import pandas as pd
import time
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import webdriver_manager
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from alive_progress import alive_bar
import sys

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")
chrome_options.add_argument('headless')
chrome_options.add_experimental_option("excludeSwitches", ["disable-popup-blocking"])
driver = webdriver.Chrome(webdriver_manager.chrome.ChromeDriverManager().install(), options=chrome_options)

number_of_scrolldowns = 50
columns = ['Tickers', 'Times', 'Headlines']
parsed_news = []

def scroll_to_bottom_of_articles_container(driver):
    # scroll to bottom of lazy loaded list to get more articles. YUM!
    for i in range(number_of_scrolldowns):
        article_container = driver.find_element_by_xpath('//*[@id="maincontent"]/div[6]/div[2]/div[1]/mw-tabs/div[2]/div[1]/mw-scrollable-news-v2/div/div')
        articles = article_container.find_elements(By.CSS_SELECTOR,'.element.element--article')
        
        driver.execute_script("arguments[0].scrollIntoView();", articles[-1])
        time.sleep(0.1)
        yield

def get_parsed_news_data(tickers):
    for ticker in tickers:
        url = 'https://www.marketwatch.com/investing/stock/{}?mod=search_symbol'.format(ticker)

        page = requests.get(url)
        driver.get(url)
            
        articles = []
        
        with alive_bar(number_of_scrolldowns) as bar:
            for i in scroll_to_bottom_of_articles_container(driver):
                bar()
        
        # get updated html after scrolldown
        html = BeautifulSoup(driver.page_source, "html.parser")
        headline_table_of_elements = html.find(class_='column column--primary j-moreHeadlineWrapper')
        headline_table = html.find_all(class_='element element--article') # get all articles
        print("{}: num articles: {}".format(ticker, len(headline_table)))
        
        parsed_headline = ""
        parsed_date = None
        
        for headline in headline_table:
            if headline:
                container = headline.findChild(class_="article__headline")
                if container:
                    a = container.findChild('a')                
                    info = headline.findChild(class_="article__details")
                    
                    if info and a:
                        date_and_time = info.findChild(class_="article__timestamp")
                        if date_and_time:
                            try:
                                parsed_date = date_and_time['data-est']
                                parsed_headline = a.text.strip()
                            except: KeyError
            
            parsed_news.append([ticker, parsed_date, parsed_headline])
            
    return parsed_news

def print_out_data(parsed_news):
    table = pd.DataFrame(parsed_news, columns=columns)    
    analyzer = SentimentIntensityAnalyzer()
    scores = table['Headlines'].apply(analyzer.polarity_scores).tolist()
    
    table = table.join(pd.DataFrame(scores), rsuffix='_right')
    unique_tickers = table['Tickers'].unique().tolist()
    ticker_table = {name: table.loc[table['Tickers'] == name] for name in unique_tickers}
    
    values = []
    for ticker in tickers:
        dataframe = ticker_table[ticker]
        dataframe = dataframe.set_index('Tickers')
        dataframe = dataframe.drop(columns=['Headlines'])
        
        mean = round(dataframe['compound'].mean(), 4)
        values.append(mean)
        
        
    df = pd.DataFrame(list(zip(tickers, values)), columns=['Ticker', 'Mean Sentiment'])
    df.set_index('Ticker')
    df = df.sort_values('Mean Sentiment', ascending=False)
    print ('\n')
    print (df)

if __name__ == '__main__':
    tickers_string = input('Enter ticker(s) you want to analyze. If you don\'t enter anything, I will analyze a predetermined list of popular tickers.\n')
    if tickers_string != None and tickers_string != '':
        tickers = tickers_string.strip().replace(' ', '').split(',')
    else:
        tickers = ['AAPL', 'GOOG', 'AMZN', 'TSLA', 'SPYD', 'SPHD', 'MPW', 'DJD', 'VOO'] 
    
    parsed_news = get_parsed_news_data(tickers)
    print_out_data(parsed_news)
    sys.exit()

            



















