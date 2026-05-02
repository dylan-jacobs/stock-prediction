# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:47:46 2022

@author: dylan
"""
import yfinance
import numpy as np
import os
import pandas as pd
import backtesting
import multiprocessing
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps
import requests
import time
import random

HISTORY = '10y'
INTERVAL = '1d' # 1h: max 2y
TICKER = 'AAPL'

class PEGYStrategy(Strategy):

        buy_threshold = 1
        sell_threshold = 1
        pred_data = None

        def init(self):
            self.pegy = self.I(lambda: self.pred_data, name='PEGY Predictions')

        def next(self):
            pegy = self.pegy[-1]

            if pegy <= 0:
                if self.position.is_long:
                    self.position.close()
                return # do not buy if pegy negative
            
            if pegy < self.buy_threshold:
                if not self.position.is_long:
                    self.buy()
            elif pegy > self.sell_threshold:
                if self.position.is_long:
                    self.position.close()

def load_close_data(ticker, period):
    data = yfinance.download(ticker, period=period, interval=INTERVAL)
    close_data = data[['Close']]
    open_data = data[['Open']]
    date_data = data.index.to_frame(index=False, name='Date')  # Extract Date from the index
    high_data = data[['High']]
    low_data = data[['Low']]
    volume = data[['Volume']]
    return open_data, close_data, date_data, high_data, low_data, volume

def get_macrotrends(ticker, company, metric="pe-ratio"):

    url = f"https://www.macrotrends.net/stocks/charts/{ticker}/{company}/{metric}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    response = requests.get(url, headers=headers)
    tables = pd.read_html(str(response.text))
    
    df = tables[0]
    df.columns = ["Date", "Price", "EPS", "PE_Ratio"]

    # Clean up
    df["Date"]     = pd.to_datetime(df["Date"])
    df["PE_Ratio"] = pd.to_numeric(df["PE_Ratio"].astype(str).str.replace(r"[$%,]", "", regex=True), errors="coerce")
    df["EPS"]      = pd.to_numeric(df["EPS"].astype(str).str.replace(r"[$%,]", "", regex=True), errors="coerce")
    df["Price"]    = pd.to_numeric(df["Price"].astype(str).str.replace(r"[$%,]", "", regex=True), errors="coerce")

    df = df.set_index("Date").sort_index()
    return df

def calculate_pegy(ticker, period):
    stock = yfinance.Ticker(ticker)
    history = stock.history(period=period)
    df = history[["Close"]].copy()

    # Dividends
    dividends = stock.dividends

    if dividends.index.tz is not None:
        dividends.index = dividends.index.tz_localize(None)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    if dividends.empty:
        df["Rolling_Annual_Div"] = 0
        df["Dividend_Yield_%"]   = 0
    else:
        # Sum last 4 dividend payments (quarterly payers = 1 year)
        # More accurate than rolling 252 days
        ttm_div = dividends.rolling(4).sum()

        # Reindex to daily and forward fill
        df["Rolling_Annual_Div"] = ttm_div.reindex(df.index, method="ffill")
        df["Dividend_Yield_%"]   = 100 * df["Rolling_Annual_Div"] / df["Close"]
    
    # EPS
    ttm_eps_table = get_macrotrends(ticker, ticker)
    ttm_eps = ttm_eps_table["EPS"]
    ttm_eps.sort_index()
    ttm_eps.index = ttm_eps.index.tz_localize(df.index.tz) if ttm_eps.index.tz is None else ttm_eps.index
    df['EPS'] = ttm_eps.reindex(df.index, method='ffill')

    # PE ratio
    df['PE'] = df['Close'] / df['EPS']

    # Growth
    growth = ttm_eps.pct_change(4) * 100
    df['EPS_Growth_%'] = growth.reindex(df.index, method='ffill')
    df['PEGY'] = df['PE'] / (df['EPS_Growth_%'] + df['Dividend_Yield_%'])
    print(df)
    return df

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # Pretend to be a browser
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    tables = pd.read_html(url, storage_options={"User-Agent": headers["User-Agent"]})
    df = tables[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df['Symbol'].to_list()

def backtest_strategy(df, predictions, graph=False, verbose=False):

    split = int(0.7 * len(df)) # for testing
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    
    PEGYStrategy.pred_data = predictions[:split].flatten()
    train_bt = Backtest(train_df, PEGYStrategy, cash=10000, commission=0, finalize_trades=True)
    buy_thresholds = [float(round(i, 2)) for i in np.arange(0.6, 2, 0.1)]
    sell_thresholds = [float(round(i, 2)) for i in np.arange(0.6, 2, 0.1)]
    stats, heatmap = train_bt.optimize(
        buy_threshold=buy_thresholds,
        sell_threshold=sell_thresholds, 
        constraint=lambda p: p.buy_threshold < p.sell_threshold, # buy threshold must be less than sell threshold
        maximize='Return [%]',
        max_tries=None,
        random_state=42,
        return_heatmap=True,
        return_optimization=False
    )
    if graph:
        plot_heatmaps(heatmap, agg='mean')
        train_bt.plot()

    # get optimal parameters
    optimal_buy_threshold = stats.at['_strategy'].buy_threshold
    optimal_sell_threshold = stats.at['_strategy'].sell_threshold

    if verbose:
        print(f'Best buy threshold: {optimal_buy_threshold}, Best sell threshold: {optimal_sell_threshold}')
    
    # test on test set
    PEGYStrategy.buy_threshold = optimal_buy_threshold
    PEGYStrategy.sell_threshold = optimal_sell_threshold
    
    PEGYStrategy.pred_data = predictions[split:].flatten()
    test_bt = Backtest(test_df, PEGYStrategy, cash=10000, commission=0, finalize_trades=True)
    stats = test_bt.run()
    if graph:
        test_bt.plot()
    
    if verbose:
        print(stats)

    return optimal_buy_threshold, optimal_sell_threshold, stats['Return [%]'], stats['Buy & Hold Return [%]']

def test_pegy_strategy(ticker=TICKER, graph=False, verbose=False):
    close_col = 'Close'
    # get close data
    open_data, close_data, dates, highs, lows, volume = load_close_data(ticker, HISTORY)
    data = pd.DataFrame(close_data.values, columns=[close_col])

    data['Date'] = dates.values.ravel()
    data['Open'] = open_data.values
    data['High'] = highs.values
    data['Low'] = lows.values
    data['Volume'] = volume.values
    
    # add pegy
    pegy_df = pd.DataFrame()
    while pegy_df.empty:
        try:
            pegy_df = calculate_pegy(ticker, HISTORY)
        except Exception as e:
            print(f'Error fetching PEGY ratio for ticker {ticker}. Trying again...')
            time.sleep(20)
    pegy = pegy_df['PEGY']
    data['PEGY'] = pegy.values

    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
        
    # extract dates
    dates = data['Date']
    pegy = data['PEGY'].values.flatten()

    close_data = data[close_col]

    # backtest
    df = pd.DataFrame({
        'Open': data['Open'].values.flatten(),
        'High': data["High"].values.flatten(),
        'Low': data["Low"].values.flatten(),
        'Close': data['Close'].values.flatten(),
        'Volume': data["Volume"].values.flatten()
    }, index=pd.DatetimeIndex(dates.values.flatten()))
    optimal_buy_threshold, optimal_sell_threshold, pegy_return, bnh_return = backtest_strategy(df, data['PEGY'].values.flatten(), graph=graph, verbose=verbose)

    return pegy[-1], optimal_buy_threshold, optimal_sell_threshold, pegy_return, bnh_return

def test_all_tickers():
    tickers = get_sp500_tickers()
    tickers = random.sample(tickers, 15)
    tickers = ['T']
    results = []
    for ticker in tickers:
        _, _, _, pegy_return, bnh_return = test_pegy_strategy(ticker, graph=True, verbose=True)
        results.append({
            "Ticker": ticker,
            "PEGY_Return_%": pegy_return,
            "BNH_Return_%": bnh_return
        })
        time.sleep(20)
        
    df = pd.DataFrame(results).set_index("Ticker")
    print(df)
    
    pegy_total_returns = df["PEGY_Return_%"].sum()
    bnh_total_returns = df["BNH_Return_%"].sum()

    print(f'PEGY strategy total returns: {pegy_total_returns}')
    print(f'Buy and hold total returns: {bnh_total_returns}')

def main():
    test_all_tickers()

if __name__=='__main__':
    main()
