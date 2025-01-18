# -*- coding: utf-8 -*-
"""
Created on Thu Dec 8 2022

@author: dylan

Strategy: semi hft?
Every five mins:
    iterate throught list of tickers (all from s&p500 index)
    find one that is at low bollinger index
    buy
    keep track of bought stocks and profits
        when owned stocks are at too high bollinger -> sell, or when price has risen by $1
        sell when lost > 5% of initial dinero
"""

import alpaca_trade_api as tradeapi
import time
from datetime import datetime
from pytz import timezone
from threading import Thread
import stock_predictor
import os
from dotenv import load_dotenv

# ALPACA API Info for fetching data, portfolio, etc. from Alpaca
BASE_URL = os.getenv("BASE_URL")
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TICKER = 'SPY'
print('hello')
print(BASE_URL)
print(ALPACA_API_KEY)
print(ALPACA_SECRET_KEY)
# Instantiate REST API Connection
api = tradeapi.REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url=BASE_URL, api_version='v2')

class NewThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)

    def run(self):
        if self._target != None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def getAlpacaQuote(ticker):
    try:
        quote = api.get_latest_quote(ticker)
    except:
        quote = None
    return quote

def placeBuyAlpacaOrder(ticker, amnt, price, orders, positions):
    return api.submit_order(symbol=ticker, qty=amnt, side='buy', type='limit', time_in_force='day', limit_price=price)    
        
def placeSellAlpacaOrder(ticker, amnt, price, limit_order):
    response = None
    
    if len(positions) > 0: # we own at least one stock
        if limit_order:
            response = api.submit_order(symbol=ticker, qty=amnt, side='sell', type='limit', time_in_force='day', limit_price=price)    
        else:
            response = api.submit_order(symbol=ticker, qty=amnt, side='sell', type='market', time_in_force='day')                
    else:
        print('We do not own this stock!')
    
    return response

def getTime():
    CURRENT_HOUR = int(datetime.now(timezone('US/Eastern')).strftime('%H'))
    CURRENT_MIN = int(datetime.now(timezone('US/Eastern')).strftime('%M'))
    CURRENT_SEC = int(datetime.now(timezone('US/Eastern')).strftime('%S'))
    return CURRENT_HOUR, CURRENT_MIN, CURRENT_SEC

def setAccountVars():
    try:
        account = api.get_account()
        positions = api.list_positions()
        orders = api.list_orders()
        CASH = float(account.cash)
        EQUITY = float(account.equity)
        PROFIT = 0
        if (len(positions) > 0):
            for pos in positions:
                PROFIT += float(pos.unrealized_pl)
        return positions, orders, CASH, EQUITY, PROFIT
    except Exception:
        time.sleep(5)
        return setAccountVars()

def buyPosition(ticker, qty, price):
    positions, orders, CASH, _, _ = setAccountVars()
            
    # buy now
    response = placeBuyAlpacaOrder(ticker, qty, price, orders, positions)
    
    # update info
    ORDER_ID = response.id
    positions, orders, CASH, _, _ = setAccountVars()
    times_to_delay = 0
    symbols = [p.symbol for p in positions]
    while (ticker not in symbols) and (times_to_delay <= 24):
        positions, orders, CASH, _, _ = setAccountVars()
        symbols = [p.symbol for p in positions]
        times_to_delay+=1
        print('Attempting to buy...')
        time.sleep(5)
    if (ticker not in symbols):
        api.cancel_order(ORDER_ID)
        
def sellPosition(ticker, qty, price, limit_order):
    # sell now
    response = placeSellAlpacaOrder(ticker, qty, price, limit_order)
    positions, orders, CASH, EQUITY, PROFIT = setAccountVars()
    
    # update info
    symbols = [p.symbol for p in positions]
    times_repeated = 0
    while (ticker in symbols) and times_repeated <= 60:
        symbols = [p.symbol for p in api.list_positions()]
        print('Attempting to sell...')
        times_repeated += 1
        time.sleep(5)
    return response

# now go!
while(True):
    
    # get time
    CURRENT_HOUR, CURRENT_MIN, _ = getTime()
    
    # is it trading hours?
    while(((CURRENT_HOUR + (CURRENT_MIN / 60)) >= 9.5) & (CURRENT_HOUR < 16) & (datetime.today().weekday() != 5) & (datetime.today().weekday() != 6)) or True:
        
        # get account info
        positions, orders, CASH, EQUITY, PROFIT = setAccountVars()
        own_stock = len(positions) > 0

        prediction = stock_predictor.load_data_train_and_predict(TICKER)[0]  
        current_close = getAlpacaQuote(TICKER).bp
        print(f'Prediction: {prediction}, Current Close: {current_close}')
        if (prediction > current_close) and not own_stock:
            buyPosition(TICKER, 1, current_close)
        elif (prediction <= current_close) and own_stock:
            sellPosition(TICKER, 1, current_close, False)
        
        time.sleep(300)











