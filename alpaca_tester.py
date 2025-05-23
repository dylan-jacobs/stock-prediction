# -*- coding: utf-8 -*-
"""
Created on Thu Dec 8 2022

@author: dylan

Strategy: 
    - Every hour, make prediction using LSTM model in stock_predictor.py
    - If prediction close > current close & don't own stock: ---> Buy
    - Else: ---> Sell
"""

import alpaca_trade_api as tradeapi
import time
from datetime import datetime
from pytz import timezone
from threading import Thread
import stock_predictor
import os
from dotenv import load_dotenv

load_dotenv() # only on local

# ALPACA API Info for fetching data, portfolio, etc. from Alpaca
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
TICKER = 'SPY'

# Instantiate REST API Connection
api = tradeapi.REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url=BASE_URL, api_version='v2')
print(api.get_account())

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
    
    if limit_order:
        response = api.submit_order(symbol=ticker, qty=amnt, side='sell', type='limit', time_in_force='day', limit_price=price)    
    else:
        response = api.submit_order(symbol=ticker, qty=amnt, side='sell', type='market', time_in_force='day')                

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

def main():    
    # get account info
    positions, orders, CASH, EQUITY, PROFIT = setAccountVars()
    own_stock = len(positions) > 0

    current_close = getAlpacaQuote(TICKER).ap
    prediction = stock_predictor.load_data_train_and_predict(TICKER)
    print(f'Prediction: {prediction}, Current Close: {current_close}')
    if (prediction > current_close) and not own_stock:
        buyPosition(TICKER, 1, current_close)
    elif (prediction <= current_close) and own_stock:
        sellPosition(TICKER, 1, current_close, False)


if __name__=='__main__':
    main()









