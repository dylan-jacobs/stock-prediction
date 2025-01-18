# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:47:46 2022

@author: dylan
"""
from itertools import combinations
import yfinance
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import preprocessing
import rsi_calculator
from dateutil.relativedelta import relativedelta
from datetime import datetime
import math
import back_tester
import seaborn as sns
import stock_predictor

np.random.seed(314)
STATE = np.random.get_state()
DAYS_IN_FUTURE_TO_PREDICT = 1
HISTORY = '10y'
INTERVAL = '1d' # 1h: max 2y
FIRST_DATE = (datetime.today() - relativedelta(days=(15 * 253) - 7)).strftime('%Y-%m-%d')
TICKER = 'SPY'
FEATURES = ['close', 'wr', 'volume', '5d-rolling-sma', '20d-rolling-avg',
       'bollinger_up_20', 'bollinger_down_20', 'bollinger_up_10',
       'bollinger_down_10', 'bollinger_up_5', 'bollinger_down_5', 'rsi',
       'highs', 'lows']

def load_close_data(ticker, period):
    data = yfinance.download(ticker, period=period, interval=INTERVAL)
    close_data = data[['Close']]
    date_data = data.index.to_frame(index=False, name='Date')  # Extract Date from the index
    high_data = data[['High']]
    low_data = data[['Low']]
    volume = data[['Volume']]
    return close_data, date_data, high_data, low_data, volume
   
def get_train_and_test_data(d):
    train_len = math.ceil(d.shape[0] * 0.9)
    return d[:train_len, :], d[train_len:, :]

def split_closes_into_X_and_y(closes):
    X = []
    y = []
    for i in range(0, len(closes)):
        X.append(closes.iloc[i:i+1].values[-1])
        try:
            gl = -1
            if closes.iloc[i+DAYS_IN_FUTURE_TO_PREDICT].values[-1] > closes.iloc[i].values[-1]:
                gl = 1
            y.append(gl)
        except:
            y.append(None) # this None value will get removed later via X.dropna()
    X, y = np.asarray(X), np.asarray(y)
    return X, y

def get_bollinger_bands(X, rate=20):
    sma = (X.iloc[:, 0]).rolling(rate).mean()
    std = X.iloc[:, 0].rolling(rate).std()
    bollinger_up = sma + (std * 2) # Calculate top band
    bollinger_down = sma - (std * 2) # Calculate bottom band
    return sma, std, bollinger_up, bollinger_down

def calculate_ema(X, days, smoothing=2):
    ema = []
    for i in range(days-1):
        ema.append(None) # this will get removed later - it's just so that initial dimensions will match
    if (X.shape[0] <= 0): return ema # if there's no data, return an empty list
    ema.append(sum(X.iloc[:days, 0]) / X.shape[0])
    for close in X.iloc[days:, 0]:
        ema.append((close * (smoothing / (1 + days))) + (ema[-1] * (1 - (smoothing / (1 + days)))))
    return ema

def get_wr(high, low, close, lookback):
    highh = high.rolling(lookback).max().values 
    lowl = low.rolling(lookback).min().values
    close = close.values.reshape(len(close), 1)
    #lowl = lowl.dropna().values
    #highh = highh.dropna().values
    #close = close.dropna().values.reshape(len(close), 1)
    highh = highh[:len(close)]
    lowl = lowl[:len(close)]
    wr = -100 * (np.subtract(highh, close) / np.subtract(highh, lowl))
    return wr

def prepare_data():
    # get close data
    close_data, dates, highs, lows, volume = load_close_data(TICKER, HISTORY)
    X, y = split_closes_into_X_and_y(close_data)
    X = pd.DataFrame(X, columns=['close'])
    
    # get bollinger bands
    sma, std, up20, down20 = get_bollinger_bands(X, 20)
    _, _, up10, down10 = get_bollinger_bands(X, 10)
    sma_5, _, up5, down5 = get_bollinger_bands(X, 5)
    
    # get emas
    ema10 = calculate_ema(X, 10)
    ema20 = calculate_ema(X, 20)
    ema50 = calculate_ema(X, 50)
    
    # get Williams %R
    wr = get_wr(highs, lows, X.iloc[:, 0], 14)
    
    # add Williams %R
    X['wr'] = wr

    # add volume
    X['volume'] = volume.values
    
    # add emas
    X['10-day-ema'] = ema10
    X['20-day-ema'] = ema20
    X['50-day-ema'] = ema50
    
    # add 7 day rolling 
    X['5d-rolling-sma'] = sma_5
    
    # add 20 moving avg
    X['20d-rolling-avg'] = sma
    
    # add bollinger bands
    X['bollinger_up_20'] = up20
    X['bollinger_down_20'] = down20
    X['bollinger_up_10'] = up10
    X['bollinger_down_10'] = down10
    X['bollinger_up_5'] = up5
    X['bollinger_down_5'] = down5
    
    # add rsi
    rsi = rsi_calculator.calculateRSI(TICKER, HISTORY, INTERVAL)
    X['rsi'] = rsi[1:]
    
    # add highs and lows
    X['highs'] = highs.values[0:]
    X['lows'] = lows.values[0:]

    # add y values for scaling
    X['y'] = y

    # remove na
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    feature_names = X.columns
    
    print(X.head())
    print(X.tail())
    
    X = np.array(X)
    
    # split into train, test
    train, test = get_train_and_test_data(X)

    # resplit into X, y
    trainX, trainy = train[:, :-1], train[:, -1]
    testX, testy = test[:, :-1], test[:, -1]

    run_random_forest(trainX, trainy, testX, testy, feature_names)

    return _, trainX, trainy, testX, testy

def calculate_indicators():
    close_col = 'close'
    # get close data
    close_data, dates, highs, lows, volume = load_close_data(TICKER, HISTORY)
    data, y = split_closes_into_X_and_y(close_data)
    data = pd.DataFrame(data, columns=[close_col])

    data['Volume'] = volume.values

    data["SMA20"] = data[close_col].rolling(20).mean()
    data["SMA50"] = data[close_col].rolling(50).mean()
    
    # Squeeze Pro: Bollinger Bands vs. Keltner Channels
    data["stddev"] = data[close_col].rolling(20).std()
    data["upper_bb"] = data["SMA20"] + 2 * data["stddev"]
    data["lower_bb"] = data["SMA20"] - 2 * data["stddev"]
    data["upper_kc"] = data["SMA20"] + 1.5 * data["stddev"]
    data["lower_kc"] = data["SMA20"] - 1.5 * data["stddev"]
    data["squeeze_pro"] = ((data["lower_bb"] > data["lower_kc"]) & (data["upper_bb"] < data["upper_kc"])).astype(int)

    # Percentage Price Oscillator (PPO)
    data["ema12"] = data[close_col].ewm(span=12, adjust=False).mean()
    data["ema26"] = data[close_col].ewm(span=26, adjust=False).mean()
    data["ppo"] = ((data["ema12"] - data["ema26"]) / data["ema26"]) * 100

    # Thermo (relative position over a range)
    data["thermo"] = (data[close_col] - data["SMA20"]) / (data["upper_bb"] - data["lower_bb"])

    # Decay (exponential decay of price changes)
    decay_factor = 0.9
    data["decay"] = data[close_col].diff().ewm(alpha=1 - decay_factor, adjust=False).mean()

    # Archer On-Balance Volume (OBV)
    data["volume_change"] = data["Volume"] * np.sign(data[close_col].diff())
    data["archer_obv"] = data["volume_change"].cumsum()

    # Bollinger Bands
    data["bb_mid"] = data["SMA20"]
    data["bb_upper"] = data["upper_bb"]
    data["bb_lower"] = data["lower_bb"]

    # Squeeze Indicator
    data["squeeze"] = ((data["bb_upper"] - data["bb_lower"]) < (1.5 * data["stddev"])).astype(int)

    # Ichimoku Indicator
    data["conversion_line"] = (data[close_col].rolling(9).max() + data[close_col].rolling(9).min()) / 2
    data["base_line"] = (data[close_col].rolling(26).max() + data[close_col].rolling(26).min()) / 2
    data["leading_span_a"] = ((data["conversion_line"] + data["base_line"]) / 2).shift(26)
    data["leading_span_b"] = ((data[close_col].rolling(52).max() + data[close_col].rolling(52).min()) / 2).shift(26)
    data["lagging_span"] = data[close_col].shift(-26)
    
    # add y values
    data['y'] = y
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    return data

def check_redundant_features(X, y):
    # Calculate correlation matrix
    corr_matrix = X.corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # Create a correlation matrix with target variable
    corr_with_target = X.corrwith(pd.Series(y))

    # Sort features by correlation with target variable
    corr_with_target = corr_with_target.sort_values(ascending=False)

    # Plot the heatmap
    plt.figure(figsize=(4, 8))
    sns.heatmap(corr_with_target.to_frame(), cmap='GnBu', annot=True)
    plt.title('Correlation with Target Variable')
    plt.show()

def test_all_feature_combinations(ticker=TICKER):

    def all_sets(a):
        sublists = []
        # Loop through all possible lengths of sublists (0 to len(lst))
        for r in range(2, 3):#len(a) + 1):
            # Generate all combinations of length 'r' and add them to the sublists list
            sublists.extend(combinations(a, r))
        # Convert tuples from combinations into lists for consistency
        return [list(sublist) for sublist in sublists]
    def stack_zeros():
        all_possible_feature_combos = all_sets([i for i in range(0, len(FEATURES))])
        print(all_possible_feature_combos)
        print(len(all_possible_feature_combos))
        profits_matrix = np.empty(len(all_possible_feature_combos), dtype=object)
        for i, item in enumerate(all_possible_feature_combos):
            profits_matrix[i] = np.array(item)
        
        # Add a column of zeros
        column_of_zeros = np.zeros(len(profits_matrix), dtype=object)
        for i in range(len(profits_matrix)):
            column_of_zeros[i] = 0

        # Combine original array with column of zeros
        result = np.empty((len(profits_matrix), 3), dtype=object)
        for i in range(len(profits_matrix)):
            result[i, 0] = profits_matrix[i]
            result[i, 1] = 0  # Add zero as a "column"
            result[i, 2] = [FEATURES[result[i, 0][j]] for j in range(len(result[i, 0]))]
        return result
    profits_matrix = stack_zeros()

    for i in range(0, len(profits_matrix)):
        input_scaler, output_scaler, trainX, trainy, testX, testy = prepare_data(ticker)
        all_data = np.append(trainX, testX[:-1, :], 0)
        all_data_y = np.append(trainy, testy[:-1], 0)
        model = stock_predictor.train_model(trainX, trainy, testX, testy, ticker, plot=False)
        all_data = np.reshape(all_data, (all_data.shape[0], all_data.shape[1]))
        all_data_y = np.reshape(all_data_y, (all_data_y.shape[0], all_data_y.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))
        testy = np.reshape(testy, (testy.shape[0], testy.shape[1]))
        #predict(model, output_scaler, testX, testy)
        bnh_returns, strategy_returns = back_tester.test_strategy(testX, model, input_scaler, output_scaler, closes_ind=0)
        profits_matrix[i, 1] = strategy_returns.iloc[-1]
        profits_matrix = profits_matrix[profits_matrix[:, 1].argsort()[::-1]]
        # Write the profits_matrix to a CSV file
        np.savetxt('profits_matrix.csv', profits_matrix, delimiter=',', fmt='%s', header='Feature Combination,Strategy Returns,Feature Names', comments='')

        print(f'Buy and hold returns: {bnh_returns.iloc[-1]}, Strategy returns: {strategy_returns.iloc[-1]}')

    # Sort the profits_matrix by the second column in descending order
    profits_matrix = profits_matrix[profits_matrix[:, 1].argsort()[::-1]]
    print(profits_matrix)

    return profits_matrix


def run_random_forest(X_train, y_train, X_test, y_test, feature_names):
    # Train the Random Forest model
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate the model
    accuracy_before = rf.score(X_test, y_test)
    print(f'Accuracy before feature selection: {accuracy_before:.2f}')
    # Extract feature importances
    importances = rf.feature_importances_
    feature_names = feature_names[:-1] # remove the y column
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Rank features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

    # Select top N features (example selecting top 10 features)
    top_features = feature_importance_df.iloc[:10, 0].index
    X_train_selected = X_train[:, top_features]
    X_test_selected = X_test[:, top_features]
    # Train the Random Forest model with selected features
    rf_selected = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_selected.fit(X_train_selected, y_train)

    # Evaluate the model
    accuracy_after = rf_selected.score(X_test_selected, y_test)
    print(f'Accuracy after feature selection: {accuracy_after:.2f}')

def run_entire_model():
    X = calculate_indicators()
    feature_names = X.columns
    
    X = np.array(X)
    
    # split into train, test
    train, test = get_train_and_test_data(X)

    # resplit into X, y
    trainX, trainy = train[:, :-1], train[:, -1]
    testX, testy = test[:, :-1], test[:, -1]

    run_random_forest(trainX, trainy, testX, testy, feature_names)

run_entire_model()