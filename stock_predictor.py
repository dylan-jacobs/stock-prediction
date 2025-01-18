# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:47:46 2022

@author: dylan
"""
from itertools import chain, combinations
import yfinance
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import preprocessing
import xgboost as xgb
import rsi_calculator
import math
from scipy.stats import pearsonr
import back_tester
from sklearn.neural_network import MLPClassifier
import seaborn as sns

SHORT_TERM_HISTORY = 1
DAYS_IN_FUTURE_TO_PREDICT = 1
HISTORY = '2y'
INTERVAL = '1h' # 1h: max 2y
TICKER = 'SPY'
FEATURES = [f'close-{i}' for i in range(SHORT_TERM_HISTORY, 0, -1)] + ['wr', 'volume', '5d-rolling-sma', '20d-rolling-avg',
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
    return d[:train_len, :], d[train_len-SHORT_TERM_HISTORY:, :]

def split_closes_into_X_and_y(closes):
    X = []
    y = []
    for i in range(SHORT_TERM_HISTORY, len(closes)):
        X.append(closes.iloc[i-(SHORT_TERM_HISTORY):i+1].values[-SHORT_TERM_HISTORY:])
        try:
            y.append(float(closes.iloc[i+DAYS_IN_FUTURE_TO_PREDICT].values[-1]))
            #y.append(float((closes.iloc[i+DAYS_IN_FUTURE_TO_PREDICT].values[0] - closes.iloc[i].values[0])/closes.iloc[i].values[0]))
        except:
            y.append(None) # this None value will get removed later via X.dropna()
    X, y = np.asarray(X).reshape((len(closes)-SHORT_TERM_HISTORY), SHORT_TERM_HISTORY), np.asarray(y)
    return X, y
    
def scale_data(d):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(d)

def create_model(trainX):
    model = tf.keras.Sequential([
        keras.layers.Input(shape=(trainX.shape[1], 1)),
        keras.layers.LSTM(200, return_sequences=(False)),
        #keras.layers.LSTM(64, return_sequences=(False)),
        keras.layers.Dense(50),
        #keras.layers.Dense(32),
        keras.layers.Dense(1, activation='linear')
    ])  
    model.compile(optimizer='adam', loss='mean_squared_error') 
    print(model.summary)
    return model

def train_model(trainX, trainy, testX, testy, batch_size=128, epochs=3, plot=False):
    model = create_model(trainX)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("training", exist_ok=True)
    checkpoint_path = "training/cp-{epoch:04d}.weights.h5"
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True, 
        save_best_only=True, 
        verbose=1,
    )
    history = model.fit(
        trainX, trainy, 
        batch_size=batch_size, 
        epochs=epochs, 
        validation_data=(testX, testy), 
        callbacks=[checkpointer],
        verbose=1
    )
    train_mse = model.evaluate(trainX, trainy, verbose=0)
    test_mse = model.evaluate(testX, testy, verbose=0)
    print(f'Training MSE: {train_mse}, Testing MSE: {test_mse}')
    if plot:
        plt.figure()
        plt.title('Loss, MSE')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        plt.pause(1)
    return model

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
    highh = highh[:len(close)]
    lowl = lowl[:len(close)]
    wr = -100 * (np.subtract(highh, close) / np.subtract(highh, lowl))
    return wr

def get_technical_indicators(ticker=TICKER):
    close_col = 'close'
    # get close data
    close_data, dates, highs, lows, volume = load_close_data(ticker, HISTORY)
    data, y = split_closes_into_X_and_y(close_data)
    data = pd.DataFrame(data, columns=[close_col])

    data['Volume'] = volume.values[1:]

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

    # get Williams %R
    wr = get_wr(highs, lows, data.iloc[:, 0], 14)
    
    # add Williams %R
    data['wr'] = wr
    
    # add rsi
    rsi = rsi_calculator.calculateRSI(TICKER, HISTORY, INTERVAL)
    data['rsi'] = rsi[SHORT_TERM_HISTORY+1:]
    
    # add y values
    data['y'] = y

    print(data.tail())

    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    return data

def get_old_technical_data():
    # get close data
    close_data, dates, highs, lows, volume = load_close_data(TICKER, HISTORY)
    X, y = split_closes_into_X_and_y(close_data)
    X = pd.DataFrame(X, columns=[f'close-{i}' for i in range(SHORT_TERM_HISTORY, 0, -1)])
    
    # get dates
    dates_array = []
    for i in range(SHORT_TERM_HISTORY, len(dates)):
        dates_array.append(dates.iloc[i].name)
    
    # get bollinger bands
    sma, std, up20, down20 = get_bollinger_bands(X, 20)
    _, _, up10, down10 = get_bollinger_bands(X, 10)
    sma_5, _, up5, down5 = get_bollinger_bands(X, 5)
    
    # get emas
    ema10 = calculate_ema(X, 10)
    ema20 = calculate_ema(X, 20)
    ema50 = calculate_ema(X, 50)
    
    # get Williams %R
    wr = get_wr(highs, lows, X.iloc[:, SHORT_TERM_HISTORY-1], min(SHORT_TERM_HISTORY, 14))
    
    # add Williams %R
    X['wr'] = wr

    # add volume
    X['volume'] = volume.values[SHORT_TERM_HISTORY:]
    
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
    X['rsi'] = rsi[SHORT_TERM_HISTORY+1:]
    
    # add highs and lows
    X['highs'] = highs.values[SHORT_TERM_HISTORY:]
    X['lows'] = lows.values[SHORT_TERM_HISTORY:]

    #X = X.iloc[:, SHORT_TERM_HISTORY-1:]
    #check_redundant_features(X, y)

    # if we want to select specific features
    if type(feature_indices) != type(None):
        #feature_indices = [i + SHORT_TERM_HISTORY - 1 for i in feature_indices]
        #feature_indices = [i for i in range(SHORT_TERM_HISTORY)] + feature_indices
        feature_indices = [SHORT_TERM_HISTORY-1] + (list(feature_indices))
        X = X.iloc[:, feature_indices]

    # add y values for scaling
    X['y'] = y

    # remove na
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    return X

def prepare_data(ticker=TICKER):
    X = get_technical_indicators(ticker)

    feature_names = X.columns
    print(feature_names)
    
    print(X.head())
    print(X.tail())
    
    X = np.array(X)
    
    # split into train, test
    train, test = get_train_and_test_data(X)

    # resplit into X, y
    trainX, trainy = train[:, :-1], train[:, -1]
    testX, testy = test[:, :-1], test[:, -1]

    input_scaler, trainX = scale_data(trainX)
    testX = input_scaler.transform(testX)
    output_scaler, trainy = scale_data(trainy.reshape(-1, 1))
    testy = output_scaler.transform(testy.reshape(-1, 1))

    # reshape
    #trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1])) # USE THIS FOR THE XGBCLASSIFIER
    #testX = np.reshape(testX, (testX.shape[0], testX.shape[1])) 
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1)) # USE THIS FOR THE TENSORFLOW NN
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    trainy = np.reshape(trainy, (trainy.shape[0], trainy.shape[1], 1))
    testy = np.reshape(testy, (testy.shape[0], testy.shape[1], 1))

    print(f'trainX: {trainX.shape}, testX: {testX.shape}, trainy: {trainy.shape}, testy: {testy.shape}')
    return input_scaler, output_scaler, trainX, trainy, testX, testy

def load_the_model(ticker=TICKER):
    _, output_scaler, trainX, _, _, _ = prepare_data()
    model = create_model(trainX)
    model.load_weights('./training/cp-0005.weights.h5')
    return model, output_scaler

def graph_prediction(y, predictions):
    y = pd.DataFrame(y, columns=['Close'])
    y['Predictions'] = predictions
    plt.figure(figsize=(8,4))
    plt.title(f'Model for {TICKER}')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(y[['Close', 'Predictions']])
    plt.legend(['Val', 'Predictions'], loc='lower right')
    plt.show()
     
def predict(model, output_scaler, testX, testy, graph=False):
    pred = model.predict(testX)
    # unscale
    pred = output_scaler.inverse_transform(pred.reshape(-1, 1))
    testy = output_scaler.inverse_transform(testy.reshape(-1, 1))
    print(f'prediction tail: {pred[-5:, :]}, test tail: {testy[-5:, :]}')
    if graph:
        graph_prediction(testy, pred)
    
    se = np.sqrt(np.mean(((pred - testy)**2)))
    
    print(f'Prediction {DAYS_IN_FUTURE_TO_PREDICT} days from now: {str(pred[-1])}')
    print(f'Error: {se}')
    return pred, se
    
def test_statistical_significance_of_features():
    _, _, trainX, trainy, _, _ = prepare_data()
    
    for i in range(0, trainX.shape[1]):
        X_var = trainX['dow-jones']
        y_var = trainy
        correlation, p = pearsonr(X_var, y_var)
        print(i, '===> Correlation coefficient: ', correlation, 'p-value: ', p)

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

def load_data_train_and_predict(ticker=TICKER, graph=False):
    input_scaler, output_scaler, trainX, trainy, testX, testy = prepare_data(ticker) #feature_indices=[0, 1, 2, 10, 12, 13]
    print(f'Training samples: {str(len(trainy))}, testing samples: {str(len(testy))}')
    model = train_model(trainX, trainy, testX, testy, graph)
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))
    testy = np.reshape(testy, (testy.shape[0], testy.shape[1]))
    predictions, _ = predict(model, output_scaler, testX, testy, graph)
    bnh_returns, strategy_returns = back_tester.test_strategy(testX, model, input_scaler, output_scaler, closes_ind=0, graph=graph)
    print(f'Buy and hold returns: {bnh_returns.iloc[-1]}, Strategy returns: {strategy_returns.iloc[-1]}')
    return predictions[-DAYS_IN_FUTURE_TO_PREDICT] # return the prediction for the nth future day

def load_model_and_test(ticker=TICKER):
    model, output_scaler = load_the_model(ticker)
    input_scaler, output_scaler, trainX, trainy, testX, testy = prepare_data(ticker)
    all_data = np.append(trainX, testX[:-1, :], 0)
    all_data_y = np.append(trainy, testy[:-1], 0)
    all_data = np.reshape(all_data, (all_data.shape[0], all_data.shape[1]))
    all_data_y = np.reshape(all_data_y, (all_data_y.shape[0], all_data_y.shape[1]))

    predict(model, output_scaler, all_data, all_data_y)
    bnh_returns, strategy_returns = back_tester.test_strategy(all_data, model, input_scaler, output_scaler)
    print(f'Buy and hold returns: {bnh_returns.iloc[-1]}, Strategy returns: {strategy_returns.iloc[-1]}')

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
        model = train_model(trainX, trainy, testX, testy, ticker, plot=False)
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

#load_data_train_and_predict()
#load_model_and_test()
#test_all_feature_combinations()
