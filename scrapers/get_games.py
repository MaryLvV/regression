import time
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def scrape_season(year_end):
    # https://www.hockey-reference.com/leagues/NHL_2019_games.html
    url = f'https://www.hockey-reference.com/leagues/NHL_{year_end}_games.html'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features='lxml')
    table = soup.find_all(class_='overthrow table_container', id='div_games')[0]
    df = pd.read_html(str(table))[0]
    df['season'] = f'{int(year_end-1)}-{str(year_end)[-2:]}'
    return df

def scrape(start=2018, end=2019):
    seasons = pd.DataFrame()
    for year in tqdm(range(start, end+1)):
        df = scrape_season(year)
        seasons = seasons.append(df)
        time.sleep(3)
    return seasons

raw = scrape(start=2018, end=2019)
df = raw.copy()

def clean(df):
    df.columns = [column.lower() for column in df.columns]
    df = df.dropna(subset=['g'])
    df = df.rename(columns={'g': 'away_goals', 'g.1': 'home_goals', 'visitor': 'away'})
    df = df[['season', 'date', 'home', 'away', 'home_goals', 'away_goals']]
    return df

df = clean(raw)
df.to_csv('data/games.csv', index=False)

# PREPARE DATA FOR 2018-2019 SEASON

df = pd.read_csv('data/games.csv')
last_year = df.query('season == "2017-18"')

home = df.groupby('home').mean().reset_index().rename(
    columns={'home_goals': 'home_for', 'away_goals': 'home_against'})
away = df.groupby('away').mean().reset_index().rename(
    columns={'home_goals': 'away_against', 'away_goals': 'away_for'})

this_year = df.query('season == "2018-19"')

df = pd.merge(this_year, home, how='left', on='home')
df = pd.merge(df, away, how='left', on='away')

#
# df['date'] = df['date'].apply(pd.to_datetime)
# df = df.sort_values('date')
# df.groupby('date')[['goals_home']].mean()
#
#
# def calculate_moving_average(df, window=5):
#     # home and visitor separation to be rejoined with concat
#     home = df.copy()
#     home = home.rename(columns={'home': 'team', 'goals_home': 'goals_for', 'goals_visitor': 'goals_against'})
#     home = home.drop(['visitor'], axis=1)
#     visitor = df.copy()
#     visitor = visitor.rename(columns={'visitor': 'team', 'goals_home': 'goals_against', 'goals_visitor': 'goals_for'})
#     visitor = visitor.drop(['home'], axis=1)
#     # rejoin the data
#     df = pd.concat([home, visitor], sort=False)
#     df = df.sort_values(['date', 'team'])
#     df = df.reset_index(drop=True)
#     # moving average
#     df[['goals_for_ma', 'goals_against_ma']] = (
#         df
#         .groupby('team')
#         [['goals_for', 'goals_against']]
#         .rolling(window=window)
#         .mean()
#         .groupby('team') ## NEED TO RE-GROUPBY
#         .shift(1) ## THIS IS REALLY IMPORTANT
#         .reset_index(0, drop=True)
#     )
#     return df
#
# def prepare_backtest(df, window=5):
#     moving_average = calculate_moving_average(df, window)
#     df = pd.merge(df, moving_average, how='left', left_on=['date', 'home'], right_on=['date', 'team'])
#     df = pd.merge(df, moving_average, how='left', left_on=['date', 'visitor'], right_on=['date', 'team'], suffixes=['_home', '_visitor'])
#     df = df[[
#         'date', 'home', 'visitor', 'goals_home', 'goals_visitor',
#         'goals_for_ma_home', 'goals_against_ma_home',
#         'goals_for_ma_visitor', 'goals_against_ma_visitor'
#     ]]
#     df = df.dropna()
#     return df

# df = prepare_backtest(df, window=10)
df['goals_total'] = df['home_goals'] + df['away_goals']

target = 'goals_total'
y = df[target]
X = df[['home_for', 'home_against', 'away_against', 'away_for']]

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
# X_train = poly.fit_transform(X_train)
# X_test = poly.transform(X_test)
poly.fit(X_train)
poly.get_feature_names(X_train.columns)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
model.coef_

from sklearn.metrics import r2_score, mean_absolute_error
y_hat = model.predict(X_train)

from matplotlib import pyplot as plt
%matplotlib inline
plt.scatter(y_train, y_hat, alpha=1/10)

r2_score(y_train, y_hat)






#
