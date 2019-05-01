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

def scrape(start=2010, end=2019):
    seasons = pd.DataFrame()
    for year in tqdm(range(start, end+1)):
        df = scrape_season(year)
        seasons = seasons.append(df)
        time.sleep(3)
    return seasons

raw = scrape(start=2010, end=2019)

def clean(df):
    df.columns = [column.lower() for column in df.columns]
    df = df.dropna(subset=['g'])
    df = df.rename(columns={'g': 'goals_visitor', 'g.1': 'goals_home'})
    df['extra_time'] = ~df['unnamed: 5'].isna() * 1
    df = df[['season', 'date', 'home', 'visitor', 'goals_home', 'goals_visitor', 'extra_time']]
    df['winner'] = np.where(df.goals_home > df.goals_visitor, df.home, df.visitor)
    return df

df = clean(raw)
df.to_csv('data/games.csv', index=False)

######

df = pd.read_csv('data/games.csv')
df = df.drop(['extra_time', 'winner'], axis=1)
df = df.query('season == "2018-19"')
df['date'] = df['date'].apply(pd.to_datetime)

# home and visitor separation to be rejoined with concat
home = df.copy()
home = home.rename(columns={'home': 'team', 'goals_home': 'goals_for', 'goals_visitor': 'goals_against'})
home = home.drop(['visitor'], axis=1)
visitor = df.copy()
visitor = visitor.rename(columns={'visitor': 'team', 'goals_home': 'goals_against', 'goals_visitor': 'goals_for'})
visitor = visitor.drop(['home'], axis=1)
# rejoin the data
df = pd.concat([home, visitor], sort=False)
df = df.reset_index(drop=True)
df = df.sort_values(['date', 'team'])
# moving average
df[['goals_for_ma', 'goals_against_ma']] = (
    df
    .groupby('team')
    [['goals_for', 'goals_against']]
    .rolling(window=5)
    .mean()
    .shift(1) # REALLY IMPORTANT!!
    .reset_index(0, drop=True)
)




df = pd.merge(df, average_goals, how='left', left_on=['date', 'home'], right_on=['date', 'team'])
df = pd.merge(df, average_goals, how='left', left_on=['date', 'visitor'], right_on=['date', 'team'], suffixes=['_home', '_visitor'])
df['goals_total'] = df['goals_home'] + df['goals_visitor']
df = df[[
    'goals_total',
    'goals_for_average_home', 'goals_against_average_home',
    'goals_for_average_visitor', 'goals_against_average_visitor'
]]
df = df.dropna()

target = 'goals_total'
y = df[target]
X = df.drop(target, axis=1)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

poly = PolynomialFeatures(interaction_only=True, include_bias=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)
# poly.get_feature_names(X_train.columns)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import r2_score, mean_absolute_error
y_hat = model.predict(X_train)

r2_score(y_train, y_hat)
mean_absolute_error(y_test, y_hat)


model.score(X_test)





#
