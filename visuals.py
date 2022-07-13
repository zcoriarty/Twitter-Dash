# import libraries

from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
import datetime as dt
from itertools import chain

import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from langdetect import detect
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Dashboard Specific
import plotly.express as px
import plotly.graph_objs as go


# design
GRAPH_FOREGROUND = '#FFFFFF'
FONT_COLOR = 'rgb(101, 119, 134)'
BACKGROUND = '#F5F8FA'

CARD_STYLE = {
	'font-family': 'Tahoma',
	'font-size': '20px',
	'text-align': 'left',
	'color': 'rgb(101, 119, 134)'
}
CARD_INFO_STYLE = {
	'font-family': 'Tahoma',
	'font-size': '15px',
	'text-align': 'left',
	'color': 'rgb(170, 184, 194)'
}

# Auth
consumerKey = 'wtNPt2e2QkphQRvSiDmmbZfeS'
consumerSecret = 'bSfmEOr6szlCdwrePgNzqIZmtakHK1fMA0LY2V1RLrDDPMvJEw'
accessToken = '1544386357409243143-j2zn9jyOXjuioG87sealMcaFy4sGXl'
accessTokenSecret = 'sMkOKTVe9yS9PNLnBtbvwl32vlQ7tnoOTD6WjKx9XD2js'

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)


def make_layout(df):
	return html.Div(
		children= html.Div([
			dbc.Card(
				dbc.CardBody([
					dbc.Row([
						dbc.Col([
							get_pie_chart(df)
						], width=3),
						dbc.Col([
							get_histogram(df)
						], width=6),
						dbc.Col([
							dbc.Row([
								dbc.Col(get_overall_sentiment(df))
								
							]),
							dbc.Row([
								dbc.Col(get_median_favorites(df)),
								dbc.Col(get_median_retweets(df))
							]),
							dbc.Row([
								dbc.Col(get_compound_avg(df)),
								dbc.Col(get_compound_median(df))
							])
							
						], width=3),
					], align='center'), 
					html.Br(),
					dbc.Row([
						
						dbc.Col([
							get_scatter_plot(df)
						], width=12),
					], align='center'),      
				]), color = BACKGROUND # all cell border
			)
		], style={'margin-bottom':'30rem'})
		)

def polarity_score(compound):
		if compound > 0.05:
			return 'positive'
		elif compound < -0.5:
			return 'negative'
		return 'neutral'

def generate_data(keyword, noOfTweets):
	if keyword is None:
		keyword = 'gun control'
	if noOfTweets is None:
		noOfTweets = 100
	today = dt.date.today()

	tweets = tweepy.Cursor(api.search_tweets, q=keyword, until=today).items(noOfTweets/7)
	for day in [1, 2, 3, 4, 5, 6]:
		minus_one = today - dt.timedelta(days=day)
		more_tweets = tweepy.Cursor(api.search_tweets, q=keyword, until=minus_one).items((noOfTweets/7)-1)
		tweets = chain(tweets, more_tweets)

	df = pd.DataFrame(columns=['text'])
	stop_words = stopwords.words('english')
	for tweet in tweets:
		
		df = pd.concat([df, pd.DataFrame({'id': tweet.user.id, 'text': tweet.text, 'date': tweet.created_at, 'retweets': tweet.retweet_count, 'favorites': tweet.favorite_count,
		'location': tweet.user.location,'followers': tweet.user.followers_count}, index=[0])])
		df['text'] = df['text'].apply(lambda txt: str(txt).lower())

		
		df['text'] = df['text'].apply(lambda txt: ' '.join([word for word in txt.split() if word not in stop_words]))

		df['text'] = df['text'].apply(lambda txt: sent_tokenize(txt))
		df['text'] = df['text'].apply(lambda txt: ' '.join(txt))

	sid = SentimentIntensityAnalyzer()
	df['score'] = df['text'].apply(lambda txt: sid.polarity_scores(txt))

	df['negative'] = df['score'].apply(lambda txt: txt['neg'])
	df['neutral'] = df['score'].apply(lambda txt: txt['neu'])
	df['positive'] = df['score'].apply(lambda txt: txt['pos'])
	df['compound'] = df['score'].apply(lambda txt: txt['compound'])


	def polarity_score(compound):
		if compound > 0.05:
			return "positive"
		elif compound < -0.5:
			return "negative"
		return "neutral"

	df['sentiment'] = df['compound'].apply(lambda val: polarity_score(val))
	df['day_of_week'] = df['date'].dt.day_name()
	df = df.reset_index(drop=True).drop('score', axis=1)
	return df

def get_pie_chart(df):
	# create an array of colors which will be custom colors to the plot
	colors_dict = {
		'Positive': 'rgb(29, 161, 242)', # positive
		'Negative': 'rgb(101, 119, 134)', # negative
		'Neutral': 'rgb(225, 232, 237)', # neutral
	}
	
	values = df['sentiment'].str.title().value_counts()
	options = values.index.tolist()
	colors = {i: colors_dict[i] for i in options}
	

	fig = px.pie(df, values=values, names=values.index.tolist(), hole=.6, color=options, color_discrete_map=colors
	)
	return html.Div([
			dbc.Card(
				dbc.CardBody([
					dcc.Graph(
						figure=fig,
					config={
						'displayModeBar': False
					}
					)
				]), color = GRAPH_FOREGROUND, style={'border-radius': 20}
			),  
		])

def get_scatter_plot(df):

	fig = px.box(df, x="day_of_week", y="compound", hover_data=['sentiment'], color_discrete_sequence=['rgb(101, 119, 134)'])
	fig.update_layout(
		yaxis_title_text='Sentiment Score', # yaxis label
		xaxis_title_text = '',
		bargap=0.2, # gap between bars of adjacent location coordinates
		bargroupgap=0.1, # gap between bars of the same location coordinates
		plot_bgcolor= GRAPH_FOREGROUND,
		paper_bgcolor= GRAPH_FOREGROUND,
	)
	fig.update_xaxes(categoryorder='array', categoryarray= ['Monday','Tuesday','Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday'])
	

	return html.Div([
			dbc.Card(
				dbc.CardBody([
					dcc.Graph(
						figure=fig,
					config={
						'displayModeBar': False
					}
					)
				]), color = GRAPH_FOREGROUND, style={'border-radius': 20}
			),  
		])

def get_histogram(df):

	fig = go.Figure()
	fig.add_trace(go.Histogram(
		x=df['day_of_week'][df['sentiment'] == 'neutral'],
		histnorm='percent',
		name='Neutral', # name used in legend and hover labels
		marker_color='rgb(225, 232, 237)',
	))
	fig.add_trace(go.Histogram(
		x=df['day_of_week'][df['sentiment'] == 'positive'],
		histnorm='percent',
		name='Positive',
		marker_color='rgb(29, 161, 242)',
	))
	fig.add_trace(go.Histogram(
		x=df['day_of_week'][df['sentiment'] == 'negative'],
		histnorm='percent',
		name='Negative',
		marker_color='rgb(101, 119, 134)',
	))

	fig.update_layout(
		yaxis_title_text='Count', # yaxis label
		bargap=0.2, # gap between bars of adjacent location coordinates
		bargroupgap=0.1, # gap between bars of the same location coordinates
		plot_bgcolor= GRAPH_FOREGROUND,
		paper_bgcolor= GRAPH_FOREGROUND,
	)
	fig.update_xaxes(categoryorder='array', categoryarray= ['Monday','Tuesday','Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday'])

	return html.Div([
			dbc.Card(
				dbc.CardBody([
					dcc.Graph(
						figure=fig,
					config={
						'displayModeBar': False
					}
					)
				]), color = GRAPH_FOREGROUND, style={'border-radius': 20}
			),  
		])

def get_compound_avg(df):
	return html.Div([
			dbc.Card([
				dbc.Row(
					[
						dbc.Col(
							dbc.CardImg(
								src='static/images/icons8-normal-distribution-histogram-96.png',
								className='img-fluid rounded-start',
							),
							className='col-md-4',
						),
						dbc.Col(
							dbc.CardBody(
								[
									html.H4('Average Score', className='card-title', style=CARD_STYLE),
									html.P(df['compound'].mean().round(1),
										className='card-text', style=CARD_INFO_STYLE
									)
								]
							),
							className='col-md-8'
						),
					],
					className='g-0 d-flex align-items-center',
				)
			],
			className='mb-3',
			style={'border-radius': 20},
			)
		])

def get_compound_median(df):
	return html.Div([
			dbc.Card([
				dbc.Row(
					[
						dbc.Col(
							dbc.CardImg(
								src='static/images/icons8-bar-chart-64.png',
								className='img-fluid rounded-start',
							),
							className='col-md-4',
						),
						dbc.Col(
							dbc.CardBody(
								[
									html.H4('Median Score', className='card-title', style=CARD_STYLE),
									html.P(df['compound'].median().round(1),
										className='card-text', style=CARD_INFO_STYLE
									)
								]
							),
							className='col-md-8', style={'border-radius': 20}
						),
					],
					className='g-0 d-flex align-items-center',
				)
			],
			className='mb-3',
			style={'border-radius': 20},
			)
		])

def get_median_favorites(df):
	return html.Div([
			dbc.Card([
				dbc.Row(
					[
						dbc.Col(
							dbc.CardImg(
								src='static/images/icons8-heart-64.png',
								className='img-fluid rounded-start',
							),
							className='col-md-4',
						),
						dbc.Col(
							dbc.CardBody(
								[
									html.H4('Median Favorites', className='card-title', style=CARD_STYLE),
									html.P(df['favorites'].median().round(1),
										className='card-text', style=CARD_INFO_STYLE
									)
								]
							),
							className='col-md-8', style={'border-radius': 20}
						),
					],
					className='g-0 d-flex align-items-center',
				)
			],
			className='mb-3',
			style={'border-radius': 20},
			)
		])


def get_median_retweets(df):
	return html.Div([
			dbc.Card([
				dbc.Row(
					[
						dbc.Col(
							dbc.CardImg(
								src='static/images/icons8-twitter-64.png',
								className='img-fluid rounded-start',
							),
							className='col-md-4',
						),
						dbc.Col(
							dbc.CardBody(
								[
									html.H4('Median Retweets', className='card-title', style=CARD_STYLE),
									html.P(df['retweets'].median().round(1),
										className='card-text', style=CARD_INFO_STYLE
									)
								]
							),
							className='col-md-8', style={'border-radius': 20}
						),
					],
					className='g-0 d-flex align-items-center',
				)
			],
			className='mb-3',
			style={'border-radius': 20},
			)
		])

def get_overall_sentiment(df):
	return  html.Div([
			dbc.Card([
				dbc.Row(
					[
						dbc.Col(
							dbc.CardImg(
								src='static/images/icons8-thermostat-100.png',
								className='img-fluid rounded-start',
							),
							className='col-md-4',
						),
						dbc.Col(
							dbc.CardBody(
								[
									html.H4('Overall Sentiment', className='card-title', style=CARD_STYLE),
									html.P(df['sentiment'].value_counts().idxmax().upper(),
										className='card-text', style=CARD_INFO_STYLE
									)
								]
							),
							className='col-md-8', style={'border-radius': 20}
						),
					],
					className='g-0 d-flex align-items-center',
				)
			], 
			# color='rgb(29, 161, 242)', 
			# outline=True, 
			className='mb-3',
			style={'border-radius': 20},
			)
		])