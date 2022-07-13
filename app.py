# import dash libraries
import dash
from dash import html, dcc, dash_table
import plotly.express as px
# from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
# import dash_mantine_components as dmc
from dash.dependencies import Input, Output, State
from datetime import timedelta, datetime, date

# import nltk
# nltk.download('stopwords')
# nltk.download('vader_lexicon')
# nltk.download('punkt')

# hosting on heroku
# import gunicorn
# from whitenoise import WhiteNoise

# import DS libraries
import pandas as pd
import numpy as np
import json

import visuals

# DESIGN
BACKGROUND = '#F5F8FA'

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# server = app.server

# server.wsgi_app = WhiteNoise(server.wsgi_app, root='Static/') 


topics_list = [ 'Britney Spears', 'Gun Control', 'Abortion', 'Healthcare', 'Joe Biden', 'Tesla', 'Tour De France']
tweet_survey_count_list = [5, 10, 100, 250, 1000]
# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 54,
    'left': 0,
    'bottom': 0,
    'width': '16rem',
    'height': '100%',
    'z-index': 1,
    'overflow-x': 'hidden',
    'transition': 'all 0.5s',
    'padding': '0.5rem 1rem',
    'background-color': BACKGROUND,
	'color': '#FFFFFF',
}

SIDEBAR_HIDEN = {
    'position': 'fixed',
    'top': 54,
    'left': '-16rem',
    'bottom': 0,
    'width': '16rem',
    'height': '100%',
    'z-index': 1,
    'overflow-x': 'hidden',
    'transition': 'all 0.5s',
    'padding': '0rem 0rem',
    'background-color': BACKGROUND,
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    'transition': 'margin-left .5s',
    'margin-left': '16rem',
    'margin-bottom': '30rem',
    'margin-right': 0,
    'padding': '2rem 1rem',
    'background-color': BACKGROUND,
}

CONTENT_STYLE1 = {
    'transition': 'margin-left .5s',
    # 'margin-left': '2rem',
    # 'margin-right': '2rem',
    'padding': '2rem 1rem',
    'background-color': BACKGROUND,
}
SEARCH_STYLE  = {
    'background-color': BACKGROUND,
    'color': 'black',
    }
DATATABLE_STYLE = {
    'color': 'white',
    'backgroundColor': BACKGROUND,
}


# Sticky dash board header
navbar = dbc.NavbarSimple(
    children=[
        dbc.Button('Sidebar', outline=True, color='secondary', className='mr-1', id='btn_sidebar'),
        
    ],
    id='navbar',
    sticky='top',
    brand='Twitter Sentiment Analysis',
    brand_href='#',
    color=BACKGROUND,
    dark=True,
    fluid=True,
)

# sidebar, including input for content window
sidebar = html.Div(
    [
        html.Br(),
		
        dbc.Nav(
            [
                html.Br(),
                html.Br(),
                dcc.Dropdown(id='selected-keyword',placeholder='Select a topic',value='Britney Spears', options = topics_list, style=SEARCH_STYLE, clearable=False),
                html.Br(),
                dcc.Dropdown(id='tweet-count',placeholder='Number of tweets to survey',value=10, options = tweet_survey_count_list, style=SEARCH_STYLE, clearable=False),
                html.Br(),
                dbc.Button("See Tweets", id="tweets-modal", n_clicks=0, outline=False, color='secondary'),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Tweets Analyzed")),
                        dbc.ModalBody(id='tweets-df'),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id="close-tweets-modal",
                                className="ms-auto",
                                n_clicks=0,
                            )
                        ),
                    ],
                    id="modal-body-scroll",
                    scrollable=True,
                    is_open=False,
                    size='xl'
                ),
                
                
            ],
            vertical=True,
            pills=True,
            style=SIDEBAR_STYLE
            
        ),
    ],
    id='sidebar',
    style=SIDEBAR_STYLE,
)


content = html.Div(
	style=CONTENT_STYLE, 
	id='page-content'
)

app.layout = html.Div([
    navbar,
    sidebar,
    dcc.Loading(
        children=[
            html.Div(
            [
                dcc.Store(id='side_click'),
                content,
                # dcc.Location(id='url'),
                
                
                
            ], style={'height': '100vh', 'width': '100vw'}
        )],
        type="cube",
    )], style={'background-color': BACKGROUND})
        

def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

app.callback(
    Output("modal-body-scroll", "is_open"),
    [
        Input("tweets-modal", "n_clicks"),
        Input("close-tweets-modal", "n_clicks"),
    ],
    [State("modal-body-scroll", "is_open")],
)(toggle_modal)


@app.callback([Output('page-content', 'children'),Output('tweets-df', 'children')], [Input('selected-keyword', 'value'), Input('tweet-count', 'value')])
def render_page_content(keyword, noOfTweets):
    df = visuals.generate_data(keyword, noOfTweets)
    modal_df = df.drop(columns=['date', 'retweets', 'favorites', 'negative', 'neutral', 'positive', 'compound', 'sentiment'], axis=1)
    return visuals.make_layout(df), dash_table.DataTable(modal_df.to_dict('records'), [{"name": i, "id": i} for i in modal_df.columns])

# @app.callback(Output('tweets-df', 'children'), [Input('selected-keyword', 'value'), Input('tweet-count', 'value')])
# def get_tweet_modal_content(keyword, noOfTweets):
#     df = visuals.generate_data(keyword, noOfTweets)
#     df = df[['text', 'sentiment']]
#     return dash_table.DataTable(modal.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
@app.callback(
    [
        Output('sidebar', 'style'),
        Output('page-content', 'style'),
        Output('side_click', 'data'),
    ],

    [Input('btn_sidebar', 'n_clicks')],
    [
        State('side_click', 'data'),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == 'SHOW':
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = 'HIDDEN'
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = 'SHOW'
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick


if __name__ == '__main__':
    app.run_server(debug=True, port=8086)
    # yfin1()