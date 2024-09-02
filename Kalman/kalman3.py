import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback, State
import json
import re
import os
import time
from datetime import datetime
from jupyterthemes import jtplot
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# currently installed theme will be used to
# set plot style if no arguments provided
from pykalman import KalmanFilter 
import numpy as np
import pandas as pd
from scipy import poly1d  
from statsmodels.tsa.stattools import adfuller
from sklearn import linear_model
import statsmodels.api as sm
from scipy import signal
import plotly.graph_objs as go
from datetime import timedelta
from dash import dash_table

jtplot.style()


data_kalman = pd.read_csv('trades1.csv')
data_kalman2 = pd.read_csv('trades2.csv')



external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "dash"

card_style = {
    "display": "flex",  # Use flexbox to create a horizontal layout
}

graph_style = {
    "flex": "82%",  # Adjust the width of the graph
}

statistics_style = {
    "flex": "40%",  # Adjust the width of the statistics section
    "padding": "20px",  # Add padding to the section for spacing
}

title_style = {
    "text-align": "center",  # Center align the title text
    "margin-top": "10px",  # Add some space above the title
    "margin-bottom": "10px",  # Add some space below the title
}
tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

def get_freq(x,n,dt):
    '''EXTRACTS TOP THREE FREQUENCIES USING DFT'''
    # Expects x as dataframe, n as integer and dt as positive number
    x = x - sum(x)/len(x)
    assert sum(x)/len(x) < 0.0001 and sum(x)/len(x) > -0.0001
    f, Pxx_den = signal.periodogram(x,1/dt)
    res,i = [],0
    while i < n:
        freq_index = np.argmax(Pxx_den)
        res.append(f[freq_index])
        Pxx_den[freq_index] = min(Pxx_den)
        i += 1
    return res


def generate_html(table):
    win_rate_counts = table['Win Rate'].value_counts()

    # Calculate the win rate
    win_rate = win_rate_counts.get('Win', 0) / len(table)
    win_rate = win_rate*100
    win_rate = round(win_rate, 2)
    name = (table.iloc[0])['name']
    error = (table.iloc[0])['error']
    error = round(error,2)
    average_count = table['Count'].mean(skipna=True)
    average_count = round(average_count, 2)
    table['Reversion'] = pd.to_numeric(table['Reversion'], errors='coerce')
    average_revert = table['Reversion'].mean(skipna=True)
    average_revert = round(average_revert,2)
    average_PnL = table['PnL'].mean(skipna=True)
    average_PnL = average_PnL*100
    average_PnL = round(average_PnL,2)
    max_value_column1 = table['PnL'].max()
    max_value_column1 = 100*max_value_column1
    max_value_column1 = round(max_value_column1,2)
    min_value_column1 = table['PnL'].min()
    min_value_column1 = 100*min_value_column1
    min_value_column1 = round(min_value_column1,2)
    worst_mkm = table['Mkm'].min()
    worst_mkm = 100*worst_mkm
    worst_mkm = round(worst_mkm,2)

    split_parts = name.split(', ', 1)

    first_part = split_parts[0]
    second_part = split_parts[1] if len(split_parts) > 1 else None


    ###### GRAPH ##############
    data_k = pd.read_csv('datafinal.csv', index_col ='Dates', parse_dates=True)
    column_names = data_k.columns
    column_names_list = column_names.tolist()
    rate_1 = data_k[first_part]
    rate_2 = data_k[second_part]
    ratio = rate_2 - rate_1
    indices = rate_1.index.tolist()

    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.0001
    )

    mean, cov = kf.filter(ratio)
    mean, std = mean.squeeze(), np.std(cov.squeeze())
    freq = get_freq((ratio.values)[-90:],3,1)
    HL = 1/(freq[1]*2)  

    curr_error = (ratio.values-mean)[-1]
    a=(ratio - mean)[-252*20:]

    columns_to_drop = ['Unnamed: 0', 'name', 'error']
    table = table.drop(columns=columns_to_drop)

    # Update values in column 'A' to contain only the first five characters
    table['PnL'] = table['PnL'].astype(str).str[:5]
    table['Mkm'] = table['Mkm'].astype(str).str[:5]

    # Create the figure for the Kalman Filter Error
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices, y=(ratio - mean)[-252*20:], mode='lines', line=dict(color='magenta', width=1), name='Error'))
    fig.add_trace(go.Scatter(x=indices, y=5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='red', width=1), name='5z'))
    fig.add_trace(go.Scatter(x=indices, y=5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='red', width=1), name='-5z'))
    fig.add_trace(go.Scatter(x=indices, y=4.5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='green', width=1), name='4.5z'))
    fig.add_trace(go.Scatter(x=indices, y=4.5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='green', width=1), name='-4.5z'))
    fig.add_trace(go.Scatter(x=indices, y=4 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='yellow', width=1), name='4z'))
    fig.add_trace(go.Scatter(x=indices, y=4 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='yellow', width=1), name='-4z'))
    fig.add_trace(go.Scatter(x=indices, y=3.5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1), name='3.5z'))
    fig.add_trace(go.Scatter(x=indices, y=3.5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1), name='-3.5z'))
    
    
    fig.update_layout(
        title=name,
        xaxis_title='Day',
        yaxis_title='Value'
    )

    fig.update_xaxes(type='date')

    #######################################

    return html.Div(
        className="wrapper",
        children=[
            html.Div(
                className="card",
                style=card_style,  # Apply the card style with flexbox
                children=[
                    html.Div(
                        style=graph_style,  # Apply the graph styles
                        children=[
                            dcc.Graph(figure=fig, config={'displayModeBar': False}),
                        ],
                    ),  
                    html.Div(
                        style=statistics_style,  # Apply the statistics section styles
                        children=[
                            html.H3("Statistics"),
                            html.Ul([
                                html.Li([html.Strong("Name: "), html.Span(str(name), style={"color": "blue"})]),
                                html.Li([html.Strong("Error "), html.Span(str(error), style={"color": "blue"})]),
                                html.Li([html.Strong("Win Rate "), html.Span(str(win_rate), style={"color": "blue"})]),
                                html.Li([html.Strong("Average days off-side: "), html.Span((str(average_count)), style={"color": "blue"})]),
                                html.Li([html.Strong("Average days to revert: "), html.Span(str(average_revert), style={"color": "blue"})]),
                                html.Li([html.Strong("Average PnL "), html.Span(str(average_PnL) + " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Max PnL: "), html.Span((str(max_value_column1))+ " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Worst PnL "), html.Span(str(min_value_column1)+ " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Worst Mtm: "), html.Span(str(worst_mkm)+ " bps", style={"color": "blue"})]),
                            ])
                        ]
                    )
                ]
            ),
            html.Div([
                html.Ul([
                html.Li([
                    html.Strong("Dates of reversion:"),
                    dash_table.DataTable(
                        id='table1',
                        columns=[{"name": col, "id": col} for col in table.columns],
                        data=table.to_dict('records'),
                        style_data={
                            'color': 'green',  # Set the font color to green
                        }
                    )
                ]),
                    
                ]),
            ], style=statistics_style),
        ]
    )

def generate_html2(table):
    #name	error	Historical Entry Dates	Historical Zero Exit	Historical Exit Dates	Count	Reversion	PnL	Mtm1	Mtm2	Mtm Date	Mtm2 Date	Win Rate
    print("hi!")
    win_rate_counts = table['Win Rate'].value_counts()

    # Calculate the win rate
    win_rate = win_rate_counts.get('Win', 0) / len(table)
    win_rate = win_rate*100
    win_rate = round(win_rate, 2)
    name = (table.iloc[0])['name']
    error = (table.iloc[0])['error']
    error = round(error,2)
    average_count = table['Count'].mean(skipna=True)
    average_count = round(average_count, 2)
    #table['Reversion'] = pd.to_numeric(table['Reversion'], errors='coerce')
    #average_revert = table['Reversion'].mean(skipna=True)
    #average_revert = round(average_revert,2)
    average_PnL = table['PnL'].mean(skipna=True)
    average_PnL = average_PnL*100
    average_PnL = round(average_PnL,2)
    max_value_column1 = table['PnL'].max()
    max_value_column1 = 100*max_value_column1
    max_value_column1 = round(max_value_column1,2)
    min_value_column1 = table['PnL'].min()
    min_value_column1 = 100*min_value_column1
    min_value_column1 = round(min_value_column1,2)
    worst_mkm1 = table['Mtm1'].min()
    worst_mkm2 = table['Mtm1'].min()
    worst_mkm = min(worst_mkm1,worst_mkm2)
    worst_mkm = 100*worst_mkm
    worst_mkm = round(worst_mkm,2)

    split_parts = name.split(', ', 1)

    first_part = split_parts[0]
    second_part = split_parts[1] if len(split_parts) > 1 else None


    ###### GRAPH ##############
    data_k = pd.read_csv('datafinal.csv', index_col ='Dates', parse_dates=True)
    column_names = data_k.columns
    column_names_list = column_names.tolist()
    rate_1 = data_k[first_part]
    rate_2 = data_k[second_part]
    ratio = rate_2 - rate_1
    indices = rate_1.index.tolist()

    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.0001
    )

    mean, cov = kf.filter(ratio)
    mean, std = mean.squeeze(), np.std(cov.squeeze())
    freq = get_freq((ratio.values)[-90:],3,1)
    HL = 1/(freq[1]*2)  

    curr_error = (ratio.values-mean)[-1]
    a=(ratio - mean)[-252*20:]

    columns_to_drop = ['Unnamed: 0', 'name', 'error']
    table = table.drop(columns=columns_to_drop)

    # Update values in column 'A' to contain only the first five characters
    table['PnL'] = table['PnL'].astype(str).str[:5]
    table['Mkm'] = table['Mtm1'].astype(str).str[:5]
    table['Mkm'] = table['Mtm2'].astype(str).str[:5]

    # Create the figure for the Kalman Filter Error
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices, y=(ratio - mean)[-252*20:], mode='lines', line=dict(color='magenta', width=1), name='Error'))
    fig.add_trace(go.Scatter(x=indices, y=5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='red', width=1), name='5z'))
    fig.add_trace(go.Scatter(x=indices, y=5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='red', width=1), name='-5z'))
    fig.add_trace(go.Scatter(x=indices, y=4.5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='green', width=1), name='4.5z'))
    fig.add_trace(go.Scatter(x=indices, y=4.5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='green', width=1), name='-4.5z'))
    fig.add_trace(go.Scatter(x=indices, y=4 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='yellow', width=1), name='4z'))
    fig.add_trace(go.Scatter(x=indices, y=4 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='yellow', width=1), name='-4z'))
    fig.add_trace(go.Scatter(x=indices, y=3.5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1), name='3.5z'))
    fig.add_trace(go.Scatter(x=indices, y=3.5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1), name='-3.5z'))
    fig.add_trace(go.Scatter(x=indices, y=3 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1), name='3z'))
    fig.add_trace(go.Scatter(x=indices, y=3* -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1), name='-3z'))
    
    
    fig.update_layout(
        title=name,
        xaxis_title='Day',
        yaxis_title='Value'
    )

    fig.update_xaxes(type='date')

    #######################################

    return html.Div(
        className="wrapper",
        children=[
            html.Div(
                className="card",
                style=card_style,  # Apply the card style with flexbox
                children=[
                    html.Div(
                        style=graph_style,  # Apply the graph styles
                        children=[
                            dcc.Graph(figure=fig, config={'displayModeBar': False}),
                        ],
                    ),  
                    html.Div(
                        style=statistics_style,  # Apply the statistics section styles
                        children=[
                            html.H3("Statistics"),
                            html.Ul([
                                html.Li([html.Strong("Name: "), html.Span(str(name), style={"color": "blue"})]),
                                html.Li([html.Strong("Error "), html.Span(str(error), style={"color": "blue"})]),
                                html.Li([html.Strong("Win Rate "), html.Span(str(win_rate), style={"color": "blue"})]),
                                html.Li([html.Strong("Average days off-side: "), html.Span((str(average_count)), style={"color": "blue"})]),
                                #html.Li([html.Strong("Average days to revert: "), html.Span(str(average_revert), style={"color": "blue"})]),
                                html.Li([html.Strong("Average PnL "), html.Span(str(average_PnL) + " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Max PnL: "), html.Span((str(max_value_column1))+ " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Worst PnL "), html.Span(str(min_value_column1)+ " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Worst Mtm: "), html.Span(str(worst_mkm)+ " bps", style={"color": "blue"})]),
                            ])
                        ]
                    )
                ]
            ),
            html.Div([
                html.Ul([
                html.Li([
                    html.Strong("Dates of reversion:"),
                    dash_table.DataTable(
                        id='table1',
                        columns=[{"name": col, "id": col} for col in table.columns],
                        data=table.to_dict('records'),
                        style_data={
                            'color': 'green',  # Set the font color to green
                        }
                    )
                ]),
                    
                ]),
            ], style=statistics_style),
        ]
    )
# Create an empty list to store the generated HTML code for each row
kalman_html = []

for name, group in data_kalman.groupby('name'):
    html_code = generate_html(group)
    kalman_html.append(html_code)

kalman_html2 = []

for name, group in data_kalman2.groupby('name'):
    html_code = generate_html2(group)
    kalman_html2.append(html_code)

#print(kalman_html, kalman_html2)

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Dashboard-Kalman" , className="header-title", style={"padding": "50px"} 
                ),
                dcc.Tabs(id="tabs-mk", value='tab-1', children=[
                    dcc.Tab(label='Kalman', value='K', style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='Kalman 2.0', value='K2', style=tab_style, selected_style=tab_selected_style),
                ], style=tabs_styles),
            ],
            className="header",
        ),
        #html.Div(children=html_rows),
        html.Div(id='tabs-content')
    ]
)


@callback(Output('tabs-content', 'children'),
              Input('tabs-mk', 'value'))


def render_content(tab):
    if tab == 'K':
        return html.Div(children=kalman_html),
    elif tab == 'K2':
        return html.Div(children=kalman_html2),



if __name__ == "__main__":
    app.run_server(debug=True)