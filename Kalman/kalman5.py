where should i update it here:
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
import dash_table

jtplot.style()

data_ST = (
    pd.read_csv('UW.csv')
)

data_MT = (
    pd.read_csv('MT.csv')
)

data_LT = (
    pd.read_csv('LT.csv')
)

data_W = (
    pd.read_csv('W.csv')
)

data_kalman = pd.read_csv('DataFinal.csv', index_col ='Dates', parse_dates=True)


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

def find_crossings_to_zero(data):
    crossings = (data.shift(1) >= 0) & (data < 0) | (data.shift(1) < 0) & (data >= 0)
    crossing_dates = data[crossings].index
    return crossing_dates

def find_crossings_to_x(data, x):
    crossings = (data.shift(1) >= x) & (data < x) | (data.shift(1) < x) & (data >= x)
    crossing_indices = data[crossings].index
    return crossing_indices

def generate_html(row):
    blue = row['Blue']
    res = (eval(blue))[0]
    exp = row['Spread']
    exp_ma = (eval(exp))[0]
    ide = row['Name']
    spread = row['End_Spread']
    tp = row['Take_Profit']
    tn = row['Total_Num']
    wr = row['Win_Ratio']
    total_pnl = row['Cum_PNL']
    average_time = row['Avg_Time']
    max_drawdown = row['Max_Drawdown']
    average_pnl = row['Average_PNL']
    max_pnl = row['Max_PNL']
    time_stop = row['Time_Stop']
    fourier = row['Fourier']
    days_until_fourier = row['ttof']
    lookback = row['Lookback']
    zscore = row['zscore']

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
                            dcc.Graph(
                                id="price-chart",
                                config={"displayModeBar": False},
                                figure={
                                    "data": [
                                        {
                                            "x": list(range(len(res))),
                                            "y": res,
                                            "type": "lines",
                                            "name": "Spread"
                                        },
                                        {
                                            "x": list(range(len(exp_ma))),
                                            "y": exp_ma,
                                            "type": "lines",
                                            "name": "Exp MA"
                                        }
                                    ],
                                    "layout": {
                                        "title": ide,
                                        "xaxis": {"title": "Days"},
                                        "yaxis": {"title": "Spread"},
                                        "legend": {"x": 1, "y": 1},
                                        "margin": {"l": 40, "r": 40, "t": 40, "b": 40}
                                    }
                                }
                            ),
                        ],
                    ),  
                    html.Div(
                        style=statistics_style,  # Apply the statistics section styles
                        children=[
                            html.H3("Statistics"),
                            html.Ul([
                                html.Li([html.Strong("Name: "), html.Span(str(ide), style={"color": "blue"})]),
                                html.Li([html.Strong("Win Rate: "), html.Span(str(wr), style={"color": "blue"})]),
                                html.Li([html.Strong("Entry Level: "), html.Span((str(spread))[:5] + " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Average PnL: "), html.Span(str(average_pnl) + " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Total Number of Trades: "), html.Span(str(tn), style={"color": "blue"})]),
                                html.Li([html.Strong("Take Profit: "), html.Span((str(tp))[:5]+ " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Fourier date: "), html.Span(str(fourier), style={"color": "blue"})]),
                                html.Li([html.Strong("Max PnL: "), html.Span(str(max_pnl)[:5]+ " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Time Stop: "), html.Span(str(time_stop), style={"color": "blue"})]),
                                html.Li([html.Strong("Total PnL: "), html.Span(str(total_pnl)+ " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Max Drawdown: "), html.Span(str(max_drawdown)+ " bps", style={"color": "blue"})]),
                                html.Li([html.Strong("Average time of trade: "), html.Span(str(average_time), style={"color": "blue"})]),
                                html.Li([html.Strong("Lookback: "), html.Span(str(lookback), style={"color": "blue"})]),
                                html.Li([html.Strong("Z-score: "), html.Span(str(zscore), style={"color": "blue"})]),
                            ])
                        ]
                    )
                ]
            )
        ]
    )
# Create an empty list to store the generated HTML code for each row
ST_rows = []

# Iterate over each row in the DataFrame and generate the HTML code
for index, row in data_ST.iterrows():
    html_code = generate_html(row)
    ST_rows.append(html_code)

# Create an empty list to store the generated HTML code for each row
MT_rows = []

# Iterate over each row in the DataFrame and generate the HTML code
for index, row in data_MT.iterrows():
    html_code = generate_html(row)
    MT_rows.append(html_code)

# Create an empty list to store the generated HTML code for each row
LT_rows = []

# Iterate over each row in the DataFrame and generate the HTML code
for index, row in data_LT.iterrows():
    html_code = generate_html(row)
    LT_rows.append(html_code)

# Create an empty list to store the generated HTML code for each row
W_rows = []

# Iterate over each row in the DataFrame and generate the HTML code
for index, row in data_W.iterrows():
    html_code = generate_html(row)
    W_rows.append(html_code)

# Specify the file path
file_path = 'Cross_market_data_final_final_final_final.xlsx'

# Get the modification timestamp of the file
modification_time = os.path.getmtime(file_path)

# Convert the timestamp to a datetime object
modification_datetime = datetime.fromtimestamp(modification_time)

# Get the current date in the same format
current_date = datetime.now().date()

updated = ""
# Check if the modification date is equal to the current date
if modification_datetime.date() == current_date:
    updated = "Current"
else:
    updated = "Outdated"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Dashboard ("+ updated + ")" , className="header-title", style={"padding": "50px"} 
                ),
                dcc.Tabs(id="tabs-mk", value='tab-1', children=[
                    dcc.Tab(label='XMKT - ST', value='ST', style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='XMKT - MT', value='MT', style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='XMKT - LT', value='LT', style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='XMKT - W', value='W', style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='Kalman', value='K', style=tab_style, selected_style=tab_selected_style),
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
    if tab == 'ST':
        return html.Div(children=ST_rows),
    elif tab == 'MT':
        return html.Div(children=MT_rows),
    elif tab == 'LT':
        return html.Div(children=LT_rows),
    elif tab == 'W':
        return html.Div(children=W_rows),
    elif tab == 'K':
        return html.Div([
            html.Div([
                html.Label("Select Action:"),
                html.Label("Weight 1: "),
                dcc.Input(id='w1', value='', type='text', style={'display': 'inline-block', 'margin-right': '10px'}),
                html.Label("Weight 2: "),
                dcc.Input(id='w2', value='', type='text', style={'display': 'inline-block', 'margin-right': '10px'}),
                html.Button('Calculate', id='calculate-button', n_clicks=0, style={'display': 'inline-block'}),
            ], style={'text-align': 'center', 'padding': '10px'}),
            html.Div(id='output', style={'text-align': 'center', 'padding': '20px'}),
            #dcc.Graph(id='kalman-filter-graph', style={'display': 'flex', 'justify-content': 'center'}),
        ])

@app.callback(
    Output('output', 'children'),
    #Output('kalman-filter-graph', 'figure'),
    Input('calculate-button', 'n_clicks'),
    State('w1', 'value'),
    State('w2', 'value'),
)

def output(n_clicks, w1, w2):
    if n_clicks > 0:
        w1 = float(w1)
        w2 = float(w2)
        # Call pnl_calculator function with user inputs
        data_k = pd.read_csv('custom_trade.csv', index_col ='Dates', parse_dates=True)
        column_names = data_k.columns
        column_names_list = column_names.tolist()
        rate_1 = data_k[column_names_list[0]]*w1
        rate_2 = data_k[column_names_list[1]]*w2
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

        # Create the figure for the Kalman Filter Error
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=indices, y=(ratio - mean)[-252*20:], mode='lines', line=dict(color='magenta', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='red', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='red', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=4.5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='green', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=4.5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='green', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=4 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='yellow', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=4 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='yellow', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=3.5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=3.5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1)))
        
        
        fig.update_layout(
            title='Kalman Filter Error',
            xaxis_title='Day',
            yaxis_title='Value'
        )

        fig.update_xaxes(type='date')

        fig.update_layout(width=1400)
        fig.update_layout(height=1200)
    

        y = (ratio - mean)[-252*20:]
        date_of_largest_value = y.abs().idxmax()
        crossing_dates = find_crossings_to_zero(y)
        x = (ratio.values - mean)[-1]  # Replace with your desired value
        crossing_indices = find_crossings_to_x(y, x)
        d1 = crossing_indices
        d2 = crossing_dates
        # Initialize a global variable to store the time differences
        global_diff = 0
        totals = 0
        nearest = datetime(2010, 10, 20)
        columns = ['Historical Entry Dates', 'Historical Exit Dates']

        main = pd.DataFrame(columns=columns)
        print("here")
        # Iterate through d1
        for date1 in d1:
            if date1 < nearest:
                continue
            else:
                if d2 is None:
                    break
                d2 = d2[d2 > date1]
                nearest_date2 = None
                min_time_diff= 0
                # Find the nearest date in d2 that is after or at the same time as date1
                time_diff = 0
                for date2 in d2:
                    if date2 >= date1:
                        time_diff = date2 - date1
                        if nearest_date2 is None or date2 < nearest_date2:
                            min_time_diff = time_diff.days
                            nearest_date2 = date2
                # print(date1, nearest_date2, time_diff)
                data = {
                    'Historical Entry Dates': date1,
                    'Historical Exit Dates': nearest_date2,
                }
                main.loc[len(main.index)] = data
                if not nearest_date2:
                    break
                if nearest_date2:
                    if min_time_diff ==0:
                        min_time_diff += 1
                    # Add the time difference to the global variable
                    global_diff += min_time_diff
                    #print(global_diff, min_time_diff)
                    totals += 1
                
                nearest = nearest_date2
            print("global diff: ",global_diff, " totals: ",totals)
            print(date1, nearest_date2)
        average = global_diff/totals
        
        ########################################################################
        ######## NEW TABLE 2 - 0.1 below 
        print("here2")
        a = x
        if a>0:
            a = a-0.05
        else:
            a = a+0.05
        crossing_indices2 = find_crossings_to_x(y, a)

        d1 = crossing_indices2

        d2 = crossing_dates
        # Initialize a global variable to store the time differences
        global_diff = 0
        totals = 0
        nearest = datetime(2010, 10, 20)
        main2 = pd.DataFrame(columns=columns)

        # Iterate through d1
        for date1 in d1:
            if date1 < nearest:
                continue
            else:
                if d2 is None:
                    break
                d2 = d2[d2 > date1]
                nearest_date2 = None
                min_time_diff= 0
                # Find the nearest date in d2 that is after or at the same time as date1
                time_diff = 0
                for date2 in d2:
                    if date2 >= date1:
                        time_diff = date2 - date1
                        if nearest_date2 is None or date2 < nearest_date2:
                            min_time_diff = time_diff.days
                            nearest_date2 = date2
                # print(date1, nearest_date2, time_diff)
                data = {
                    'Historical Entry Dates': date1,
                    'Historical Exit Dates': nearest_date2,
                }
                main2.loc[len(main2.index)] = data
                if not nearest_date2:
                    break
                if nearest_date2:
                    if min_time_diff ==0:
                        min_time_diff += 1
                    # Add the time difference to the global variable
                    global_diff += min_time_diff
                    totals += 1
                
                nearest = nearest_date2
            print("global diff: ",global_diff, " totals: ",totals)
            print(date1, nearest_date2)
            print(main2)
        average2 = global_diff/totals
        print(main2)

        container_style = {
            'display': 'flex',  # Use flex display
            'flex-wrap': 'wrap',  # Allow wrapping to the next line if needed
            'justify-content': 'space-between',  # Align items with space between them
        }

        graph_style = {
            'width': '69%',  # 49% width for the graph
        }

        statistics_style = {
            'width': '49%',  # 49% width for the statistics
            'padding-left': '20%',  # Add a little spacing between the sections
        }

        fig.update_layout(width=1000)
        fig.update_layout(height=800)

        content = html.Div(style=container_style, children=[
            html.Div(style=graph_style, children=[
                dcc.Graph(figure=fig, config={'displayModeBar': False}),
            ]),

            html.Div([
                html.H3("Statistics"),
                html.Ul([
                    html.Li([html.Strong("Name: "), html.Span(str(column_names_list[0] + " ," + column_names_list[1]))]),
                    html.Li([html.Strong("Current error: "), html.Span(str((ratio.values - mean)[-1]))]),
                    html.Li([html.Strong("Discrete Fourier Transformation Half-Life: "), html.Span((str(HL)))]),
                    html.Li([html.Strong("Time taken for error to revert: "), html.Span(str(average))]),
                    html.Li([
                        html.Strong("Dates of reversion: "),
                        dash_table.DataTable(
                            id='table1',
                            columns=[{"name": col, "id": col} for col in main.columns],
                            data=main.to_dict('records')
                        )
                    ]),
                    html.Li([html.Strong("Time taken for error to revert (5 bp range): "), html.Span(str(average2))]),
                    html.Li([
                        html.Strong("Dates of reversion:"),
                        dash_table.DataTable(
                            id='table2',
                            columns=[{"name": col, "id": col} for col in main2.columns],
                            data=main2.to_dict('records'),
                        )
                    ]),
                    html.Li([html.Strong("Max error: "), html.Span(str(y.abs().max()))]),
                    html.Li([html.Strong("Max error date: "), html.Span(str(date_of_largest_value))]),
                ]),
            ], style=statistics_style),
        ])

        return html.Div(
            className="wrapper",
            children=[
                html.Div(
                    className="card",
                    style=card_style,  # Apply the card style with flexbox
                    children=[
                        content,
                    ]
                )
            ]
        )

"""
def update(n_clicks, t1, t2):
    if n_clicks > 0:
        # Call pnl_calculator function with user inputs
        rate_1 = data_kalman[t1]
        rate_2 = data_kalman[t2]
        ratio = rate_2 - rate_1
        indices = data_kalman.index.tolist()
        print(indices)
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

        # Create the figure for the Kalman Filter Error
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=indices, y=(ratio - mean)[-252*20:], mode='lines', line=dict(color='magenta', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='red', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='red', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=4.5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='green', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=4.5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='green', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=4 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='yellow', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=4 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='yellow', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=3.5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1)))
        fig.add_trace(go.Scatter(x=indices, y=3.5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1)))
        
        
        fig.update_layout(
            title='Kalman Filter Error',
            xaxis_title='Day',
            yaxis_title='Value'
        )

        fig.update_xaxes(type='date')


        result_text = html.Div([
            html.P(f"Kalman Relative Value in bps: {(ratio.values - mean)[-1]}", style={'font-size': '24px', 'font-weight': 'bold'}),
            html.P(f"Discrete Fourier Transformation Half-Life: {HL}", style={'font-size': '24px', 'font-weight': 'bold'})
        ])

        fig.update_layout(width=1400)
        fig.update_layout(height=1200)
        
        centered_plots = html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
        ], style={'display': 'flex', 'justify-content': 'center'})

        return result_text, fig
"""
if __name__ == "__main__":
    app.run_server(debug=True)