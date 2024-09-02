import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
import os
import pandas as pd
from xbbg import blp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from datetime import timedelta as td
import statsmodels.api as sm
import re
import matplotlib.image as image
from ipywidgets import interact, IntSlider, Checkbox, Dropdown, Output, HBox, VBox, interactive, interactive_output, ToggleButton,Text, Button, DatePicker, IntText, ToggleButtons, RadioButtons,SelectMultiple
from IPython.display import display, clear_output
import itertools
from scipy import stats
from scipy.optimize import minimize 
from scipy.special import ndtr
from scipy import stats
from scipy.optimize import minimize 
from scipy.special import ndtr
from datetime import datetime as dt, timedelta
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


# File path for the CSV
csv_file_path = "cix.csv"

# Create Dash app
app = dash.Dash(__name__)
app.title = 'Financial Data Dashboard'

# App layout
app.layout = html.Div(children=[
    html.H1(children='Financial Data Dashboard', style={'textAlign': 'center'}),
    
    html.Div(children=[
        html.Div([
            html.Label('Ticker'),
            dcc.Input(id='ticker-input', type='text', value=''),
        ], style={'display': 'inline-block', 'margin-right': '10px'}),
        
        html.Div([
            html.Label('Description'),
            dcc.Input(id='description-input', type='text', value=''),
        ], style={'display': 'inline-block', 'margin-right': '10px'}),
        
        html.Button('Save', id='save-button', n_clicks=0),
    ], style={'textAlign': 'center', 'margin-top': '20px'}),
    
    html.Div(id='save-output', style={'textAlign': 'center', 'margin-top': '10px'}),
    
    html.Button('My Cix\'s', id='my-cix-button', n_clicks=0, style={'display': 'block', 'margin': '20px auto'}),
    
    html.Div(id='cix-output', style={'textAlign': 'center', 'margin-top': '20px'}),
    
    html.Hr(),
    
    html.Div(id='data-table-container', children=[
        html.H2('Current Data'),
        dash_table.DataTable(id='data-table', style_table={'margin': '0 auto'})
    ], style={'textAlign': 'center', 'margin-top': '20px'}),
    
    html.Hr(),
    
    html.Div(children=[
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=dt.today() - timedelta(days=365),
            end_date=dt.today(),
            display_format='YYYY-MM-DD'
        ),
        html.Button('Pull BBG Data', id='pull-bbg-data-button', n_clicks=0)
    ], style={'textAlign': 'center', 'margin-top': '20px'}),
    
    html.Div(id='bbg-data-output', style={'textAlign': 'center', 'margin-top': '20px'}),
    
    html.Hr(),
    
    html.Div(children=[
        html.Label('Study Target:'),
        dcc.Dropdown(id='study-target-dropdown', options=[], value=''),
        html.Button('Show Z-Scores', id='show-z-scores-button', n_clicks=0)
    ], style={'textAlign': 'center', 'margin-top': '20px'}),
    
    html.Div(id='z-scores-output', style={'textAlign': 'center', 'margin-top': '20px'}),
])

# Function to save entries to CSV
def save_to_csv(ticker, description):
    df = pd.DataFrame([[ticker, description]], columns=['String', 'Description'])
    if os.path.exists(csv_file_path):
        df.to_csv(csv_file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file_path, index=False)

# Function to display contents of CSV
def display_cix():
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        return df.to_dict('records')
    else:
        return []

# Callbacks
@app.callback(
    Output('save-output', 'children'),
    Input('save-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('description-input', 'value')
)
def update_output(n_clicks, ticker, description):
    if n_clicks > 0 and ticker and description:
        save_to_csv(ticker, description)
        return f'Saved: {ticker}, {description}'
    return ''

@app.callback(
    Output('cix-output', 'children'),
    Input('my-cix-button', 'n_clicks')
)
def display_my_cix(n_clicks):
    if n_clicks > 0:
        records = display_cix()
        if records:
            return dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in records[0].keys()],
                data=records
            )
        return 'No entries found.'
    return ''

@app.callback(
    Output('bbg-data-output', 'children'),
    Input('pull-bbg-data-button', 'n_clicks'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date')
)
def pull_bbg_data(n_clicks, start_date, end_date):
    if n_clicks > 0:
        # Dummy implementation - replace with actual data pulling
        return f'Data pulled from {start_date} to {end_date}'
    return ''

@app.callback(
    Output('study-target-dropdown', 'options'),
    Input('pull-bbg-data-button', 'n_clicks')
)
def update_study_target_dropdown(n_clicks):
    # Dummy implementation - replace with actual data pulling
    return [{'label': f'Target {i}', 'value': f'Target {i}'} for i in range(5)]

@app.callback(
    Output('z-scores-output', 'children'),
    Input('show-z-scores-button', 'n_clicks'),
    State('study-target-dropdown', 'value'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date')
)
def show_z_scores(n_clicks, target, start_date, end_date):
    if n_clicks > 0 and target:
        # Dummy implementation - replace with actual data processing
        return f'Z-scores for {target} from {start_date} to {end_date}'
    return ''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
