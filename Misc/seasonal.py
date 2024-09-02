import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback, State
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api
import statsmodels as sm
import datetime  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api
import statsmodels as sm
import datetime  
from statsmodels.tsa.seasonal import STL

"""
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in table.columns])] +
                # Body
                [html.Tr([html.Td(data) for data in row]) for row in table.values],
                style={
                    'width': '50%',  # Adjust the width as needed
                    'margin': 'auto',  # Center the table horizontally
                    'border-collapse': 'collapse',  # Collapse table borders
                    'font-size': '14px',  # Adjust font size
                    'border': '1px solid #ddd',  # Add borders to the table
                    'text-align': 'center',  # Center align text
                    'padding': '100px'
                }
            )
            """

suppress_callback_exceptions=True

def get_business_day_number(date_str):
    # Convert the input date string to a pandas Timestamp object
    date = pd.to_datetime(date_str)

    # Create a DatetimeIndex with all the business days in the year
    year = date.year
    business_days = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='B')

    # Find the closest business day to the input date
    closest_business_day = min(business_days, key=lambda x: abs((x - date).days))

    # Calculate the business day number (0-indexed)
    business_day_number = business_days.get_loc(closest_business_day)

    return business_day_number


def get_business_day_by_number_one(year, business_day_number1):
    business_days = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='B')

    if business_day_number1 < 1 or business_day_number1 > len(business_days):
        raise ValueError("Invalid business day number")

    start = business_days[business_day_number1 - 1]

    formatted_date = start.strftime("%m-%d")

    return formatted_date


def get_business_day_by_number(year, business_day_number1, business_day_number2):
    business_days = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='B')

    if business_day_number1 < 1 or business_day_number2 > len(business_days):
        raise ValueError("Invalid business day number")

    start = business_days[business_day_number1 - 1]

    formatted_date = start.strftime("%m-%d")
    
    end = business_days[business_day_number2 - 1]

    formatted_date2 = end.strftime("%m-%d")

    return formatted_date, formatted_date2


def pnl_calculator(trade, data, day, type, year, start, end, verbose=False):
    print("trade: ", trade)

    start, end = get_business_day_by_number(year, start, end)
    pnls = []
    print(data)
    Index = data.index  
    print(Index)
    Years = list(set(Index.year))
    Years.sort()
    Years = list(map(str, Years))

    df = data.loc[:, trade]
    res = STL(df).fit()
    starts = [str(i) + '-' + start for i in Years]
    ends = [str(i) + '-' + end for i in Years]

    if type == 'Pay':

        for i in range(len(Years)):
            try:
                Data_Seasonality = res.seasonal[str(Years[i])]
                Data_Seasonality = Data_Seasonality.loc[Data_Seasonality.index >=starts[i]]
                Data_Seasonality = Data_Seasonality[Data_Seasonality.index <= ends[i]]

                min_date = Data_Seasonality.idxmin()
                start_date = min_date
                end_date = min_date + pd.DateOffset(days = day)
                mask = (Data_Seasonality.index >= start_date) & (Data_Seasonality.index <= end_date)
                Min_Months = Data_Seasonality.idxmin().month
                Max_Month = Data_Seasonality.loc[mask].idxmax().month
                Min_Day = Data_Seasonality.idxmin().day
                Max_Day = Data_Seasonality.loc[mask].idxmax().day

                if len(str(Min_Months)) < 2:
                    Min_Months = '0' + str(Min_Months)
                if len(str(Max_Month)) < 2:
                    Max_Month = '0' + str(Max_Month)

                try:
                    pnl = df[str(Years[i]) + '-' + str(Max_Month) + '-' + str(Max_Day)] - df[str(Years[i]) + '-' + str(Min_Months) + '-' + str(Min_Day)]
                    
                    pnls.append((pnl, str(Years[i]) + '-' + str(Min_Months) + '-' + str(Min_Day), str(Years[i]) + '-' + str(Max_Month) + '-' + str(Max_Day)))
                except:
                    pass
            except:
                pass

    
    elif type == 'Receive':
        for i in range(len(Years)):
            try:
                Data_Seasonality = res.seasonal[str(Years[i])]
                Data_Seasonality = Data_Seasonality.loc[Data_Seasonality.index >=starts[i]]
                Data_Seasonality = Data_Seasonality[Data_Seasonality.index <= ends[i]]
                max_date = Data_Seasonality.idxmax()
                start_date = max_date
                end_date = start_date + pd.DateOffset(days = day)
                mask = (Data_Seasonality.index >= start_date) & (Data_Seasonality.index <= end_date)
                Max_Month = Data_Seasonality.idxmax().month
                Min_Month = Data_Seasonality.loc[mask].idxmin().month
                Max_Day = Data_Seasonality.idxmax().day
                Min_Day = Data_Seasonality.loc[mask].idxmin().day

                if len(str(Max_Month)) < 2:
                    Max_Month = '0' + str(Max_Month)
                if len(str(Min_Month)) < 2:
                    Min_Month = '0' + str(Min_Month)

                try:
                    pnl = df[str(Years[i]) + '-' + str(Max_Month) + '-' + str(Max_Day)] - df[str(Years[i]) + '-' + str(Min_Month) + '-' + str(Min_Day)]
                    pnls.append((pnl, str(Years[i]) + '-' + str(Max_Month) + '-' + str(Max_Day), str(Years[i]) + '-' + str(Min_Month) + '-' + str(Min_Day)))
                except:
                    pass
            except:
                pass
            
            
 
    finaldf = pd.DataFrame(pnls, columns = ['PnL', 'Entry', 'Exit'])
    mean = finaldf['PnL'].mean()
    std = finaldf['PnL'].std()
    finaldf['Win Rate'] = ((finaldf['PnL'] > 0)*1)
    winrate = finaldf['Win Rate'].mean()
    finaldf.drop('Win Rate', axis = 1, inplace=True)
    
    if verbose == True:
        print(type, trade)
        print('Average PnL: ' + str(round(mean, 5)))
        print('Std: ' +  str(round(std, 5)))
        for column in finaldf.columns:
            if column == 'PnL':
                finaldf[column] = finaldf[column].round(3)
        return finaldf
    
    return mean, std, trade, winrate

def pnl_scanner(data, day, type, year, start, end):
    print("here!")
    pnl = []
    print("data222: ", data)
    for column in data.columns:
        stats = pnl_calculator(column, data, day, type, year, start, end, verbose=False)
        pnl.append(stats)
        df = pd.DataFrame(pnl, columns = ['PnL', 'Std', ' Trade', 'Win Rate'])
        df['PnL/Std'] = df['PnL'] / df['Std']

    for column in df.columns:
        if column != ' Trade':
            df[column] = df[column].round(3)
    df = df.sort_values(by='PnL/Std', ascending = False)
    return df

def pnl_calculator2(trade, data, day, type, year, start, end):
    start, end = get_business_day_by_number(year, start, end)
    pnls = []
    Index = data.index  
    Years = list(set(Index.year))
    Years.sort()
    Years = list(map(str, Years))

    df = data.loc[:, trade]

    for i in range(len(Years)):
        try:
            Data = df[str(Years[i])]
            min_date = Data.idxmin()
            max_date = Data.idxmax()

            start_date = min_date
            end_date = min_date + pd.DateOffset(days=day) if type == 'Pay' else max_date + pd.DateOffset(days=day)

            pnl = Data.loc[end_date] - Data.loc[start_date]

            pnls.append((pnl, str(Years[i]) + '-' + start_date.strftime('%m-%d'), str(Years[i]) + '-' + end_date.strftime('%m-%d')))
        except:
            pass

    finaldf = pd.DataFrame(pnls, columns=['PnL', 'Entry', 'Exit'])
    mean = finaldf['PnL'].mean()
    std = finaldf['PnL'].std()
    finaldf['Win Rate'] = ((finaldf['PnL'] > 0) * 1)
    winrate = finaldf['Win Rate'].mean()
    finaldf.drop('Win Rate', axis=1, inplace=True)

    return mean, std, finaldf, winrate

def pnl_scanner2(data, day, type, year, start, end):
    pnl = []
    for column in data.columns:
        print(column)
        mean, std, pnl_df, winrate = pnl_calculator2(column, data, day, type, year, start, end)
        pnl_df['PnL/Std'] = pnl_df['PnL'] / std
        pnl.append((mean, std, pnl_df, winrate, column))

    pnl.sort(key=lambda x: x[0] / x[1], reverse=True)
    result = pd.concat([x[2] for x in pnl], ignore_index=True)
    result = result.sort_values(by='PnL/Std', ascending=False)

    return result

def find_best_pnl(data, day, type, year, start, end):
    if end - start < day:
        print("The disparity between the dates and the trade range provided is not correct")
        return
    else:
        best = 0 
        best_index = 0 
        for i in range(start, end-19):
            pnl = []
            for column in data.columns:
                stats = pnl_calculator(column, data, day, type, year, i, i+20, verbose=False)
                pnl.append(stats)
                df = pd.DataFrame(pnl, columns = ['PnL', 'Std', ' Trade', 'Win Rate'])
                df['PnL/Std'] = df['PnL'] / df['Std']

            df = df.sort_values(by='PnL/Std', ascending = False)
            avg = df['PnL/Std'].mean()
            if avg > best:
                best_index = i
                best = avg
        pnl = []
        for column in data.columns:
            stats = pnl_calculator(column, data, day, type, year, best_index, best_index+20, verbose=False)
            pnl.append(stats)
            df = pd.DataFrame(pnl, columns = ['PnL', 'Std', ' Trade', 'Win Rate'])
            df['PnL/Std'] = df['PnL'] / df['Std']
        df = df.sort_values(by='PnL/Std', ascending = False)  
        print(get_business_day_by_number_one(year, best_index))
        return df

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
        
    },
    {
        "href": (
            "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
        ),
        "rel": "stylesheet"
    }

]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Seasonal"

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

data=pd.read_excel('xkmt.xlsx', index_col=0, parse_dates = True) 
data = data.dropna()

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Seasonal", className="header-title", style={"padding": "50px"} 
                ),
                dcc.Tabs(id="tabs-mk", value='tab-1', children=[
                    dcc.Tab(label='LOESS', value='ST', style=tab_style, selected_style=tab_selected_style),
                    dcc.Tab(label='NAIVE - SMA', value='MT', style=tab_style, selected_style=tab_selected_style),
                ], style=tabs_styles),
                
            ],
            className="header",
        ),
        html.Div(id='tabs-content')
    ]
)

def generate_html_table():
    # Convert the DataFrame to its HTML representation
    data=pd.read_excel('M:\\BHCM\\Trading\\TeamSK\\Library of Strategies\\Nimit\\Seasonality_Trader\\Seasonal_Final_Data.xlsx', index_col=0, parse_dates = True) 
    table = pnl_scanner(data, 20, 'Receive', '2021', 1, 30)
    table_html = table.to_html(index=False, escape=False, classes='table table-bordered table-hover table-condensed')
    return table_html


@callback(Output('tabs-content', 'children'),
              Input('tabs-mk', 'value'))

def render_content(tab):
    if tab == 'ST':
        return html.Div([
            html.Div([
                html.H2("Change date to number:", style={'text-align': 'center', 'padding': '20px'}),
                html.Div([
                    "Input: ",
                    dcc.Input(id='my-input', value='Date as Year-Month-day', type='text', style={'display': 'inline-block'}),
                    html.Button('Calculate', id='submit-button', n_clicks=0, style={'display': 'inline-block', 'margin-left': '10px'}),
                ], style={'text-align': 'center', 'padding': '10px'}),
                html.Div(id='my-output', style={'text-align': 'center', 'padding': '20px'}),
            ]),
            html.Div([
                html.H2("Generate Table", style={'text-align': 'center', 'padding': '20px'}),
                html.Div([
                    html.Label("Select Action:"),
                    dcc.RadioItems(
                        id='action-radio',
                        options=[
                            {'label': 'Receive', 'value': 'Receive'},
                            {'label': 'Pay', 'value': 'Pay'}
                        ],
                        value='Receive',  # Default value
                        style={'display': 'inline-block', 'margin-left': '10px'}
                    ),
                    html.Label("Day 1:  "),
                    dcc.Input(id='value1-input', value='', type='text', style={'display': 'inline-block'}),
                    html.Label("Day 2:  "),
                    dcc.Input(id='value2-input', value='', type='text', style={'display': 'inline-block'}),
                    html.Button('Generate Table', id='generate-button', n_clicks=0, style={'display': 'inline-block', 'margin-left': '10px'}),
                ], style={'text-align': 'center', 'padding': '10px'}),
            ]),
            html.Div(id='table-output'),
            html.Div([
                html.H2("Change date to number:", style={'text-align': 'center', 'padding': '20px'}),
                html.Div([
                    "Input: ",
                    dcc.Input(id='my-input', value='Date as Year-Month-day', type='text', style={'display': 'inline-block'}),
                    html.Button('Calculate', id='submit-button', n_clicks=0, style={'display': 'inline-block', 'margin-left': '10px'}),
                ], style={'text-align': 'center', 'padding': '10px'}),
                html.Div(id='my-output', style={'text-align': 'center', 'padding': '20px'}),
            ]),
            html.H2("pnl_calculator Results:", style={'text-align': 'center', 'padding': '20px'}),
            html.Div([
                html.Label("Select Action:"),
                dcc.RadioItems(
                    id='pnl-action-radio',
                    options=[
                        {'label': 'Receive', 'value': 'Receive'},
                        {'label': 'Pay', 'value': 'Pay'}
                    ],
                    value='Receive',  # Default value
                    style={'display': 'inline-block', 'margin-right': '10px'}
                ),
                html.Label("Ticker:"),
                dcc.Input(id='pnl-year-input', value='', type='text', style={'display': 'inline-block', 'margin-right': '10px'}),
                html.Label("Start Day:"),
                dcc.Input(id='pnl-start-day-input', value='', type='text', style={'display': 'inline-block', 'margin-right': '10px'}),
                html.Label("End Day:"),
                dcc.Input(id='pnl-end-day-input', value='', type='text', style={'display': 'inline-block', 'margin-right': '10px'}),
                html.Button('Calculate', id='pnl-calculate-button', n_clicks=0, style={'display': 'inline-block'}),
            ], style={'text-align': 'center', 'padding': '10px'}),
            html.Div(id='pnl-output', style={'text-align': 'center', 'padding': '20px'}),
        ])
    elif tab == 'MT':  
        return html.Div([
            html.Div([
                html.H2("Change date to number:", style={'text-align': 'center', 'padding': '20px'}),
                html.Div([
                    "Input: ",
                    dcc.Input(id='my-input', value='Date as Year-Month-day', type='text', style={'display': 'inline-block'}),
                    html.Button('Calculate', id='submit-button', n_clicks=0, style={'display': 'inline-block', 'margin-left': '10px'}),
                ], style={'text-align': 'center', 'padding': '10px'}),
                html.Div(id='my-output', style={'text-align': 'center', 'padding': '20px'}),
            ]),
            html.Div([
                html.H2("Generate Table", style={'text-align': 'center', 'padding': '20px'}),
                html.Div([
                    html.Label("Select Action:"),
                    dcc.RadioItems(
                        id='action-radio',
                        options=[
                            {'label': 'Receive', 'value': 'Receive'},
                            {'label': 'Pay', 'value': 'Pay'}
                        ],
                        value='Receive',  # Default value
                        style={'display': 'inline-block', 'margin-left': '10px'}
                    ),
                    html.Label("Window: "),
                    dcc.Input(id='day2', value='', type='text', style={'display': 'inline-block', 'margin-right': '10px'}),
                    html.Label("Day 1:  "),
                    dcc.Input(id='value1-input2', value='', type='text', style={'display': 'inline-block'}),
                    html.Label("Day 2:  "),
                    dcc.Input(id='value2-input2', value='', type='text', style={'display': 'inline-block'}),
                    html.Button('Generate Table', id='generate-button2', n_clicks=0, style={'display': 'inline-block', 'margin-left': '10px'}),
                ], style={'text-align': 'center', 'padding': '10px'}),
            ]),
            html.Div(id='table-output2'),
            html.H2("pnl_calculator Results:", style={'text-align': 'center', 'padding': '20px'}),
            html.Div([
                html.Label("Select Action:"),
                dcc.RadioItems(
                    id='pnl-action-radio2',
                    options=[
                        {'label': 'Receive', 'value': 'Receive'},
                        {'label': 'Pay', 'value': 'Pay'}
                    ],
                    value='Receive',  # Default value
                    style={'display': 'inline-block', 'margin-right': '10px'}
                ),
                html.Label("Window: "),
                dcc.Input(id='day', value='', type='text', style={'display': 'inline-block', 'margin-right': '10px'}),
                html.Label("Ticker: "),
                dcc.Input(id='pnl-year-input2', value='', type='text', style={'display': 'inline-block', 'margin-right': '10px'}),
                html.Label("Start Day: "),
                dcc.Input(id='pnl-start-day-input2', value='', type='text', style={'display': 'inline-block', 'margin-right': '10px'}),
                html.Label("End Day: "),
                dcc.Input(id='pnl-end-day-input2', value='', type='text', style={'display': 'inline-block', 'margin-right': '10px'}),
                html.Button('Calculate', id='pnl-calculate-button2', n_clicks=0, style={'display': 'inline-block'}),
            ], style={'text-align': 'center', 'padding': '10px'}),
            html.Div(id='pnl-output2', style={'text-align': 'center', 'padding': '20px'}),
        ])

@callback(
    Output('pnl-output2', 'children'),
    Input('pnl-calculate-button2', 'n_clicks'),
    State('pnl-action-radio2', 'value'),
    State('day', 'value'),
    State('pnl-year-input2', 'value'),
    State('pnl-start-day-input2', 'value'),
    State('pnl-end-day-input2', 'value')
)
def update_pnl_output2(n_clicks, selected_action, day, year, start_day, end_day):
    if n_clicks > 0:
        # Call pnl_calculator function with user inputs
        trade = year
        years = "2022"
        day = int(day)
        verbose = True
        start_day = int(start_day)
        end_day = int(end_day)
        stats = pnl_calculator2(trade, data, day, selected_action, years, start_day, end_day)
        datafr = stats[2]
        for column in datafr.columns:
            if column == 'PnL':
                datafr[column] = datafr[column].round(3)
        
        return [
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in datafr.columns])] +
                # Body
                [html.Tr([html.Td(data) for data in row]) for row in datafr.values],
                style={
                    'width': '50%',  # Adjust the width as needed
                    'margin': 'auto',  # Center the table horizontally
                    'border-collapse': 'collapse',  # Collapse table borders
                    'font-size': '14px',  # Adjust font size
                    'border': '1px solid #ddd',  # Add borders to the table
                    'text-align': 'center',  # Center align text
                    'padding': '20px'
                }
            )
        ]

    else:
        return []
    
@callback(
    Output('table-output2', 'children'),
    Input('generate-button2', 'n_clicks'),
    State('action-radio', 'value'),
    State('day2', 'value'),
    State('value1-input2', 'value'),
    State('value2-input2', 'value')
)

def generate_table2(n_clicks, selected_action, day,value1, value2):
    if n_clicks > 0:
        # Convert year to integer and other processing as needed
        year = "2022"
        value1 = int(value1)
        value2 = int(value2)
        print("day: ",day)
        day = int(day)
        # Call your pnl_scanner function with the user input values
        table = pnl_scanner(data, day, selected_action, year, value1, value2)
        table_html = table.to_html(index=False, escape=False)
        
        return [
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in table.columns])] +
                # Body
                [html.Tr([html.Td(data) for data in row]) for row in table.values],
                style={
                    'width': '50%',  # Adjust the width as needed
                    'margin': 'auto',  # Center the table horizontally
                    'border-collapse': 'collapse',  # Collapse table borders
                    'font-size': '14px',  # Adjust font size
                    'border': '1px solid #ddd',  # Add borders to the table
                    'text-align': 'center',  # Center align text
                    'padding': '20px'
                }
            )
        ]
    else:
        return []



@callback(
    Output('pnl-output', 'children'),
    Input('pnl-calculate-button', 'n_clicks'),
    State('pnl-action-radio', 'value'),
    State('pnl-year-input', 'value'),
    State('pnl-start-day-input', 'value'),
    State('pnl-end-day-input', 'value')
)
def update_pnl_output(n_clicks, selected_action, year, start_day, end_day):
    if n_clicks > 0:
        # Call pnl_calculator function with user inputs
        trade = year
        years = "2022"
        day = 20
        verbose = True
        start_day = int(start_day)
        end_day = int(end_day)
        stats = pnl_calculator(trade, data, day, selected_action, years, start_day, end_day, verbose=True)
        df_copy = stats.copy()
        mean = df_copy['PnL'].mean()
        std = df_copy['PnL'].std()
        df_copy['Win Rate'] = ((df_copy['PnL'] > 0)*1)
        winrate = df_copy['Win Rate'].mean()
        df_copy.drop('Win Rate', axis = 1, inplace=True)
        sharp = (mean / std)
        df_copy['Entry'] = pd.to_datetime(df_copy['Entry'])
        df_copy['Exit'] = pd.to_datetime(df_copy['Exit'])

        # Calculate the differences between exit and entry dates
        df_copy['TradeDuration'] = df_copy['Exit'] - df_copy['Entry']

        # Calculate the average time of the trade
        average_trade_duration = df_copy['TradeDuration'].mean()

        result = {
            'PnL': df_copy['PnL'].mean(),
            'Win Rate': winrate,
            'Sharp': sharp,
            'Average Time of Trade': [average_trade_duration]
        }

        # Convert the dictionary into a DataFrame
        new_df = pd.DataFrame(result)
        
        """
        mean = finaldf['PnL'].mean()
        std = finaldf['PnL'].std()
        finaldf['Win Rate'] = ((finaldf['PnL'] > 0)*1)
        winrate = finaldf['Win Rate'].mean()
        finaldf.drop('Win Rate', axis = 1, inplace=True)
        
        if verbose == True:
            print(type, trade)
            print('Average PnL: ' + str(round(mean, 5)))
            print('Std: ' +  str(round(std, 5)))
            for column in finaldf.columns:
                if column == 'PnL':
                    finaldf[column] = finaldf[column].round(3)
            return finaldf
        
        return mean, std, trade, winrate
        """
        return [
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in new_df.columns])] +
                # Body
                [html.Tr([html.Td(data) for data in row]) for row in new_df.values],
                style={
                    'width': '50%',  # Adjust the width as needed
                    'margin': 'auto',  # Center the table horizontally
                    'border-collapse': 'collapse',  # Collapse table borders
                    'font-size': '14px',  # Adjust font size
                    'border': '1px solid #ddd',  # Add borders to the table
                    'text-align': 'center',  # Center align text
                    'padding': '20px'
                }
            ),
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in stats.columns])] +
                # Body
                [html.Tr([html.Td(data) for data in row]) for row in stats.values],
                style={
                    'width': '50%',  # Adjust the width as needed
                    'margin': 'auto',  # Center the table horizontally
                    'border-collapse': 'collapse',  # Collapse table borders
                    'font-size': '14px',  # Adjust font size
                    'border': '1px solid #ddd',  # Add borders to the table
                    'text-align': 'center',  # Center align text
                    'padding': '20px'
                }
            )
        ]

    else:
        return []
    
@callback(
    Output('table-output', 'children'),
    Input('generate-button', 'n_clicks'),
    State('action-radio', 'value'),
    State('value1-input', 'value'),
    State('value2-input', 'value')
)

def generate_table(n_clicks, selected_action, value1, value2):
    if n_clicks > 0:
        # Your table generation logic based on user input
        # ...

        # Convert year to integer and other processing as needed
        year = "2022"
        value1 = int(value1)
        value2 = int(value2)
        print("data: ", data)
        # Call your pnl_scanner function with the user input values
        table = pnl_scanner(data, 20, selected_action, year, value1, value2)
        table_html = table.to_html(index=False, escape=False)
        
        return [
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in table.columns])] +
                # Body
                [html.Tr([html.Td(data) for data in row]) for row in table.values],
                style={
                    'width': '50%',  # Adjust the width as needed
                    'margin': 'auto',  # Center the table horizontally
                    'border-collapse': 'collapse',  # Collapse table borders
                    'font-size': '14px',  # Adjust font size
                    'border': '1px solid #ddd',  # Add borders to the table
                    'text-align': 'center',  # Center align text
                    'padding': '20px'
                }
            )
        ]
    else:
        return []


@callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    string = "2021-08-17"
    try:
        string = get_business_day_number(input_value)
        return f'Output: {string}'
    except:
        return f'Output: {"Please get date in correct format"}'
    


if __name__ == "__main__":
    app.run_server(debug=True)