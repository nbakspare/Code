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

def date_to_day_of_year(date_str):
    date = datetime.datetime.strptime(date_str, "%m-%d")
    day_of_year = date.timetuple().tm_yday
    return day_of_year

def day_of_year_to_date(day_of_year):
    base_date = datetime.datetime(datetime.datetime.now().year, 1, 1)
    result_date = base_date + datetime.timedelta(days=day_of_year - 1)
    return result_date.strftime("%m-%d")

def find_nearest_business_day(date_str):
    date = pd.to_datetime(date_str, format='%Y-%m-%d')
    
    while date.weekday() >= 5:
        date += pd.DateOffset(days=1)

    formatted_date = date.strftime("%Y-%m-%d")
    return formatted_date

def correctdate(date):
    if date in data.index:
        return date
    else:
        return find_nearest_business_day(date)

def get_day_and_month(date_str):
    # Parse the date string into a datetime object
    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    
    # Extract day and month components
    day = date.day
    month = date.month
    
    return day, month

def pnl_calculator1(trade, data, day, type, start, end, verbose=False):
    start = day_of_year_to_date(start)
    end = day_of_year_to_date(end)
    print(start, end)
    pnls = []
    Index = data.index  
    Years = list(set(Index.year))
    Years.sort()
    Years = list(map(str, Years))

    df = data.loc[:, trade]
    res = STL(df).fit()
    starts = [str(i) + '-' + start for i in Years]
    ends = [str(i) + '-' + end for i in Years]
    #print(starts)
    #print(ends)
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
                    #print(str(Years[i]) + '-' + str(Max_Month) + '-' + str(Max_Day), str(Years[i]) + '-' + str(Min_Months) + '-' + str(Min_Day))
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
                except Exception as e:
                    print("Error in pnl calculation:", e)
            except Exception as e_outer:
                print("Error in main loop:", e_outer)
            
            
 
    finaldf = pd.DataFrame(pnls, columns = ['PnL', 'Entry', 'Exit'])
    mean = finaldf['PnL'].mean()
    std = finaldf['PnL'].std()
    finaldf['Win Rate'] = ((finaldf['PnL'] > 0)*1)
    winrate = finaldf['Win Rate'].mean()
    finaldf.drop('Win Rate', axis = 1, inplace=True)
    finaldf['PnL/Std'] = finaldf['PnL'] / std

    if verbose == True:
        print(type, trade)
        print('Average PnL: ' + str(round(mean, 5)))
        print('Std: ' +  str(round(std, 5)))
        for column in finaldf.columns:
            if column == 'PnL':
                finaldf[column] = finaldf[column].round(3)
        return finaldf
    
    return mean, std, trade, winrate

def pnl_scanner1(data, day, type, start, end):
    pnl = []
    for column in data.columns:
        stats = pnl_calculator1(column, data, day, type, start, end, verbose=False)
        pnl.append(stats)
        df = pd.DataFrame(pnl, columns = ['PnL', 'Std', ' Trade', 'Win Rate'])
        df['PnL/Std'] = df['PnL'] / df['Std']

    for column in df.columns:
        if column != ' Trade':
            df[column] = df[column].round(3)
    df = df.sort_values(by='PnL/Std', ascending = False)
    return df

def pnl_calculator(trade, data, s, e, verbose=False):
    s = day_of_year_to_date(s)
    e = day_of_year_to_date(e)
    pnls = []
    Index = data.index  
    Years = list(set(Index.year))
    Years.sort()
    Years = list(map(str, Years))
    df = data.loc[:, trade]
    starts = []
    ends = []
    result_df = pd.DataFrame()

    for i in range(len(Years)):
        try:
            start = s
            end = e
            start = correctdate(str(Years[i])+'-'+start)
            end = correctdate(str(Years[i])+'-'+end)
            date1 = pd.to_datetime(start)
            date2 = pd.to_datetime(end)
            mask = df.loc[date1:date2]
            mask.reset_index(drop=True, inplace=True)
            year = date1.year
            result_df[year] = mask
            d1, m1 = get_day_and_month(start)
            d2, m2 = get_day_and_month(end)
            Min_Months = m1
            Max_Month = m2
            Min_Day = d1
            Max_Day = d2

            if len(str(Min_Months)) < 2:
                Min_Months = '0' + str(Min_Months)
            if len(str(Max_Month)) < 2:
                Max_Month = '0' + str(Max_Month)

            try:
                pnl = df[str(Years[i]) + '-' + str(Max_Month) + '-' + str(Max_Day)] - df[str(Years[i]) + '-' + str(Min_Months) + '-' + str(Min_Day)]
                pnls.append((pnl, str(Years[i]) + '-' + str(Min_Months) + '-' + str(Min_Day), str(Years[i]) + '-' + str(Max_Month) + '-' + str(Max_Day)))
            except Exception as e:
                    print("Error in pnl calculation:", e)
        except Exception as e_outer:
            print("Error in main loop:", e_outer)
            
    finaldf = pd.DataFrame(pnls, columns = ['PnL', 'Entry', 'Exit'])
    mean = finaldf['PnL'].mean()
    std = finaldf['PnL'].std()
    finaldf['Win Rate'] = ((finaldf['PnL'] > 0)*1)
    winrate = finaldf['Win Rate'].mean()
    finaldf.drop('Win Rate', axis = 1, inplace=True)
    finaldf['PnL/Std'] = finaldf['PnL'] / std

    if verbose == True:
        print(type, trade)
        print('Average PnL: ' + str(round(mean, 5)))
        print('Std: ' +  str(round(std, 5)))
        for column in finaldf.columns:
            if column == 'PnL':
                finaldf[column] = finaldf[column].round(3)
        return finaldf, result_df
    
    return mean, std, trade, winrate

def pnl_scanner(data, start, end):
    pnl = []
    for column in data.columns:
        stats = pnl_calculator(column, data, start, end, verbose=False)
        pnl.append(stats)
        df = pd.DataFrame(pnl, columns = ['PnL', 'Std', ' Trade', 'Win Rate'])
        df['PnL/Std'] = df['PnL'] / df['Std']

    for column in df.columns:
        if column != ' Trade':
            df[column] = df[column].round(3)
    df = df.sort_values(by='PnL/Std', ascending = False)
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

data=pd.read_excel('M:\\BHCM\\Trading\\TeamSK\\Library of Strategies\\Nimit\\Seasonality_Trader\\Copy of Seasonal_Final_Data_Final.xlsx', index_col=0, parse_dates = True) 
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


@callback(Output('tabs-content', 'children'),
              Input('tabs-mk', 'value'))

def render_content(tab):
    if tab == 'ST':
        return html.Div([
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
    State('pnl-year-input2', 'value'),
    State('pnl-start-day-input2', 'value'),
    State('pnl-end-day-input2', 'value')
)
def update_pnl_output2(n_clicks, year, start_day, end_day):
    if n_clicks > 0:
        # Call pnl_calculator function with user inputs
        start_day = int(start_day)
        end_day = int(end_day)
        stats, graph_data = pnl_calculator(year, data, start_day, end_day, verbose=True)
        stats['Action'] = stats['PnL'].apply(lambda x: 'Receive' if x < 0 else 'Pay')
        stats = stats[stats['PnL'] >= 0.029]
        table = html.Table(
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
        
        fig, ax = plt.subplots(figsize=(10, 6))

        for year, values in graph_data.items():
            ax.plot(range(len(values)), values, label=f'Year {year}')

        ax.set_title('Performance by Year')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()
        plt.show()
        
        
        
        return table

    else:
        return []

    
@callback(
    Output('table-output2', 'children'),
    Input('generate-button2', 'n_clicks'),
    State('value1-input2', 'value'),
    State('value2-input2', 'value')
)

def generate_table2(n_clicks,value1, value2):
    if n_clicks > 0:
        value1 = int(value1)
        value2 = int(value2)
        # Call your pnl_scanner function with the user input values
        table = pnl_scanner(data, value1, value2)
        table_html = table.to_html(index=False, escape=False)
        table['Action'] = table['PnL'].apply(lambda x: 'Receive' if x < 0 else 'Pay')
        table = table[table['PnL'] >= 0.029]
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
        day = 20
        start_day = int(start_day)
        end_day = int(end_day)
        stats = pnl_calculator1(trade, data, day, selected_action, start_day, end_day, verbose=True)
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
            'PnL': round(df_copy['PnL'].mean(),3),
            'Sharp': round(sharp,3),
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
        value1 = int(value1)
        value2 = int(value2)
        # Call your pnl_scanner function with the user input values
        table = pnl_scanner1(data, 20, selected_action, value1, value2)
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
    string = ""
    try:
        string = date_to_day_of_year(input_value)
        return f'Output: {string}'
    except:
        return f'Output: {"Please get date in correct format"}'
    


if __name__ == "__main__":
    app.run_server(debug=True)