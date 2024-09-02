from dash import Dash, dcc, html, Input, Output, callback, State
import pandas as pd
from dash_table import DataTable
from dash.exceptions import PreventUpdate

# Read the CSV file or create an empty DataFrame if it doesn't exist
csv_path = 'data.csv'
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=['Name', 'Age', 'City'])

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Editable Table"),
    
    DataTable(
        id='editable-table',
        columns=[
            {'name': col, 'id': col, 'editable': True} for col in df.columns
        ],
        data=df.to_dict('records'),
        editable=True,
        row_deletable=True,
        page_current=0,  # Current page
        page_size=10,    # Number of rows per page
    ),

    html.Button('Add Row', id='add-row-button', n_clicks=0),
    html.Button('Save Changes', id='save-button', n_clicks=0),
    html.Div(id='output-message', style={'margin-top': 20}),
    
    # Hidden div to store the data
    html.Div(id='hidden-div', style={'display': 'none'}, children=df.to_json(orient='split'))
])

@app.callback(
    Output('editable-table', 'data'),
    [Input('add-row-button', 'n_clicks')],
    [State('hidden-div', 'children')],
    prevent_initial_call=True
)
def add_row(n_clicks, hidden_data):
    df_from_json = pd.read_json(hidden_data, orient='split')
    if n_clicks > 0:
        new_row = dict(zip(df_from_json.columns, [None] * len(df_from_json.columns)))
        df_from_json = pd.concat([df_from_json, pd.DataFrame([new_row])], ignore_index=True)
    return df_from_json.to_dict('records')

@app.callback(
    Output('output-message', 'children'),
    [Input('save-button', 'n_clicks')],
    [State('editable-table', 'data')],
    prevent_initial_call=True
)
def save_changes(n_clicks, table_data):
    if n_clicks > 0:
        df = pd.DataFrame(table_data)
        df.to_csv(csv_path, index=False)
        return "Changes saved successfully!"
    else:
        raise PreventUpdate()

if __name__ == '__main__':
    app.run_server(debug=True)