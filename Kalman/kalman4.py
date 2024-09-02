def generate_html(table):
    name = (table.iloc[0])['name']
    error = (table.iloc[0])['error']
    average_count = table['Count'].mean(skipna=True)
    table['Reversion'] = pd.to_numeric(table['Reversion'], errors='coerce')
    average_revert = table['Reversion'].mean(skipna=True)
    average_PnL = table['PnL'].mean(skipna=True)
    max_value_column1 = table['PnL'].max()
    min_value_column1 = table['PnL'].min()
    worst_mkm = table['Mkm'].min()

    split_parts = name.split(', ', 1)

    first_part = split_parts[0]
    second_part = split_parts[1] if len(split_parts) > 1 else None


    ###### GRAPH ##############
    data_k = pd.read_csv('datafinal.csv', index_col ='Dates', parse_dates=True)
    column_names = data_k.columns
    column_names_list = column_names.tolist()
    rate_1 = data_k[column_names_list[0]]
    rate_2 = data_k[column_names_list[1]]
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
                                html.Li([html.Strong("Average days off-side: "), html.Span((str(average_count)), style={"color": "blue"})]),
                                html.Li([html.Strong("Average days to revert: "), html.Span(str(average_revert), style={"color": "blue"})]),
                                html.Li([html.Strong("Average PnL "), html.Span(str(average_PnL), style={"color": "blue"})]),
                                html.Li([html.Strong("Max PnL: "), html.Span((str(max_value_column1)), style={"color": "blue"})]),
                                html.Li([html.Strong("Worst PnL "), html.Span(str(min_value_column1), style={"color": "blue"})]),
                                html.Li([html.Strong("Worst Mkm: "), html.Span(str(worst_mkm), style={"color": "blue"})]),
                            ])
                        ]
                    )
                ]
            )
        ]
    )