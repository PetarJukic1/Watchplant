from dash import Dash, dcc, html, Input, Output
import dash_daq as daq
import zmq
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import timedelta
from datetime import date
from waitress import serve
from flask import Flask

rpi = []

rpi.append(pd.DataFrame())
rpi.append(pd.DataFrame())
rpi.append(pd.DataFrame())
rpi.append(pd.DataFrame())

def double_y_subplots(dfs, x_label, y_labels, x_value, y_values, y_names, title):
    # Create figure with secondary y-axis
    fig = make_subplots(rows=len(dfs), cols=1, shared_xaxes=False, x_title=x_label,
                        specs=[[{"secondary_y": True}] for _ in range(len(dfs))])

    for i in range(len(dfs)):
        # Add traces
        if not rpi[i].empty:
            fig.add_trace(
                go.Scatter(x=dfs[i][x_value], y=dfs[i][y_values[0]], 
                        name=y_names[0].format(i),
                        mode='lines'),
                row=i+1, col=1,
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=dfs[i][x_value], y=dfs[i][y_values[1]], 
                        name=y_names[1].format(i),
                        mode='lines'),
                row=i+1, col=1,
                secondary_y=True,
            )
            # Add figure title
            fig.update_layout(
                title_text=title,
                height=250*len(dfs), width=max(1000, 250*len(dfs)),
            )

            # Set y-axes titles
            fig.update_yaxes(title_text=y_labels[0], secondary_y=False)
            fig.update_yaxes(title_text=y_labels[1], secondary_y=True)
        else:
            fig.add_trace(
                go.Scatter(
                        name=y_names[0].format(i),
                        mode='lines'),
                row=i+1, col=1,
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(name=y_names[1].format(i)),
                row=i+1, col=1,
                secondary_y=True,
            )
            # Add figure title
            fig.update_layout(
                title_text=title,
                height=250*len(dfs), width=max(1000, 250*len(dfs)),
            )

            # Set y-axes titles
            fig.update_yaxes(title_text=y_labels[0], secondary_y=False)
            fig.update_yaxes(title_text=y_labels[1], secondary_y=True)

    return fig

def demogrify(topicmsg):
    """ Inverse of mogrify() """
    json0 = topicmsg.find('{')
    topic = topicmsg[0:json0].strip()
    msg = json.loads(topicmsg[json0:])
    return topic, msg 

def create_zmq_socket(zmq_port="5556", topicfilter="data"):
    """ Create a ZMQ SUBSCRIBE socket """
    context = zmq.Context()
    zmq_socket = context.socket(zmq.SUB)
    zmq_socket.connect ("tcp://server:%s" % zmq_port)
    zmq_socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)
    return zmq_socket

def recv_zmq(topic='data'):
    """ Receive data over ZMQ PubSub socket
    Args:
        socket: zmq.socket
        topic: topic subscribed to
    Returns numpy array data
    """
    with create_zmq_socket() as socket:
        topic, msg = demogrify(socket.recv_string())
    return msg

 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.Label(['Graphs']),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'External temperature', 'value': 'External temperature'},
            {'label': 'External light', 'value': 'External light'},
            {'label': 'Humidity and transpiration', 'value': 'Humidity and transpiration'},
            {'label': 'Soil temperature and moisture', 'value': 'Soil temperature and moisture'},
            {'label': 'Soil moisture and transpiration', 'value': 'Soil moisture and transpiration'},
            {'label': 'Differential potential', 'value': 'Differential potential'},
        ],
        value='External temperature',
        multi=False,
        clearable=False,
        style={"width":"50%"},
    ),  
    html.Label(['Select amount of days to show:']),
    html.Div([dcc.RangeSlider(
        id='time-slider',
        marks={
            1: '1 day',
            2: '2 days',
            3: '3 days',
            4: '4 days',
            5: '5 days',
            6: '6 days',
            7: '7 days',
        },
        step=1,
        min=1,
        max=7,
        value=[1],
        allowCross=False,
        updatemode='mouseup',
        included=True,
    ),
    ], style={"width": "80%"}),
    html.Div(
        id='graph-div',
        children=[]
    ),
    dcc.Interval(
        id='interval-component',
        interval=1*1500, # in milliseconds
        n_intervals=0
    )
])

server = Flask(__name__)
server = app.server
@server.route("/")
def display():
    return "hello"

# The updating is done with this callback
@app.callback(
    Output('graph-div', 'children'),
    Input(component_id='interval-component', component_property='n_intervals'),
    Input(component_id='dropdown', component_property='value'),
    Input(component_id='time-slider', component_property='value'))
def update(n, dropdown, days):
    data = recv_zmq('data')
    dataframe = pd.DataFrame([data])
    #print(dataframe)
    #dataframe['timestamp'] = datetime.fromtimestamp(dataframe['timestamp']/1000)
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='ms')
    if (dataframe['sender_hostname'] == 'rpi0').values[0]:
        rpi[0] = rpi[0].append(dataframe, ignore_index=True)
    elif (dataframe['sender_hostname'] == 'rpi1').values[0]:
        rpi[1] = rpi[1].append(dataframe, ignore_index=True)
    elif (dataframe['sender_hostname'] == 'rpi2').values[0]:
        rpi[2] = rpi[2].append(dataframe, ignore_index=True)
    elif (dataframe['sender_hostname'] == 'rpi3').values[0]:
        rpi[3] = rpi[3].append(dataframe, ignore_index=True)
    
    rpi_temp = []
    rpi_temp.append(pd.DataFrame())
    rpi_temp.append(pd.DataFrame())
    rpi_temp.append(pd.DataFrame())
    rpi_temp.append(pd.DataFrame())

    max_time = date(2000,1,1)
    select_range = True
    if select_range:
        for n in range(len(rpi)):
            if not rpi[n].empty:
                max_time = max(rpi[n]["timestamp"][len(rpi[n]["timestamp"])-1], max_time)
        start_datetime  = (max_time)-timedelta(days=days[0])
        end_datetime = max_time  
        for n in range(len(rpi)):
            if not rpi[n].empty:    
                mask = (rpi[n]['timestamp'] > start_datetime) & (rpi[n]['timestamp'] <= end_datetime)
                rpi_temp[n] = rpi[n].loc[mask]

    print(rpi_temp[0])
    frames = [i for i in rpi]
    rpi_concat = pd.concat(frames)
    mask_concat =  (rpi_concat['timestamp'] > start_datetime) & (rpi_concat['timestamp'] <= end_datetime)

    if dropdown == "External temperature":    
        fig1 = px.line(rpi_concat.loc[mask_concat], x="timestamp", y="temp-external", 
            color='sender_hostname',
            labels={"sender_hostname":"Raspberry Pi name", "temp-external":"Temperature [°C]", "timestamp":"Time"}, 
            title='External temperature')

        fig2 = go.Figure()
        for n in range(len(rpi_temp)):
            if not rpi_temp[n].empty:
                fig2.add_trace(go.Violin(x=rpi_temp[n]['sender_hostname'],
                                        y=rpi_temp[n]['temp-external'],
                                        name=rpi_temp[n]['sender_hostname'].iloc[0],
                                        box_visible=True,
                                        meanline_visible=True))
        fig2.update_layout(title='External temperature comparison',showlegend=False)
        fig2.update_yaxes(title_text="Temperature [°C]")
        fig2.update_xaxes(title_text="Measurement units")
        
        fig3 = double_y_subplots(rpi_temp,
                        "Time",
                        ["Temperature [°C]", "Diff. potential"],
                        "timestamp",
                        ["temp-external", "differential_potential_CH1"],
                        ["RPi{}: External temperature", "RPi{}: Diff. potential CH1"],
                        "Temperature vs. Differential potential of CH1")

        fig4 = double_y_subplots(rpi_temp,
                        "Time",
                        ["Temperature [°C]", "Diff. potential"],
                        "timestamp",
                        ["temp-external", "differential_potential_CH2"],
                        ["RPi{}: External temperature", "RPi{}: Diff. potential CH2"],
                        "Temperature vs. Differential potential of CH2")

        return [
            dcc.Graph(id='my-graph1', figure=fig1, clickData=None, hoverData=None,
                config={
                    'staticPlot':False,
                    'scrollZoom':False,
                    'doubleClick':'reset',
                    'showTips':True,
                    'displayModeBar':True,
                    'watermark':True,
                },
                className='six colimns'),

                dcc.Graph(id='my-graph2', figure=fig2, clickData=None, hoverData=None,
                config={
                    'staticPlot':False,
                    'scrollZoom':False,
                    'doubleClick':'reset',
                    'showTips':True,
                    'displayModeBar':True,
                    'watermark':True,
                },
                className='six colimns'),

                dcc.Graph(id='my-graph3', figure=fig3, clickData=None, hoverData=None,
                config={
                    'staticPlot':False,
                    'scrollZoom':False,
                    'doubleClick':'reset',
                    'showTips':True,
                    'displayModeBar':True,
                    'watermark':True,
                },
                className='six colimns'),

                dcc.Graph(id='my-graph4', figure=fig4, clickData=None, hoverData=None,
                config={
                    'staticPlot':False,
                    'scrollZoom':False,
                    'doubleClick':'reset',
                    'showTips':True,
                    'displayModeBar':True,
                    'watermark':True,
                },
                className='six colimns'),
            ]

    if dropdown == "External light":

        fig5 = px.line(rpi_concat.loc[mask_concat], x="timestamp", y="light-external", 
                        color='sender_hostname',
                        labels={"sender_hostname":"Raspberry Pi name", "light-external":"Light [Lux]", "timestamp":"Time"}, 
                        title='External light')

        fig6 = double_y_subplots(rpi_temp,
                        "Time",
                        ["Light [Lux]", "Diff. potential"],
                        "timestamp",
                        ["light-external", "differential_potential_CH1"],
                        ["RPi{}: External light", "RPi{}: Diff. potential CH1"],
                        "Light vs. Differential potential of CH1")

        fig7 = double_y_subplots(rpi_temp,
                        "Time",
                        ["Light [Lux]", "Diff. potential"],
                        "timestamp",
                        ["light-external", "differential_potential_CH2"],
                        ["RPi{}: External light", "RPi{}: Diff. potential CH2"],
                        "Light vs. Differential potential of CH2")

        return [
            dcc.Graph(id='my-graph5', figure=fig5, clickData=None, hoverData=None,
                config={
                    'staticPlot':False,
                    'scrollZoom':False,
                    'doubleClick':'reset',
                    'showTips':True,
                    'displayModeBar':True,
                    'watermark':True,
                },
                className='six colimns'),

                dcc.Graph(id='my-graph6', figure=fig6, clickData=None, hoverData=None,
                config={
                    'staticPlot':False,
                    'scrollZoom':False,
                    'doubleClick':'reset',
                    'showTips':True,
                    'displayModeBar':True,
                    'watermark':True,
                },
                className='six colimns'),

                dcc.Graph(id='my-graph7', figure=fig7, clickData=None, hoverData=None,
                config={
                    'staticPlot':False,
                    'scrollZoom':False,
                    'doubleClick':'reset',
                    'showTips':True,
                    'displayModeBar':True,
                    'watermark':True,
                },
                className='six colimns'),
            ]

    if dropdown == "Humidity and transpiration":

        fig8 = double_y_subplots(rpi_temp,
                        "Time",
                        ["External humidity [%]", "Transpiration"],
                        "timestamp",
                        ["humidity-external", "transpiration"],
                        ["RPi{}: External humidity", "RPi{}: Transpiration"],
                        "External humidity vs. transpiration")

        return [
            dcc.Graph(id='my-graph8', figure=fig8, clickData=None, hoverData=None,
            config={
                'staticPlot':False,
                'scrollZoom':False,
                'doubleClick':'reset',
                'showTips':True,
                'displayModeBar':True,
                'watermark':True,
            },
            className='six colimns'),
        ]

    if dropdown == "Soil temperature and moisture":
                
        fig9 = double_y_subplots(rpi_temp,
                        "Time",
                        ["Soil moisture [?]", "Soil temperature [°C]"],
                        "timestamp",
                        ["soil_moisture", "soil_temperature"],
                        ["RPi{}: Soil moisture", "RPi{}: Soil temperature"],
                        "Soil moisture and temperature")

        fig10 = double_y_subplots(rpi_temp,
                        "Time",
                        ["Soil moisture [?]", "Diff. potential"],
                        "timestamp",
                        ["soil_moisture", "differential_potential_CH1"],
                        ["RPi{}: Soil moisture", "RPi{}: Diff. potential CH1"],
                        "Soil moisture vs. Differential potential of CH1")

        fig11 = double_y_subplots(rpi_temp,
                        "Time",
                        ["Soil moisture [?]", "Diff. potential"],
                        "timestamp",
                        ["soil_moisture", "differential_potential_CH2"],
                        ["RPi{}: Soil moisture", "RPi{}: Diff. potential CH2"],
                        "Soil moisture vs. Differential potential of CH2")

        return [
            dcc.Graph(id='my-graph9', figure=fig9, clickData=None, hoverData=None,
                config={
                    'staticPlot':False,
                    'scrollZoom':False,
                    'doubleClick':'reset',
                    'showTips':True,
                    'displayModeBar':True,
                    'watermark':True,
                },
                className='six colimns'),

                dcc.Graph(id='my-graph10', figure=fig10, clickData=None, hoverData=None,
                config={
                    'staticPlot':False,
                    'scrollZoom':False,
                    'doubleClick':'reset',
                    'showTips':True,
                    'displayModeBar':True,
                    'watermark':True,
                },
                className='six colimns'),

                dcc.Graph(id='my-graph11', figure=fig11, clickData=None, hoverData=None,
                config={
                    'staticPlot':False,
                    'scrollZoom':False,
                    'doubleClick':'reset',
                    'showTips':True,
                    'displayModeBar':True,
                    'watermark':True,
                },
                className='six colimns'),
        ]

    if dropdown == "Soil moisture and transpiration":
        
        fig12 = double_y_subplots(rpi_temp,
                "Time",
                ["Soil moisture [?]", "Transpiration [%]"],
                "timestamp",
                ["soil_moisture", "transpiration"],
                ["RPi{}: Soil moisture", "RPi{}: Transpiration"],
                "Soil moisture and transpiration")

        return [
            dcc.Graph(id='my-graph12', figure=fig12, clickData=None, hoverData=None,
                config={
                    'staticPlot':False,
                    'scrollZoom':False,
                    'doubleClick':'reset',
                    'showTips':True,
                    'displayModeBar':True,
                    'watermark':True,
                },
                className='six colimns'),
        ]
    
    if dropdown == "Differential potential":
            
        # Create figure with secondary y-axis
        fig13 = make_subplots(rows=len(rpi_temp), cols=1, shared_xaxes=False, x_title='Time')

        for i in range(len(rpi_temp)):
            mini = rpi_temp[i]['differential_potential_CH1'].min()
            maxi = rpi_temp[i]['differential_potential_CH1'].max()
            if (maxi != mini):
                ch1 = (rpi_temp[i]['differential_potential_CH1'] - mini) / (maxi - mini)
            else:
                ch1 = rpi_temp[i]['differential_potential_CH1'] / mini
            
            mini = rpi_temp[i]['differential_potential_CH2'].min()
            maxi = rpi_temp[i]['differential_potential_CH2'].max()
            ch2 = (rpi_temp[i]['differential_potential_CH2'] - mini) / (maxi - mini)
            if (maxi != mini):
                ch2 = (rpi_temp[i]['differential_potential_CH2'] - mini) / (maxi - mini)
            else:
                ch2 = rpi_temp[i]['differential_potential_CH2'] / mini
            
            
            # Add traces
            fig13.add_trace(
                go.Scatter(x=rpi_temp[i]['timestamp'], y=ch1, 
                        name="Differential potential CH1 RPi{}".format(i),
                        mode='lines'),
                row=i+1, col=1,
            )
            fig13.add_trace(
                go.Scatter(x=rpi_temp[i]['timestamp'], y=ch2, 
                        name="Differential potential CH2 RPi{}".format(i),
                        mode='lines'),
                row=i+1, col=1,
            )
            # Add figure title
            fig13.update_layout(
                title_text="Differential potential of CH1 and CH2",
                height=1000, width=1500,
                legend = dict(font = dict(size = 16, color = "black")),
            )

            # Set y-axes titles
            fig13.update_yaxes(title_text="Differential potential, [uV]")

            return [
                dcc.Graph(id='my-graph13', figure=fig13, clickData=None, hoverData=None,
                    config={
                        'staticPlot':False,
                        'scrollZoom':False,
                        'doubleClick':'reset',
                        'showTips':True,
                        'displayModeBar':True,
                        'watermark':True,
                    },
                    className='six colimns'),
            ]

if __name__ == '__main__':
    app.title = 'Watchplant'
    # app.run_server(debug=False, host="0.0.0.0", port=8050)
    serve(server, host = '0.0.0.0', port = 8080)