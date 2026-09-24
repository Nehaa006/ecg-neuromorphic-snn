import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

app = dash.Dash(__name__, title="ZedBoard SNN ECG Dashboard")

app.layout = html.Div(style={'backgroundColor': '#121212', 'color': '#ffffff', 'fontFamily': 'Arial', 'padding': '20px'}, children=[
    html.H2("⚡ ZedBoard Neuromorphic ECG Classifier Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([html.H4("Active Model"), html.H3("6-Bit Quantized rMLP", style={'color': '#00e5ff'})], 
                 style={'width': '22%', 'backgroundColor': '#1e1e1e', 'padding': '15px', 'borderRadius': '10px'}),
        html.Div([html.H4("Target Accuracy"), html.H3("98.22% (AAMI)", style={'color': '#00ff88'})], 
                 style={'width': '22%', 'backgroundColor': '#1e1e1e', 'padding': '15px', 'borderRadius': '10px'}),
        html.Div([html.H4("Firing Sparsity"), html.H3("15.4% Activity", style={'color': '#ffcc00'})], 
                 style={'width': '22%', 'backgroundColor': '#1e1e1e', 'padding': '15px', 'borderRadius': '10px'}),
        html.Div([html.H4("Energy Metric"), html.H3("80.4% Savings (AC)", style={'color': '#ff4081'})], 
                 style={'width': '22%', 'backgroundColor': '#1e1e1e', 'padding': '15px', 'borderRadius': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),

    html.Div([
        dcc.Graph(id='live-ecg-spikes', style={'width': '65%', 'display': 'inline-block'}),
        dcc.Graph(id='class-probabilities', style={'width': '33%', 'display': 'inline-block'}),
    ]),

    dcc.Interval(id='timer-refresh', interval=500, n_intervals=0)
])

@app.callback(
    [Output('live-ecg-spikes', 'figure'), Output('class-probabilities', 'figure')],
    [Input('timer-refresh', 'n_intervals')]
)
def refresh_views(n):
    t = np.linspace(0, 1, 180)
    signal = np.sin(2 * np.pi * 2 * t) + 0.3 * np.random.randn(180)
    spikes = (signal > 0.5).astype(int)

    fig_ecg = go.Figure()
    fig_ecg.add_trace(go.Scatter(y=signal, mode='lines', name='ECG Signal', line=dict(color='#00e5ff')))
    fig_ecg.add_trace(go.Scatter(y=spikes * 1.2, mode='markers', name='LC Spikes', marker=dict(color='#ff4081', size=6)))
    fig_ecg.update_layout(title="Live Level-Crossing Stream", template="plotly_dark", height=380)

    classes = ['N', 'S', 'V', 'F', 'Q']
    probs = [0.95, 0.02, 0.01, 0.01, 0.01]
    fig_prob = go.Figure(go.Bar(x=classes, y=probs, marker_color='#00ff88'))
    fig_prob.update_layout(title="AAMI Classification Logits", template="plotly_dark", height=380)

    return fig_ecg, fig_prob

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)