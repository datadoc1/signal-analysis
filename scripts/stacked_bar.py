import plotly.graph_objects as go

# Data
strategies = ['Random', 'Optimal']
gold_values = [2.7, 5.25]
silver_values = [2.0, 4.67]

# Create stacked bar chart
fig = go.Figure(data=[
    go.Bar(name='Gold', x=strategies, y=gold_values, marker_color='gold'),
    go.Bar(name='Silver', x=strategies, y=silver_values, marker_color='silver')
])

# Update layout for stacked bars
fig.update_layout(
    barmode='stack',
    title='Expected Interviews by Signal Strategy',
    yaxis_title='Total Expected Interviews',
    xaxis_title='Signal Strategy',
    showlegend=True,
)

import os

# Create the directory if it doesn't exist
os.makedirs('reports/Diagnostic Radiology', exist_ok=True)

# Save the figure as PNG
fig.write_image('reports/Diagnostic Radiology/stacked_bar.png')