import plotly.graph_objects as go

# Data for the plot
categories = ['Category A', 'Category B', 'Category C']
counts = [10, 20, 15]

# Create a bar chart
trace = go.Bar(
    x=categories,
    y=counts,
    name='Counts',
    marker_color='skyblue'
)

# Layout for the chart
layout = go.Layout(
    title='Simple Bar Chart',
    xaxis=dict(title='Category'),
    yaxis=dict(title='Count'),
)

# Create a figure and show it
fig = go.Figure(data=[trace], layout=layout)
fig.show()