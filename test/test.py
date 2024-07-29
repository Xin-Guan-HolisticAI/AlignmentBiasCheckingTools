import plotly.graph_objects as go
import plotly.io as pio

# Create a simple scatter plot
fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode='markers'))


fig.to_image(format="png", engine="kaleido")
# Save the plot as a static image
fig.write_image('scatter_plot.png')
