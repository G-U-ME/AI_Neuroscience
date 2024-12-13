import plotly.graph_objects as go

# Data from the estimation above (using lists and dictionaries)
data_app_pb1 = {
    'Plaque Size': [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 100, 200, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050],
    'Ratio': [25, 30, 28, 15, 65, 35, 20, 40, 38, 42, 25, 50, 40, 30, 75, 45, 35, 25, 30, 32, 38, 18, 15, 22, 28, 18, 25, 5, 10, 12, 55, 45, 40, 48, 42, 35, 38, 28, 30, 25, 28],
    'Group': 'APP/PB1'
}

data_app_ps1_pb1_ko = {
    'Plaque Size': [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 100, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450],
    'Ratio': [20, 25, 22, 5, 10, 15, 8, 35, 25, 28, 15, 20, 18, 30, 15, 25, 18, 22, 15, 12, 20, 18, 15, 10, 8, 5, 50, 40, 35, 30, 25, 18, 22, 20, 15, 10, 8, 12, 15, 25],
    'Group': 'APP/PS1 PB1-KO'
}

# Create the 3D scatter plot
fig = go.Figure(data=[
    go.Scatter3d(
        x=data_app_pb1['Plaque Size'],
        y=[0] * len(data_app_pb1['Plaque Size']),
        z=data_app_pb1['Ratio'],
        mode='markers',
        marker=dict(
            size=5,
            color='black',
            symbol='circle',
            opacity=0.8
        ),
        name=data_app_pb1['Group']
    ),
    go.Scatter3d(
        x=data_app_ps1_pb1_ko['Plaque Size'],
        y=[1] * len(data_app_ps1_pb1_ko['Plaque Size']),
        z=data_app_ps1_pb1_ko['Ratio'],
        mode='markers',
        marker=dict(
            size=5,
            color='green',
            symbol='circle',
            opacity=0.8
        ),
        name=data_app_ps1_pb1_ko['Group']
    )
])

# Update layout for better visualization
fig.update_layout(
    scene=dict(
        xaxis_title='Plaque Size (μm²)',
        yaxis_title='Group',
        zaxis_title='Ratio of CD68+ area/plaque area (%)',
        yaxis=dict(tickvals=[0, 1], ticktext=['APP/PB1', 'APP/PS1 PB1-KO'])
    ),
    title='3D Scatter Plot of Plaque Size, Group, and CD68+/Plaque Area Ratio',
    legend_title='Group'
)

fig.show()