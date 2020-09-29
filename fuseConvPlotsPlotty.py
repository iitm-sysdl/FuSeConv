import plotly.graph_objects as go
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
from operator import add
import plotly.express as px


#####

networks = ['MobileNet-V1', 'MobileNet-V2', 'MnasNet-B1', 'MobileNet<br>Small', 'MobileNet-V3<br>Large']

baseline = [2536900, 2979156, 2748684, 700161, 2018189]
v1 = [375260, 411910, 384202, 167935, 370097]
v2 = [621300, 576770, 542818, 231735, 558553]
h1 = [1072300, 1375054, 1394346, 414921, 1102045]
h2 = [1151124, 1453926, 1458602, 437559, 1144825]
baselineSpeedup = [1, 1, 1, 1, 1]
v1Speedup = [6.760379470233971, 7.232541089072856, 7.154267807039005, 4.169238098073659, 5.4531352591347675]
v2Speedup = [4.08321261870272, 5.165240910588276, 5.063730384769849, 3.021386497507929, 3.613245296328191]
h1Speedup = [2.3658491093910285, 2.1665738218280883, 1.9713069783253223, 1.687456166354559, 1.8313126959425432]
h2Speedup = [2.2038459801029253, 2.049042385926106, 1.8844647134722152, 1.6001522080450865, 1.7628799161443889]


fig = go.Figure()
fig.add_trace(go.Bar(
    x=networks,
    y=baseline,
    name='Baseline',
    marker_color='lightslategray'
))
fig.add_trace(go.Bar(
    x=networks,
    y=v2,
    name='Full FuSeConv',
    marker_color='rgb(99,110,250)'
))
fig.add_trace(go.Bar(
    x=networks,
    y=v1,
    name='Half FuSeConv',
    marker_color='rgb(239,85,59)'
))
fig.add_trace(go.Bar(
    x=networks,
    y=h2,
    name='Full-50% FuSeConv',
    marker_color='rgba(99,110,250, 0.5)'
))
fig.add_trace(go.Bar(
    x=networks,
    y=h1,
    name='Half-50% FuSeConv',
    marker_color='rgba(239,85,59, 0.5)'
))


fig.update_layout(
    # font_family="Source Code Pro",
    font_family="Arial",
    font_color="gray",
    title_font_color="red", font_size=14, width=600, height=400,
    legend_x = 0.43, legend_y = 1,
    legend_font_color = 'black',
    paper_bgcolor='rgba(0,0,0,0)'
    )

fig.update_layout(
    annotations=[
        dict(
            x=-0.15,
            y=0.5,
            showarrow=False,
            text="Latency in cycles",
            xref="paper",
            yref="paper",
            textangle=-90,
            font_color="black"
        )
    ]
)

fig.update_layout(barmode='group', xaxis_tickangle=0)
fig.write_image("fig1.pdf")

#####


speedup = [5.703469079939668, 6.754551295780681, 9.913109756097562, 7.354194544518785, 7.778477779669675, 7.778477779669675, 4.450152078404867, 5.015578947368421, 5.015578947368421, 5.015578947368421, 3.913938260056127, 4.264058679706602, 4.264058679706602, 2.674855668836752, 2.5994878995622366, 2.5994878995622366, 2.1198661393367813];
import plotly.graph_objects as go

layers = []

for i in range(17):
    layers.append('MB'+str(i+1))

fig = go.Figure(go.Bar(
            x=speedup,
            y=layers,
            marker_color=["rgb(99,110,250)"]*6 + ["lightslategray"]*11,
            orientation='h'))

fig.update_layout(
    # font_family="Source Code Pro",
    font_family="Arial",
    font_color="gray",
    title_font_color="red", font_size=14, width=300, height=400,
    legend_x = 0.43, legend_y = 1,
    legend_font_color = 'black',
    paper_bgcolor='rgba(0,0,0,0)'
    )

fig.update_layout(
    annotations=[
        dict(
            x=0.5,
            y=-0.18,
            showarrow=False,
            text="Speed-up",
            xref="paper",
            yref="paper",
            font_color="black"
        ),
        dict(
            x=-0.4,
            y=0.25,
            showarrow=False,
            text="Layers",
            xref="paper",
            yref="paper",
            textangle=-90,
            font_color="black"
        )
    ]
)

fig.write_image('fig2.pdf')


######
## Variant-2

otherConv = [265913, 265913, 265913, 132956, 132956] 
pointConv = [9329664, 5327220, 5309432, 1068178, 3987903] 
depthConv = [4162560, 4968416, 5652608, 1407400, 3794168]
linear = [17384, 21480, 21480, 32890, 51546]
fusepointconv =  [18021376, 7264628, 7405192, 1360322, 5318975]
fusedepthconv =  [837760, 1010240, 869288, 224896, 661472]
linearfuse = [17384, 21480, 21480, 57764, 130032]

otherConv = list(map(add, otherConv, linear))
otherConvfuse= list(map(add, otherConv, linearfuse))
total = list(map( add, list(map(add, otherConv, pointConv)), depthConv))
totalFriendly = list(map( add, list(map(add, fusedepthconv, fusepointconv)), otherConvfuse))

percentoc = [i / j * 100 for i,j in zip(otherConv, total)]
percentpc = [i / j * 100 for i,j in zip(pointConv, total)]
percentdc = [i / j * 100 for i,j in zip(depthConv, total)]

percentocf = [i / j * 100 for i,j in zip(otherConvfuse, totalFriendly)]
percentpcf = [i / j * 100 for i,j in zip(fusepointconv, totalFriendly)]
percentdcf = [i / j * 100 for i,j in zip(fusedepthconv, totalFriendly)]


networks =  ['MobileNet-V1', 'MobileNet-V2', 'MnasNet-B1', 'MobileNet-V3<br>Small', 'MobileNet-V3<br>Large']

cbar1 = px.colors.qualitative.Set2;
cbar2 = px.colors.qualitative.Pastel2;


fig = go.Figure(
    data=[
        go.Bar(
            name="Standard",
            y=networks,
            x=percentoc,
            offsetgroup=0,
            orientation="h",
            marker_color=cbar1[0],
            legendgroup="depthwise"
        ),
        go.Bar(
            name="Point-wise",
            y=networks,
            x=percentpc,
            offsetgroup=0,
            base=percentoc,
            orientation="h",
            marker_color=cbar1[1],
            legendgroup="depthwise"
        ),
        go.Bar(
            name="Depthwise",
            y=networks,
            x=percentdc,
            offsetgroup=0,
            base=list(map(add, percentpc, percentoc)),
            orientation="h",
            marker_color=cbar1[2],
            legendgroup="depthwise"
        ),
        go.Bar(
            name="Standard",
            y=networks,
            x=percentocf,
            offsetgroup=1,
            orientation="h",
            marker_color=cbar2[0],
            legendgroup="FuSeConv"
        ),
        go.Bar(
            name="Point-wise",
            y=networks,
            x=percentpcf,
            offsetgroup=1,
            base=percentocf,
            orientation="h",
            marker_color=cbar2[1],
            legendgroup="FuSeConv"
        ),
        go.Bar(
            name="FuSe",
            y=networks,
            x=percentdcf,
            offsetgroup=1,
            base=list(map(add, percentpcf, percentocf)),
            orientation="h",
            marker_color=cbar2[2],
            legendgroup="FuSeConv"
        ),
    ]
)
fig.update_layout(
    font_family="Arial",
    font_color="gray",
    title_font_color="red", font_size=14, width=400, height=400,
    legend_x = 1, legend_y = 1,
    legend_font_color = 'black',
    paper_bgcolor='rgba(0,0,0,0)'
    )

fig.update_layout(
    annotations=[
        dict(
            x=0.5,
            y=-0.18,
            showarrow=False,
            text="Percentage of cycles",
            xref="paper",
            yref="paper",
            font_color="black"
        ),
        dict(
        	x=1.4,
            y=1.05,
            showarrow=False,
            text="Baseline",
            xref="paper",
            yref="paper",
            font_color="black"
        ),
        dict(
        	x=1.46,
            y=0.71,
            showarrow=False,
            text="FuSeConv",
            xref="paper",
            yref="paper",
            font_color="black"
        )
    ]
)
fig.write_image('fig3New.pdf')

#####

Speedup =[[2.023718468729478, 4.08321261870272, 6.777745410408133, 8.636542215565186], [3.350260103199137, 5.165240910588276, 6.83670060674575, 7.891341918938507], [3.31582778991125, 5.063730384769849, 6.456343997050969, 7.171005322794138], [2.3800222973367635, 3.021386497507929, 3.618686195106574, 3.9708378515144944], [2.480129194282618, 3.613245296328191, 4.819822954009899, 5.639453565269347]]
speedup = np.array(Speedup)


# Create traces
fig = go.Figure()
x = ['32x32', '64x64', '128x128', '256x256']
fig.add_trace(go.Scatter(x=x, y=speedup[3],
                    mode='lines+markers',
                    name='MobileNet-V3 Small'))
fig.add_trace(go.Scatter(x=x, y=speedup[4],
                    mode='lines+markers',
                    name='MobileNet-V3 Large'))
fig.add_trace(go.Scatter(x=x, y=speedup[2],
                    mode='lines+markers',
                    name='MnasNet-B1'))
fig.add_trace(go.Scatter(x=x, y=speedup[1],
                    mode='lines+markers',
                    name='MobileNet-V2'))
fig.add_trace(go.Scatter(x=x, y=speedup[0],
                    mode='lines+markers',
                    name='MobileNet-V1'))


fig.update_layout(
    font_family="Arial",
    font_color="gray",
    legend_font_family="Arial",
    legend_font_color="black",
    title_font_color="red", font_size=14, width=375, height=400,
    legend_x = 0, legend_y = 1,
    paper_bgcolor='rgba(0,0,0,0)'
    )

fig.update_layout(
    annotations=[
        dict(
            x=0.5,
            y=-0.18,
            showarrow=False,
            text="Systolic Array Size",
            xref="paper",
            yref="paper",
            font_color="black"
        ),
        dict(
            x=-0.18,
            y=0.5,
            showarrow=False,
            text="Speed-up",
            xref="paper",
            yref="paper",
            textangle=-90,
            font_color="black"
        ),
    ]
)

fig.write_image('fig4.pdf')