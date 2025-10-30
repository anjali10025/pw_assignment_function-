#Q:5
from bokeh.plotting import figure, show
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.transform import transform
import numpy as np

# Create random 10x10 dataset
data_heatmap = np.random.rand(10, 10)

# Create coordinate arrays for the rectangles
xx, yy = np.meshgrid(np.arange(10), np.arange(10))
xx = xx.flatten()
yy = yy.flatten()

# Create a color mapper for the heatmap
mapper = LinearColorMapper(
    palette="Viridis256",
    low=data_heatmap.min(),
    high=data_heatmap.max()
)

# Create figure
p = figure(
    title="Bokeh Heatmap",
    x_axis_label='X',
    y_axis_label='Y',
    x_range=(-0.5, 9.5),
    y_range=(-0.5, 9.5),
    width=600,
    height=600
)

# Draw rectangles for heatmap
p.rect(
    x=xx,
    y=yy,
    width=1,
    height=1,
    fill_color=transform('color', mapper),
    line_color=None,
    color=data_heatmap.flatten()
)
 #Add color bar
color_bar = ColorBar(color_mapper=mapper,
                     label_standoff=12,location=(0,0))
p.add_layout(color_bar, 'right')

#show plot
show(p)