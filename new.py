#Q:1
from bokeh.plotting import figure,show
import numpy as np

#create data
x=np.linspace(0,10,100)
y=np.sin(x)

#create bokeh figure
p=figure (title="Sine Wave",x_axis_label='x',
y_axis_label='sin(x)')

#ADD LINE TO THE FIGURE
p.line(x,y,line_with=2,color="blue")

#show thw plot
show(p)
