

#Q:1

import matplotlib.pyplot as plt
import numpy as np

#given data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 5, 7, 6, 8, 9, 10, 12, 13])

#create scatter plot
plt.scatter(x,y)

#add lables  and titles
plt.title("our own scatter plot")
plt.xlabel("x-axis hai")
plt.ylabel("y-axis hai")

#display the plot
plt.show()

#Q:2
import matplotlib.pyplot as plt
import numpy as np

#given data
data = np.array([3, 7, 9, 15, 22, 29, 35])
 
#labels ad title
plt.title("my line plot")

#display plot
plt.show()

#Q:3
import matplotlib.pyplot as plt
import numpy as np

#given data

categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 35, 20]
 
 #create bar plot
plt.bar(categories,values,color='red') 

#display plot
plt.show()


#Q:no.4
import matplotlib.pyplot as  plt
import numpy as np

#given data
data = np.random.normal(0, 1, 1000) 
plt.hist(data)
#display plot
plt.show()


#Q:NO.5
import matplotlib.pyplot as plt
import numpy as np
#given data
sections = ['Section A', 'Section B', 'Section C', 'Section D']
sizes = [25, 30, 15, 30]
plt.pie(sizes,labels=sections,shadow="true")
#add titles
plt.title("students pie chart")
plt.show()

                       #####SEABORN ASSINGMENT#######
#Q:1
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate synthetic dataset
np.random.seed(10)   # for reproducibility
x = np.random.rand(50) * 10   # 50 random values for X
y = 2 * x + np.random.randn(50) * 3   # Y related to X with some noise

# Create DataFrame
data = pd.DataFrame({'X': x, 'Y': y})

# Scatter plot
sns.scatterplot(x='X', y='Y', data=data)
plt.title("Scatter Plot of Synthetic Data")
plt.show()

#Q:n0.2

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# generate a dataset of randm numbers
data = np.random.randn(1000)

#visualize the distribution 
sns.histplot(data,kde=True,color='grey')

#add labels and tiltle
plt.title("distribution of random numbers")
plt.xlabel("value")
plt.ylabel("frequency")

#display plot
plt.show()


#Q:3
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#create a siple dataset
data = {'Category':['A','B','C','D','E']
        ,'values':np.random.randint(10,100,5)}
df = pd.DataFrame(data)

#visualizing comparison using barplot
sns.barplot(x='Category', y='values', data=df, hue='Category', legend=False)

#Add labels and title
plt.title("Comparison of Categories Based on Values")
plt.xlabel("Category")
plt.ylabel("Values")

plt.show()


#Q:no.4
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#create a data set with categories and numerical values
np.random.seed(1)
data = {'Category':np.random.choice(['A','B','C'],size=100),
        'Values':np.random.randint(1,100,size=100)}
df = pd.DataFrame(data)

#visualizing distribution using boxplot
sns.boxplot(x='Category', y='Values', data=df, hue='Category', legend=False)

#Add labels and title
plt.title("Distribution olf Values across Categories")
plt.xlabel("Category")
plt.ylabel("Values")

plt.show()


#Q:no.5
import seaborn as sns
import matplotlib.pyplot as plt
import  numpy as np
import pandas as pd
#generate a synthetic dataset with correlated feature
np.random.seed(0)
data = {
    'Feature1':np.random.randn(100),
    'Feature2':np.random.randn(100)*0.8+0.5,
    'Feature3':np.random.randn(100)*0.5+2,
    'Feature4':np.random.randn(100)*1.2 - 1
}
df =pd.DataFrame(data)
#create correlation matrix
corr=df.corr()
#visualize the correlation matrix using heatmap
sns.heatmap(corr,annot=True,cmap='cool')
plt.title("Correlation Matrix Heatmap")
plt.show()

        
        #######PLOTLY ASSINGMENT#######
#Q:1
import numpy as np
import pandas as pd
import plotly.express as px

#Generate the dataset
np.random.seed(30)
data = {
    'X': np.random.uniform(-10, 10, 300),
    'Y': np.random.uniform(-10, 10, 300),
    'Z': np.random.uniform(-10, 10, 300)
}
df = pd.DataFrame(data)

#create a 3D scatter plot
fig = px.scatter_3d(df,x='X',y='Y',z='Z',color='Z',
                    title="3D scatter plot of randomdata")
fig.show()

#Q:2
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

#create random data
np.random.seed(15)
data ={
    'Grade':np.random.choice(['A','B','C','D','F'],200),
    'Score':np.random.randint(50,100,200)

}
df=pd.DataFrame(data)

#create voilin plot
sns.violinplot(x='Grade',y='Score',data=df,palette='Set2')

#ADD TITLE AND LABLES
plt.title('Distribution of scores scross different grades')
plt.xlabel('Grade')
plt.ylabel('Score')

plt.show()
               

#Q:3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#given data
np.random.seed(20)
data = {
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
    'Day': np.random.choice(range(1, 31), 100),
    'Sales': np.random.randint(1000, 5000, 100)
}
df = pd.DataFrame(data)

#create pivot table for heatmap
pivot = df.pivot_table(values='Sales',index='Month',columns='Day',aggfunc='mean')

#plot heatmap
plt.figure(figsize=(10,5))
sns.heatmap(pivot,cmap='YlGnBu')
plt.title('Sales Heatmap Across Months and Days')
plt.show()

#Q:5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#set random seed
np.random.seed(25)
#creat a dataset
data = {
    'Country': ['USA', 'Canada', 'UK',
'Germany', 'France'],
    'Population':
np.random.randint(100, 1000, 5),
    'GDP': np.random.randint(500, 2000,
5)
}
df = pd.DataFrame(data)

#create bubble sort
plt.figure(figsize=(8,6))
plt.scatter(df['GDP'],df['Population'],
            s=df['Population']*2,
            alpha=0.6,edgecolor='w')

#Add labels and title
plt.title('bubble chart :GDP VS POPULATION')
plt.xlabel('GDP')
plt.ylabel('Population')
plt.grid(True)
plt.show()

#Q:4
import numpy as np
import matplotlib.pyplot as plt

#create x and y values
x =np.linspace(-5,5,100)
y = np.linspace(-5,5,100)

#create a grid of x and y
x,y=np.meshgrid(x,y)

#calculate z values
z=np.sin(np.sqrt(x**2 + y**2))

#plot the 3d surface
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(x,y,z,cmap='viridis')

#add labels
ax.set_xlabels('x axis')
ax.set_ylabels('y axis')
ax.set_zlabels('z = sin(sqrt(x^2 +y^2))')

plt.show()

#Q:5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#given data
np.random.seed(25)
np.random.seed(25)
data = {
    'Country': ['USA', 'Canada', 'UK',
'Germany', 'France'],
    'Population':
np.random.randint(100, 1000, 5),
    'GDP': np.random.randint(500, 2000,
5)
}
df = pd.DataFrame(data)

#create bubble chart
plt.scatter(df['GDP'],df['Population'],s=df['Population']*2,
            alpha=0.5,color='skyblue',edgecolor='black')

#add labels
for i in range(len(df)):
    plt.text(df['GDP'][i],df['Population'][i],df['Country'][i],fontsize=9,
             ha='center')
    
    plt.xlabel('GDP')
    plt.ylabel('Population')
    plt.title('bubble chart of countries:GDP vs Population')
    plt.show()
    

           
###########BOKEH ASSINGMENT##########
#Q:1

from bokeh.plotting import figure, show
import numpy as np

#create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

#create bokeh figure
p = figure(title="Sine Wave", x_axis_label='x', y_axis_label='sin(x)')

#ADD LINE TO THE FIGURE
p.line(x, y, line_width=2, color="blue")

#show the plot
show(p)


#Q:2
from bokeh.plotting import figure, show
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

#random data
x = np.random.rand(50)
y = np.random.rand(50)
sizes = np.random.randint(10, 50, size=50)
colors = ["#%02x%02x%02x" % (int(r), int(g), int(b))
         for r,g,b in np.random.randint(0, 255, (50, 3))]

#create figure
p = figure(title="Bokeh Scatter Plot", 
           x_axis_label='X',
           y_axis_label='Y',
           width=600,
           height=400)

#SCATTER PLOT
p.scatter(x, y, size=sizes, color=colors, alpha=0.6)

#show plot
show(p)


#Q:3
from bokeh.plotting import figure, show
from bokeh.io import output_file

# Save the plot to an HTML file that will open in your browser
output_file("fruits_bar_chart.html")

#Data
fruits = ['Apples', 'Oranges', 'Bananas', 'Pears']
counts = [20, 25, 30, 35]

#create figure
p = figure(x_range=fruits, 
           title="Fruits Counts",
           height=350,
           toolbar_location=None,
           tools="")

#Add bars
p.vbar(x=fruits, top=counts, width=0.5, color="orange")

#customize appearance
p.xgrid.grid_line_color = None
p.y_range.start = 0
p.xaxis.axis_label = 'Fruits'
p.yaxis.axis_label = 'Count'

#show plot
show(p)

#Q:4
from bokeh.plotting import figure, show
import numpy as np

#create random data
data_hist = np.random.randn(1000)

#create histogram data
hist, edges = np.histogram(data_hist, bins=30)

#create figure
p = figure(title="Bokeh Histogram",
x_axis_label='Value',y_axis_label='Frequency')

#draw histogram as quad bar
p.quad(bottom=0, top=hist, left=edges[:-1],right=edges[1:],
       fill_color='skyblue', line_color="white", alpha=0.7)

#show the plot
show(p)

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
          

                        



 



    








 









