

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










