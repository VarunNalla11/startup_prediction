import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# Load data
data = pd.read_csv(r"C:/Users/asus/OneDrive/Desktop/Machine-Learning/50_Startups.csv")
x=data.iloc[:,3]
y=data.iloc[:,4]

plt.bar(x,y)
plt.show()