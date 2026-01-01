import pandas as pd 
import numpy as np 
import xgboost as xgb
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import shap 
shap.initjs() #Display plots in the notebook



data = fetch_openml(data_id= 43611 , version='active', as_frame=True)
df = data.frame.copy()
plt.scatter(df['class'], df['thickness'])


