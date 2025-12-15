import pandas as pd
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
import xgboost as xgb 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from LIME import LIME as lime
from SHAP import SHAP as shap

def load_file():
    # Load Breast Cancer Wisconsin (Original) dataset from OpenML
    data = fetch_openml(data_id= 43611 , version='active', as_frame=True)
    df = data.frame.copy()
    df.replace('?', np.nan, inplace=True)
    x = df.drop(columns=['class'])
    y = df['class']
    #convert object columns to numeric
    x = x.apply(pd.to_numeric, errors='coerce')
    #Handle missing values by imputing with column mean
    imputer = SimpleImputer(strategy='mean')
    x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)
    
    return x,y

def train_and_test(x,y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    #Create a map to understand which class is which 
    class_names = list(le.classes_)
    print (f"Class Mapping: {dict(zip(range(len(class_names)), class_names))}")
    
    #Split data
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=42)
    
    #Train Model
    forest_clf = RandomForestClassifier(random_state=42)
    forest_clf.fit(x_train, y_train)
    print(f"Model Accuracy: {forest_clf.score(x_test,y_test):.4f}")
    
    #Initialize LIME Explainer
    lime.explainer(x_train, x_test, y_test, x.columns.tolist(), class_names, forest_clf)
    

def main():

    x,y = load_file()
    train_and_test(x,y)
    
if __name__ == "__main__":
    main()