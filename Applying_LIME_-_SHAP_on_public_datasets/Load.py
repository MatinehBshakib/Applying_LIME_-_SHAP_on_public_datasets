import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
      
class LoadData:
      def advanced_imputation(self, x, drop_threshold=0.5):
          # Drop columns with missing value percentage above the threshold
          x=x.drop(columns=x.columns[x.isnull().mean()>drop_threshold])
          # Separate numerical and categorical columns
          num_cols = x.select_dtypes(include=[np.number]).columns
          cat_cols = x.select_dtypes(exclude=[np.number]).columns
          # Impute numerical columns with mean
          if len(num_cols) > 0:
                  imputer_num = SimpleImputer(strategy='mean')
                  x[num_cols] = imputer_num.fit_transform(x[num_cols])
          # Impute categorical columns with mode
          if len(cat_cols) > 0:
                  imputer_cat = SimpleImputer(strategy='most_frequent')
                  x[cat_cols] = imputer_cat.fit_transform(x[cat_cols])
                  
          return x
      def load_dataset(self):
            #Load the dataset from OpenML
            data = fetch_openml(data_id=46591, version='active', as_frame=True) 
            df = data.frame.copy()
            #drop id column
            if 'id' in df.columns:
               df = df.drop(columns=['id'], inplace=True)
               
            df.replace('?', np.nan, inplace=True)
            X = df.drop(columns=['Class']).apply(pd.to_numeric, errors='coerce')
            y = df['Class']
            
            return self.advanced_imputation(X), y
            