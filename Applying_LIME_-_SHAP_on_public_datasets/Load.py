import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
      
class LoadData:
      def advanced_imputation(self, x, drop_threshold=0.5):
          x = x.copy()
          for col in x.columns:
            if x[col].dtype == 'object':
                  # Try to convert, if it fails, check if it was mostly numbers
                  temp_col = pd.to_numeric(x[col], errors='coerce')
                  if temp_col.isna().mean() < drop_threshold: # If valid numbers remain
                        x[col] = temp_col
          # Drop columns with missing value percentage above the threshold
          x=x.drop(columns=x.columns[x.isnull().mean()>drop_threshold])
          # Separate numerical and categorical columns
          num_cols = x.select_dtypes(include=[np.number]).columns
          cat_cols = x.select_dtypes(exclude=[np.number]).columns
          # Impute numerical columns with mean
          if len(num_cols) > 0:
                  imputer_num = SimpleImputer(strategy='mean')
                  x[num_cols] = pd.DataFrame(
                        imputer_num.fit_transform(x[num_cols]),
                        columns=num_cols,
                        index=x.index
                  )
          # Impute categorical columns with mode
          if len(cat_cols) > 0:
                  imputer_cat = SimpleImputer(strategy='most_frequent')
                  x[cat_cols] = pd.DataFrame(
                        imputer_cat.fit_transform(x[cat_cols]),
                        columns=cat_cols,
                        index=x.index
                  )     
                  for col in cat_cols:
                      # Convert column to category type, then to integer codes
                      x[col] = x[col].astype('category').cat.codes   
          return x
    
      def load_dataset(self, data_id, target_cols=None):
            #Load the dataset from OpenML
            data = fetch_openml(data_id=data_id, version='active', as_frame=True) 
            df = data.frame.copy()
            #Drop id column
            if 'id' in df.columns:
               df.drop(columns=['id'], inplace=True)
              
            df.replace(['?', 'NA', '', 'null'], np.nan, inplace=True)
            if target_cols:
                  missing_targets = [col for col in target_cols if col not in df.columns]
                  if missing_targets:
                      raise ValueError(f"Requested target columns not found in dataset: {missing_targets}")
                  y = df[target_cols].copy()
                  X = df.drop(columns=target_cols)
                  y = y.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
            else:
                  #Get the exact target column name
                  target_name = data.default_target_attribute
                  #Separate features and target
                  X = df.drop(columns=[target_name])
                  y = df[target_name]
            return self.advanced_imputation(X), y
            