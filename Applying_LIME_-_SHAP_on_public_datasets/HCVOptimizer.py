import pandas as pd
import numpy as np

class HCVOptimizer:
    def __init__(self):
        # Define the binary columns that need 1/2 -> 0/1 correction
        self.binary_cols = [
            'Gender', 'Fever', 'Nausea/Vomting', 'Headache', 
            'Diarrhea', 'Fatigue & generalized bone ache', 
            'Jaundice', 'Epigastric pain'
        ]

    def optimize(self, X, y, target_name):
        print("\n>>> Starting HCV Specific Optimization...")

        # 1. Clean Column Names (Remove trailing spaces)
        X.columns = X.columns.str.strip()
        
        # 2. Drop Leaky Columns
        # We remove 'Baseline histological Grading' because it reveals the answer
        if 'Baseline histological Grading' in X.columns:
            X = X.drop(columns=['Baseline histological Grading'])
            print("Dropped 'Baseline histological Grading' to prevent data leakage.")

        # 3. Fix Binary Columns (1/2 -> 0/1)
        # Your specific logic: shift values if max is 2
        for col in self.binary_cols:
            if col in X.columns:
                # Check if values are mostly 1 and 2
                if X[col].max() == 2:
                    X[col] = X[col] - 1
                    print(f"Shifted {col} from [1,2] to [0,1]")

        # 4. Transform Target
        # Logic: 3, 4 -> 1 (Hazardous); Others -> 0
        # Check if we have the column in y, otherwise assume y is the series
        if target_name in y.columns:
            target_series = y[target_name]
        else:
            target_series = y.iloc[:, 0]

        y_new = target_series.apply(lambda val: 1 if val in [3, 4] else 0)
        y_final = y_new.to_frame(name='Stage')

        print(f"Target Distribution (1=Severe, 0=Mild):\n{y_final['Stage'].value_counts()}")
        
        return X, y_final