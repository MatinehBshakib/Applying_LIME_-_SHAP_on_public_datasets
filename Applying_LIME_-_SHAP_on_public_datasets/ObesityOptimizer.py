import pandas as pd
import numpy as np

class ObesityOptimizer:
    def optimize(self, X, y, target_name="NObeyesdad"):
        print("\n>>> Starting Obesity Dataset Optimization...")
        # A. Binary Columns
        binary_mapping = {'yes': 1, 'no': 0, 'Male': 1, 'Female': 0}
        binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map(binary_mapping)

        # B. Ordinal Columns
        ordinal_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        ordinal_cols = ['CAEC', 'CALC']
        
        for col in ordinal_cols:
            if col in X.columns:
                X[col] = X[col].map(ordinal_mapping)

        # C. Nominal Columns (One-Hot)
        if 'MTRANS' in X.columns:
            # dtype=int ensures we get 0/1 instead of True/False
            dummies = pd.get_dummies(X['MTRANS'], prefix='Transport', dtype=int)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=['MTRANS'])
            print("One-hot encoded 'MTRANS' column.")

        obesity_mapping = {
            'Insufficient_Weight': 0, 'Normal_Weight': 0,
            'Overweight_Level_I': 0, 'Overweight_Level_II': 0,
            'Obesity_Type_I': 1, 'Obesity_Type_II': 1, 'Obesity_Type_III': 1
        }

        # Handle input type (Series vs DataFrame)
        if isinstance(y, pd.DataFrame):
            raw_target = y[target_name]
        else:
            raw_target = y

        # Apply map
        y_new = raw_target.map(obesity_mapping)
        y_final = y_new.to_frame(name='Obesity')

        if y_final.isnull().any().any():
            print("Warning: Found unmapped target labels. Filling with 0.")
            y_final = y_final.fillna(0)
            
        # Drop original target if it exists in X
        if target_name in X.columns:
            X = X.drop(columns=[target_name])

        # Drop Height/Weight to prevent BMI leakage 
        leakage_cols = ['Height', 'Weight']
        cols_to_drop = [c for c in leakage_cols if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            print(f"Dropped leakage columns: {cols_to_drop}")
        
        # Final numeric conversion safety check
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        print(f"Optimization Complete.")
        print(f"Target Distribution (1=Obese, 0=Non-Obese):\n{y_final['Obesity'].value_counts()}")
        
        return X, y_final