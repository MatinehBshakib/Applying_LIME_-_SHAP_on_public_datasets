import pandas as pd
import numpy as np

class DiabetesOptimizer:
    def optimize(self, X, y=None, target_name='readmitted'):
        print("\n>>> Starting Diabetes Dataset Optimization...")
        if y is not None:
            X = X.copy()
            if isinstance(y, pd.DataFrame):
                X[target_name] = y[target_name]
            else:
                X[target_name] = y
        # Replace ? with NaN
        X = X.replace('?', np.nan)

        # Drop columns with too many missing values
        drop_cols = ['weight', 'payer_code', 'encounter_id', 'patient_nbr', 'medical_specialty']
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

        # Drop rows with missing critical info
        subset_cols = ['race', 'diag_1', 'diag_2', 'diag_3', 'gender']
        existing_subset = [c for c in subset_cols if c in X.columns]
        X = X.dropna(subset=existing_subset)
        
        # Drop Invalid Gender
        if 'gender' in X.columns:
            X = X[X['gender'] != 'Unknown/Invalid']

        # Remove "Dead" Patients (cannot be readmitted)
        if 'discharge_disposition_id' in X.columns:
            dead_ids = [11, 13, 14, 19, 20, 21]
            # Ensure numeric comparison
            X['discharge_disposition_id'] = pd.to_numeric(X['discharge_disposition_id'], errors='coerce')
            X = X[~X['discharge_disposition_id'].isin(dead_ids)]

        if target_name in X.columns:
            # Extract the cleaned target column
            y_clean = X[target_name]
            X = X.drop(columns=[target_name])
            
            # Logic: <30 days = 1 (Early Readmission), All else = 0
            y_new = y_clean.apply(lambda x: 1 if str(x) == '<30' else 0)
            y_final = y_new.to_frame(name='Readmitted')
        else:
            print("Warning: Target column lost during optimization.")
            y_final = y # Fallback (though sizes might mismatch if this happens)
        
        # A. Map Age
        if 'age' in X.columns:
            age_map = {
                '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, 
                '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, 
                '[80-90)': 8, '[90-100)': 9
            }
            X['age'] = X['age'].map(age_map)

        # B. Map Medications (No/Steady/Up/Down -> 0/1/2)
        med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
                    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
                    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                    'miglitol', 'troglitazone', 'tolazamide', 'examide', 
                    'citoglipton', 'insulin', 'glyburide-metformin', 
                    'glipizide-metformin', 'glimepiride-pioglitazone', 
                    'metformin-rosiglitazone', 'metformin-pioglitazone']
        
        med_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 2}
        for col in med_cols:
            if col in X.columns:
                X[col] = X[col].map(med_map).fillna(0)

        # C. Binary Columns
        binary_cols = ['change', 'diabetesMed']
        bin_map = {'No': 0, 'Yes': 1, 'Ch': 1}
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map(bin_map).fillna(0)

        # D. ICD-9 Diagnosis Grouping
        def get_diag_category(code):
            try:
                if str(code).startswith(('V', 'E')): return 'Other'
                n = float(code)
                if 390 <= n <= 459 or n == 785: return 'Circulatory'
                if 460 <= n <= 519 or n == 786: return 'Respiratory'
                if 520 <= n <= 579 or n == 787: return 'Digestive'
                if str(n).startswith('250'): return 'Diabetes'
                if 800 <= n <= 999: return 'Injury'
                if 710 <= n <= 739: return 'Musculoskeletal'
                if 580 <= n <= 629 or n == 788: return 'Genitourinary'
                if 140 <= n <= 239: return 'Neoplasms'
                return 'Other'
            except:
                return 'Other'

        diag_cols = ['diag_1', 'diag_2', 'diag_3']
        for col in diag_cols:
            if col in X.columns:
                X[col] = X[col].apply(get_diag_category)
                # One-Hot Encode
                dummies = pd.get_dummies(X[col], prefix=col, dtype=int)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(columns=[col])

        # E. One-Hot Encoding for remaining Nominal columns
        nominal_cols = ['race', 'gender', 'max_glu_serum', 'A1Cresult']
        for col in nominal_cols:
            if col in X.columns:
                dummies = pd.get_dummies(X[col], prefix=col, dtype=int)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(columns=[col])

        # Final Numeric Check
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        print(f"Optimization Complete. Final Shape: {X.shape}")
        if y_final is not None:
            print(f"Target Distribution (1=<30 days, 0=Other):\n{y_final['Readmitted'].value_counts()}")
        
        return X, y_final