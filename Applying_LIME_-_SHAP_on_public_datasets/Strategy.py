
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from Explainability import Explainability
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import numpy as np

class BaseStrategy(Explainability):
      def execute(self, x, y):
            raise NotImplementedError()
class SingleOutput(BaseStrategy):
      def __init__(self, algo='rf'):
            self.algo = algo
      def execute(self, x, y):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            class_names = list(le.classes_) # Get original class names
            class_names = [str(name) for name in class_names] # Ensure all class names are strings for LIME
            print(f"Class Mapping: {dict(zip(range(len(class_names)), class_names))}")
            
            #Split the data
            x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.3, random_state=42)
            
            #Train the model 
            if self.algo == 'xgb':
                  clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            else:
                  clf = RandomForestClassifier(random_state=42)
            clf.fit(x_train, y_train)
            print(f"Training Features: {x_train.columns.tolist()}")
            print(f"Model Accuracy: {clf.score(x_test, y_test):.4f}")
            self.run_shap(clf, x_train, x_test)
            self.run_lime(clf, x_train, x_test, class_names)
            return clf
class HierarchicalStrategy(BaseStrategy):
      def __init__(self, group_mapping, algo='xgb'):
            self.group_mapping = group_mapping
            self.algo = algo
            
      def execute(self, x, y):
            if not isinstance(y, pd.DataFrame):
                  raise ValueError("Target y must be a DataFrame for Hierarchical Strategy")    
            #Split the data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)
            results = {}
            for category, subtypes in self.group_mapping.items():
                  print(f"\n>>> PROCESSING FLOW: {category}")
                  valid_subtypes = [c for c in subtypes if c in y_train.columns]
                  if not valid_subtypes:
                        print(f"Skipping {category}: columns not found in dataset")
                        continue
                  
                  #Level 1: Gatekeeper
                  y_train_gate = y_train[valid_subtypes].max(axis=1) #create parent label
                  y_test_gate = y_test[valid_subtypes].max(axis=1)
                  
                  print(f"Training Gatekeeper for {category}...")
                  if self.algo == 'xgb':
                        gate_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                  else:
                        gate_model = RandomForestClassifier(random_state=42)
                  gate_model.fit(x_train, y_train_gate)
                  #Evaluate 
                  gate_pred = gate_model.predict(x_test)
                  print(f"Gatekeeper Accuracy: {accuracy_score(y_test_gate, gate_pred):.4f}")
                  #Level 2: Specialist 
                  mask_train = y_train_gate == 1
                  x_spec_train = x_train[mask_train] # Select rows where gatekeeper predicts 1
                  y_spec_train = y_train.loc[mask_train, valid_subtypes] # Corresponding subtypes
                  
                  # Base model for MultiOutput
                  if self.algo == 'xgb':
                        base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                  else:
                        base = RandomForestClassifier(random_state=42)
                  
                  spec_model = None  # Initialize as None
                  if len(x_spec_train) > 0:
                        spec_model = MultiOutputClassifier(base)
                        spec_model.fit(x_spec_train, y_spec_train)
                  else:
                        print(f"Warning: No positive training examples for {category}.")
                  
                  #Evaluate Specialist
                  #conditional prediction on gatekeeper positive
                  final_pred = pd.DataFrame(0, index=y_test.index, columns=valid_subtypes)
                  pos_indices = np.where(gate_pred == 1)[0]

                  if len(pos_indices) > 0 and spec_model is not None:
                        spec_pred = spec_model.predict(x_test.iloc[pos_indices])
                        final_pred.iloc[pos_indices] = spec_pred
                        
                        x_test_spec = x_test.iloc[pos_indices] # Subset of test data relevant to specialist
                        # Iterate through each sub-category column and its corresponding trained estimator
                        for idx, sub_col in enumerate(valid_subtypes):
                              estimator = spec_model.estimators_[idx] # Extract the specific model for this sub-category
                              print(f"Visualizing SHAP for Specialist Subtype {sub_col}...")
                              self.run_shap(estimator, 
                                            x_spec_train, 
                                            x_test_spec, 
                                            output_filename=f"shap_{category}_{sub_col}.csv")
                              print(f"Visualizing LIME for Specialist Subtype {sub_col}...")
                              sub_class_names = [f"No_{sub_col}", sub_col]
                              self.run_lime(estimator, 
                                            x_spec_train, 
                                            x_test_spec, 
                                            class_names= sub_class_names,
                                            output_filename=f"lime_{category}_{sub_col}.csv")
                  results[category] = (gate_model, spec_model)
            return results

                  
                   