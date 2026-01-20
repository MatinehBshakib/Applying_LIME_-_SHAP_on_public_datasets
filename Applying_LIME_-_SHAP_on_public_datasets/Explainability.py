from lime import lime_tabular
import pandas as pd

class Explainability:
      def run_lime(clf, x_train, x_test, y_test, class_names, output_filename="lime_explanation_results.csv"):
            #initialize LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                  training_data=x_train.values,
                  feature_names=x_train.columns.tolist(),
                  class_names=class_names,
                  mode='classification'
            )
            #generate LIME explanations for each instance in the test set
            lime_rows = []
            n_features = len(x_train.columns)
            print(f"Calculating LIME values for {len(x_test)} instances. This may take a while...")
            
            #Iterate through each instance in the test set
            for i in range(len(x_test)):
                  exp= explainer.explain_instance(
                        data_row=x_test.values[i],
                        predict_fn=clf.predict_proba,
                        labels=[1], # Assuming binary classification with positive class as 1
                        num_features=n_features
                  )
                  
            #extract values
            base_value = exp.intercept[1]  # Base value for positive class
            local_weight = dict(exp.local_exp[1])  # Local weights for positive class
            current_id = x_test.index[i]  # Get the original index of the instance
            
            for feat_idx, feat_name in enumerate(x_train.columns):
                  feat_val = x_test.values[i][feat_idx]
                  lime_value = local_weight.get(feat_idx, 0.0)
                  lime_rows.append({
                        "id": current_id,
                        "feature": feat_name,
                        "feature_value": feat_val,
                        "base_value": base_value,
                        "lime_value": lime_value
                  })
            
            #convert to dataframe and save to csv
            lime_df = pd.DataFrame(lime_rows)
            sort_lime_df = lime_df.sort_values(by=["id", "lime_value"], ascending=[True, False])
            sort_lime_df.to_csv(output_filename, index=False)
            print(f"LIME explanations saved to {output_filename}")
            return sort_lime_df
                        
            
    