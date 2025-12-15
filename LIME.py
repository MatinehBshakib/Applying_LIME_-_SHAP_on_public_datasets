import lime
from lime import lime_tabular
import matplotlib.pyplot as plt




class LIME:
    def explainer(self, x_train, x_test, y_test, feature_names, class_names, forest_clf):
        #Initialize LIME Explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=x_train.values,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )
        #Explain first two instances in test set
        for i in range(2):
            actual_class = class_names[int(y_test.iloc[i])]
            print(f"Actual Status: {actual_class}")
            print(dict(zip(x_test.columns, x_test.values[i])))

            explanation = explainer.explain_instance(
                data_row=x_test.values[i],
                predict_fn=forest_clf.predict_proba,
                num_features=9
            )
            
            # Visualize LIME explanation
            fig = explanation.as_pyplot_figure()
            plt.tight_layout()
            plt.show()