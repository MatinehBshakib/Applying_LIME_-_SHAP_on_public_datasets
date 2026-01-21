
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import sklearn.ensemble as RandomForestClassifier
from Explainability import Explainability

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
            print(f"Class Mapping: {dict(zip(len(class_names), class_names))}")
            
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