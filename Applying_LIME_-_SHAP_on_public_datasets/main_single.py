from Load import LoadData
from sklearn.preprocessing import LabelEncoder
from Strategy import SingleOutput
from PostProcessor import PostProcessor
import pandas as pd
from sklearn.utils import shuffle  # <--- Make sure this is imported

def main():
    loader = LoadData()
    target_list = ["class"] 
    url = "Applying_LIME_-_SHAP_on_public_datasets\Applying_LIME_-_SHAP_on_public_datasets\messidor_features.arff"
    X, y = loader.load_csv(file_path=url, target_cols=target_list)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.values.ravel())
    y_final = pd.DataFrame(y_encoded, index=y.index, columns=target_list)

    X, y_final = shuffle(X, y_final, random_state=42)  # Shuffle the data to ensure randomness

    # 3. Split data 
    x_train, x_test, y_train, y_test = loader.export_data_for_rulex(X, y_final)
    # 4. Execute Strategy
    strategy = SingleOutput(algo='xgb')
    strategy.execute(x_train, x_test, y_train, y_test)
    
    aggregator = PostProcessor()
    aggregator.aggregate_and_clean()

if __name__ == "__main__":
    main()