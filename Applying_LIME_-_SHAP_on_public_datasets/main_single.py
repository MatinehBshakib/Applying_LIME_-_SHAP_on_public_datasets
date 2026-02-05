from Load import LoadData
from sklearn.preprocessing import LabelEncoder
from Strategy import SingleOutput
from PostProcessor import PostProcessor
import pandas as pd
from sklearn.utils import shuffle  # <--- Make sure this is imported

def main():
    loader = LoadData()
    target_list = ["class"] 
    X, y = loader.load_dataset(data_id=43611, target_cols=target_list)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.values.ravel())
    y_final = pd.DataFrame(y_encoded, index=y.index, columns=target_list)

    X, y_final = shuffle(X, y_final, random_state=42)  # Shuffle the data to ensure randomness

    # 3. Split data 
    x_train, x_test, y_train, y_test = loader.export_data_for_rulex(X, y_final)
    # 4. Execute Strategy
    strategy = SingleOutput(algo='rf')
    strategy.execute(x_train, x_test, y_train, y_test)
    
    aggregator = PostProcessor()
    aggregator.aggregate_and_clean()

if __name__ == "__main__":
    main()