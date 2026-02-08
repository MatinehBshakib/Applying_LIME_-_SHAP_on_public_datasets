from Load import LoadData
from sklearn.preprocessing import LabelEncoder
from Strategy import SingleOutput
from HCVOptimizer import HCVOptimizer as HCVOpt
from PostProcessor import PostProcessor
import pandas as pd
from sklearn.utils import shuffle  # <--- Make sure this is imported

def main():
    loader = LoadData()
    target_list = ["Baselinehistological staging"] 
    dataset_name = "Hepatitis"
    url = "HCV-Egy-Data.csv"
    X, y = loader.load_file(file_path=url, target_cols=target_list)
    le = LabelEncoder()
    if dataset_name == "Hepatitis":
        hcv = HCVOpt()
        X, y_final = hcv.optimize(X, y, target_name=target_list[0])
    else: 
        y_encoded = le.fit_transform(y.values.ravel())
        print(f"Mapping: 0 = {le.classes_[0]}, 1 = {le.classes_[1]}")
        y_final = pd.DataFrame(y_encoded, index=y.index, columns=target_list)

    X, y_final = shuffle(X, y_final, random_state=42)  # Shuffle the data to ensure randomness

    # 3. Split data 
    x_train, x_test, y_train, y_test = loader.export_data_for_rulex(X, y_final, dataset_name=dataset_name)
    # 4. Execute Strategy
    strategy = SingleOutput(algo='rf')
    strategy.execute(x_train, x_test, y_train, y_test)
    
    aggregator = PostProcessor()
    aggregator.aggregate_and_clean(database_name=dataset_name)

if __name__ == "__main__":
    main()