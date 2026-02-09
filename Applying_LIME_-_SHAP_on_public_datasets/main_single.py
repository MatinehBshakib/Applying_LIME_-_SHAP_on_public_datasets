from sklearn.model_selection import train_test_split
from Load import LoadData
from sklearn.preprocessing import LabelEncoder
from Strategy import SingleOutput
from HCVOptimizer import HCVOptimizer as HCVOpt
from ObesityOptimizer import ObesityOptimizer as ObesityOpt
from DiabetesOptimizer import DiabetesOptimizer as DiabetesOpt
from PostProcessor import PostProcessor
import pandas as pd
from sklearn.utils import shuffle  

def main():
    loader = LoadData()
    target_list = ["readmitted"]  # Change this to the desired target column name(s) for the desired dataset
    dataset_name = "Diabetes_130_US"  # Change this to the desired dataset name
    url = "diabetic_data.csv" # Change this to the desired dataset URL if needed, otherwise it will load from the local database
    X, y = loader.load_file(file_path=url, target_cols=target_list)
    le = LabelEncoder()
    if dataset_name == "Hepatitis":
        hcv = HCVOpt()
        X, y_final = hcv.optimize(X, y, target_name=target_list[0])
    elif dataset_name == "Obesity_level":
        obesity = ObesityOpt()
        X, y_final = obesity.optimize(X, y, target_name=target_list[0])
    elif dataset_name == "Diabetes_130_US":
        diabetes = DiabetesOpt()
        X, y_final = diabetes.optimize(X, y, target_name=target_list[0])
    else: 
        y_encoded = le.fit_transform(y.values.ravel())
        print(f"Mapping: 0 = {le.classes_[0]}, 1 = {le.classes_[1]}")
        y_final = pd.DataFrame(y_encoded, index=y.index, columns=target_list)

    X, y_final = shuffle(X, y_final, random_state=42)  # Shuffle the data to ensure randomness
    SAMPLE_SIZE = 2500  # Adjustable number 
        
    if len(X) > SAMPLE_SIZE:
        print(f"\n>>> Downsampling dataset from {len(X)} to {SAMPLE_SIZE} rows...")
        X, _, y_final, _ = train_test_split(
            X, y_final, 
            train_size=SAMPLE_SIZE, 
            stratify=y_final, 
            random_state=42
        )
    # 3. Split data 
    x_train, x_test, y_train, y_test = loader.export_data_for_rulex(X, y_final, dataset_name=dataset_name)
    # 4. Execute Strategy
    strategy = SingleOutput(algo='xgb') # Change this to the desired algorithm ('xgb', 'rf')
    strategy.execute(x_train, x_test, y_train, y_test)
    
    aggregator = PostProcessor()
    aggregator.aggregate_and_clean(database_name=dataset_name)

if __name__ == "__main__":
    main()