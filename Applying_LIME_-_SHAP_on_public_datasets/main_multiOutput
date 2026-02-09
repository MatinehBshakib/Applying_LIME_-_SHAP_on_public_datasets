from Load import LoadData
from sklearn.preprocessing import LabelEncoder
from Strategy import MultiLabelStrategy
from Load import LoadData as loader
from sklearn.preprocessing import LabelEncoder as le
from sklearn.utils import shuffle
import pandas as pd
from PostProcessor import PostProcessor
def main():
      # Pass a list of multiple columns
      loader = LoadData()
      dataset_name = "Cervical_Cancer"
      url = "risk_factors_cervical_cancer.csv"  
      target_list = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
      X, y = loader.load_file(file_path=url, target_cols=target_list)
      X, y = shuffle(X, y, random_state=42)  # Shuffle the data to ensure randomness
      # 3. Split data 
      x_train, x_test, y_train, y_test = loader.export_data_for_rulex(X, y, dataset_name=dataset_name)
      # Use MultiLabel Strategy
      strategy = MultiLabelStrategy(algo='xgb')
      strategy.execute(x_train, x_test, y_train, y_test)
      aggregator = PostProcessor()
      aggregator.aggregate_and_clean(database_name=dataset_name)

if __name__ == "__main__":
      main()
