from Load import LoadData
from sklearn.preprocessing import LabelEncoder
from Strategy import MultiLabelStrategy
def main():
      loader = LoadData()
      # Pass a list of multiple columns
      target_list = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
      X, y = loader.load_dataset(data_id=42912, target_cols=target_list)
      # Split data for Rulex export
      x_train, x_test, y_train, y_test = loader.export_data_for_rulex(X, y)
      # Use MultiLabel Strategy
      strategy = MultiLabelStrategy(algo='xgb')
      strategy.execute(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()