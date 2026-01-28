from Load import LoadData
from sklearn.preprocessing import LabelEncoder
from Strategy import MultiLabelStrategy, HierarchicalStrategy
from Config import MycordinalConfig as config
def main():
      loader = LoadData()
      # Pass a list of multiple columns
      target_list = config.get_all_target_cols()
      X, y = loader.load_dataset(data_id=46943, target_cols=target_list)
      # Split data for Rulex export
      x_train, x_test, y_train, y_test = loader.export_data_for_rulex(X, y)
      # Use MultiLabel Strategy
      strategy = HierarchicalStrategy(
            group_mapping=config.Hierarchy_mapping,
            algo='xgb')
      strategy.execute(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()