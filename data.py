import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor


class PointDataset:
    def __init__(self, csv_path):
        self.scaler = RobustScaler()
        self.path = csv_path
        self._data = self._load_data()

    def _load_data(self):
        try:
            data = pd.read_csv(self.path)
            data_array = data.to_numpy()
            num_rows, num_cols = data_array.shape
            # 标准化处理，第一列为标签
            features = data_array[:, 1:]
            labels = data_array[:, 0]
            scaled_features = self.scaler.fit_transform(features)
            data_array = np.hstack((labels.reshape(-1, 1), scaled_features))
            return data_array
        except Exception as e:
            print(f"Error loading or preprocessing data: {e}")
            return None

    def get_data(self):
        return self._data

    def train_val_split(self, val_split=0.2):
        if self._data is None:
            print("Data not loaded. Please load the data first.")
            return None

        # 假设第一列是标签，其余是特征
        features = self._data[:, 1:]
        labels = self._data[:, 0]

        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=val_split)
        return x_train, x_test, y_train, y_test

def parameter_search(param_grid, model_selected, search_strategy):
    model = search_strategy(
    estimator=model_selected,
    param_grid=param_grid,
    scoring="r2",
    verbose=10,
    n_jobs=-1,
    cv=5
    )
    # fit the model and extract best score
    model.fit(x_train, y_train)
    print(f"Best score: {model.best_score_}")
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    model_final = model_selected(**best_parameters)
    return model_final

#GPT润色后的代码
def perform_parameter_search(param_grid, model_selected, search_strategy, x_train, y_train):
    """
    Perform parameter search using the specified search strategy.

    Parameters:
    - param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    - model_selected: The model to be tuned.
    - search_strategy: The search strategy to use (e.g., GridSearchCV, RandomizedSearchCV).
    - x_train: Training data features.
    - y_train: Training data labels.

    Returns:
    - model_final: The model with the best parameters.
    """
    # Initialize the search strategy
    search = search_strategy(
        estimator=model_selected,
        param_grid=param_grid,
        scoring="r2",
        verbose=10,
        n_jobs=-1,
        cv=5
    )

    # Fit the model and extract the best score
    search.fit(x_train, y_train)
    print(f"Best score: {search.best_score_}")
   
    best_parameters = search.best_estimator_.get_params()
    print(best_parameters)

    # Create the final model with the best parameters
    model_final = model_selected.__class__(**best_parameters)
    model_final.fit(x_train, y_train)
    return model_final

#设置好的
xgb_paramgrid = {
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
        'n_estimators': [100],
        'max_depth': [3, 4, 5],
        'min_child_weight': [1, 2, 3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }


Chla_dataset = PointDataset(r"C:\Users\lijian\Desktop\MSI\chla_rgb_nir.csv")
TSS_dataset = PointDataset(r"C:\Users\lijian\Desktop\MSI\TSS_rgb_nir.csv")
chla_x_train, chla_x_test, chla_y_train, chla_y_test = Chla_dataset.train_val_split()
TSS_x_train, TSS_x_test, TSS_y_train, TSS_y_test = TSS_dataset.train_val_split()

