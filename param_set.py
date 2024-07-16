from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from data import chla_x_train, chla_y_train, TSS_x_train, TSS_y_train

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
    # print(best_parameters)

    # # Create the final model with the best parameters
    # model_final = model_selected.__class__(**best_parameters)
    # model_final.fit(x_train, y_train)
    return best_parameters

#预先设置好的参数网格
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

best_parameters = perform_parameter_search(xgb_paramgrid, XGBRegressor(), GridSearchCV, chla_x_train, chla_y_train) 
TSS_parameters = perform_parameter_search(xgb_paramgrid, XGBRegressor(), GridSearchCV, TSS_x_train, TSS_y_train)