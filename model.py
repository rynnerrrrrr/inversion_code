import joblib
from xgboost import XGBRegressor
from data import chla_x_train, chla_y_train, TSS_x_train, TSS_y_train
from param_set import best_parameters, TSS_parameters

#实例化并训练模型
xgb_model = XGBRegressor(**best_parameters).fit(chla_x_train, chla_y_train)
TSS_model_xgb = XGBRegressor(**TSS_parameters).fit(TSS_x_train, TSS_y_train)

#保存模型
joblib.dump(xgb_model, 'xgb_chla_model.pkl')
joblib.dump(TSS_model_xgb, 'xgb_tss_model.pkl')

