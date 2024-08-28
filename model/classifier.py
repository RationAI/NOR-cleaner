import xgboost as xgb
from model.hyperparams import get_xgbc_hyperparams


SELECTED_MODEL = xgb.XGBClassifier(**get_xgbc_hyperparams())
