import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

RANDOM_SEED = 6

def cv_models(features, target):
    kfolds = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    ridge = make_pipeline(RobustScaler(), RidgeCV(cv=kfolds))
    lasso = make_pipeline(RobustScaler(), LassoCV(random_state=RANDOM_SEED, cv=kfolds))
    elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(cv=kfolds))                                
    svr = make_pipeline(RobustScaler(), SVR())
    rfr = RandomForestRegressor(random_state=RANDOM_SEED)
    gbr = GradientBoostingRegressor(random_state=RANDOM_SEED)
    lightgbm = LGBMRegressor(random_state=RANDOM_SEED)
    xgboost = XGBRegressor(seed=RANDOM_SEED)

    results = {}
    scoring = 'neg_mean_squared_error'
    models = {
        'Ridge': ridge,
        'Lasso': lasso,
        'ElasticNet': elasticnet,
        'SVR': svr,
        'RandomForest': rfr,
        'GradientBoostingRegressor': gbr,
        'LightGBM': lightgbm,
        'XGBoost': xgboost
    }

    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        scores = cross_val_score(model, features, target, cv=kfolds, scoring=scoring)
        
        rmse_scores = np.sqrt(-scores)
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        training_time = time.time() - start_time
        
        results[name] = {
            'Mean RMSE': mean_rmse,
            'Std RMSE': std_rmse,
            'Training Time (s)': training_time
        }
        
        print(f"{name} - Training completed in {training_time:.2f} seconds.")
        print("-" * 50)

    results_df = pd.DataFrame(results).T.reset_index()
    results_df.columns = ['Model', 'Mean RMSE', 'Std RMSE', 'Training Time (s)']
    results_df.sort_values(by='Mean RMSE', inplace=True) 

    return results_df, models

def create_submission(test, model, target_csv):
    scaled_predictions = np.expm1(model.predict(test))
    submission = pd.DataFrame({
        'Id': list(range(1461, 2920)),
        'SalePrice': scaled_predictions
    })
    submission.to_csv(target_csv, index=False)