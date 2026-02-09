from xgboost import XGBClassifier
from concurrent import futures
from sklearn.metrics import (accuracy_score, 
                             roc_auc_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             log_loss)
from PreparedData import (X_train, y_train, X_valid, y_valid, 
                          SEED, BOOSTER, N_TRIALS, load_params, TREE_METHOD, 
                          THRESHOLD, scale_pos_weight)
from Constraints import *
import nevergrad as ng
import pandas as pd
import os
from datetime import datetime


"""METRICS"""
LEARNING_SCORES = []
ACCURACY_SCORES = []
PRECISION_SCORES = []
RECALL_SCORES = []
F1_SCORES = []
GINI_SCORES = []


def objective(params: ng.p.Dict) -> float:
    model = XGBClassifier(
        booster = BOOSTER,
        objective = "binary:logistic",
        random_state = SEED,
        scale_pos_weight = scale_pos_weight,
        tree_method = TREE_METHOD,
        **params
    )

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_valid)[:, 1]

    y_pred = (y_proba >= THRESHOLD).astype(int)
    
    auc = roc_auc_score(y_valid, y_proba)
    bce = log_loss(y_valid, y_proba)
    acc = accuracy_score(y_valid, y_pred)
    prec = precision_score(y_valid, y_pred, zero_division = 0)
    f1 = f1_score(y_valid, y_pred, zero_division = 0)
    rec = recall_score(y_valid, y_pred, zero_division = 0)
    gini = 2 * auc - 1

    ACCURACY_SCORES.append(acc)
    PRECISION_SCORES.append(prec)
    RECALL_SCORES.append(rec)
    F1_SCORES.append(f1)
    GINI_SCORES.append(gini)
    LEARNING_SCORES.append(bce)

    return -gini 

search_space = load_params()

if __name__ == "__main__":
    optimizer = ng.optimizers.RandomSearch(
        parametrization = search_space,
        budget = N_TRIALS
    )
    with futures.ThreadPoolExecutor(max_workers = optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(objective, executor = executor)
    best_params = recommendation.value

    best_params_df = pd.DataFrame(
        best_params.items(),
        columns=['parameter', 'value']
    )

    optimizer_name = os.path.basename(__file__).replace('.py', '')

    """SAVE SINGLE TRIAL WITH MEASUREMENTS""" 
    metric_df = pd.DataFrame({
    'accuracy': ACCURACY_SCORES, 
    'precision': PRECISION_SCORES, 
    'recall': RECALL_SCORES,
    'f1': F1_SCORES,
    'learning curve': LEARNING_SCORES,
    'gini': GINI_SCORES
    })

    current_time = datetime.now()

    metric_df.to_csv(f'metrics/{BOOSTER}/{optimizer_name}/metrics_{current_time}.csv')
    best_params_df.to_json(f'metrics/{BOOSTER}/{optimizer_name}/best_params_{current_time}.json', orient = 'records', indent = 2)
