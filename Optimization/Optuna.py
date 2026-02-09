import optuna
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score
from PreparedData import (X_train, X_valid, y_train, y_valid, 
                          SEED, BOOSTER, N_TRIALS, TREE_METHOD, THRESHOLD, 
                          scale_pos_weight)
from xgboost import XGBClassifier
from Constraints import *
import pandas as pd
from datetime import datetime
import os


LEARNING_SCORES = []
ACCURACY_SCORES = []
PRECISION_SCORES = []
RECALL_SCORES = []
F1_SCORES = []
GINI_SCORES = [] 


def objective(trial):
    params = {
        "lambda": trial.suggest_float("lambda", LAMBDAP, LAMBDAK),
        "alpha": trial.suggest_float("alpha", ALPHAP, ALPHAK),
        "n_estimators": trial.suggest_int("n_estimators", N_ESTIMATORSP, N_ESTIMATORSK),
        "eta": trial.suggest_float("eta", ETAP, ETAK)
    }
    if BOOSTER == "gbtree" or BOOSTER == "dart":
        params["max_depth"] = trial.suggest_int("max_depth", MAX_DEPTHP, MAX_DEPTHK)
        params["subsample"] = trial.suggest_float("subsample", SUBSAMPLEP, SUBSAMPLEK)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", COLSAMPLE_BYTREEP, COLSAMPLE_BYTREEK)
        params["min_child_weight"] = trial.suggest_int("min_child_weight", MIN_CHILD_WEIGHTP, MIN_CHILD_WEIGHTK)
        params["gamma"] = trial.suggest_float("gamma", GAMMAP, GAMMAK)
    if BOOSTER == 'gblinear':
        params['updater'] = trial.suggest_categorical('updater', UPDATER_LIST)
        params['feature_selector'] = trial.suggest_categorical('feature_selector', FEATURE_SELECTOR_LIST)
    if BOOSTER == "dart":
        params["sample_type"] = trial.suggest_categorical("sample_type", SAMPLE_TYPE_LIST)
        params["normalize_type"] = trial.suggest_categorical("normalize_type", NORMALIZE_TYPE_LIST)
        params["rate_drop"] = trial.suggest_float("rate_drop", RATE_DROPP, RATE_DROPK)
        params["skip_drop"] = trial.suggest_float("skip_drop", SKIP_DROPP, SKIP_DROPK)
        params["one_drop"] = trial.suggest_categorical('one_drop', ONE_DROP_LIST)
    model = XGBClassifier(
        booster = BOOSTER,
        objective = "binary:logistic",
        tree_method = TREE_METHOD,
        random_state = SEED,
        scale_pos_weight = scale_pos_weight,
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

    return gini


if __name__ == "__main__":
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials = N_TRIALS)

    best_params_df = pd.DataFrame(
        study.best_params.items(),
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
