import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from sklearn.model_selection import train_test_split
import nevergrad as ng
from Optimization.Constraints import *

SEED = 0

"""CONFIGURATION PARAMETERS"""
PROPORTION = 0.2
SHUFFLED = True
BOOSTER = 'gbtree'
N_TRIALS = 250
TREE_METHOD = "hist"


"""Search spaces for specific boosters"""
def load_params() -> ng.p.Dict:
    if BOOSTER == 'gbtree':
        parametrization = ng.p.Dict(
            n_estimators = ng.p.Scalar(lower = N_ESTIMATORSP, upper = N_ESTIMATORSK).set_integer_casting(),
            max_depth = ng.p.Scalar(lower = MAX_DEPTHP, upper = MAX_DEPTHK).set_integer_casting(),
            eta = ng.p.Scalar(lower = ETAP, upper = ETAK),
            reg_lambda = ng.p.Scalar(lower = LAMBDAP, upper = LAMBDAK),
            alpha = ng.p.Scalar(lower = ALPHAP, upper = ALPHAK),
            subsample = ng.p.Scalar(lower = SUBSAMPLEP, upper = SUBSAMPLEK),
            colsample_bytree = ng.p.Scalar(lower = COLSAMPLE_BYTREEP, upper = COLSAMPLE_BYTREEK),
            gamma = ng.p.Scalar(lower = GAMMAP, upper = GAMMAK),
            min_child_weight = ng.p.Scalar(lower = MIN_CHILD_WEIGHTP, upper = MIN_CHILD_WEIGHTK).set_integer_casting(),
        )
    if BOOSTER == 'gblinear':
        parametrization = ng.p.Dict(
            n_estimators = ng.p.Scalar(lower = N_ESTIMATORSP, upper = N_ESTIMATORSK).set_integer_casting(),
            eta = ng.p.Scalar(lower = ETAP, upper = ETAK),
            reg_lambda = ng.p.Scalar(lower = LAMBDAP, upper = LAMBDAK),
            alpha = ng.p.Scalar(lower = ALPHAP, upper = ALPHAK),
            updater = ng.p.Choice(UPDATER_LIST),
            feature_selector = ng.p.Choice(FEATURE_SELECTOR_LIST)
        )
    if BOOSTER == 'dart':
        parametrization = ng.p.Dict(
            n_estimators = ng.p.Scalar(lower = N_ESTIMATORSP, upper = N_ESTIMATORSK).set_integer_casting(),
            max_depth = ng.p.Scalar(lower = MAX_DEPTHP, upper = MAX_DEPTHK).set_integer_casting(),
            eta = ng.p.Scalar(lower = ETAP, upper = ETAK),
            reg_lambda = ng.p.Scalar(lower = LAMBDAP, upper = LAMBDAK),
            alpha = ng.p.Scalar(lower = ALPHAP, upper = ALPHAK),
            subsample = ng.p.Scalar(lower = SUBSAMPLEP, upper = SUBSAMPLEK),
            colsample_bytree = ng.p.Scalar(lower = COLSAMPLE_BYTREEP, upper = COLSAMPLE_BYTREEK),
            gamma = ng.p.Scalar(lower = GAMMAP, upper = GAMMAK),
            min_child_weight = ng.p.Scalar(lower = MIN_CHILD_WEIGHTP, upper = MIN_CHILD_WEIGHTK).set_integer_casting(),
            sample_type = ng.p.Choice(SAMPLE_TYPE_LIST),
            normalize_type = ng.p.Choice(NORMALIZE_TYPE_LIST),
            rate_drop = ng.p.Scalar(lower = RATE_DROPP, upper = RATE_DROPK),
            one_drop = ng.p.Choice(ONE_DROP_LIST),
            skip_drop = ng.p.Scalar(lower = SKIP_DROPP, upper = SKIP_DROPK),
        )
    return parametrization


train_df = pd.read_csv('datasets/train.csv')
test_df = pd.read_csv('datasets/test.csv')


def clean_data(df1: pd.DataFrame = train_df, df2: pd.DataFrame = test_df, 
               test_size: float = PROPORTION, 
               shuffled: bool = SHUFFLED,
               seed: int = SEED) -> tuple[pd.DataFrame, 
                                          pd.DataFrame, 
                                          pd.DataFrame, 
                                          pd.DataFrame,
                                          pd.DataFrame]:
    onehot_encoder = OneHotEncoder()    

    all_df = pd.concat([df1, df2], ignore_index = True)
    all_df = all_df.drop('target', axis = 1)
    all_features = all_df.columns
    cat_featuers = [feature for feature in all_features if 'cat' in feature]

    encoded_cat_matrix = onehot_encoder.fit_transform(all_df[cat_featuers])

    all_df['num_missing'] = (all_df == -1).sum(axis = 1)
    remaining_features = [feature for feature in all_features if ('cat' not in feature and 'calc' not in feature)]
    remaining_features.append('num_missing')
    ind_features = [feature for feature in all_features if 'ind' in feature]

    is_first_feature = True
    for ind_feature in ind_features:
        if is_first_feature:
            all_df['mix_ind'] = all_df[ind_feature].astype(str) + '_'
            is_first_feature = False
        else:
            all_df['mix_ind'] += all_df[ind_feature].astype(str) + '_'
    
    cat_count_features = []
    for feature in cat_featuers + ['mix_ind']:
        val_counts_dict = all_df[feature].value_counts().to_dict()
        all_df[f'{feature}_count'] = all_df[feature].apply(lambda x: val_counts_dict[x])
        cat_count_features.append(f'{feature}_count')

    drop_features = ['ps_ind_14', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_14']
    all_df_remaining = all_df[remaining_features + cat_count_features].drop(drop_features, axis=1)

    bad_cols = all_df_remaining.select_dtypes(include=["object"])

    all_df_remaining[bad_cols.columns] = (all_df_remaining[bad_cols.columns].apply(pd.to_numeric, errors="coerce"))
    all_df_sprs = sparse.hstack([sparse.csr_matrix(all_df_remaining), encoded_cat_matrix], format='csr')
    
    num_train = len(df1)
    X = all_df_sprs[:num_train]
    X_test = all_df_sprs[num_train:]

    y = df1['target'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                          test_size = test_size, 
                                                          shuffle = shuffled,
                                                          random_state = seed)
    
    return X_train, X_valid, y_train, y_valid, X_test


X_train, X_valid, y_train, y_valid, X_test = clean_data()


THRESHOLD = 0.5
scale_pos_weight = (sum(y_train == 0) / sum(y_train == 1))