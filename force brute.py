import os
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
dtypes = {'gare': str, 'arret': str}
X_train_df = pd.read_csv('x_train_final.csv', dtype=dtypes)
y_train_df = pd.read_csv('y_train_final_j5KGWWK.csv')
X_test_df = pd.read_csv('x_test_final.csv', dtype=dtypes)

X_train_df.columns = X_train_df.columns.astype(str).str.strip()
X_test_df.columns = X_test_df.columns.astype(str).str.strip()
y_train = y_train_df['p0q0'].values

def extract_pro_features(df):
    df_feat = df.copy()
    
    lag_cols = ['p0q2', 'p0q3', 'p0q4', 'p2q0', 'p3q0', 'p4q0']
    for col in lag_cols:
        if col in df_feat.columns:
            df_feat.loc[df_feat[col] < -180, col] = np.nan
            df_feat[col] = df_feat[col].fillna(0)

    if all(c in df_feat.columns for c in ['p0q2', 'p0q3', 'p0q4']):
        df_feat['vel_spat_1'] = df_feat['p0q2'] - df_feat['p0q3']
        df_feat['vel_spat_2'] = df_feat['p0q3'] - df_feat['p0q4']

        df_feat['acc_spat'] = df_feat['vel_spat_1'] - df_feat['vel_spat_2']

        df_feat['exp_mom_spat'] = (df_feat['p0q2'] * 0.5) + (df_feat['p0q3'] * 0.3) + (df_feat['p0q4'] * 0.2)

    if all(c in df_feat.columns for c in ['p2q0', 'p3q0', 'p4q0']):

        df_feat['vel_temp_1'] = df_feat['p2q0'] - df_feat['p3q0']
        df_feat['vel_temp_2'] = df_feat['p3q0'] - df_feat['p4q0']
        df_feat['acc_temp'] = df_feat['vel_temp_1'] - df_feat['vel_temp_2']

    if 'date' in df_feat.columns:
        df_feat['date_d'] = pd.to_datetime(df_feat['date'])
        df_feat['month'] = df_feat['date_d'].dt.month
        df_feat['day_of_week'] = df_feat['date_d'].dt.dayofweek
        
    return df_feat

X_tr_pro = extract_pro_features(X_train_df)
X_te_pro = extract_pro_features(X_test_df)

num_features = [
    'p0q2', 'p0q3', 'p0q4', 'p2q0', 'p3q0', 'p4q0',
    'vel_spat_1', 'vel_spat_2', 'acc_spat', 'exp_mom_spat',
    'vel_temp_1', 'vel_temp_2', 'acc_temp',
    'month', 'day_of_week'
]

cat_features = ['gare', 'arret']

features = [f for f in (cat_features + num_features) if f in X_tr_pro.columns]

for col in cat_features:
    X_tr_pro[col] = X_tr_pro[col].astype('category')
    X_te_pro[col] = X_te_pro[col].astype('category')

X = X_tr_pro[features]
X_test_final = X_te_pro[features]


SEEDS = [10, 20, 30, 42, 60, 70, 80, 100, 123, 222, 
         333, 456, 500, 666, 777, 888, 999, 1337, 2024, 9999]
meta_preds = np.zeros(len(X_test_final))

for i, seed in enumerate(SEEDS):
    t_seed = time.time()

    model_lgb = lgb.LGBMRegressor(
        objective='mae',
        n_estimators=2500,       
        learning_rate=0.015,     
        num_leaves=255,          
        max_depth=12,
        min_child_samples=250, 
        subsample=0.8,          
        colsample_bytree=0.8, 
        random_state=seed,       
        n_jobs=-1,
        verbose=-1
    )

    model_lgb.fit(X, y_train)
    meta_preds += model_lgb.predict(X_test_final)
    print(f"    LGBM Colosse {i+1}/20 [Seed {seed}] : {time.time()-t_seed:.1f}s")

preds_mean_float = meta_preds / len(SEEDS)
preds_final = np.round(preds_mean_float).astype(int)

nom_index = X_test_df.columns[0]
sub = pd.DataFrame({nom_index: X_test_df[nom_index], 'p0q0': preds_final})
sub_path = 'force brute.csv'
sub.to_csv(sub_path, index=False)