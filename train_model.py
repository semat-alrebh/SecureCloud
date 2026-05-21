"""
train_model.py
شغّلي هذا الملف مرة وحدة لتدريب الـ 6 خوارزميات على الثلاثة datasets
"""

import pandas as pd
import numpy as np
import pickle
import os
import glob
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

NSLKDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]
NSLKDD_CATEGORICAL = ['protocol_type', 'service', 'flag']

def get_models():
    return {
        'Decision Tree':       DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        'KNN':                 KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'SVM':                 SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
        'ANN':                 MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
    }

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'f1':        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

def train_and_save(X_train_s, y_train, X_test_s, y_test, scaler, encoders, feature_names, output_file, dataset_name):
    models = get_models()
    results = {}
    best_model = None
    best_f1 = 0
    best_model_name = ''

    for name, model in models.items():
        print(f"  Training {name}...")
        if name == 'SVM':
            idx = np.random.choice(len(X_train_s), min(50000, len(X_train_s)), replace=False)
            model.fit(X_train_s[idx], y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx])
        else:
            model.fit(X_train_s, y_train)

        metrics = evaluate(model, X_test_s, y_test)
        results[name] = metrics
        print(f"    F1: {metrics['f1']}% | Accuracy: {metrics['accuracy']}%")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = model
            best_model_name = name

    # Feature importance
    top_features = {}
    if 'Random Forest' in models:
        rf = models['Random Forest']
        if hasattr(rf, 'feature_importances_'):
            fi = pd.Series(rf.feature_importances_, index=feature_names)
            top_features = fi.nlargest(10).to_dict()

    save_data = {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'all_models': models,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': feature_names,
        'all_results': results,
        'top_features': top_features,
        'dataset': dataset_name,
        'signature_columns': set(feature_names)
    }

    with open(output_file, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\n  Best: {best_model_name} (F1: {best_f1}%)")
    print(f"  Saved: {output_file}\n")

# ── NSL-KDD ──────────────────────────────────────────────────────────────────
def train_nslkdd():
    print("\n" + "="*50)
    print("1. NSL-KDD")
    print("="*50)

    train = pd.read_csv('KDDTrain+.txt', header=None, names=NSLKDD_COLUMNS)
    test  = pd.read_csv('KDDTest+.txt',  header=None, names=NSLKDD_COLUMNS)

    train['target'] = (train['label'] != 'normal').astype(int)
    test['target']  = (test['label']  != 'normal').astype(int)
    train.drop(columns=['label', 'difficulty'], inplace=True)
    test.drop(columns=['label', 'difficulty'],  inplace=True)

    encoders = {}
    for col in NSLKDD_CATEGORICAL:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])
        encoders[col] = le

    X_train = train.drop('target', axis=1)
    y_train = train['target']
    X_test  = test.drop('target', axis=1)
    y_test  = test['target']

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    train_and_save(X_train_s, y_train, X_test_s, y_test,
                   scaler, encoders, X_train.columns.tolist(),
                   'model_nslkdd.pkl', 'nslkdd')

# ── UNSW-NB15 ────────────────────────────────────────────────────────────────
def train_unsw():
    print("\n" + "="*50)
    print("2. UNSW-NB15")
    print("="*50)

    train_files = glob.glob('unsw_nb15/*train*.parquet') + glob.glob('unsw_nb15/*training*.parquet')
    test_files  = glob.glob('unsw_nb15/*test*.parquet')  + glob.glob('unsw_nb15/*testing*.parquet')

    if not train_files or not test_files:
        print("  ERROR: Files not found in unsw_nb15/")
        return

    train = pd.read_parquet(train_files[0])
    test  = pd.read_parquet(test_files[0])
    print(f"  Train: {len(train):,} | Test: {len(test):,}")

    label_col = next((c for c in ['label', 'Label', 'class', 'Class'] if c in train.columns), None)
    if not label_col:
        print("  ERROR: label column not found")
        return

    if train[label_col].dtype == object:
        train['target'] = (train[label_col].str.lower() != 'normal').astype(int)
        test['target']  = (test[label_col].str.lower()  != 'normal').astype(int)
    else:
        train['target'] = train[label_col].astype(int)
        test['target']  = test[label_col].astype(int)

    drop_cols = [label_col, 'attack_cat', 'id', 'proto', 'service', 'state']
    train.drop(columns=[c for c in drop_cols if c in train.columns], inplace=True, errors='ignore')
    test.drop(columns=[c for c in drop_cols if c in test.columns],   inplace=True, errors='ignore')

    encoders = {}
    for col in train.select_dtypes(include='object').columns:
        if col == 'target': continue
        le = LabelEncoder()
        le.fit(pd.concat([train[col].astype(str), test[col].astype(str)]))
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))
        encoders[col] = le

    train.dropna(inplace=True)
    test.dropna(inplace=True)

    X_train = train.drop('target', axis=1)
    y_train = train['target']
    X_test  = test.drop('target', axis=1)
    y_test  = test['target']

    # align columns
    common = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common]
    X_test  = X_test[common]

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    train_and_save(X_train_s, y_train, X_test_s, y_test,
                   scaler, encoders, common,
                   'model_unsw.pkl', 'unsw')

# ── CIC-IDS2017 ──────────────────────────────────────────────────────────────
def train_cicids():
    print("\n" + "="*50)
    print("3. CIC-IDS2017")
    print("="*50)

    files = glob.glob('cicids2017/*.parquet')
    if not files:
        print("  ERROR: Files not found in cicids2017/")
        return

    print(f"  Loading {len(files)} parquet files...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  Total: {len(df):,}")

    label_col = next((c for c in ['Label', 'label', 'Class', 'class'] if c in df.columns), None)
    if not label_col:
        print("  ERROR: label column not found")
        return

    df['target'] = (df[label_col].str.upper() != 'BENIGN').astype(int)
    df.drop(columns=[label_col], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if len(df) > 500000:
        print(f"  Sampling 500,000 records...")
        df = df.sample(500000, random_state=42)

    from sklearn.model_selection import train_test_split
    X = df.drop('target', axis=1)
    y = df['target']

    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        le.fit(X[col].astype(str))
        X[col] = le.transform(X[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    train_and_save(X_train_s, y_train, X_test_s, y_test,
                   scaler, encoders, X.columns.tolist(),
                   'model_cicids.pkl', 'cicids')

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("SecureFlow — Training 6 Models on 3 Datasets")
    print("Estimated time: 15-30 minutes\n")

    train_nslkdd()
    train_unsw()
    train_cicids()

    print("="*50)
    print("DONE! Files saved:")
    for f in ['model_nslkdd.pkl', 'model_unsw.pkl', 'model_cicids.pkl']:
        if os.path.exists(f):
            print(f"  {f}")
