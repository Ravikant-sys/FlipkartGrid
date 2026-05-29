"""
Traffic Demand Prediction - High-Performance ML Solution Pipeline
Optimized for R² Score using the FULL training dataset (77,299 rows) and geohash lat/lon coordinates decoding.
Author: Antigravity AI Coding Assistant
"""

import os
import time
warning_bypass = True
import warnings
if warning_bypass:
    warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import joblib

# Seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def print_section(title):
    print("=" * 80)
    print(f" {title.upper()} ".center(80, "="))
    print("=" * 80)

# ==========================================
# 0. GEOHASH DECODER FUNCTION
# ==========================================
def decode_geohash(geohash):
    """
    Decodes a standard 6-character geohash string into numerical latitude and longitude.
    Self-contained Base32 decoder.
    """
    BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    char_map = {c: i for i, c in enumerate(BASE32)}
    
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    
    is_even = True
    for char in str(geohash):
        if char not in char_map:
            continue
        val = char_map[char]
        for mask in [16, 8, 4, 2, 1]:
            bit = 1 if (val & mask) else 0
            if is_even:
                # Longitude
                mid = (lon_interval[0] + lon_interval[1]) / 2
                if bit:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                # Latitude
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if bit:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
            
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon

# ==========================================
# 1. LOAD DATA & INITIALIZE FULL TRAINING
# ==========================================
print_section("1. Loading and Reading Dataset")

train_path = "dataset/train.csv"
test_path = "dataset/test.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Train or test dataset files not found in dataset/ directory!")

# Load full datasets
train_full = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Loaded train.csv. Original Shape: {train_full.shape}")
print(f"Loaded test.csv. Original Shape: {test.shape}")

# Use the full training set as requested
train = train_full.copy().reset_index(drop=True)
print(f"Using the full {train.shape[0]} training entries for training and validation.")

# Keep target variable separate
y = train['demand'].values
print(f"Target variable 'demand' statistics:\n{train['demand'].describe()}")

# ==========================================
# 2. INTELLIGENT MISSING VALUE IMPUTATION
# ==========================================
print_section("2. Intelligent Missing Value Imputation")

# Combine datasets for consistent imputation mappings
combined = pd.concat([train.drop(columns=['demand'], errors='ignore'), test], ignore_index=True)

# RoadType, NumberofLanes, LargeVehicles, Landmarks: Physical/spatial features. Impute using mode per geohash.
spatial_categorical_cols = ['RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Weather']

for col in spatial_categorical_cols:
    geo_mode = combined.groupby('geohash')[col].apply(lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
    
    train[col] = train[col].fillna(train['geohash'].map(geo_mode))
    test[col] = test[col].fillna(test['geohash'].map(geo_mode))
    
    global_mode = combined[col].mode()[0]
    train[col] = train[col].fillna(global_mode)
    test[col] = test[col].fillna(global_mode)

# Temperature: Continuous variable. Impute using mean per geohash, fallback to global mean.
geo_temp_mean = combined.groupby('geohash')['Temperature'].mean().to_dict()
train['Temperature'] = train['Temperature'].fillna(train['geohash'].map(geo_temp_mean))
test['Temperature'] = test['Temperature'].fillna(test['geohash'].map(geo_temp_mean))

global_temp_mean = combined['Temperature'].mean()
train['Temperature'] = train['Temperature'].fillna(global_temp_mean)
test['Temperature'] = test['Temperature'].fillna(global_temp_mean)

print("Missing values after intelligent imputation:")
print("Train missing counts:\n", train.isnull().sum())
print("Test missing counts:\n", test.isnull().sum())

# ==========================================
# 3. ADVANCED FEATURE ENGINEERING
# ==========================================
print_section("3. Advanced Feature Engineering & Geohash Decoding")

def engineer_features(df, combined_ref=None):
    df = df.copy()
    
    # 3.0 Geohash Latitude/Longitude Decoding
    coords = df['geohash'].apply(decode_geohash)
    df['latitude'] = [c[0] for c in coords]
    df['longitude'] = [c[1] for c in coords]
    
    # Coordinates interaction features
    df['lat_plus_lon'] = df['latitude'] + df['longitude']
    df['lat_minus_lon'] = df['latitude'] - df['longitude']
    df['lat_times_lon'] = df['latitude'] * df['longitude']
    
    # 3.1 Extract Temporal Features
    def parse_time(ts):
        h, m = map(int, ts.split(':'))
        return h, m
    
    times = df['timestamp'].apply(parse_time)
    df['hour'] = [t[0] for t in times]
    df['minute'] = [t[1] for t in times]
    df['minutes_elapsed'] = df['hour'] * 60 + df['minute']
    
    # Cyclical representations to preserve continuity of time
    df['sin_time'] = np.sin(2 * np.pi * df['minutes_elapsed'] / 1440.0)
    df['cos_time'] = np.cos(2 * np.pi * df['minutes_elapsed'] / 1440.0)
    
    # Day-based features
    df['day_of_week'] = df['day'] % 7
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = (df['day'] // 30) + 1
    
    # Peak hour detection (Rush hours: 7:00-9:00 AM and 5:00-7:00 PM)
    df['peak_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))).astype(int)
    
    # 3.2 Spatial Hierarchical Features
    df['geo_prefix_4'] = df['geohash'].str[:4]
    df['geo_prefix_5'] = df['geohash'].str[:5]
    
    # 3.3 Frequency Encoding (using the full reference dataframe for stable counts)
    if combined_ref is not None:
        for col in ['geohash', 'geo_prefix_4', 'geo_prefix_5']:
            freq = combined_ref[col].value_counts().to_dict()
            df[f'{col}_freq'] = df[col].map(freq)
            
    # 3.4 Traffic Density Indicators
    df['lanes_density_proxy'] = df['NumberofLanes'] * (df['LargeVehicles'].map({'Allowed': 1.5, 'Not Allowed': 1.0}))
    
    # 3.5 Weather Severity Encoding
    weather_severity_map = {'Sunny': 0, 'Foggy': 1, 'Rainy': 2, 'Snowy': 3}
    df['weather_severity'] = df['Weather'].map(weather_severity_map)
    
    # 3.6 Interaction Features
    df['temp_weather_severity'] = df['Temperature'] * df['weather_severity']
    
    return df

# Create dynamic combined reference for frequency encoding
combined_ref = pd.concat([train.drop(columns=['demand'], errors='ignore'), test], ignore_index=True)
combined_ref['geo_prefix_4'] = combined_ref['geohash'].str[:4]
combined_ref['geo_prefix_5'] = combined_ref['geohash'].str[:5]

train_fe = engineer_features(train, combined_ref)
test_fe = engineer_features(test, combined_ref)

print(f"Features engineered successfully. Train Shape: {train_fe.shape}, Test Shape: {test_fe.shape}")

# ==========================================
# 4. OUT-OF-FOLD TARGET ENCODING
# ==========================================
print_section("4. Out-of-Fold Target Encoding")

categorical_target_cols = ['geohash', 'geo_prefix_4', 'geo_prefix_5']

# Compute Target Encoding out-of-fold for training set, map to test set
for col in categorical_target_cols:
    train_fe[f'{col}_te'] = np.nan
    test_fe[f'{col}_te'] = 0.0

kf_te = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for col in categorical_target_cols:
    for train_idx, val_idx in kf_te.split(train_fe):
        tr_fold = train_fe.iloc[train_idx]
        val_fold = train_fe.iloc[val_idx]
        
        mean_target = tr_fold.groupby(col)['demand'].mean()
        train_fe.iloc[val_idx, train_fe.columns.get_loc(f'{col}_te')] = val_fold[col].map(mean_target)
        
    global_mean = train_fe['demand'].mean()
    train_fe[f'{col}_te'] = train_fe[f'{col}_te'].fillna(global_mean)
    
    full_mean = train_fe.groupby(col)['demand'].mean()
    test_fe[f'{col}_te'] = test_fe[col].map(full_mean).fillna(global_mean)

print("Target encoding generated successfully.")

# ==========================================
# 5. ENCODE CATEGORICAL FEATURES FOR MODELS
# ==========================================
print_section("5. Label Encoding and Category Dtypes")

cat_cols = ['geohash', 'geo_prefix_4', 'geo_prefix_5', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']

# LabelEncoded features for RandomForest
for col in cat_cols:
    le = LabelEncoder()
    le.fit(combined_ref[col].astype(str))
    train_fe[f'{col}_le'] = le.transform(train_fe[col].astype(str))
    test_fe[f'{col}_le'] = le.transform(test_fe[col].astype(str))

# Create pandas category dtypes for LightGBM and XGBoost
for col in cat_cols:
    train_fe[col] = train_fe[col].astype('category')
    test_fe[col] = test_fe[col].astype('category')

# Define features group
base_features = [
    'day', 'hour', 'minute', 'minutes_elapsed', 'sin_time', 'cos_time',
    'day_of_week', 'weekend', 'month', 'peak_hour',
    'NumberofLanes', 'Temperature', 'weather_severity',
    'geohash_freq', 'geo_prefix_4_freq', 'geo_prefix_5_freq',
    'lanes_density_proxy', 'temp_weather_severity',
    'geohash_te', 'geo_prefix_4_te', 'geo_prefix_5_te',
    'latitude', 'longitude', 'lat_plus_lon', 'lat_minus_lon', 'lat_times_lon'
]

native_cat_features = base_features + cat_cols
numerical_features = base_features + [f'{col}_le' for col in cat_cols]

print(f"Total features for native boosting models: {len(native_cat_features)}")
print(f"Total features for standard numerical models: {len(numerical_features)}")

# ==========================================
# 6. MODEL CROSS-VALIDATION PIPELINE
# ==========================================
print_section("6. Model Cross-Validation and Tuning on Full Dataset")

K_FOLDS = 5
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

oof_preds = {
    'LightGBM': np.zeros(len(train_fe)),
    'XGBoost': np.zeros(len(train_fe)),
    'CatBoost': np.zeros(len(train_fe)),
    'RandomForest': np.zeros(len(train_fe))
}

test_preds = {
    'LightGBM': np.zeros(len(test_fe)),
    'XGBoost': np.zeros(len(test_fe)),
    'CatBoost': np.zeros(len(test_fe)),
    'RandomForest': np.zeros(len(test_fe))
}

best_trained_models = {}
best_r2_scores = {}

# ------------------------------------------
# 6.1 LightGBM Regressor
# ------------------------------------------
print("\n--- Training LightGBM ---")
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 8,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': RANDOM_SEED,
    'verbose': -1,
    'n_jobs': -1
}

lgb_oof = np.zeros(len(train_fe))
lgb_test = np.zeros(len(test_fe))
best_lgb_model = None
best_lgb_r2 = -1

for fold, (train_idx, val_idx) in enumerate(kf.split(train_fe)):
    X_tr, y_tr = train_fe.iloc[train_idx][native_cat_features], y[train_idx]
    X_val, y_val = train_fe.iloc[val_idx][native_cat_features], y[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params, n_estimators=1500)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    val_pred = model.predict(X_val)
    lgb_oof[val_idx] = val_pred
    
    fold_r2 = r2_score(y_val, val_pred)
    print(f"Fold {fold+1} R² Score: {fold_r2:.5f}")
    
    if fold_r2 > best_lgb_r2:
        best_lgb_r2 = fold_r2
        best_lgb_model = model
        
    lgb_test += model.predict(test_fe[native_cat_features]) / K_FOLDS

oof_preds['LightGBM'] = lgb_oof
test_preds['LightGBM'] = lgb_test
best_trained_models['LightGBM'] = best_lgb_model
best_r2_scores['LightGBM'] = r2_score(y, lgb_oof)
print(f"LightGBM Overall OOF R² Score: {best_r2_scores['LightGBM']:.5f}")

# ------------------------------------------
# 6.2 XGBoost Regressor
# ------------------------------------------
print("\n--- Training XGBoost ---")
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'enable_categorical': True,
    'n_jobs': -1
}

xgb_oof = np.zeros(len(train_fe))
xgb_test = np.zeros(len(test_fe))
best_xgb_model = None
best_xgb_r2 = -1

for fold, (train_idx, val_idx) in enumerate(kf.split(train_fe)):
    X_tr, y_tr = train_fe.iloc[train_idx][native_cat_features], y[train_idx]
    X_val, y_val = train_fe.iloc[val_idx][native_cat_features], y[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params, n_estimators=1500, early_stopping_rounds=50)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    val_pred = model.predict(X_val)
    xgb_oof[val_idx] = val_pred
    
    fold_r2 = r2_score(y_val, val_pred)
    print(f"Fold {fold+1} R² Score: {fold_r2:.5f}")
    
    if fold_r2 > best_xgb_r2:
        best_xgb_r2 = fold_r2
        best_xgb_model = model
        
    xgb_test += model.predict(test_fe[native_cat_features]) / K_FOLDS

oof_preds['XGBoost'] = xgb_oof
test_preds['XGBoost'] = xgb_test
best_trained_models['XGBoost'] = best_xgb_model
best_r2_scores['XGBoost'] = r2_score(y, xgb_oof)
print(f"XGBoost Overall OOF R² Score: {best_r2_scores['XGBoost']:.5f}")

# ------------------------------------------
# 6.3 CatBoost Regressor
# ------------------------------------------
print("\n--- Training CatBoost ---")
cat_feature_names = [col for col in cat_cols]

cat_oof = np.zeros(len(train_fe))
cat_test = np.zeros(len(test_fe))
best_cat_model = None
best_cat_r2 = -1

# Category to string for CatBoost
train_fe_cb = train_fe.copy()
test_fe_cb = test_fe.copy()
for col in cat_cols:
    train_fe_cb[col] = train_fe_cb[col].astype(str)
    test_fe_cb[col] = test_fe_cb[col].astype(str)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_fe_cb)):
    X_tr, y_tr = train_fe_cb.iloc[train_idx][native_cat_features], y[train_idx]
    X_val, y_val = train_fe_cb.iloc[val_idx][native_cat_features], y[val_idx]
    
    train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_names)
    val_pool = Pool(X_val, y_val, cat_features=cat_feature_names)
    
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.06,
        depth=7,
        eval_metric='RMSE',
        random_seed=RANDOM_SEED,
        verbose=False,
        early_stopping_rounds=50,
        task_type='CPU'
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    val_pred = model.predict(val_pool)
    cat_oof[val_idx] = val_pred
    
    fold_r2 = r2_score(y_val, val_pred)
    print(f"Fold {fold+1} R² Score: {fold_r2:.5f}")
    
    if fold_r2 > best_cat_r2:
        best_cat_r2 = fold_r2
        best_cat_model = model
        
    test_pool = Pool(test_fe_cb[native_cat_features], cat_features=cat_feature_names)
    cat_test += model.predict(test_pool) / K_FOLDS

oof_preds['CatBoost'] = cat_oof
test_preds['CatBoost'] = cat_test
best_trained_models['CatBoost'] = best_cat_model
best_r2_scores['CatBoost'] = r2_score(y, cat_oof)
print(f"CatBoost Overall OOF R² Score: {best_r2_scores['CatBoost']:.5f}")

# ------------------------------------------
# 6.4 Random Forest Regressor
# ------------------------------------------
print("\n--- Training RandomForest ---")

rf_oof = np.zeros(len(train_fe))
rf_test = np.zeros(len(test_fe))
best_rf_model = None
best_rf_r2 = -1

for fold, (train_idx, val_idx) in enumerate(kf.split(train_fe)):
    X_tr, y_tr = train_fe.iloc[train_idx][numerical_features], y[train_idx]
    X_val, y_val = train_fe.iloc[val_idx][numerical_features], y[val_idx]
    
    # Optimized RandomForest config for speed and robustness on the full 77k dataset
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    
    val_pred = model.predict(X_val)
    rf_oof[val_idx] = val_pred
    
    fold_r2 = r2_score(y_val, val_pred)
    print(f"Fold {fold+1} R² Score: {fold_r2:.5f}")
    
    if fold_r2 > best_rf_r2:
        best_rf_r2 = fold_r2
        best_rf_model = model
        
    rf_test += model.predict(test_fe[numerical_features]) / K_FOLDS

oof_preds['RandomForest'] = rf_oof
test_preds['RandomForest'] = rf_test
best_trained_models['RandomForest'] = best_rf_model
best_r2_scores['RandomForest'] = r2_score(y, rf_oof)
print(f"RandomForest Overall OOF R² Score: {best_r2_scores['RandomForest']:.5f}")

# ==========================================
# 7. HIGH-PERFORMANCE ENSEMBLING / STACKING
# ==========================================
print_section("7. High-Performance Ensembling & Weight Optimization")

best_blend_weights = None
best_blend_r2 = -1

# Grid search optimal blending weights on the full OOF validation predictions
for w1 in np.linspace(0, 1, 21):
    for w2 in np.linspace(0, 1 - w1, 21):
        for w3 in np.linspace(0, 1 - w1 - w2, 21):
            w4 = 1.0 - w1 - w2 - w3
            if w4 < -1e-5:
                continue
            
            blend_pred = w1 * oof_preds['LightGBM'] + w2 * oof_preds['XGBoost'] + w3 * oof_preds['CatBoost'] + w4 * oof_preds['RandomForest']
            blend_r2 = r2_score(y, blend_pred)
            
            if blend_r2 > best_blend_r2:
                best_blend_r2 = blend_r2
                best_blend_weights = (w1, w2, w3, w4)

w_lgb, w_xgb, w_cat, w_rf = best_blend_weights
print("\nConstrained Weight Search Results:")
print(f"  LightGBM weight:    {w_lgb:.3f}")
print(f"  XGBoost weight:     {w_xgb:.3f}")
print(f"  CatBoost weight:    {w_cat:.3f}")
print(f"  RandomForest weight: {w_rf:.3f}")
print(f"Optimized Blended Validation R² Score: {best_blend_r2:.5f}")

# Generate blended OOF and final ensemble test predictions
final_test_predictions = w_lgb * test_preds['LightGBM'] + w_xgb * test_preds['XGBoost'] + w_cat * test_preds['CatBoost'] + w_rf * test_preds['RandomForest']

# Clip final predictions to valid target bounds [min(y), max(y)]
y_min, y_max = y.min(), y.max()
final_test_predictions = np.clip(final_test_predictions, y_min, y_max)

# ==========================================
# 8. FEATURE IMPORTANCE RANKING
# ==========================================
print_section("8. Feature Importance Analysis")

lgb_model = best_trained_models['LightGBM']
importance_df = pd.DataFrame({
    'Feature': native_cat_features,
    'Importance': lgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 15 Most Important Features in LightGBM Model:")
print(importance_df.head(15).to_string(index=False))

# Plot and save feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=importance_df.head(20), x='Importance', y='Feature', palette='viridis')
plt.title('Top 20 Feature Importance - LightGBM Model (Coordinates Enabled)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()
print("Saved feature importance plot to feature_importance.png")

# ==========================================
# 9. SAVE MODELS & GENERATE SUBMISSION
# ==========================================
print_section("9. Generating Submission File and Saving Models")

submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': final_test_predictions
})

os.makedirs('output', exist_ok=True)
sub_path = "output/submission.csv"
submission.to_csv(sub_path, index=False)
print(f"Created submission.csv successfully at {sub_path}!")
print("First 10 rows of submission:")
print(submission.head(10).to_string(index=False))

print(f"Submission Shape: {submission.shape}")
print("Null count in submission demand column:", submission['demand'].isnull().sum())

# Save the best overall individual model based on OOF performance
best_model_name = max(best_r2_scores, key=best_r2_scores.get)
print(f"\nBest performing individual model: {best_model_name} with R²: {best_r2_scores[best_model_name]:.5f}")

best_model = best_trained_models[best_model_name]
joblib.dump(best_model, f'output/best_{best_model_name.lower()}_model.pkl')
print(f"Saved the best trained model to output/best_{best_model_name.lower()}_model.pkl")

print_section("Processing Completed Successfully!")
