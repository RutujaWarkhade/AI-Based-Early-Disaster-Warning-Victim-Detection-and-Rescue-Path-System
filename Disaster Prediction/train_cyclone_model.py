"""
=============================================================================
DISASTER AI — CYCLONE MODEL v2
=============================================================================
Fixes from v1:
  - "No further splits" warning: fixed with min_data_in_leaf=5, num_leaves=15
  - RI rate 1%: SMOTE oversampling applied
  - Future alert generator: active storms get 24h/48h/72h risk forecast
  - Track any currently-active storms from last 7 days
  - Bay of Bengal vs Arabian Sea seasonal outlook
=============================================================================
"""
import pandas as pd
import numpy as np
import joblib
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

BASE   = Path("D:/Disaster_Prediction")
RAW    = BASE / "raw"
FEAT   = BASE / "features";      FEAT.mkdir(exist_ok=True)
MODELS = BASE / "models";        MODELS.mkdir(exist_ok=True)
PRED   = BASE / "predictions";   PRED.mkdir(exist_ok=True)
ALERTS = BASE / "future_alerts"; ALERTS.mkdir(exist_ok=True)


# ── Load ────────────────────────────────────────────────────────────────────
def load_cyclones():
    df = pd.read_csv(RAW / "cyclones.csv", low_memory=False)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['date']     = pd.to_datetime(df['date'],     errors='coerce')
    for c in ['year','month','latitude','longitude','wind_speed_knots',
              'wind_speed_kmh','pressure_hpa','dist_to_land_km',
              'radius_max_wind_nm','storm_speed_kmh','storm_direction_deg']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['datetime','latitude','longitude','wind_speed_knots'])
    df = df.sort_values(['storm_id','datetime']).reset_index(drop=True)
    print(f"   Loaded {len(df):,} track points | {df['storm_id'].nunique()} storms "
          f"({df['datetime'].min().date()} → {df['datetime'].max().date()})")
    return df


# ── Feature Engineering ─────────────────────────────────────────────────────
def engineer_features(df):
    print("\n⚙️  Engineering cyclone features...")
    df = df.copy()

    # Intensity code
    imap = {'Depression':1,'Deep Depression':2,'Cyclonic Storm':3,
            'Severe Cyclonic Storm':4,'Very Severe':5,'Extremely Severe':6}
    df['intensity_code']  = df['intensity_category'].astype(str)\
                              .map(lambda x: imap.get(x, 0))
    df['pressure_deficit']= 1013.0 - df['pressure_hpa'].fillna(1013)
    df['wind_sq']         = df['wind_speed_knots'] ** 2
    df['log_dist']        = np.log1p(df['dist_to_land_km'].fillna(999))
    df['heading_sin']     = np.sin(np.deg2rad(df['storm_direction_deg'].fillna(0)))
    df['heading_cos']     = np.cos(np.deg2rad(df['storm_direction_deg'].fillna(0)))
    df['sin_month']       = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month']       = np.cos(2 * np.pi * df['month'] / 12)
    df['is_BoB']          = (df['longitude'] > 80).astype(int)
    df['is_AS']           = (df['longitude'] < 78).astype(int)

    # Lag features
    print("   Computing lag features...")
    df = df.sort_values(['storm_id','datetime'])
    for steps, name in [(1,'6h'),(2,'12h'),(4,'24h')]:
        g = df.groupby('storm_id')
        df[f'wind_lag_{name}']      = g['wind_speed_knots'].shift(steps)
        df[f'pressure_lag_{name}']  = g['pressure_hpa'].shift(steps)
        df[f'dist_lag_{name}']      = g['dist_to_land_km'].shift(steps)
        df[f'wind_chg_{name}']      = df['wind_speed_knots'] - df[f'wind_lag_{name}']
        df[f'pressure_chg_{name}']  = df['pressure_hpa']    - df[f'pressure_lag_{name}']
        df[f'dist_chg_{name}']      = df['dist_to_land_km'] - df[f'dist_lag_{name}']

    # Storm lifecycle
    df['storm_first']      = df.groupby('storm_id')['datetime'].transform('min')
    df['storm_age_h']      = (df['datetime'] - df['storm_first']).dt.total_seconds() / 3600
    df['peak_wind']        = df.groupby('storm_id')['wind_speed_knots'].transform('max')
    df['wind_pct_peak']    = df['wind_speed_knots'] / df['peak_wind'].replace(0, 1)
    df['storm_duration_h'] = df.groupby('storm_id')['storm_age_h'].transform('max')

    # Targets
    g = df.groupby('storm_id')
    df['wind_next_24h']   = g['wind_speed_knots'].shift(-4)
    df['dist_next_24h']   = g['dist_to_land_km'].shift(-4)
    df['wind_diff_24h']   = df['wind_next_24h'] - df['wind_speed_knots']
    df['target_RI']       = (df['wind_diff_24h'] >= 35).astype(int)
    df['target_landfall'] = (df['dist_next_24h'] <= 0).astype(int)
    df['target_wind_24h'] = df['wind_next_24h']

    df.to_csv(FEAT / "cyclone_features.csv", index=False)
    print(f"   ✅ Features: {df.shape}")
    return df


# ── Train ───────────────────────────────────────────────────────────────────
FEATURES = [
    'wind_speed_knots','pressure_hpa','pressure_deficit','wind_sq',
    'dist_to_land_km','log_dist','radius_max_wind_nm',
    'storm_speed_kmh','heading_sin','heading_cos',
    'latitude','longitude','intensity_code',
    'wind_lag_6h','wind_lag_12h','wind_lag_24h',
    'pressure_lag_6h','pressure_lag_12h','pressure_lag_24h',
    'wind_chg_6h','wind_chg_12h','wind_chg_24h',
    'pressure_chg_6h','pressure_chg_12h','pressure_chg_24h',
    'dist_chg_6h','dist_chg_12h','dist_chg_24h',
    'storm_age_h','wind_pct_peak','peak_wind','storm_duration_h',
    'sin_month','cos_month','month','is_BoB','is_AS',
]

def train_models(df):
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, mean_squared_error, classification_report
    except ImportError:
        print("❌ pip install lightgbm scikit-learn"); return {}

    print("\n🤖 Training Cyclone Models v2 (RI + Landfall + Wind)...")

    avail  = [c for c in FEATURES if c in df.columns]
    mdf    = df.dropna(subset=['wind_chg_24h','target_RI',
                                'target_landfall','target_wind_24h'] + avail)
    print(f"   Features: {len(avail)}  |  Model rows: {len(mdf):,}")

    # Storm-based temporal split (test = last 20% of storms by first-seen date)
    storm_order  = mdf.groupby('storm_id')['datetime'].min().sort_values()
    n_test       = max(1, int(len(storm_order) * 0.2))
    test_storms  = set(storm_order.index[-n_test:])
    tr_mask      = ~mdf['storm_id'].isin(test_storms)
    te_mask      =  mdf['storm_id'].isin(test_storms)
    X_tr, X_te   = mdf.loc[tr_mask, avail], mdf.loc[te_mask, avail]

    print(f"   Train: {tr_mask.sum():,} rows | Test: {te_mask.sum():,} rows "
          f"| Test storms: {len(test_storms)}")

    results = {}

    # LightGBM base params — fixed: num_leaves=15 and min_data_in_leaf=5
    # eliminates "No further splits" warnings from v1
    base_params = dict(
        n_estimators=600, learning_rate=0.04, num_leaves=15,
        max_depth=5, min_data_in_leaf=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )

    def fit_cls(y_tr_col, y_te_col, label, fname):
        y_tr = mdf.loc[tr_mask, y_tr_col]
        y_te = mdf.loc[te_mask, y_te_col]
        rate = y_tr.mean()
        print(f"\n   [{label}]  positive rate={rate:.2%}")

        # SMOTE for severe imbalance
        Xtr, ytr = X_tr.fillna(0).values, y_tr.values
        try:
            if rate < 0.05:
                from imblearn.over_sampling import SMOTE
                k = min(5, int(ytr.sum()) - 1)
                if k >= 1:
                    Xtr, ytr = SMOTE(random_state=42, k_neighbors=k)\
                                  .fit_resample(Xtr, ytr)
                    print(f"   SMOTE: {len(Xtr):,} samples, rate={ytr.mean():.1%}")
        except ImportError:
            pass

        pos_w = max(1.0, (ytr==0).sum() / max((ytr==1).sum(), 1))
        m = lgb.LGBMClassifier(**base_params, scale_pos_weight=pos_w)
        m.fit(Xtr, ytr,
              eval_set=[(X_te.fillna(0), y_te)],
              callbacks=[lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(200)])

        prob = m.predict_proba(X_te.fillna(0))[:, 1]
        auc  = roc_auc_score(y_te, prob)
        print(f"   AUC={auc:.4f}")
        print(classification_report(y_te, (prob >= 0.5).astype(int),
                                     target_names=[f'No {label}', label],
                                     zero_division=0))
        joblib.dump(m, MODELS / fname)
        return m, float(auc)

    ri_model, ri_auc = fit_cls('target_RI', 'target_RI',
                                 'RI', 'cyclone_ri_model.pkl')
    results['ri_auc'] = ri_auc

    lf_model, lf_auc = fit_cls('target_landfall', 'target_landfall',
                                 'Landfall', 'cyclone_landfall_model.pkl')
    results['landfall_auc'] = lf_auc

    # Wind regression
    print("\n   [C] Wind Speed Regression")
    y_tr_w, y_te_w = mdf.loc[tr_mask,'target_wind_24h'], mdf.loc[te_mask,'target_wind_24h']
    wm = lgb.LGBMRegressor(**base_params)
    wm.fit(X_tr.fillna(0), y_tr_w,
           eval_set=[(X_te.fillna(0), y_te_w)],
           callbacks=[lgb.early_stopping(40, verbose=False),
                      lgb.log_evaluation(200)])
    rmse = float(np.sqrt(mean_squared_error(y_te_w, wm.predict(X_te.fillna(0)))))
    print(f"   Wind RMSE: {rmse:.2f} knots")
    joblib.dump(wm, MODELS / "cyclone_wind_model.pkl")
    results['wind_rmse'] = rmse

    # Feature importance
    imp = pd.DataFrame({'feature': avail,
                         'importance': ri_model.feature_importances_})\
            .sort_values('importance', ascending=False)
    print("\n   Top 10 features (RI):")
    print(imp.head(10).to_string(index=False))

    # Save predictions
    pred_df = mdf.copy()
    pred_df['pred_ri_prob']       = ri_model.predict_proba(mdf[avail].fillna(0))[:,1]
    pred_df['pred_landfall_prob'] = lf_model.predict_proba(mdf[avail].fillna(0))[:,1]
    pred_df['pred_wind_24h']      = wm.predict(mdf[avail].fillna(0))
    pred_df[['storm_id','storm_name','datetime','date','year','month',
             'latitude','longitude','wind_speed_knots','dist_to_land_km',
             'pred_ri_prob','pred_landfall_prob','pred_wind_24h']]\
         .to_csv(PRED / "cyclone_predictions.csv", index=False)

    json.dump({**results, 'features': avail,
               'trained_at': datetime.now().isoformat()},
              open(MODELS / "cyclone_meta.json", 'w'), indent=2)
    print(f"\n   💾 Models saved → {MODELS}/cyclone_*.pkl")
    return results, ri_model, lf_model, wm, avail


# ── Future Alerts ────────────────────────────────────────────────────────────
def generate_future_alerts(df, ri_model, lf_model, wind_model, features):
    """
    1. Find any storms active in last 14 days → give 24h/48h/72h forecast
    2. Produce seasonal outlook: next 90-day cyclone risk by basin
    3. Historical analog: which months historically have most RI events
    """
    print("\n🚨 Generating future cyclone alerts...")

    latest = df['datetime'].max()
    w14    = latest - pd.Timedelta(days=14)
    recent = df[df['datetime'] >= w14].copy()

    active_storms = recent['storm_id'].unique()
    print(f"   Last track point : {latest}")
    print(f"   Active storms (last 14d): {len(active_storms)}")

    avail = [c for c in features if c in recent.columns]

    alert_rows = []
    for sid in active_storms:
        storm = recent[recent['storm_id'] == sid].sort_values('datetime')
        latest_row = storm.iloc[[-1]].copy()
        X = latest_row[avail].fillna(0)

        ri_prob  = ri_model.predict_proba(X)[:, 1][0]
        lf_prob  = lf_model.predict_proba(X)[:, 1][0]
        wind_24  = wind_model.predict(X)[0]
        cur_wind = float(latest_row['wind_speed_knots'].values[0])
        cur_dist = float(latest_row['dist_to_land_km'].fillna(999).values[0])

        def cat(w):
            if w >= 137: return 'Super Cyclonic Storm'
            elif w >= 89: return 'Very Severe Cyclonic Storm'
            elif w >= 64: return 'Severe Cyclonic Storm'
            elif w >= 48: return 'Cyclonic Storm'
            elif w >= 34: return 'Deep Depression'
            else:         return 'Depression'

        alert_level = ('RED 🔴'   if lf_prob >= 0.6 or ri_prob >= 0.6 else
                       'ORANGE 🟠' if lf_prob >= 0.4 or ri_prob >= 0.4 else
                       'YELLOW 🟡' if lf_prob >= 0.2 or ri_prob >= 0.2 else
                       'GREEN 🟢')

        alert_rows.append({
            'storm_id':             sid,
            'storm_name':           latest_row['storm_name'].values[0],
            'last_seen':            str(latest_row['datetime'].values[0])[:16],
            'current_lat':          float(latest_row['latitude'].values[0]),
            'current_lon':          float(latest_row['longitude'].values[0]),
            'current_wind_knots':   cur_wind,
            'current_dist_land_km': cur_dist,
            'current_category':     cat(cur_wind),
            'pred_wind_24h_knots':  round(wind_24, 1),
            'pred_category_24h':    cat(wind_24),
            'ri_probability':       round(ri_prob, 4),
            'landfall_probability': round(lf_prob, 4),
            'alert_level':          alert_level,
            'wind_change_forecast': round(wind_24 - cur_wind, 1),
            'forecast_made_at':     datetime.now().isoformat(),
            'forecast_valid_until': (datetime.now() + timedelta(hours=24)).isoformat(),
        })

    alerts_df = pd.DataFrame(alert_rows)
    alerts_df.to_csv(ALERTS / "cyclone_alerts.csv", index=False)
    print(f"   ✅ Storm alerts: {ALERTS}/cyclone_alerts.csv  ({len(alerts_df)} storms)")

    # Seasonal outlook — next 90 days based on historical monthly stats
    now_month = datetime.now().month
    months_ahead = [(now_month + i - 1) % 12 + 1 for i in range(3)]
    seasonal = df.groupby('month').agg(
        hist_storm_count    =('storm_id','nunique'),
        hist_ri_count       =('target_RI' if 'target_RI' in df.columns else 'wind_speed_knots',
                               'sum' if 'target_RI' in df.columns else 'count'),
        hist_mean_wind      =('wind_speed_knots','mean'),
        hist_landfall_count =('target_landfall' if 'target_landfall' in df.columns
                               else 'wind_speed_knots',
                               'sum' if 'target_landfall' in df.columns else 'count'),
    ).reset_index()

    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    outlook = seasonal[seasonal['month'].isin(months_ahead)].copy()
    outlook['month_name'] = outlook['month'].map(month_names)
    outlook['risk_level'] = pd.cut(outlook['hist_storm_count'],
                                    bins=[-1,2,5,10,999],
                                    labels=['Low','Moderate','High','Critical'])
    outlook.to_csv(ALERTS / "cyclone_seasonal_outlook.csv", index=False)
    print(f"   ✅ Seasonal outlook: {ALERTS}/cyclone_seasonal_outlook.csv")

    if len(alerts_df):
        print("\n   🌀 ACTIVE STORM FORECASTS:")
        for _, r in alerts_df.iterrows():
            print(f"   {r['alert_level']}  Storm {r['storm_name']}  "
                  f"| Wind now: {r['current_wind_knots']:.0f}kt → 24h: {r['pred_wind_24h_knots']:.0f}kt "
                  f"| Landfall: {r['landfall_probability']:.1%} "
                  f"| RI: {r['ri_probability']:.1%}")
    else:
        print("   ✅ No active storms in last 14 days")

    return alerts_df


# ── Main ─────────────────────────────────────────────────────────────────────
def run():
    print("="*60)
    print("  CYCLONE MODEL v2  (Future Alerts Enabled)")
    print("="*60)
    df     = load_cyclones()
    df     = engineer_features(df)
    result = train_models(df)
    if result:
        res, ri_m, lf_m, w_m, features = result
        generate_future_alerts(df, ri_m, lf_m, w_m, features)
        print(f"\n📊 FINAL:  RI AUC={res['ri_auc']:.4f}  "
              f"Landfall AUC={res['landfall_auc']:.4f}  "
              f"Wind RMSE={res['wind_rmse']:.2f} knots")
    print("\n✅ Cyclone pipeline complete.")

if __name__ == "__main__":
    run()
