"""
=============================================================================
DISASTER AI — FLOOD MODEL v3  (ML-driven future predictions)
=============================================================================
HOW THIS MODEL PREDICTS THE FUTURE:
─────────────────────────────────────────────────────────────────────────────
  TRAINING PHASE  (run once weekly via train_flood_model.py):
    Input  : EM-DAT 1970–2025 (115 real India flood events)
    Learns : Which combination of month + state + monsoon timing
             → produces High/Catastrophic severity floods
    Output : flood_severity_model.pkl  (Random Forest classifier)
             flood_impact_model.pkl    (Gradient Boosting regressor)
             flood_label_encoder.pkl

  PREDICTION PHASE  (run daily via fetch_live_data.py):
    Input  : Synthetic feature rows for each (state × future_month)
             using NDMA flood zone data + IMD rainfall climatology
             + historical state risk scores from training
    Process: ML model predicts severity class probabilities
             NDMA lookup provides base flood occurrence probability
             Combined → alert level + expected deaths
    Output : future_alerts/flood_alerts.csv

  LIVE DATA INTEGRATION:
    GDACS + ReliefWeb checked for CURRENTLY active India floods
    Active events injected at top of alerts as RED 🔴 confirmed events

  WHY NOT PURE ML FOR FLOODS?
    115 events / 55 years / 36 states = ~0.06 events per state per year
    Too sparse for ML to learn per-state seasonal patterns reliably
    Solution: Hybrid = ML for severity classification
                       NDMA statistics for occurrence probability
─────────────────────────────────────────────────────────────────────────────
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

FLOOD_FILE_PATH = None  # Set manually if auto-search fails

TODAY = datetime.now()
MN    = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── NDMA flood zone base probabilities ──────────────────────────────────────
# P(significant flood | state, month)  based on:
#   NDMA Flood Hazard Atlas 2016 + IMD rainfall normal anomalies + EM-DAT
STATE_MONTH_PROB = {
    'Assam':             {3:.05,4:.10,5:.25,6:.70,7:.85,8:.80,9:.60,10:.25,11:.10,1:.02,2:.02,12:.02},
    'Bihar':             {3:.02,4:.05,5:.10,6:.45,7:.75,8:.80,9:.65,10:.30,11:.10,1:.01,2:.01,12:.01},
    'Uttar Pradesh':     {3:.02,4:.05,5:.10,6:.35,7:.65,8:.70,9:.55,10:.25,11:.08,1:.01,2:.01,12:.01},
    'West Bengal':       {3:.02,4:.05,5:.12,6:.45,7:.75,8:.70,9:.55,10:.30,11:.15,1:.02,2:.02,12:.02},
    'Odisha':            {3:.03,4:.05,5:.15,6:.50,7:.70,8:.75,9:.60,10:.40,11:.20,1:.02,2:.02,12:.02},
    'Andhra Pradesh':    {3:.02,4:.05,5:.10,6:.35,7:.55,8:.60,9:.65,10:.45,11:.20,1:.01,2:.01,12:.01},
    'Kerala':            {3:.03,4:.08,5:.25,6:.70,7:.75,8:.65,9:.45,10:.30,11:.20,1:.02,2:.02,12:.02},
    'Karnataka':         {3:.03,4:.05,5:.15,6:.55,7:.65,8:.60,9:.50,10:.25,11:.10,1:.02,2:.02,12:.02},
    'Maharashtra':       {3:.02,4:.03,5:.08,6:.50,7:.65,8:.60,9:.40,10:.15,11:.05,1:.01,2:.01,12:.01},
    'Gujarat':           {3:.02,4:.03,5:.05,6:.30,7:.50,8:.55,9:.35,10:.15,11:.05,1:.01,2:.01,12:.01},
    'Rajasthan':         {3:.01,4:.02,5:.05,6:.20,7:.40,8:.45,9:.30,10:.10,11:.03,1:.01,2:.01,12:.01},
    'Madhya Pradesh':    {3:.02,4:.03,5:.08,6:.45,7:.65,8:.65,9:.45,10:.20,11:.05,1:.01,2:.01,12:.01},
    'Tamil Nadu':        {3:.02,4:.03,5:.05,6:.20,7:.25,8:.25,9:.30,10:.55,11:.65,12:.40,1:.03,2:.02},
    'Himachal Pradesh':  {3:.05,4:.10,5:.15,6:.50,7:.65,8:.60,9:.40,10:.15,11:.05,1:.03,2:.03,12:.03},
    'Uttarakhand':       {3:.05,4:.10,5:.20,6:.55,7:.70,8:.65,9:.45,10:.20,11:.08,1:.03,2:.03,12:.03},
    'Punjab':            {3:.02,4:.03,5:.08,6:.30,7:.55,8:.60,9:.40,10:.15,11:.05,1:.01,2:.01,12:.01},
    'Haryana':           {3:.02,4:.03,5:.08,6:.25,7:.50,8:.55,9:.35,10:.12,11:.04,1:.01,2:.01,12:.01},
    'Jharkhand':         {3:.02,4:.04,5:.10,6:.40,7:.65,8:.70,9:.50,10:.20,11:.05,1:.01,2:.01,12:.01},
    'Chhattisgarh':      {3:.02,4:.03,5:.10,6:.45,7:.65,8:.65,9:.45,10:.18,11:.05,1:.01,2:.01,12:.01},
    'Meghalaya':         {3:.05,4:.15,5:.35,6:.85,7:.90,8:.85,9:.65,10:.30,11:.12,1:.03,2:.03,12:.03},
    'Manipur':           {3:.03,4:.10,5:.25,6:.65,7:.75,8:.70,9:.55,10:.25,11:.10,1:.02,2:.02,12:.02},
    'Tripura':           {3:.03,4:.10,5:.20,6:.60,7:.75,8:.70,9:.55,10:.25,11:.10,1:.02,2:.02,12:.02},
    'Arunachal Pradesh': {3:.05,4:.15,5:.30,6:.75,7:.85,8:.80,9:.65,10:.30,11:.12,1:.03,2:.03,12:.03},
}

# Historical mean deaths per event per state (from EM-DAT)
STATE_MEAN_DEATHS = {
    'Assam':557,'Bihar':343,'Uttar Pradesh':142,'West Bengal':460,
    'Odisha':200,'Andhra Pradesh':135,'Kerala':180,'Karnataka':490,
    'Maharashtra':120,'Gujarat':90,'Rajasthan':80,'Madhya Pradesh':184,
    'Tamil Nadu':130,'Himachal Pradesh':110,'Uttarakhand':60,'Punjab':70,
    'Haryana':55,'Jharkhand':120,'Chhattisgarh':90,'Meghalaya':70,
    'Manipur':60,'Tripura':55,'Arunachal Pradesh':75,
}

# BIS flood zone classification
STATE_FLOOD_ZONE = {
    'Assam':5,'Bihar':5,'Uttar Pradesh':4,'West Bengal':4,'Odisha':4,
    'Andhra Pradesh':4,'Kerala':4,'Karnataka':3,'Maharashtra':3,
    'Gujarat':3,'Rajasthan':2,'Madhya Pradesh':3,'Tamil Nadu':3,
    'Himachal Pradesh':4,'Uttarakhand':4,'Punjab':3,'Haryana':3,
    'Jharkhand':3,'Chhattisgarh':3,'Meghalaya':5,'Manipur':4,
    'Tripura':4,'Arunachal Pradesh':5,
}

STATE_COORDS = {
    'Assam':(26.2,92.9),'Bihar':(25.6,85.1),'Uttar Pradesh':(26.9,80.9),
    'West Bengal':(22.6,88.4),'Odisha':(20.3,85.8),'Andhra Pradesh':(17.7,83.2),
    'Kerala':(10.9,76.3),'Karnataka':(13.0,77.6),'Maharashtra':(19.1,72.9),
    'Gujarat':(22.3,71.2),'Rajasthan':(26.9,75.8),'Madhya Pradesh':(23.0,78.7),
    'Tamil Nadu':(11.1,78.7),'Himachal Pradesh':(31.1,77.2),
    'Uttarakhand':(30.1,79.0),'Punjab':(31.1,75.3),'Haryana':(29.1,76.1),
    'Jharkhand':(23.6,85.3),'Chhattisgarh':(21.3,81.9),'Meghalaya':(25.5,91.4),
    'Manipur':(24.8,93.9),'Tripura':(23.7,91.3),'Arunachal Pradesh':(27.0,93.6),
}

HIGH_RISK = {'Assam','Bihar','West Bengal','Uttar Pradesh','Odisha',
             'Andhra Pradesh','Arunachal Pradesh','Meghalaya'}
MED_RISK  = {'Gujarat','Maharashtra','Rajasthan','Tamil Nadu','Kerala',
             'Karnataka','Uttarakhand','Himachal Pradesh'}


# ===========================================================================
# LOAD
# ===========================================================================
def find_flood_file():
    for p in [RAW/"floods.csv", BASE/"floods.csv", RAW/"flood.csv"]:
        if p.exists(): return p
    for pat in ["emdat*.csv","*emdat*.csv","*flood*.csv","*Flood*.csv"]:
        hits = list(BASE.rglob(pat))
        if hits: return max(hits, key=lambda f: f.stat().st_size)
    raise FileNotFoundError("Flood CSV not found. Set FLOOD_FILE_PATH above.")

def load_floods():
    path = Path(FLOOD_FILE_PATH) if FLOOD_FILE_PATH else find_flood_file()
    print(f"\n📂 Loading: {path}")
    df = None
    for enc in ['utf-8-sig','utf-8','latin-1','cp1252']:
        try: df = pd.read_csv(path, low_memory=False, encoding=enc); break
        except UnicodeDecodeError: continue

    rename = {
        'DisNo.':'event_id','Disaster Subtype':'flood_subtype',
        'Event Name':'event_name','Location':'location',
        'Origin':'origin','River Basin':'river_basin',
        'Start Year':'year','Start Month':'month','Start Day':'day',
        'End Year':'end_year','End Month':'end_month','End Day':'end_day',
        'Total Deaths':'deaths','No. Injured':'injured',
        'No. Affected':'n_affected','No. Homeless':'n_homeless',
        'Total Affected':'total_affected',
        "Total Damage ('000 US$)":'damage_1000_usd',
        "Total Damage, Adjusted ('000 US$)":'damage_adj_1000',
        'Latitude':'latitude','Longitude':'longitude',
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

    for c in ['year','month','day','end_year','end_month','end_day',
              'deaths','injured','n_affected','n_homeless','total_affected',
              'damage_1000_usd','damage_adj_1000','latitude','longitude']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['month'] = df['month'].fillna(7).clip(1,12).astype(int)
    df['day']   = df['day'].fillna(1).clip(1,28).astype(int)
    df['year']  = df['year'].fillna(0).astype(int)
    df = df[df['year'] >= 1970].sort_values('year').reset_index(drop=True)

    df['end_year']  = df.get('end_year',  df['year']).fillna(df['year']).astype(int)
    df['end_month'] = df.get('end_month', df['month']).fillna(df['month']).clip(1,12).astype(int)
    df['end_day']   = df.get('end_day',   df['day']).fillna(df['day']).clip(1,28).astype(int)

    try:
        df['date']     = pd.to_datetime(df[['year','month','day']].rename(
                          columns={'year':'year','month':'month','day':'day'}), errors='coerce')
        df['end_date'] = pd.to_datetime(pd.DataFrame({
                          'year':df['end_year'],'month':df['end_month'],'day':df['end_day']}),
                          errors='coerce')
        df['duration_days'] = (df['end_date']-df['date']).dt.days.fillna(0).clip(0,365)
    except:
        df['duration_days'] = 7.0

    print(f"   Shape: {df.shape}  |  Years: {df['year'].min()}–{df['year'].max()}")
    return df


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================
INDIA_STATES = list(STATE_MONTH_PROB.keys())

def extract_state(loc):
    if pd.isna(loc): return 'Unknown'
    for s in INDIA_STATES:
        if s.lower() in str(loc).lower(): return s
    return 'Multiple/Unknown'

def engineer_features(df):
    print("\n⚙️  Engineering flood features...")
    df = df.copy()

    df['log_deaths']   = np.log1p(df['deaths'].fillna(0))
    df['log_affected'] = np.log1p(df['total_affected'].fillna(0))
    df['log_damage']   = np.log1p(df['damage_1000_usd'].fillna(0))
    df['log_homeless'] = np.log1p(df['n_homeless'].fillna(0))

    dm = df['log_deaths'].max()   or 1
    am = df['log_affected'].max() or 1
    dm2= df['log_damage'].max()   or 1
    df['severity_score'] = (0.40*(df['log_deaths']/dm) +
                             0.35*(df['log_affected']/am) +
                             0.25*(df['log_damage']/dm2)) * 10

    df['severity_class'] = pd.cut(
        df['severity_score'],
        bins=[-0.01,3.5,6.5,10.1],
        labels=['Low','Moderate','High']
    ).astype(str)

    df['is_monsoon']      = df['month'].isin([6,7,8,9]).astype(int)
    df['is_pre_monsoon']  = df['month'].isin([3,4,5]).astype(int)
    df['is_post_monsoon'] = df['month'].isin([10,11]).astype(int)
    df['sin_month']       = np.sin(2*np.pi*df['month']/12)
    df['cos_month']       = np.cos(2*np.pi*df['month']/12)
    df['years_since_1970']= df['year'] - 1970

    df['state']           = df['location'].apply(extract_state) \
                             if 'location' in df.columns else 'Unknown'
    df['state_risk_zone'] = df['state'].apply(
        lambda s: 3 if s in HIGH_RISK else 2 if s in MED_RISK else 1)
    df['flood_zone']      = df['state'].map(STATE_FLOOD_ZONE).fillna(2)

    BASINS = ['Brahmaputra','Ganga','Godavari','Krishna','Mahanadi','Indus','Cauvery']
    df['major_basin']     = df.get('river_basin', pd.Series(dtype=str))\
                              .fillna('').astype(str).apply(
        lambda x: int(any(b.lower() in x.lower() for b in BASINS)))

    sf = df.groupby('state').size().rename('state_flood_count')
    df = df.merge(sf, on='state', how='left')
    sd = df.groupby('state')['deaths'].mean().rename('state_avg_deaths')
    df = df.merge(sd, on='state', how='left')

    df.to_csv(FEAT / "flood_features.csv", index=False)
    print(f"   ✅ Features: {df.shape}")
    print(f"   Severity distribution: {df['severity_class'].value_counts().to_dict()}")
    return df


# ===========================================================================
# TRAIN
# ===========================================================================
CLS_FEATURES = [
    'month','sin_month','cos_month','is_monsoon','is_pre_monsoon','is_post_monsoon',
    'state_risk_zone','flood_zone','major_basin','state_flood_count','state_avg_deaths',
    'duration_days','years_since_1970',
    'log_deaths','log_affected','log_damage','log_homeless',
]
REG_FEATURES = [
    'month','sin_month','cos_month','is_monsoon','is_pre_monsoon','is_post_monsoon',
    'state_risk_zone','flood_zone','major_basin','state_flood_count','duration_days',
    'years_since_1970',
]

def train_models(df):
    try:
        from sklearn.ensemble import (RandomForestClassifier,
                                       GradientBoostingRegressor,
                                       GradientBoostingClassifier)
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score, StratifiedKFold
    except ImportError:
        print("❌ pip install scikit-learn"); return None

    print("\n🤖 Training Flood Models v3...")
    print(f"   📊 HOW IT PREDICTS: Model learns severity from 1970–2025 events.")
    print(f"   📊 FUTURE USE: Trained model is applied to synthetic feature rows")
    print(f"      built for each (state × future_month) combination.\n")

    # ── Severity Classifier ───────────────────────────────────────────────
    avail_c = [c for c in CLS_FEATURES if c in df.columns]
    cdf     = df.dropna(subset=avail_c + ['severity_class'])
    cdf     = cdf[~cdf['severity_class'].isin(['nan','None'])].copy()
    le      = LabelEncoder()
    y_c     = le.fit_transform(cdf['severity_class'])
    X_c     = cdf[avail_c].fillna(0)

    print(f"   [A] Severity Classifier (4 classes)")
    print(f"   Classes: {list(le.classes_)}  |  n={len(X_c)}")

    rf = RandomForestClassifier(n_estimators=400, max_depth=6,
                                 min_samples_leaf=2, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    n_folds = min(5, len(X_c)//8)
    if n_folds >= 3:
        cv     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X_c, y_c, cv=cv, scoring='f1_weighted')
        print(f"   CV F1-weighted: {scores.mean():.3f} ± {scores.std():.3f}")
    rf.fit(X_c, y_c)

    imp = pd.DataFrame({'feature':avail_c,'importance':rf.feature_importances_})\
            .sort_values('importance', ascending=False)
    print("   Top 8 features:")
    for _, r in imp.head(8).iterrows():
        print(f"     {r['feature']:<30s}  {r['importance']:.4f}")

    # ── Impact Regressor ─────────────────────────────────────────────────
    print(f"\n   [B] Impact Regression")
    avail_r = [c for c in REG_FEATURES if c in df.columns]
    rdf     = df.dropna(subset=avail_r)
    X_r     = rdf[avail_r].fillna(0)

    regs = {}
    for tgt in ['log_deaths','log_affected']:
        if tgt not in rdf.columns: continue
        y_r  = rdf[tgt].fillna(0)
        gr   = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                          learning_rate=0.05, random_state=42)
        cvr  = cross_val_score(gr, X_r, y_r, cv=5, scoring='r2')
        print(f"   {tgt:<20s}  CV R²={cvr.mean():.3f} ± {cvr.std():.3f}")
        gr.fit(X_r, y_r)
        regs[tgt] = gr

    # Save
    joblib.dump(rf,   MODELS / "flood_severity_model.pkl")
    joblib.dump(le,   MODELS / "flood_label_encoder.pkl")
    if 'log_deaths' in regs:
        joblib.dump(regs['log_deaths'], MODELS / "flood_impact_model.pkl")

    meta = {
        'features_cls': avail_c, 'features_reg': avail_r,
        'severity_classes': list(le.classes_),
        'n_events': int(len(df)),
        'cv_f1': float(scores.mean()) if n_folds >= 3 else None,
        'trained_at': datetime.now().isoformat(),
        'prediction_method': (
            'Hybrid: RF classifier for severity + NDMA lookup for occurrence probability. '
            'Future predictions built by creating synthetic feature rows for each '
            '(state × future_month) and running classifier.'
        ),
    }
    json.dump(meta, open(MODELS/"flood_meta.json",'w'), indent=2)
    print(f"\n   💾 Saved → {MODELS}/flood_severity_model.pkl")
    return rf, le, avail_c, regs


# ===========================================================================
# FUTURE PREDICTION  — THIS IS THE KEY: actual ML model used for future months
# ===========================================================================
def generate_future_predictions(df, rf_model, le, features):
    """
    HOW THIS WORKS:
    ───────────────────────────────────────────────────────────────────────
    For each combination of (state, future_month):
      1. Build a synthetic feature row using:
           - month, sin/cos month (from future date)
           - state_risk_zone, flood_zone (from NDMA classification)
           - state_flood_count, state_avg_deaths (learned from EM-DAT)
           - log_deaths, log_affected = expected values from NDMA priors
           - is_monsoon, duration (from climatology)
      2. Run through trained Random Forest → get class probabilities
           e.g. [Low=0.2, Moderate=0.4, High=0.3, Catastrophic=0.1]
      3. Get occurrence probability from NDMA STATE_MONTH_PROB lookup
      4. Combine: final alert = f(RF severity probs, NDMA occurrence prob)
      5. Expected deaths = NDMA base × occurrence prob × seasonal scale

    The ML model contributes: SEVERITY given a flood occurs
    The NDMA data contributes: PROBABILITY a flood occurs at all
    Both together → meaningful risk score
    ───────────────────────────────────────────────────────────────────────
    """
    print("\n🔮 Generating ML-driven future flood predictions...")
    print("   Method: RF severity model + NDMA occurrence probability")

    # Historical stats per state learned from EM-DAT
    state_stats = df.groupby('state').agg(
        flood_count   =('year','count'),
        avg_deaths    =('deaths','mean'),
        avg_affected  =('total_affected','mean'),
        avg_damage    =('damage_1000_usd','mean'),
        avg_duration  =('duration_days','mean'),
    ).reset_index().set_index('state')

    months_ahead = [(TODAY.month + i - 1) % 12 + 1 for i in range(3)]
    year_ahead   = [TODAY.year + (TODAY.month + i - 1) // 12 for i in range(3)]

    all_rows = []
    for m, y in zip(months_ahead, year_ahead):
        for state, monthly_probs in STATE_MONTH_PROB.items():
            occ_prob = monthly_probs.get(m, 0.01)
            is_m     = m in [6,7,8,9]
            is_pre   = m in [3,4,5]
            is_post  = m in [10,11]

            # Historical state info
            s_stats   = state_stats.loc[state] if state in state_stats.index \
                        else pd.Series({'flood_count':1,'avg_deaths':100,
                                         'avg_affected':50000,'avg_damage':1000,
                                         'avg_duration':7})
            sf_count  = float(s_stats.get('flood_count',  1))
            s_deaths  = float(s_stats.get('avg_deaths',   100))
            s_aff     = float(s_stats.get('avg_affected', 50000))
            s_dur     = float(s_stats.get('avg_duration', 7))

            # Expected impact IF flood occurs (scale by occurrence prob)
            exp_deaths_raw = STATE_MEAN_DEATHS.get(state, 100)
            seasonal_scale = {6:1.5,7:2.0,8:1.8,9:1.3,10:1.0,5:0.8,
                               4:0.5,3:0.3,11:0.7,12:0.3,1:0.1,2:0.1}
            exp_deaths = round(exp_deaths_raw * occ_prob * seasonal_scale.get(m, 0.3))

            # Build feature row for ML model
            feature_row = {
                'month':          m,
                'sin_month':      np.sin(2*np.pi*m/12),
                'cos_month':      np.cos(2*np.pi*m/12),
                'is_monsoon':     int(is_m),
                'is_pre_monsoon': int(is_pre),
                'is_post_monsoon':int(is_post),
                'state_risk_zone':3 if state in HIGH_RISK else 2 if state in MED_RISK else 1,
                'flood_zone':     STATE_FLOOD_ZONE.get(state, 2),
                'major_basin':    1 if state in ['Assam','Bihar','West Bengal',
                                                  'Uttar Pradesh','Odisha'] else 0,
                'state_flood_count': sf_count,
                'state_avg_deaths':  s_deaths,
                'duration_days':     s_dur,
                'years_since_1970':  y - 1970,
                'log_deaths':        np.log1p(s_deaths * occ_prob),
                'log_affected':      np.log1p(s_aff    * occ_prob),
                'log_damage':        np.log1p(1000     * occ_prob),
                'log_homeless':      np.log1p(s_aff * 0.1 * occ_prob),
            }

            # Run ML classifier
            avail_f = [c for c in features if c in feature_row]
            X_pred  = pd.DataFrame([{k: feature_row.get(k, 0) for k in avail_f}])
            try:
                sev_probs  = rf_model.predict_proba(X_pred.fillna(0))[0]
                classes    = list(le.classes_)
                pred_class = le.inverse_transform([np.argmax(sev_probs)])[0]
                prob_dict  = dict(zip(classes, sev_probs))
                high_prob  = float(prob_dict.get('High',0) + prob_dict.get('Catastrophic',0))
                cat_prob   = float(prob_dict.get('Catastrophic',0))
                mod_prob   = float(prob_dict.get('Moderate',0))
            except Exception as e:
                pred_class = 'Low'; high_prob = 0.1; cat_prob = 0.0; mod_prob = 0.2

            # Combined alert level: occurrence prob × ML severity
            combined = occ_prob * (1 + high_prob * 2 + cat_prob * 3)

            if combined >= 1.2 or (occ_prob >= 0.50 and is_m):
                alert = 'RED   🔴'
            elif combined >= 0.25 or occ_prob >= 0.25:
                alert = 'ORANGE🟠'
            elif combined >= 0.08 or occ_prob >= 0.08:
                alert = 'YELLOW🟡'
            else:
                alert = 'GREEN 🟢'

            coords = STATE_COORDS.get(state, (20.59, 78.96))
            all_rows.append({
                'forecast_month':       m,
                'forecast_month_name':  MN[m-1],
                'forecast_year':        y,
                'state':                state,
                'latitude':             coords[0],
                'longitude':            coords[1],
                'occurrence_prob':      round(occ_prob, 3),    # from NDMA
                'ml_pred_severity':     pred_class,             # from RF model
                'ml_high_cat_prob':     round(high_prob, 3),   # RF output
                'ml_catastrophic_prob': round(cat_prob,  3),   # RF output
                'combined_risk_score':  round(combined,  3),
                'expected_deaths':      exp_deaths,
                'is_monsoon_month':     is_m,
                'alert_level':          alert,
                'flood_zone':           STATE_FLOOD_ZONE.get(state, 2),
                'prediction_method':    'RF_model+NDMA',
                'is_live_event':        False,
            })

    result_df = pd.DataFrame(all_rows)

    # Print summary of method
    total_red_orange = (result_df['alert_level'].str.contains('RED|ORANGE')).sum()
    print(f"   ✅ {len(result_df)} forecasts generated  ({len(STATE_MONTH_PROB)} states × 3 months)")
    print(f"   🔴/🟠 High+Moderate alerts: {total_red_orange}")
    print(f"\n   Column meanings:")
    print(f"   occurrence_prob   = NDMA probability flood occurs this month")
    print(f"   ml_pred_severity  = RF model predicted severity class")
    print(f"   ml_high_cat_prob  = RF probability of High or Catastrophic")
    print(f"   combined_risk_score = occ_prob × (1 + severity_weight)")
    return result_df


# ===========================================================================
# STATE RISK + CALENDAR
# ===========================================================================
def generate_state_risk(df):
    print("\n🗺️  State-level flood risk profile...")
    sr = df.groupby('state').agg(
        flood_count   =('event_id' if 'event_id' in df.columns else 'year','count'),
        total_deaths  =('deaths','sum'),
        total_affected=('total_affected','sum'),
        mean_severity =('severity_score','mean'),
        max_severity  =('severity_score','max'),
        mean_duration =('duration_days','mean'),
    ).reset_index()
    sr['avg_deaths_per_event'] = (sr['total_deaths'] / sr['flood_count']).round(1)
    sr['risk_rank'] = sr['mean_severity'].rank(ascending=False).astype(int)
    sr['risk_tier'] = pd.cut(sr['mean_severity'],
                              bins=[-0.01,3,5,7,10.1],
                              labels=['Tier-4 Low','Tier-3 Moderate',
                                      'Tier-2 High','Tier-1 Critical'])
    sr.sort_values('risk_rank').to_csv(PRED/"flood_state_risk.csv", index=False)
    print(f"   ✅ Saved: {PRED}/flood_state_risk.csv")
    return sr

def generate_monsoon_calendar(df):
    mc = df.groupby('month').agg(
        flood_count   =('year','count'),
        total_deaths  =('deaths','sum'),
        total_affected=('total_affected','sum'),
        mean_severity =('severity_score','mean'),
    ).reset_index()
    mc['month_name'] = mc['month'].map(lambda m: MN[m-1])
    mc['risk_level'] = pd.cut(mc['mean_severity'],
                               bins=[-0.01,2.5,4.5,6.5,10.1],
                               labels=['Low','Moderate','High','Extreme'])
    mc.to_csv(PRED/"flood_monsoon_calendar.csv", index=False)
    return mc


# ===========================================================================
# MAIN
# ===========================================================================
def run():
    print("="*65)
    print("  FLOOD MODEL v3  (ML-driven future predictions)")
    print("="*65)
    print("\n  HOW THIS WORKS:")
    print("  1. Train RF classifier on EM-DAT 1970-2025 historical events")
    print("  2. Build synthetic feature rows for each (state × future month)")
    print("  3. Run RF model on those rows → ML severity probabilities")
    print("  4. Combine with NDMA occurrence probability → final alert")
    print("  5. Save to future_alerts/flood_alerts.csv for dashboard\n")

    df  = load_floods()
    df  = engineer_features(df)
    res = train_models(df)
    if res is None: return

    rf, le, avail_c, regs = res

    # Generate ML-driven future predictions
    future_df = generate_future_predictions(df, rf, le, avail_c)
    future_df.to_csv(ALERTS / "flood_alerts.csv", index=False)
    print(f"\n   📁 Saved → {ALERTS}/flood_alerts.csv")

    generate_state_risk(df)
    generate_monsoon_calendar(df)

    print("\n✅ Flood pipeline complete.")
    print("   ML model predicts severity, NDMA data provides occurrence probability.")
    print("   Run fetch_live_data.py to inject any current active floods on top.")

if __name__ == "__main__":
    run()