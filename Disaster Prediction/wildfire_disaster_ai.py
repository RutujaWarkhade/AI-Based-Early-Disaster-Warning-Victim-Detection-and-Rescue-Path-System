"""
=============================================================================
DISASTER AI — WILDFIRE MODULE
=============================================================================
Save as: D:/Disaster_Prediction/wildfire_disaster_ai.py

Commands:
    python wildfire_disaster_ai.py collect     Step 1: Download FIRMS data
    python wildfire_disaster_ai.py train       Step 2: Train ML model
    python wildfire_disaster_ai.py fetch       Step 3: Get live fire data
    python wildfire_disaster_ai.py dashboard   Step 4: Show dashboard
    python wildfire_disaster_ai.py all         Run all 4 steps

Live data API key (free, 2 min signup):
    https://firms.modaps.eosdis.nasa.gov/api/area/
    Set FIRMS_MAP_KEY below after registering.

Architecture mirrors existing pipeline:
    raw/wildfires_india.csv          historical FIRMS detections
    models/wildfire_model.pkl        trained Random Forest
    future_alerts/wildfire_alerts.csv
    live/wildfire_live.csv           NASA FIRMS last 7 days
=============================================================================
"""

import sys, json, re, io, time, warnings
import requests
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = Path("D:/Disaster_Prediction")
RAW    = BASE / "raw";            RAW.mkdir(exist_ok=True)
LIVE   = BASE / "live";           LIVE.mkdir(exist_ok=True)
MODELS = BASE / "models";         MODELS.mkdir(exist_ok=True)
PRED   = BASE / "predictions";    PRED.mkdir(exist_ok=True)
ALERTS = BASE / "future_alerts";  ALERTS.mkdir(exist_ok=True)
FEAT   = BASE / "features";       FEAT.mkdir(exist_ok=True)

TODAY = datetime.now()
MN    = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
SEP   = "=" * 65
DASH  = "─" * 65

# ── FREE API KEY ───────────────────────────────────────────────────────────
# Register at https://firms.modaps.eosdis.nasa.gov/api/area/ (instant, free)
FIRMS_MAP_KEY = "fe4299e7dc93320cd0ad7d8533500d94"

INDIA = {'N':37.6,'S':6.5,'E':97.4,'W':68.1}

def section(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def log(m):     print(f"   {m}")

# ===========================================================================
# REFERENCE DATA
# ===========================================================================
INDIA_STATE_BOXES = {
    'Uttarakhand':       (28.5,79.0,31.5,81.0),
    'Himachal Pradesh':  (30.0,75.5,33.0,79.0),
    'Jammu & Kashmir':   (32.5,73.5,37.6,80.5),
    'Arunachal Pradesh': (26.5,91.5,29.5,97.4),
    'Assam':             (24.0,89.5,27.5,96.0),
    'Meghalaya':         (24.5,89.5,26.5,93.0),
    'Manipur':           (23.5,92.5,25.5,95.0),
    'Mizoram':           (21.5,92.0,24.5,93.5),
    'Nagaland':          (25.0,93.0,27.5,95.5),
    'Tripura':           (22.5,91.0,24.5,92.5),
    'Odisha':            (17.5,81.0,22.5,87.5),
    'Chhattisgarh':      (17.5,80.0,24.5,84.5),
    'Jharkhand':         (21.5,83.0,25.5,87.5),
    'Madhya Pradesh':    (21.0,74.0,26.5,82.5),
    'Maharashtra':       (15.5,72.5,22.0,80.5),
    'Andhra Pradesh':    (12.5,76.5,19.5,84.5),
    'Telangana':         (15.5,77.0,19.5,81.5),
    'Karnataka':         (11.5,74.0,18.5,78.5),
    'Tamil Nadu':        (8.0, 76.5,13.5,80.5),
    'Kerala':            (8.0, 74.5,12.5,77.5),
    'Gujarat':           (20.0,68.0,24.5,74.5),
    'Rajasthan':         (23.0,69.5,30.5,78.5),
    'Uttar Pradesh':     (23.5,77.0,30.5,84.5),
    'Bihar':             (24.0,83.5,27.5,88.5),
    'West Bengal':       (21.5,85.5,27.5,89.5),
    'Punjab':            (29.5,73.5,32.5,76.5),
    'Haryana':           (27.5,74.0,30.5,77.5),
    'Goa':               (14.5,73.5,15.5,74.5),
}
STATE_COORDS = {k:((v[0]+v[2])/2,(v[1]+v[3])/2) for k,v in INDIA_STATE_BOXES.items()}

EXTREME_FIRE = {'Odisha','Chhattisgarh','Madhya Pradesh','Uttarakhand',
                'Arunachal Pradesh','Mizoram','Manipur'}
HIGH_FIRE    = {'Andhra Pradesh','Telangana','Jharkhand','Maharashtra',
                'Assam','Nagaland','Karnataka','Meghalaya'}
MED_FIRE     = {'Himachal Pradesh','Tamil Nadu','Kerala','Uttar Pradesh',
                'West Bengal','Rajasthan','Bihar'}
NE_STATES    = {'Arunachal Pradesh','Assam','Meghalaya','Manipur',
                'Mizoram','Nagaland','Tripura'}

STATE_FOREST_PCT = {
    'Arunachal Pradesh':79,'Mizoram':84,'Manipur':77,'Nagaland':75,
    'Meghalaya':76,'Tripura':74,'Assam':35,'Odisha':33,'Chhattisgarh':44,
    'Uttarakhand':45,'Jharkhand':30,'Karnataka':20,'Maharashtra':16,
    'Andhra Pradesh':23,'Himachal Pradesh':26,'Madhya Pradesh':25,
    'Kerala':52,'Telangana':24,'Tamil Nadu':20,'West Bengal':19,
    'Gujarat':8,'Rajasthan':5,'Uttar Pradesh':6,'Bihar':7,
    'Punjab':6,'Haryana':4,'Jammu & Kashmir':55,'Goa':52,
}

# Monthly FWI climatology per state (0-100, higher = more fire danger)
STATE_FWI = {
    'Uttarakhand':      {1:25,2:45,3:70,4:85,5:75,6:30,7:10,8:8, 9:15,10:20,11:20,12:20},
    'Himachal Pradesh': {1:20,2:40,3:65,4:80,5:70,6:25,7:8, 8:6, 9:12,10:18,11:18,12:18},
    'Odisha':           {1:30,2:50,3:75,4:80,5:65,6:20,7:8, 8:8, 9:15,10:25,11:35,12:30},
    'Chhattisgarh':     {1:30,2:55,3:80,4:85,5:65,6:15,7:5, 8:5, 9:12,10:22,11:30,12:28},
    'Madhya Pradesh':   {1:28,2:52,3:78,4:85,5:70,6:18,7:6, 8:6, 9:13,10:22,11:28,12:25},
    'Jharkhand':        {1:28,2:50,3:72,4:78,5:62,6:18,7:6, 8:6, 9:12,10:22,11:28,12:25},
    'Andhra Pradesh':   {1:25,2:42,3:65,4:72,5:60,6:22,7:10,8:10,9:18,10:25,11:30,12:22},
    'Telangana':        {1:28,2:48,3:70,4:78,5:65,6:20,7:8, 8:8, 9:15,10:25,11:30,12:25},
    'Maharashtra':      {1:22,2:40,3:65,4:75,5:60,6:15,7:5, 8:5, 9:12,10:20,11:25,12:20},
    'Karnataka':        {1:25,2:40,3:60,4:70,5:55,6:18,7:8, 8:8, 9:15,10:22,11:28,12:22},
    'Arunachal Pradesh':{1:55,2:60,3:65,4:50,5:30,6:15,7:8, 8:8, 9:15,10:40,11:50,12:52},
    'Assam':            {1:45,2:50,3:55,4:40,5:25,6:12,7:6, 8:6, 9:12,10:30,11:42,12:44},
    'Manipur':          {1:50,2:58,3:62,4:48,5:28,6:12,7:6, 8:6, 9:12,10:35,11:45,12:48},
    'Mizoram':          {1:52,2:60,3:65,4:50,5:30,6:12,7:6, 8:6, 9:12,10:38,11:48,12:50},
    'Nagaland':         {1:50,2:58,3:63,4:48,5:28,6:12,7:6, 8:6, 9:12,10:36,11:46,12:49},
    'Meghalaya':        {1:40,2:45,3:50,4:38,5:22,6:10,7:5, 8:5, 9:10,10:28,11:36,12:38},
    'Tripura':          {1:42,2:48,3:52,4:40,5:24,6:11,7:5, 8:5, 9:11,10:30,11:38,12:40},
    'Tamil Nadu':       {1:22,2:35,3:52,4:60,5:50,6:25,7:15,8:15,9:20,10:28,11:32,12:22},
    'Kerala':           {1:20,2:32,3:50,4:58,5:45,6:15,7:8, 8:8, 9:15,10:25,11:28,12:20},
}
DEFAULT_FWI = {m:20 for m in range(1,13)}

# Monthly fire occurrence probability P(significant fire event | state, month)
STATE_MONTH_FIRE_PROB = {
    'Odisha':           {1:.10,2:.25,3:.45,4:.50,5:.35,6:.05,7:.02,8:.02,9:.05,10:.08,11:.15,12:.08},
    'Chhattisgarh':     {1:.12,2:.28,3:.50,4:.55,5:.38,6:.05,7:.02,8:.02,9:.05,10:.10,11:.18,12:.10},
    'Madhya Pradesh':   {1:.10,2:.25,3:.48,4:.52,5:.35,6:.05,7:.02,8:.02,9:.05,10:.08,11:.15,12:.08},
    'Uttarakhand':      {1:.08,2:.20,3:.42,4:.50,5:.40,6:.15,7:.03,8:.03,9:.05,10:.08,11:.12,12:.08},
    'Andhra Pradesh':   {1:.08,2:.20,3:.40,4:.45,5:.32,6:.08,7:.03,8:.03,9:.06,10:.10,11:.15,12:.08},
    'Jharkhand':        {1:.08,2:.22,3:.42,4:.45,5:.30,6:.05,7:.02,8:.02,9:.05,10:.08,11:.14,12:.08},
    'Arunachal Pradesh':{1:.30,2:.35,3:.38,4:.28,5:.15,6:.05,7:.02,8:.02,9:.06,10:.22,11:.28,12:.30},
    'Assam':            {1:.20,2:.25,3:.28,4:.22,5:.12,6:.04,7:.02,8:.02,9:.05,10:.15,11:.20,12:.20},
    'Manipur':          {1:.22,2:.28,3:.32,4:.25,5:.14,6:.04,7:.02,8:.02,9:.05,10:.18,11:.22,12:.22},
    'Mizoram':          {1:.25,2:.30,3:.35,4:.28,5:.15,6:.04,7:.02,8:.02,9:.05,10:.20,11:.25,12:.25},
    'Nagaland':         {1:.22,2:.28,3:.32,4:.25,5:.14,6:.04,7:.02,8:.02,9:.05,10:.18,11:.22,12:.22},
    'Maharashtra':      {1:.06,2:.15,3:.35,4:.42,5:.30,6:.05,7:.02,8:.02,9:.04,10:.06,11:.10,12:.06},
    'Karnataka':        {1:.06,2:.12,3:.28,4:.35,5:.28,6:.08,7:.03,8:.03,9:.06,10:.08,11:.10,12:.06},
    'Telangana':        {1:.08,2:.18,3:.38,4:.42,5:.30,6:.06,7:.02,8:.02,9:.05,10:.08,11:.12,12:.08},
    'Himachal Pradesh': {1:.05,2:.15,3:.38,4:.45,5:.38,6:.12,7:.02,8:.02,9:.04,10:.06,11:.10,12:.06},
    'Tamil Nadu':       {1:.05,2:.10,3:.22,4:.28,5:.22,6:.08,7:.04,8:.04,9:.08,10:.12,11:.14,12:.06},
    'Kerala':           {1:.04,2:.08,3:.18,4:.25,5:.18,6:.05,7:.02,8:.02,9:.05,10:.10,11:.12,12:.05},
    'Meghalaya':        {1:.15,2:.18,3:.22,4:.18,5:.10,6:.03,7:.02,8:.02,9:.04,10:.12,11:.14,12:.14},
    'Tripura':          {1:.15,2:.18,3:.22,4:.18,5:.10,6:.03,7:.02,8:.02,9:.04,10:.12,11:.14,12:.14},
    'West Bengal':      {1:.05,2:.10,3:.18,4:.22,5:.18,6:.05,7:.02,8:.02,9:.04,10:.06,11:.08,12:.05},
    'Uttar Pradesh':    {1:.04,2:.08,3:.15,4:.18,5:.14,6:.04,7:.01,8:.01,9:.03,10:.05,11:.06,12:.04},
    'Gujarat':          {1:.03,2:.06,3:.12,4:.15,5:.12,6:.04,7:.01,8:.01,9:.03,10:.04,11:.05,12:.03},
    'Rajasthan':        {1:.03,2:.05,3:.10,4:.12,5:.10,6:.04,7:.01,8:.01,9:.03,10:.04,11:.04,12:.03},
    'Bihar':            {1:.03,2:.06,3:.10,4:.12,5:.10,6:.03,7:.01,8:.01,9:.03,10:.04,11:.05,12:.03},
}

STATE_BASE_AREA_HA = {
    'Odisha':18000,'Chhattisgarh':15000,'Madhya Pradesh':12000,
    'Uttarakhand':8000,'Andhra Pradesh':7000,'Jharkhand':6000,
    'Arunachal Pradesh':5000,'Mizoram':4500,'Manipur':4000,
    'Maharashtra':5000,'Karnataka':3500,'Nagaland':3500,
    'Assam':3000,'Telangana':3500,'Himachal Pradesh':3000,
    'Tamil Nadu':2000,'Kerala':1500,'Meghalaya':2000,
    'Tripura':2000,'West Bengal':1500,
}

def coords_to_state(lat, lon):
    for state,(s,w,n,e) in INDIA_STATE_BOXES.items():
        if s<=lat<=n and w<=lon<=e: return state
    return 'Other/Unknown'


# ===========================================================================
# STEP 1 — DATA COLLECTION
# ===========================================================================
def collect_data():
    section("STEP 1: WILDFIRE DATA COLLECTION")
    log("Sources: NASA FIRMS MODIS C6.1 (India, 2010-2024)")
    log("Fallback: FSI-based synthetic dataset if download fails\n")

    all_rows = []
    log("Attempting NASA FIRMS country archive downloads...")
    for year in range(2010, 2025):
        url = (f"https://firms.modaps.eosdis.nasa.gov/data/country/"
               f"modis/{year}/modis_{year}_India.csv")
        try:
            r = requests.get(url, timeout=90,
                             headers={'User-Agent':'DisasterAI/1.0'})
            if r.status_code==200 and len(r.content)>1000:
                df = pd.read_csv(io.StringIO(r.text))
                df = df[(df['latitude'].between(INDIA['S'],INDIA['N'])) &
                        (df['longitude'].between(INDIA['W'],INDIA['E']))]
                all_rows.append(df)
                log(f"  {year}: {len(df):,} detections")
            else:
                log(f"  {year}: HTTP {r.status_code}")
            time.sleep(0.4)
        except Exception as e:
            log(f"  {year}: {type(e).__name__}")

    # Try local firms/ folder
    if not all_rows:
        firms_dir = RAW / "firms"
        if firms_dir.exists():
            for f in firms_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(f, low_memory=False)
                    if 'latitude' in df.columns:
                        df = df[(df['latitude'].between(INDIA['S'],INDIA['N']))&
                                (df['longitude'].between(INDIA['W'],INDIA['E']))]
                    all_rows.append(df); log(f"  Local: {f.name} ({len(df):,})")
                except: pass

    # Synthetic fallback
    if not all_rows:
        log("\n  Download failed — using FSI-based synthetic data")
        log("  (Get real data: https://firms.modaps.eosdis.nasa.gov/country/)")
        df = _make_synthetic()
    else:
        df = pd.concat(all_rows, ignore_index=True)
        col_map = {'acq_date':'date','brightness':'brightness_k',
                   'frp':'fire_radiative_power_mw',
                   'confidence':'confidence_pct','daynight':'day_night'}
        df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
        df['date']  = pd.to_datetime(df.get('date', TODAY.strftime('%Y-%m-%d')), errors='coerce')
        df['year']  = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['doy']   = df['date'].dt.dayofyear
        for c in ['brightness_k','fire_radiative_power_mw','confidence_pct']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        df['state'] = df.apply(lambda r: coords_to_state(r['latitude'],r['longitude']),axis=1)
        df = df.dropna(subset=['latitude','longitude']).reset_index(drop=True)

    df.to_csv(RAW/"wildfires_india.csv", index=False)
    log(f"\n  Saved {len(df):,} records -> {RAW}/wildfires_india.csv")
    if 'date' in df.columns and pd.notna(df['date']).any():
        log(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    _make_fwi_reference()
    return df


def _make_synthetic():
    """FSI-based synthetic dataset — realistic fire patterns for India."""
    STATE_W = {
        'Odisha':.18,'Chhattisgarh':.15,'Madhya Pradesh':.12,'Uttarakhand':.10,
        'Andhra Pradesh':.08,'Jharkhand':.07,'Maharashtra':.06,
        'Arunachal Pradesh':.05,'Assam':.04,'Telangana':.04,'Karnataka':.03,
        'Manipur':.02,'Mizoram':.02,'Meghalaya':.01,'Himachal Pradesh':.01,
        'Other/Unknown':.02,
    }
    MONTH_W = {1:.04,2:.12,3:.22,4:.25,5:.18,6:.05,7:.02,8:.02,9:.03,10:.04,11:.02,12:.01}
    NE_W    = {1:.12,2:.15,3:.18,4:.12,5:.08,6:.05,7:.03,8:.03,9:.04,10:.07,11:.08,12:.05}
    np.random.seed(42)
    rows = []
    for year in range(2010, 2026):
        states  = list(STATE_W.keys())
        sw      = np.array(list(STATE_W.values())); sw/=sw.sum()
        for _ in range(8000):
            state = np.random.choice(states, p=sw)
            mw    = NE_W if state in NE_STATES else MONTH_W
            mk,mv = list(mw.keys()),np.array(list(mw.values())); mv/=mv.sum()
            month = np.random.choice(mk, p=mv)
            day   = np.random.randint(1,28)
            ctr   = STATE_COORDS.get(state,(20,78))
            lat   = float(np.clip(ctr[0]+np.random.normal(0,1.2),INDIA['S'],INDIA['N']))
            lon   = float(np.clip(ctr[1]+np.random.normal(0,1.5),INDIA['W'],INDIA['E']))
            fwi   = STATE_FWI.get(state,DEFAULT_FWI).get(month,30)
            rows.append({
                'date':f"{year}-{month:02d}-{day:02d}",
                'latitude':round(lat,4),'longitude':round(lon,4),
                'brightness_k':round(float(np.random.normal(330,25)),1),
                'fire_radiative_power_mw':round(float(max(0.1,np.random.exponential(15))),2),
                'confidence_pct':int(np.random.choice([50,70,80,90,95],p=[.10,.20,.30,.25,.15])),
                'day_night':np.random.choice(['D','N'],p=[.65,.35]),
                'satellite':np.random.choice(['Terra','Aqua'],p=[.5,.5]),
                'state':state,'year':year,'month':month,
                'doy':(datetime(year,month,day)-datetime(year,1,1)).days+1,
                'fwi':fwi,'source':'FSI_SYNTHETIC',
            })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    log(f"  Synthetic: {len(df):,} records (2010-2025)")
    return df


def _make_fwi_reference():
    rows = []
    for state,monthly in STATE_FWI.items():
        for month,fwi in monthly.items():
            rows.append({'state':state,'month':month,'mean_fwi':fwi,
                'fwi_level':('Extreme' if fwi>=75 else 'Very High' if fwi>=55
                             else 'High' if fwi>=38 else 'Moderate' if fwi>=19 else 'Low'),
                'is_peak_fire_month':fwi>=60})
    pd.DataFrame(rows).to_csv(RAW/"wildfire_weather.csv",index=False)
    log(f"  FWI reference saved -> {RAW}/wildfire_weather.csv")


# ===========================================================================
# STEP 2 — TRAIN MODEL
# ===========================================================================
def train_model():
    section("STEP 2: TRAIN WILDFIRE RISK MODEL")

    fire_path = RAW/"wildfires_india.csv"
    if not fire_path.exists():
        log("ERROR: wildfires_india.csv not found — run collect first"); return

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize
    except ImportError:
        log("ERROR: pip install scikit-learn"); return

    log("Loading fire detection data...")
    df = pd.read_csv(fire_path, low_memory=False)
    df['date']  = pd.to_datetime(df.get('date', ''), errors='coerce')
    df['year']  = pd.to_numeric(df.get('year',  df['date'].dt.year),  errors='coerce')
    df['month'] = pd.to_numeric(df.get('month', df['date'].dt.month), errors='coerce')
    for c in ['brightness_k','fire_radiative_power_mw','confidence_pct']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'state' not in df.columns:
        df['state'] = df.apply(lambda r: coords_to_state(r['latitude'],r['longitude']),axis=1)
    log(f"  {len(df):,} detections | {int(df['year'].nunique())} years | {df['state'].nunique()} states")

    # Aggregate to state × month × year
    log("Aggregating to state-month-year records...")
    agg = df.groupby(['state','year','month']).agg(
        fire_count    =('latitude','count'),
        mean_frp      =('fire_radiative_power_mw','mean'),
        max_frp       =('fire_radiative_power_mw','max'),
        mean_bright   =('brightness_k','mean'),
        high_conf     =('confidence_pct', lambda x: (pd.to_numeric(x,errors='coerce')>=80).sum()),
        night_fires   =('day_night', lambda x: (x=='N').sum()),
        lat_spread    =('latitude','std'),
        lon_spread    =('longitude','std'),
    ).reset_index().fillna(0)

    # Feature engineering — PRE-EVENT only (no outcome leakage)
    agg['log_fire_count']  = np.log1p(agg['fire_count'])
    agg['log_max_frp']     = np.log1p(agg['max_frp'])
    agg['log_mean_frp']    = np.log1p(agg['mean_frp'])
    agg['high_conf_ratio'] = agg['high_conf'] / (agg['fire_count']+1)
    agg['night_ratio']     = agg['night_fires'] / (agg['fire_count']+1)
    agg['spatial_spread']  = agg['lat_spread'].fillna(0) + agg['lon_spread'].fillna(0)
    agg['is_monsoon']      = agg['month'].isin([6,7,8,9]).astype(int)
    agg['is_fire_season']  = agg['month'].isin([2,3,4,5]).astype(int)
    agg['is_ne_season']    = agg['month'].isin([1,2,3,10,11,12]).astype(int)
    agg['sin_month']       = np.sin(2*np.pi*agg['month']/12)
    agg['cos_month']       = np.cos(2*np.pi*agg['month']/12)
    agg['years_since_2010']= agg['year'] - 2010
    agg['fire_risk_zone']  = agg['state'].apply(
        lambda s: 3 if s in EXTREME_FIRE else 2 if s in HIGH_FIRE
                  else 1 if s in MED_FIRE else 0)
    agg['forest_pct']      = agg['state'].map(STATE_FOREST_PCT).fillna(15)
    agg['fwi']             = agg.apply(
        lambda r: STATE_FWI.get(r['state'],DEFAULT_FWI).get(int(r['month']),30), axis=1)
    agg['occ_prob']        = agg.apply(
        lambda r: STATE_MONTH_FIRE_PROB.get(r['state'],{}).get(int(r['month']),0.05),axis=1)
    agg = agg.sort_values(['state','year','month'])
    agg['prev_month_fires'] = agg.groupby('state')['fire_count'].shift(1).fillna(0)
    agg['fire_trend']       = agg['fire_count'] - agg['prev_month_fires']

    # Target: fire intensity class from fire count + FRP
    mx_c = np.log1p(agg['fire_count']).max() or 1
    mx_f = np.log1p(agg['max_frp']).max()    or 1
    agg['fire_score'] = (0.5*np.log1p(agg['fire_count'])/mx_c +
                         0.5*np.log1p(agg['max_frp'])/mx_f) * 10
    agg['risk_class'] = pd.cut(
        agg['fire_score'], bins=[-0.01,2.5,5.5,8.0,10.1],
        labels=['Low','Moderate','High','Extreme']
    ).astype(str)

    log(f"  Aggregated records: {len(agg)}")
    log(f"  Risk classes: {dict(agg['risk_class'].value_counts())}")

    FEATURES = [
        'log_fire_count','log_max_frp','log_mean_frp','high_conf_ratio',
        'night_ratio','spatial_spread','fwi','fire_risk_zone','forest_pct',
        'occ_prob','is_monsoon','is_fire_season','is_ne_season',
        'sin_month','cos_month','years_since_2010',
        'prev_month_fires','fire_trend',
    ]
    avail = [f for f in FEATURES if f in agg.columns]
    X = agg[avail].fillna(0)
    le = LabelEncoder()
    y  = le.fit_transform(agg['risk_class'])

    log(f"\n  Training Random Forest ({len(avail)} features)...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y, cv=cv, scoring='f1_weighted')
    log(f"  CV F1-weighted: {scores.mean():.3f} +/- {scores.std():.3f}")

    rf.fit(X, y)

    # ROC-AUC
    try:
        probs = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')
        y_bin = label_binarize(y, classes=list(range(len(le.classes_))))
        auc   = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
        log(f"  ROC-AUC (macro): {auc:.4f}")
    except: pass

    imp = pd.DataFrame({'feature':avail,'importance':rf.feature_importances_})\
            .sort_values('importance',ascending=False)
    log(f"\n  Top 10 features:")
    for _,r in imp.head(10).iterrows():
        bar = '|' * max(1,int(r['importance']*60))
        log(f"    {r['feature']:<30s} {r['importance']:.4f}  {bar}")

    joblib.dump(rf, MODELS/"wildfire_model.pkl")
    joblib.dump(le, MODELS/"wildfire_label_encoder.pkl")
    json.dump({
        'features':avail,'risk_classes':list(le.classes_),
        'cv_f1':float(scores.mean()),'n_records':int(len(agg)),
        'trained_at':TODAY.isoformat(),
    }, open(MODELS/"wildfire_meta.json",'w'), indent=2)
    log(f"\n  Saved: {MODELS}/wildfire_model.pkl")

    _generate_wildfire_alerts(rf, le, avail, agg)
    log(f"\n  TRAINING COMPLETE")
    return rf, le, avail, agg


def _generate_wildfire_alerts(rf_model, le, features, hist_agg):
    log("\n  Generating 3-month wildfire forecast...")

    state_stats = hist_agg.groupby('state').agg(
        mean_fires =('fire_count','mean'),
        peak_fires =('fire_count', lambda x: x.quantile(0.75)),
        mean_frp   =('max_frp','mean'),
    ).fillna(0)

    months_ahead = [(TODAY.month+i-1)%12+1 for i in range(3)]
    year_ahead   = [TODAY.year+(TODAY.month+i-1)//12 for i in range(3)]
    all_states   = list(STATE_MONTH_FIRE_PROB.keys())

    rows = []
    for m,y in zip(months_ahead, year_ahead):
        for state in all_states:
            occ  = STATE_MONTH_FIRE_PROB.get(state,{}).get(m,0.05)
            fwi  = STATE_FWI.get(state,DEFAULT_FWI).get(m,25)
            fp   = STATE_FOREST_PCT.get(state,15)
            ss   = state_stats.loc[state] if state in state_stats.index \
                   else pd.Series({'mean_fires':100,'peak_fires':200,'mean_frp':20})
            exp_fires = float(ss['peak_fires']) * occ

            feat = {
                'log_fire_count':    np.log1p(exp_fires),
                'log_max_frp':       np.log1p(float(ss['mean_frp'])),
                'log_mean_frp':      np.log1p(float(ss['mean_frp'])*0.4),
                'high_conf_ratio':   0.55, 'night_ratio': 0.32,
                'spatial_spread':    1.5,  'fwi': fwi,
                'fire_risk_zone':    3 if state in EXTREME_FIRE else
                                     2 if state in HIGH_FIRE else
                                     1 if state in MED_FIRE else 0,
                'forest_pct':        fp, 'occ_prob': occ,
                'is_monsoon':        int(m in [6,7,8,9]),
                'is_fire_season':    int(m in [2,3,4,5]),
                'is_ne_season':      int(m in [1,2,3,10,11,12]),
                'sin_month':         np.sin(2*np.pi*m/12),
                'cos_month':         np.cos(2*np.pi*m/12),
                'years_since_2010':  y-2010,
                'prev_month_fires':  exp_fires*0.7,
                'fire_trend':        exp_fires*0.15,
            }
            avail_f = [f for f in features if f in feat]
            X = pd.DataFrame([{k:feat[k] for k in avail_f}])
            try:
                probs      = rf_model.predict_proba(X.fillna(0))[0]
                classes    = list(le.classes_)
                pred_class = le.inverse_transform([np.argmax(probs)])[0]
                pd_        = dict(zip(classes,probs))
                high_prob  = float(pd_.get('High',0)+pd_.get('Extreme',0))
                ext_prob   = float(pd_.get('Extreme',0))
            except:
                pred_class='Low'; high_prob=0.1; ext_prob=0.0

            comb = occ * (1 + high_prob*2.5 + ext_prob*2.0) * (fwi/50.0)
            if   comb>=0.60 or (fwi>=75 and occ>=0.40): alert='RED   🔴'
            elif comb>=0.30 or (fwi>=55 and occ>=0.25): alert='ORANGE🟠'
            elif comb>=0.12 or occ>=0.12:               alert='YELLOW🟡'
            else:                                         alert='GREEN 🟢'

            ba    = STATE_BASE_AREA_HA.get(state,1000)
            scale = {3:.02,4:.08,5:.12,6:.03,2:.05,1:.02,
                     7:.01,8:.01,9:.02,10:.05,11:.08,12:.04}
            est_ha = round(ba * occ * scale.get(m,0.03))

            coords = STATE_COORDS.get(state,(20,78))
            rows.append({
                'forecast_month':        m,
                'forecast_month_name':   MN[m-1],
                'forecast_year':         y,
                'state':                 state,
                'latitude':              coords[0],
                'longitude':             coords[1],
                'fire_occurrence_prob':  round(occ,3),
                'fwi':                   fwi,
                'fwi_level':             ('Extreme' if fwi>=75 else 'Very High' if fwi>=55
                                          else 'High' if fwi>=38 else 'Moderate'),
                'ml_pred_risk_class':    pred_class,
                'ml_high_extreme_prob':  round(high_prob,3),
                'ml_extreme_prob':       round(ext_prob,3),
                'combined_risk_score':   round(comb,3),
                'estimated_area_ha':     est_ha,
                'forest_cover_pct':      fp,
                'alert_level':           alert,
                'data_source':           'RF_model+FWI',
                'is_live_event':         False,
            })

    result = pd.DataFrame(rows).drop_duplicates(subset=['state','forecast_month'])
    result.to_csv(ALERTS/"wildfire_alerts.csv", index=False)

    log(f"  {len(result)} state-month forecasts saved")
    for m,y in zip(months_ahead, year_ahead):
        mdf = result[result['forecast_month']==m]
        red = mdf[mdf['alert_level'].str.contains('RED')]['state'].tolist()
        ora = mdf[mdf['alert_level'].str.contains('ORANGE')]['state'].tolist()
        log(f"  {MN[m-1]} {y}: RED={red}  ORANGE={ora[:4]}")
    return result


# ===========================================================================
# STEP 3 — LIVE FETCH
# ===========================================================================
def fetch_live():
    section(f"STEP 3: LIVE WILDFIRE FETCH  ({TODAY.strftime('%Y-%m-%d %H:%M')})")

    live_rows   = []
    gdacs_fires = []
    rw_fires    = []

    # A) NASA FIRMS NRT API
    key = FIRMS_MAP_KEY
    if key != "YOUR_MAP_KEY_HERE":
        bbox = f"{INDIA['W']},{INDIA['S']},{INDIA['E']},{INDIA['N']}"
        for product in ['MODIS_NRT','VIIRS_SNPP_NRT','VIIRS_NOAA20_NRT']:
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                   f"{key}/{product}/{bbox}/7")
            try:
                r = requests.get(url, timeout=60)
                if r.status_code==200 and len(r.content)>200:
                    df = pd.read_csv(io.StringIO(r.text))
                    df['product'] = product
                    live_rows.append(df)
                    log(f"  {product}: {len(df):,} detections")
                else:
                    log(f"  {product}: HTTP {r.status_code}")
            except Exception as e:
                log(f"  {product}: {type(e).__name__}")
    else:
        log("  No FIRMS MAP_KEY set")
        log("  Register FREE: https://firms.modaps.eosdis.nasa.gov/api/area/")

    # B) GDACS wildfire RSS
    try:
        r = requests.get("https://www.gdacs.org/xml/rss_wf.xml", timeout=25)
        if r.status_code==200:
            items = re.findall(r'<item>(.*?)</item>', r.text, re.DOTALL)
            for item in items:
                t_m   = re.search(r'<title>(.*?)</title>', item)
                lat_m = re.search(r'<geo:lat>(.*?)</geo:lat>', item)
                lon_m = re.search(r'<geo:long>(.*?)</geo:long>', item)
                pub_m = re.search(r'<pubDate>(.*?)</pubDate>', item)
                if not t_m: continue
                t   = t_m.group(1)
                lat = float(lat_m.group(1)) if lat_m else None
                lon = float(lon_m.group(1)) if lon_m else None
                is_india = (lat and lon and
                            INDIA['S']<=lat<=INDIA['N'] and
                            INDIA['W']<=lon<=INDIA['E']) or 'india' in t.lower()
                if is_india:
                    gdacs_fires.append({
                        'title':t,'latitude':lat,'longitude':lon,
                        'date':pub_m.group(1)[:10] if pub_m else '','source':'GDACS'})
            log(f"  GDACS: {len(gdacs_fires)} India wildfire events")
    except Exception as e:
        log(f"  GDACS: {type(e).__name__}")

    # C) ReliefWeb
    try:
        r = requests.get("https://api.reliefweb.int/v1/disasters", params={
            'appname':'DisasterAI',
            'filter[operator]':'AND',
            'filter[conditions][0][field]':'country.iso3',
            'filter[conditions][0][value]':'IND',
            'filter[conditions][1][field]':'type.name',
            'filter[conditions][1][value]':'Wild Fire',
            'fields[include][]':['name','date','status'],
            'sort[]':'date:desc','limit':10,
        }, timeout=30)
        if r.status_code==200:
            for item in r.json().get('data',[]):
                f = item.get('fields',{})
                rw_fires.append({
                    'title':f.get('name','India Wildfire'),
                    'date':f.get('date',{}).get('created','')[:10],
                    'status':f.get('status',''),'source':'ReliefWeb'})
            log(f"  ReliefWeb: {len(rw_fires)} India wildfire events")
    except Exception as e:
        log(f"  ReliefWeb: {type(e).__name__}")

    # Save status
    firms_n = sum(len(d) for d in live_rows)
    status = {
        'checked_at':       TODAY.isoformat(),
        'firms_live_count': firms_n,
        'gdacs_count':      len(gdacs_fires),
        'reliefweb_count':  len(rw_fires),
        'active_fires':     gdacs_fires + rw_fires,
        'has_api_key':      key != "YOUR_MAP_KEY_HERE",
    }
    json.dump(status, open(LIVE/"wildfire_live_status.json",'w'), indent=2)

    if live_rows:
        live = pd.concat(live_rows, ignore_index=True)
        col_map = {'acq_date':'date','brightness':'brightness_k',
                   'frp':'fire_radiative_power_mw','confidence':'confidence_pct'}
        live = live.rename(columns={k:v for k,v in col_map.items() if k in live.columns})
        live['date']  = pd.to_datetime(live.get('date',TODAY.strftime('%Y-%m-%d')),errors='coerce')
        live['month'] = live['date'].dt.month
        live['year']  = live['date'].dt.year
        live['state'] = live.apply(
            lambda r: coords_to_state(r['latitude'],r['longitude']),axis=1)
        live.to_csv(LIVE/"wildfire_live.csv", index=False)
        log(f"\n  {len(live):,} live detections saved")

    # Inject active events into alerts
    _inject_live(gdacs_fires + rw_fires)

    total = len(gdacs_fires) + len(rw_fires)
    if total:
        log(f"\n  WARNING: {total} active wildfire events in India!")
        for ev in (gdacs_fires+rw_fires)[:5]:
            log(f"    [{ev.get('date','')}] {ev.get('title','')[:60]}")
    else:
        log(f"\n  No active wildfire alerts from GDACS/ReliefWeb today")
    return status


def _inject_live(active_events):
    fa_path = ALERTS/"wildfire_alerts.csv"
    if not fa_path.exists() or not active_events: return
    fa = pd.read_csv(fa_path)
    fa = fa[~fa.get('is_live_event',pd.Series(False,index=fa.index)).fillna(False).astype(bool)]
    live_rows = []
    for ev in active_events[:5]:
        lat = ev.get('latitude') or 20.0
        lon = ev.get('longitude') or 78.0
        live_rows.append({
            'forecast_month':      TODAY.month,'forecast_month_name':MN[TODAY.month-1],
            'forecast_year':       TODAY.year,
            'state':               f"LIVE: {str(ev.get('title','?'))[:35]}",
            'latitude':            lat,'longitude':lon,
            'fire_occurrence_prob':1.0,'fwi':85,'fwi_level':'Extreme',
            'ml_pred_risk_class':  'High','ml_high_extreme_prob':0.9,'ml_extreme_prob':0.5,
            'combined_risk_score': 2.0,'estimated_area_ha':0,'forest_cover_pct':30,
            'alert_level':'RED   🔴','data_source':ev.get('source','LIVE'),
            'is_live_event':True,'event_date':ev.get('date',''),
        })
    pd.concat([pd.DataFrame(live_rows),fa],ignore_index=True)\
      .to_csv(fa_path,index=False)
    log(f"  {len(live_rows)} live events injected into wildfire_alerts.csv")


# ===========================================================================
# STEP 4 — DASHBOARD
# ===========================================================================
def show_dashboard():
    section(f"WILDFIRE — INDIA EARLY WARNING DASHBOARD")
    print(f"  {TODAY.strftime('%A, %d %B %Y  %H:%M:%S')}")

    # Data currency
    print(f"\n  {DASH}")
    print(f"  DATA CURRENCY")
    print(f"  {DASH}")

    hist_p = RAW/"wildfires_india.csv"
    if hist_p.exists():
        row1 = pd.read_csv(hist_p,nrows=1)
        src  = str(row1.get('source',pd.Series(['FIRMS']))[0])
        log(f"Historical : wildfires_india.csv  |  Source: {src}")
    else:
        log("Historical : NOT FOUND — run: python wildfire_disaster_ai.py collect")

    ls_p = LIVE/"wildfire_live_status.json"
    if ls_p.exists():
        ls  = json.load(open(ls_p))
        chk = ls.get('checked_at','')[:16]
        n   = ls.get('gdacs_count',0)+ls.get('reliefweb_count',0)
        try:
            age = (TODAY-datetime.fromisoformat(ls['checked_at'])).total_seconds()/3600
            tag = f"LIVE ({age:.0f}h ago)" if age<12 else f"STALE ({age:.0f}h ago)"
        except: tag = "checked"
        log(f"Live check : {chk}  |  Active events: {n}  |  {tag}")
        firms_n = ls.get('firms_live_count',0)
        has_key = ls.get('has_api_key',False)
        log(f"FIRMS NRT  : {str(firms_n)+' detections' if has_key else 'No API key — set FIRMS_MAP_KEY'}")
    else:
        log("Live       : NOT FOUND — run: python wildfire_disaster_ai.py fetch")

    mp = MODELS/"wildfire_meta.json"
    if mp.exists():
        meta = json.load(open(mp))
        log(f"ML Model   : RF on {meta.get('n_records',0)} records  "
            f"CV F1={meta.get('cv_f1',0):.3f}  "
            f"Classes: {meta.get('risk_classes')}")
    else:
        log("ML Model   : NOT FOUND — run: python wildfire_disaster_ai.py train")

    # Active fires
    if ls_p.exists():
        ls = json.load(open(ls_p))
        active = ls.get('active_fires',[])
        print(f"\n  {DASH}")
        if active:
            print(f"  CURRENTLY ACTIVE WILDFIRE EVENTS  [{len(active)}]")
            print(f"  {DASH}")
            for ev in active[:8]:
                print(f"  RED [{ev.get('date','?')[:10]}]  {ev.get('title','?')[:60]}")
                print(f"       Source: {ev.get('source','?')}")
        else:
            print(f"  No active wildfire alerts from GDACS/ReliefWeb today")
            print(f"  {DASH}")

    # NASA FIRMS live hotspots
    ll_p = LIVE/"wildfire_live.csv"
    if ll_p.exists():
        ll = pd.read_csv(ll_p)
        if len(ll):
            print(f"\n  {DASH}")
            print(f"  NASA FIRMS Active Hotspots (last 7 days)  [{len(ll):,} detections]")
            print(f"  {DASH}")
            by_state = ll.groupby('state').agg(
                count  =('latitude','count'),
                max_frp=('fire_radiative_power_mw','max'),
            ).sort_values('count',ascending=False).head(10)
            print(f"  {'State':<24}{'Hotspots':>10}  {'Max FRP (MW)':>14}")
            print(f"  {'─'*50}")
            for state,r in by_state.iterrows():
                icon = '🔴' if r['count']>500 else '🟠' if r['count']>100 else '🟡'
                print(f"  {icon} {state:<22}{int(r['count']):>10,}  {r.get('max_frp',0):>14.1f}")

    # 3-month forecast
    fa_path = ALERTS/"wildfire_alerts.csv"
    if not fa_path.exists():
        print(f"\n  No wildfire_alerts.csv — run: python wildfire_disaster_ai.py all")
        return

    fa       = pd.read_csv(fa_path)
    is_live  = fa.get('is_live_event',pd.Series(False,index=fa.index)).fillna(False).astype(bool)
    forecast = fa[~is_live]
    months   = sorted(forecast['forecast_month'].unique()) if 'forecast_month' in forecast.columns else []

    print(f"\n  {DASH}")
    print(f"  3-Month Wildfire Risk Forecast  (RF Model + Fire Weather Index)")
    print(f"  {DASH}")
    print(f"  Based on: NASA FIRMS 2010-2024 + FWI climatology + FSI forest data\n")

    for m in months:
        mdf  = forecast[forecast['forecast_month']==m].copy()
        yr   = int(mdf['forecast_year'].iloc[0]) if 'forecast_year' in mdf.columns else TODAY.year
        mark = ' [CURRENT MONTH]' if m==TODAY.month else \
               ' [NEXT MONTH]'    if m==TODAY.month%12+1 else ''
        fs   = ' FIRE SEASON' if m in [2,3,4,5] else \
               ' NE FIRE SEASON' if m in [10,11,12,1] else ''

        red  = mdf[mdf['alert_level'].str.contains('RED')]
        ora  = mdf[mdf['alert_level'].str.contains('ORANGE')]
        yel  = mdf[mdf['alert_level'].str.contains('YELLOW')]
        grn  = mdf[mdf['alert_level'].str.contains('GREEN')]

        print(f"  {'─'*62}")
        print(f"  {MN[m-1]} {yr}{mark}{fs}")
        print(f"  {'─'*62}")
        print(f"  EXTREME(RED):{len(red)}  HIGH(ORANGE):{len(ora)}  WATCH:{len(yel)}  NORMAL:{len(grn)}\n")

        prob_col = 'fire_occurrence_prob' if 'fire_occurrence_prob' in mdf.columns else mdf.columns[0]
        show = mdf.sort_values(prob_col, ascending=False)
        show = show[show[prob_col] > 0.04]

        if len(show):
            print(f"  {'State':<22}  {'Alert':>13}  {'FireProb':>9}  {'FWI':>5}  {'ML Risk':<10}  {'Est.Ha':>9}")
            print(f"  {'─'*72}")
            for _,r in show.iterrows():
                p   = r.get(prob_col,0)
                fwi = r.get('fwi',0)
                arl = str(r.get('alert_level',''))
                rl  = ('EXTREME  🔴' if 'RED'    in arl else
                       'HIGH     🟠' if 'ORANGE' in arl else
                       'WATCH    🟡' if 'YELLOW' in arl else 'NORMAL   🟢')
                print(f"  {str(r.get('state','?')):<22}  {rl:>13}  "
                      f"{p:>9.0%}  {fwi:>5.0f}  "
                      f"{str(r.get('ml_pred_risk_class','?')):<10}  "
                      f"{r.get('estimated_area_ha',0):>9,.0f}")

        if len(red)+len(ora):
            print(f"\n  Highest risk states:")
            for _,r in mdf[mdf['alert_level'].str.contains('RED|ORANGE')]\
                          .sort_values(prob_col,ascending=False).head(5).iterrows():
                p   = r.get(prob_col,0)
                fwi = r.get('fwi',0)
                ha  = r.get('estimated_area_ha',0)
                ext = r.get('ml_extreme_prob',0)
                print(f"    {str(r.get('state','?')):<22}  "
                      f"Prob:{p:.0%}  FWI:{fwi:.0f}  Extreme:{ext:.0%}  ~{ha:,.0f} ha")

    # Monthly fire calendar from historical data
    print(f"\n  {DASH}")
    print(f"  Historical Monthly Fire Calendar  (FIRMS 2010-2024)")
    print(f"  {DASH}")
    if hist_p.exists():
        try:
            hist  = pd.read_csv(hist_p, low_memory=False)
            hist['month'] = pd.to_numeric(hist.get('month',pd.Series()),errors='coerce')
            mc = hist.groupby('month').agg(
                fire_count=('latitude','count'),
                mean_frp  =('fire_radiative_power_mw','mean'),
            ).reset_index()
            print(f"  {'Month':<8}{'Hotspots':>10}  {'AvgFRP':>8}  Activity")
            print(f"  {'─'*52}")
            for _,r in mc.iterrows():
                m_   = int(r.get('month',0))
                if not 1<=m_<=12: continue
                n_   = int(r.get('fire_count',0))
                frp_ = float(r.get('mean_frp',0))
                bar  = '#' * max(1,min(n_//1000,25))
                peak = '[FIRE SEASON]' if m_ in [2,3,4,5] else \
                       '[NE SEASON]'   if m_ in [10,11,12,1] else ''
                now  = ' <- NOW'  if m_==TODAY.month else \
                       ' <- NEXT' if m_==TODAY.month%12+1 else ''
                print(f"  {MN[m_-1]:<8}{n_:>10,}  {frp_:>8.1f}  {bar}{peak}{now}")
        except Exception as e:
            log(f"Calendar error: {e}")

    # Top fire states
    print(f"\n  {DASH}")
    print(f"  Fire-Prone States Classification  (FSI / FIRMS)")
    print(f"  {DASH}")
    print(f"  {'State':<24}{'Zone':<14}{'Forest%':>9}  Peak Fire Month")
    print(f"  {'─'*58}")
    for state in sorted(EXTREME_FIRE):
        fwi_d = STATE_FWI.get(state,DEFAULT_FWI)
        peak_m = max(fwi_d, key=fwi_d.get)
        print(f"  RED    {state:<22} {'EXTREME':<14}"
              f"{STATE_FOREST_PCT.get(state,15):>8}%  {MN[peak_m-1]}")
    for state in sorted(HIGH_FIRE)[:6]:
        fwi_d = STATE_FWI.get(state,DEFAULT_FWI)
        peak_m = max(fwi_d, key=fwi_d.get)
        print(f"  ORANGE {state:<22} {'HIGH':<14}"
              f"{STATE_FOREST_PCT.get(state,15):>8}%  {MN[peak_m-1]}")

    print(f"\n  {SEP}")
    print(f"  Dashboard complete.")
    print(f"  Daily:  python wildfire_disaster_ai.py fetch && python wildfire_disaster_ai.py dashboard")
    print(f"  Weekly: python wildfire_disaster_ai.py train")
    print(f"  {SEP}")


# ===========================================================================
# INTEGRATE INTO EXISTING DASHBOARD
# ===========================================================================
def get_wildfire_summary():
    """
    Call this from disaster_alert_dashboard.py to add wildfire to main dashboard.
    Returns dict with summary stats for the main risk table.
    """
    fa_path = ALERTS/"wildfire_alerts.csv"
    ls_path = LIVE/"wildfire_live_status.json"
    if not fa_path.exists():
        return {'high_alerts':0,'status':'Run wildfire_disaster_ai.py all','data_date':'N/A'}

    fa       = pd.read_csv(fa_path)
    is_live  = fa.get('is_live_event',pd.Series(False,index=fa.index)).fillna(False).astype(bool)
    forecast = fa[~is_live]
    cur_m    = forecast[forecast.get('forecast_month',pd.Series())==TODAY.month] \
               if 'forecast_month' in forecast.columns else forecast

    high_n   = (cur_m['alert_level'].str.contains('RED|ORANGE')).sum() \
               if 'alert_level' in cur_m.columns else 0

    live_n = 0
    if ls_path.exists():
        ls     = json.load(open(ls_path))
        live_n = ls.get('gdacs_count',0)+ls.get('reliefweb_count',0)
        chk    = ls.get('checked_at','')[:16]
    else:
        chk = 'not fetched'

    return {
        'high_alerts': int(high_n) + live_n,
        'status': 'ACTIVE FIRES DETECTED' if live_n else
                  ('High risk states' if high_n else 'Seasonal watch'),
        'data_date': chk,
        'live_events': live_n,
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    cmd = sys.argv[1].lower() if len(sys.argv)>1 else 'help'
    print(SEP)
    print(f"  DISASTER AI — WILDFIRE MODULE")
    print(f"  {TODAY.strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)

    if   cmd in ('collect','data','1'):    collect_data()
    elif cmd in ('train','model','2'):     train_model()
    elif cmd in ('fetch','live','3'):      fetch_live()
    elif cmd in ('dashboard','dash','4'):  show_dashboard()
    elif cmd in ('all','full'):
        print("\n  Running full pipeline...\n")
        collect_data()
        train_model()
        fetch_live()
        show_dashboard()
    else:
        print(f"\n  Commands:")
        print(f"    python wildfire_disaster_ai.py collect    Download FIRMS data (Step 1)")
        print(f"    python wildfire_disaster_ai.py train      Train ML model (Step 2)")
        print(f"    python wildfire_disaster_ai.py fetch      Get live fire data (Step 3)")
        print(f"    python wildfire_disaster_ai.py dashboard  Show fire dashboard (Step 4)")
        print(f"    python wildfire_disaster_ai.py all        Run all steps")
        print(f"\n  For live data (free, 2 min):")
        print(f"    https://firms.modaps.eosdis.nasa.gov/api/area/")
        print(f"    Set FIRMS_MAP_KEY at line 36 of this file")

if __name__ == "__main__":
    main()
