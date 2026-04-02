"""
=============================================================================
DISASTER AI — UNIFIED ALERT DASHBOARD  v2
=============================================================================
Save as: D:/Disaster_Prediction/disaster_alert_dashboard.py

HOW TO RUN (NO TRAINING NEEDED):
    python disaster_alert_dashboard.py            # Full dashboard
    python disaster_alert_dashboard.py eq         # Earthquake only
    python disaster_alert_dashboard.py cy         # Cyclone only
    python disaster_alert_dashboard.py fl         # Flood only
    python disaster_alert_dashboard.py wf         # Wildfire only
    python disaster_alert_dashboard.py summary    # Quick summary table only

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DAILY  (~1 min)  — keeps live data fresh, NO retraining:
    python fetch_live_data.py
    python wildfire_disaster_ai.py fetch
    python disaster_alert_dashboard.py

  VIEW ANYTIME  (~3 sec)  — just read pre-built alerts:
    python disaster_alert_dashboard.py

  WEEKLY  (~5 min)  — retrain earthquake model on new USGS data
                       + regenerate earthquake alerts with location names:
    python earthquake_model_v2.py
    python disaster_alert_dashboard.py eq

  MONTHLY  (~20 min)  — full retrain of ALL 4 models:
    python earthquake_model_v2.py
    python cyclone_model_v2.py
    python flood_model_v3.py
    python wildfire_disaster_ai.py all
    python fetch_live_data.py
    python disaster_alert_dashboard.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EARTHQUAKE ALERTS NOW INCLUDE EXACT LOCATION NAMES (v2 model):
    location_name  e.g. "Chamoli/Uttarkashi, Uttarakhand"
    district       e.g. "Chamoli/Uttarkashi"
    state_name     e.g. "Uttarakhand"
    nearest_city   e.g. "42 km NNE of Chamoli"
    tectonic_zone  e.g. "Main Central Thrust — High Himalayan Seismic Belt"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This dashboard ONLY READS pre-generated CSV/JSON files.
It does NOT retrain anything. Runtime: ~3 seconds.
=============================================================================
"""

import sys
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = Path("D:/Disaster_Prediction")
LIVE   = BASE / "live"
MODELS = BASE / "models"
ALERTS = BASE / "future_alerts"
PRED   = BASE / "predictions"
RAW    = BASE / "raw"

TODAY = datetime.now()
MN    = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
SEP   = "=" * 70
DASH  = "─" * 70
THIN  = "·" * 70

# ── Helpers ────────────────────────────────────────────────────────────────
def section(title, icon=""):
    print(f"\n{SEP}")
    print(f"  {icon}  {title}" if icon else f"  {title}")
    print(SEP)

def sub(title):
    print(f"\n  {DASH}")
    print(f"  {title}")
    print(f"  {DASH}")

def log(msg):
    print(f"   {msg}")

def warn(msg):
    print(f"   ⚠️  {msg}")

def ok(msg):
    print(f"   ✅ {msg}")

def fmt_date(s):
    try:
        return str(s)[:10]
    except:
        return str(s)

def load_csv(path, label=""):
    """Safely load a CSV, return empty DataFrame on failure."""
    p = Path(path)
    if not p.exists():
        if label:
            warn(f"{label} not found: {p.name}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, low_memory=False)
        return df
    except Exception as e:
        warn(f"Could not read {p.name}: {e}")
        return pd.DataFrame()

def load_json(path, label=""):
    """Safely load a JSON file."""
    p = Path(path)
    if not p.exists():
        if label:
            warn(f"{label} not found: {p.name}")
        return {}
    try:
        return json.load(open(p))
    except Exception as e:
        warn(f"Could not read {p.name}: {e}")
        return {}

def data_age(dt_str):
    """Return human-readable age of data."""
    try:
        dt = datetime.fromisoformat(str(dt_str)[:19])
        diff = TODAY - dt
        h = diff.total_seconds() / 3600
        if h < 1:    return f"{int(diff.total_seconds()/60)}m ago"
        elif h < 24: return f"{int(h)}h ago"
        else:        return f"{int(diff.days)}d ago"
    except:
        return "unknown age"

def months_ahead_list(n=3):
    return [(TODAY.month + i - 1) % 12 + 1 for i in range(n)], \
           [TODAY.year + (TODAY.month + i - 1) // 12 for i in range(n)]

# ===========================================================================
# DATA STATUS CHECK
# ===========================================================================
def print_data_status():
    sub("DATA STATUS  (pre-trained model outputs)")

    files = {
        "Earthquake alerts":      ALERTS / "earthquake_alerts.csv",
        "EQ zone forecast":       ALERTS / "earthquake_zone_forecast.csv",
        "EQ location forecast":   ALERTS / "earthquake_location_forecast.csv",
        "EQ zone outlook":        ALERTS / "earthquake_zone_outlook.csv",
        "Cyclone alerts":         ALERTS / "cyclone_alerts.csv",
        "Cyclone seasonal":       ALERTS / "cyclone_seasonal_outlook.csv",
        "Flood alerts":           ALERTS / "flood_alerts.csv",
        "Wildfire alerts":        ALERTS / "wildfire_alerts.csv",
        "EQ live":                LIVE   / "earthquakes_live.csv",
        "Cyclone live status":    LIVE   / "cyclone_live_status.json",
        "Flood live status":      LIVE   / "flood_live_status.json",
        "Wildfire live status":   LIVE   / "wildfire_live_status.json",
        "EQ model":               MODELS / "earthquake_meta.json",
        "Cyclone model":          MODELS / "cyclone_meta.json",
        "Flood model":            MODELS / "flood_meta.json",
        "Wildfire model":         MODELS / "wildfire_meta.json",
    }

    print(f"\n  {'File':<28}  {'Status':<10}  {'Info'}")
    print(f"  {'─'*65}")
    for label, path in files.items():
        p = Path(path)
        if p.exists():
            sz = p.stat().st_size
            mt = datetime.fromtimestamp(p.stat().st_mtime)
            age = data_age(mt.isoformat())
            sz_s = f"{sz//1024}KB" if sz > 1024 else f"{sz}B"
            print(f"  {'✅'} {label:<26}  {sz_s:<10}  {age}")
        else:
            print(f"  {'❌'} {label:<26}  {'MISSING':<10}  Run training/fetch first")


# ===========================================================================
# MODEL PERFORMANCE SUMMARY
# ===========================================================================
def print_model_summary():
    sub("TRAINED MODEL PERFORMANCE SUMMARY")

    for label, meta_path in [
        ("Earthquake Model",  MODELS / "earthquake_meta.json"),
        ("Cyclone Model",     MODELS / "cyclone_meta.json"),
        ("Flood Model",       MODELS / "flood_meta.json"),
        ("Wildfire Model",    MODELS / "wildfire_meta.json"),
    ]:
        meta = load_json(meta_path)
        if not meta:
            log(f"{label}: ❌ meta.json missing")
            continue

        trained_at = fmt_date(meta.get('trained_at', 'unknown'))
        n_rec = meta.get('n_records') or meta.get('n_events') or '?'

        scores = []
        if 'roc_auc'    in meta: scores.append(f"ROC-AUC={meta['roc_auc']:.4f}")
        if 'cv_f1'      in meta and meta['cv_f1']: scores.append(f"CV-F1={meta['cv_f1']:.3f}")
        if 'ri_auc'     in meta: scores.append(f"RI-AUC={meta['ri_auc']:.4f}")
        if 'landfall_auc' in meta: scores.append(f"LF-AUC={meta['landfall_auc']:.4f}")
        if 'wind_rmse'  in meta: scores.append(f"Wind-RMSE={meta['wind_rmse']:.1f}kt")

        score_str = "  |  ".join(scores) if scores else "No metrics recorded"
        log(f"{label:<22}  Trained: {trained_at}  |  n={n_rec}  |  {score_str}")


# ===========================================================================
# EARTHQUAKE DASHBOARD
# ===========================================================================
def print_earthquake_dashboard():
    section("EARTHQUAKE — AFTERSHOCK ALERT SYSTEM + 3-MONTH FORECAST", "🌍")

    # ── Live status ────────────────────────────────────────────────────────
    live_df = load_csv(LIVE / "earthquakes_live.csv", "EQ live data")
    if not live_df.empty:
        live_df['date'] = pd.to_datetime(live_df['date'], errors='coerce')
        n7  = live_df[live_df['date'] >= TODAY - timedelta(days=7)]
        n30 = live_df[live_df['date'] >= TODAY - timedelta(days=30)]
        m5  = live_df[live_df['magnitude'] >= 5.0]

        sub("Live USGS Data Summary  (last 90 days)")
        log(f"Total events loaded  : {len(live_df):,}")
        log(f"Last 7 days          : {len(n7):,}  |  Last 30 days: {len(n30):,}")
        log(f"M≥5.0 events total   : {len(m5):,}")
        if not live_df.empty and 'date' in live_df.columns:
            vd = live_df['date'].dropna()
            if len(vd):
                log(f"Date range           : {vd.min().date()} → {vd.max().date()}")

        if len(m5):
            sub("Recent M≥5.0 Earthquakes  (top 8 by magnitude)")
            print(f"  {'Date':<12}  {'Mag':>5}  {'Depth':>7}  {'Zone':>6}  {'Place'}")
            print(f"  {'─'*65}")
            for _, r in m5.nlargest(8, 'magnitude').iterrows():
                z = r.get('seismic_zone', '?')
                print(f"  {fmt_date(r['date']):<12}  M{r['magnitude']:>4.1f}  "
                      f"{r.get('depth_km',0):>6.1f}km  "
                      f"Z-{z}     {str(r.get('place',''))[:40]}")

    # ── Current alerts (live USGS scored events) ──────────────────────────
    alerts = load_csv(ALERTS / "earthquake_alerts.csv", "EQ alerts")
    if not alerts.empty:
        sub("ML Aftershock Alerts  (next 48h — based on live USGS events)")
        alerts['date'] = pd.to_datetime(alerts.get('date', pd.Series()), errors='coerce')
        has_location   = 'location_name' in alerts.columns

        if 'aftershock_prob_48h' in alerts.columns:
            alerts    = alerts.sort_values('aftershock_prob_48h', ascending=False)
            triggered = alerts[alerts.get('alert_triggered', pd.Series(0)) == 1] \
                        if 'alert_triggered' in alerts.columns else \
                        alerts[alerts['aftershock_prob_48h'] >= 0.20]

            if len(triggered):
                print(f"\n  ⚠️  {len(triggered)} AFTERSHOCK ALERT(S) ACTIVE")
                if has_location:
                    print(f"  {'─'*70}")
                    for i, (_, r) in enumerate(triggered.head(10).iterrows()):
                        alv  = r.get('alert_level', '?')
                        mag  = r.get('magnitude', 0)
                        dep  = r.get('depth_km', 0)
                        prob = r['aftershock_prob_48h']
                        dt   = fmt_date(r.get('date', '?'))
                        zone = r.get('seismic_zone', '?')
                        loc_name = r.get('location_name',
                                         f"{r.get('latitude',0):.2f}°N {r.get('longitude',0):.2f}°E")
                        city     = r.get('nearest_city', '')
                        tectonic = r.get('tectonic_zone', '')
                        lat      = r.get('latitude', 0)
                        lon      = r.get('longitude', 0)
                        print(f"\n  [{i+1}] {alv}  M{mag:.1f}  depth {dep:.0f}km  "
                              f"prob {prob:.1%}  Zone-{zone}  [{dt}]")
                        print(f"       📍 Location  : {loc_name}")
                        if city:     print(f"       🏙️  Nearest    : {city}")
                        if tectonic: print(f"       🌐 Tectonic  : {tectonic}")
                        print(f"       🗺️  Coords    : {lat:.3f}°N, {lon:.3f}°E")
                    print()
                else:
                    print(f"  {'Date':<12}  {'Mag':>5}  {'Zone':>6}  {'Prob':>8}  {'Alert':<14}  Coords")
                    print(f"  {'─'*68}")
                    for _, r in triggered.head(10).iterrows():
                        print(f"  {fmt_date(r.get('date','?')):<12}  "
                              f"M{r.get('magnitude',0):>4.1f}  "
                              f"Z-{r.get('seismic_zone','?'):>4}  "
                              f"{r['aftershock_prob_48h']:>8.1%}  "
                              f"{r.get('alert_level','?'):<14}  "
                              f"{r.get('latitude',0):.2f}°N {r.get('longitude',0):.2f}°E")
                    print(f"\n  💡 Run earthquake_model_v2.py to get exact location names")
            else:
                log("No high-risk aftershock zones currently active (next 48h)")

            if 'alert_level' in alerts.columns:
                log(f"\n  Alert distribution:")
                for lvl, cnt in alerts['alert_level'].value_counts().items():
                    log(f"    {lvl:<15}  {cnt:>5} events")

    # ── 3-month zone-level forecast ────────────────────────────────────────
    zone_fc = load_csv(ALERTS / "earthquake_zone_forecast.csv")
    if not zone_fc.empty:
        months_list, years_list = months_ahead_list(3)
        for m, y in zip(months_list, years_list):
            mdf  = zone_fc[zone_fc['forecast_month'] == m] \
                   if 'forecast_month' in zone_fc.columns else zone_fc
            if mdf.empty: continue

            mark = ' ◄ CURRENT MONTH' if m == TODAY.month else \
                   ' ◄ NEXT MONTH'    if m == (TODAY.month % 12) + 1 else ''

            sub(f"3-Month Seismic Zone Forecast: {MN[m-1]} {y}{mark}")
            print(f"  {'Zone':<8}  {'Zone Name':<38}  {'Occ%':>6}  "
                  f"{'ExpMag':<12}  {'M5+/30d':>8}  {'Alert'}")
            print(f"  {'─'*78}")
            for _, r in mdf.sort_values('seismic_zone', ascending=False).iterrows():
                zn   = str(r.get('zone_name', '?'))[:37]
                occ  = float(r.get('occurrence_prob', 0))
                emag = str(r.get('expected_mag_range', '?'))
                m5c  = int(r.get('recent_m5_30d', 0))
                alv  = str(r.get('alert_level', '?'))
                alv_s = ('CRITICAL 🔴' if 'RED'    in alv else
                         'HIGH     🟠' if 'ORANGE' in alv else
                         'WATCH    🟡' if 'YELLOW' in alv else
                         'LOW      🟢')
                print(f"  Z-{r.get('seismic_zone','?'):<5}  {zn:<38}  "
                      f"{occ:>6.0%}  {emag:<12}  {m5c:>8}  {alv_s}")
    else:
        warn("earthquake_zone_forecast.csv not found — run earthquake_model_v2.py")

    # ── 3-month per-location forecast ─────────────────────────────────────
    loc_fc = load_csv(ALERTS / "earthquake_location_forecast.csv")
    if not loc_fc.empty:
        months_list, years_list = months_ahead_list(3)
        for m, y in zip(months_list, years_list):
            mdf = loc_fc[loc_fc['forecast_month'] == m] \
                  if 'forecast_month' in loc_fc.columns else loc_fc
            if mdf.empty: continue

            mark = ' ◄ CURRENT MONTH' if m == TODAY.month else \
                   ' ◄ NEXT MONTH'    if m == (TODAY.month % 12) + 1 else ''

            # Show only RED + ORANGE locations
            high_risk = mdf[mdf.get('alert_level', pd.Series(dtype=str))
                            .str.contains('RED|ORANGE', na=False)]\
                        .sort_values('combined_risk', ascending=False)
            all_locs  = mdf.sort_values('combined_risk', ascending=False)

            sub(f"High-Risk Locations Forecast: {MN[m-1]} {y}{mark}")
            log(f"RED: {mdf['alert_level'].str.contains('RED',na=False).sum()}  "
                f"ORANGE: {mdf['alert_level'].str.contains('ORANGE',na=False).sum()}  "
                f"YELLOW: {mdf['alert_level'].str.contains('YELLOW',na=False).sum()}  "
                f"GREEN: {mdf['alert_level'].str.contains('GREEN',na=False).sum()}")

            show = high_risk if len(high_risk) else all_locs.head(10)
            print(f"\n  {'Location':<42}  {'Zone':>6}  {'Prob':>6}  "
                  f"{'ExpMag':<12}  {'Alert'}")
            print(f"  {'─'*80}")
            for _, r in show.head(12).iterrows():
                loc  = str(r.get('location_name', '?'))[:41]
                z    = r.get('seismic_zone', '?')
                occ  = float(r.get('occurrence_prob', 0))
                emag = str(r.get('expected_mag_range', '?'))
                alv  = str(r.get('alert_level', '?'))
                alv_s= ('CRITICAL 🔴' if 'RED'    in alv else
                        'HIGH     🟠' if 'ORANGE' in alv else
                        'WATCH    🟡' if 'YELLOW' in alv else
                        'LOW      🟢')
                city = str(r.get('nearest_city', ''))
                tect = str(r.get('tectonic_zone', ''))
                print(f"  {loc:<42}  Z-{z:<4}  {occ:>6.0%}  {emag:<12}  {alv_s}")
                if tect and tect != 'nan':
                    print(f"  {'':>42}        🌐 {tect}")
    else:
        warn("earthquake_location_forecast.csv not found — run earthquake_model_v2.py")

    # ── Zone current outlook ───────────────────────────────────────────────
    outlook = load_csv(ALERTS / "earthquake_zone_outlook.csv")
    if not outlook.empty:
        sub("Current Seismic Activity Outlook  (based on last 30 days live data)")
        print(f"  {'Zone':<8}  {'Zone Name':<40}  {'Last 30d':>10}  {'Outlook'}")
        print(f"  {'─'*68}")
        for _, r in outlook.iterrows():
            zn  = r.get('zone_name') or r.get('next_30d_outlook', 'Zone ' + str(r.get('seismic_zone','')))
            ol  = r.get('next_30d_outlook') or r.get('outlook', 'NORMAL')
            m5c = r.get('events_m5_30d') or r.get('recent_m5_count', 0)
            icon = '🔴' if ol == 'ELEVATED' else '🟢' if ol == 'QUIET' else '🟡'
            print(f"  Z-{r.get('seismic_zone','?'):<5}  {str(zn):<40}  "
                  f"M5+:{int(m5c):>5}     {icon} {ol}")




# ===========================================================================
# CYCLONE DASHBOARD
# ===========================================================================
def print_cyclone_dashboard():
    section("CYCLONE — RAPID INTENSIFICATION & LANDFALL FORECAST", "🌀")

    # ── Live status ────────────────────────────────────────────────────────
    live_status = load_json(LIVE / "cyclone_live_status.json", "Cyclone live status")
    if live_status:
        checked = live_status.get('checked_at', '')
        active  = live_status.get('active_count', 0)
        storms  = live_status.get('active_storms', [])

        sub("Live Cyclone Status")
        log(f"Checked at   : {fmt_date(checked)}  ({data_age(checked)})")
        log(f"Active storms: {active}")

        if active and storms:
            print(f"\n  ⚠️  ACTIVE STORM(S) IN INDIAN OCEAN:")
            for s in storms[:5]:
                log(f"  🌀 {s.get('storm_name','?')}  |  "
                    f"Lat:{s.get('latitude','?')}  Lon:{s.get('longitude','?')}  |  "
                    f"Source: {s.get('source','?')}")
        else:
            log("No active tropical cyclones in Indian Ocean today")
            log("(Dec 2025 storms DITWAH/SENYAR have dissipated)")

    # ── Alert file ─────────────────────────────────────────────────────────
    alerts = load_csv(ALERTS / "cyclone_alerts.csv", "Cyclone alerts")
    if not alerts.empty:
        sub("Cyclone Alert File")
        active_alerts = alerts[~alerts.get('storm_name',
                        pd.Series(dtype=str)).fillna('').str.contains('NO_ACTIVE', na=False)]

        if len(active_alerts):
            print(f"  {'Storm':<25}  {'Wind(kt)':>9}  {'Category':<28}  "
                  f"{'LF%':>6}  {'RI%':>6}  {'Alert'}")
            print(f"  {'─'*68}")
            for _, r in active_alerts.iterrows():
                print(f"  {str(r.get('storm_name','?'))[:24]:<25}  "
                      f"{r.get('current_wind_knots',0):>9.0f}  "
                      f"{str(r.get('current_category','?'))[:27]:<28}  "
                      f"{r.get('landfall_probability',0):>6.1%}  "
                      f"{r.get('ri_probability',0):>6.1%}  "
                      f"{r.get('alert_level','?')}")
        else:
            log("No active storm alerts in cyclone_alerts.csv")
            log("File shows: NO_ACTIVE_STORMS — Indian Ocean is clear")

    # ── Seasonal outlook ───────────────────────────────────────────────────
    seasonal = load_csv(ALERTS / "cyclone_seasonal_outlook.csv")
    if not seasonal.empty:
        sub("3-Month Cyclone Seasonal Outlook")
        months_list, years_list = months_ahead_list(3)

        print(f"  {'Month':<10}  {'Year':>5}  {'Hist.Storms':>12}  "
              f"{'Mean Wind':>10}  {'Peak?':>7}  {'Risk'}")
        print(f"  {'─'*58}")
        for _, r in seasonal.iterrows():
            m_  = int(r.get('forecast_month') or r.get('month', 0))
            y_  = int(r.get('forecast_year', TODAY.year))
            mn_ = r.get('month_name') or r.get('forecast_month_name') or MN[m_-1]
            hs  = r.get('hist_storm_trackpts') or r.get('hist_storm_count', 0)
            mw  = r.get('hist_mean_wind_kt') or r.get('hist_mean_wind', 0)
            pk  = r.get('is_peak_season', False)
            rl  = r.get('risk_level', 'Low')
            icon = '🔴' if str(rl) in ['Critical','High'] else \
                   '🟠' if rl == 'Moderate' else '🟢'
            peak_s = '★ PEAK' if pk else '      '
            print(f"  {str(mn_):<10}  {y_:>5}  {int(hs):>12,}  "
                  f"{float(mw):>10.1f}kt  {peak_s:>7}  {icon} {rl}")

        if 'season_note' in seasonal.columns:
            print()
            for _, r in seasonal.iterrows():
                mn_ = r.get('month_name', '') or r.get('forecast_month_name', '')
                note = r.get('season_note', '')
                if note:
                    log(f"{mn_}: {note}")


# ===========================================================================
# FLOOD DASHBOARD
# ===========================================================================
def print_flood_dashboard():
    section("FLOOD — MONSOON & INUNDATION EARLY WARNING", "🌊")

    # ── Live status ────────────────────────────────────────────────────────
    live_status = load_json(LIVE / "flood_live_status.json", "Flood live status")
    if live_status:
        checked = live_status.get('checked_at', '')
        active  = live_status.get('active_floods', [])
        n_india = live_status.get('india_count', 0)

        sub("Live Flood Status")
        log(f"Checked at   : {fmt_date(checked)}  ({data_age(checked)})")
        log(f"Active India floods: {n_india}")

        if active:
            print(f"\n  ⚠️  ACTIVE FLOOD EVENTS:")
            for ev in active[:6]:
                days_old = ev.get('days_old', 0)
                status   = ev.get('status', '')
                icon = '🔴' if days_old <= 3 else '🟠'
                log(f"  {icon} [{ev.get('date','?')}] {ev.get('title','?')[:55]}  "
                    f"({ev.get('source','?')})")
        else:
            log("No active flood events from GDACS/ReliefWeb today")

    # ── 3-month forecast ───────────────────────────────────────────────────
    alerts = load_csv(ALERTS / "flood_alerts.csv", "Flood alerts")
    if not alerts.empty:
        is_live = alerts.get('is_live_event', pd.Series(False)).fillna(False).astype(bool)
        forecast = alerts[~is_live]
        months_list, years_list = months_ahead_list(3)

        # Determine probability column
        prob_col = next((c for c in ['occurrence_prob', 'historical_flood_prob',
                                     'high_severity_prob'] if c in forecast.columns), None)

        for m, y in zip(months_list, years_list):
            mdf = forecast[forecast.get('forecast_month', pd.Series()) == m] \
                  if 'forecast_month' in forecast.columns else forecast
            if mdf.empty:
                continue

            mark = ' ◄ CURRENT MONTH' if m == TODAY.month else \
                   ' ◄ NEXT MONTH'    if m == (TODAY.month % 12) + 1 else ''
            season = ' [MONSOON PEAK]' if m in [7, 8] else \
                     ' [MONSOON]'      if m in [6, 9] else \
                     ' [NE MONSOON]'   if m in [10, 11] else ''

            sub(f"Flood Forecast: {MN[m-1]} {y}{mark}{season}")

            red  = mdf[mdf.get('alert_level', pd.Series(dtype=str)).str.contains('RED',  na=False)]
            ora  = mdf[mdf.get('alert_level', pd.Series(dtype=str)).str.contains('ORANGE', na=False)]
            yel  = mdf[mdf.get('alert_level', pd.Series(dtype=str)).str.contains('YELLOW', na=False)]
            grn  = mdf[mdf.get('alert_level', pd.Series(dtype=str)).str.contains('GREEN', na=False)]
            log(f"RED:{len(red)}  ORANGE:{len(ora)}  YELLOW:{len(yel)}  GREEN:{len(grn)}")

            # Show top alerts
            show = mdf.copy()
            if prob_col:
                show = show.sort_values(prob_col, ascending=False)
            show = show[show.get('alert_level', pd.Series(dtype=str)).str.contains('RED|ORANGE', na=False)]

            if not show.empty:
                print(f"\n  {'State':<22}  {'Alert':>13}  "
                      f"{'Prob':>8}  {'ML Severity':<14}  {'ExpDeaths':>10}")
                print(f"  {'─'*70}")
                for _, r in show.head(12).iterrows():
                    p   = float(r.get(prob_col, 0)) if prob_col else 0
                    arl = str(r.get('alert_level', ''))
                    rl  = ('CRITICAL 🔴' if 'RED'    in arl else
                           'HIGH     🟠' if 'ORANGE' in arl else
                           'WATCH    🟡' if 'YELLOW' in arl else 'LOW      🟢')
                    sv  = r.get('ml_pred_severity') or r.get('predicted_severity', '?')
                    ed  = r.get('expected_deaths', 0) or 0
                    print(f"  {str(r.get('state','?'))[:21]:<22}  {rl:>13}  "
                          f"{p:>8.0%}  {str(sv)[:13]:<14}  {int(ed):>10,}")

    # ── Historical monsoon calendar ────────────────────────────────────────
    cal = load_csv(PRED / "flood_monsoon_calendar.csv")
    if not cal.empty:
        sub("Historical Flood Activity by Month  (EM-DAT 1970-2025)")
        print(f"  {'Month':<8}  {'Floods':>8}  {'Deaths':>10}  {'Risk Level':<14}  Activity")
        print(f"  {'─'*62}")
        for _, r in cal.iterrows():
            m_  = int(r.get('month', 0))
            if not 1 <= m_ <= 12: continue
            fc  = int(r.get('flood_count', 0))
            td  = int(r.get('total_deaths', 0))
            rl  = str(r.get('risk_level', 'Low'))
            bar = '█' * max(1, min(fc // 2, 20))
            icon = '🔴' if rl == 'Extreme' else '🟠' if rl == 'High' else \
                   '🟡' if rl == 'Moderate' else '🟢'
            now = ' ◄ NOW'  if m_ == TODAY.month else \
                  ' ◄ NEXT' if m_ == TODAY.month % 12 + 1 else ''
            print(f"  {MN[m_-1]:<8}  {fc:>8}  {td:>10,}  "
                  f"{icon} {rl:<12}  {bar}{now}")


# ===========================================================================
# WILDFIRE DASHBOARD
# ===========================================================================
def print_wildfire_dashboard():
    section("WILDFIRE — INDIA FOREST FIRE EARLY WARNING", "🔥")

    # ── Live status ────────────────────────────────────────────────────────
    live_status = load_json(LIVE / "wildfire_live_status.json", "Wildfire live status")
    if live_status:
        checked   = live_status.get('checked_at', '')
        firms_n   = live_status.get('firms_live_count', 0)
        gdacs_n   = live_status.get('gdacs_count', 0)
        rw_n      = live_status.get('reliefweb_count', 0)
        active    = live_status.get('active_fires', [])
        has_key   = live_status.get('has_api_key', False)

        sub("Live Wildfire Status")
        log(f"Checked at   : {fmt_date(checked)}  ({data_age(checked)})")
        log(f"FIRMS NRT    : {'Key set — ' + str(firms_n) + ' detections' if has_key else 'No API key — Register at firms.modaps.eosdis.nasa.gov'}")
        log(f"GDACS events : {gdacs_n}  |  ReliefWeb events: {rw_n}")

        if active:
            print(f"\n  ⚠️  ACTIVE WILDFIRE EVENTS:")
            for ev in active[:6]:
                log(f"  🔴 [{ev.get('date','?')}] {ev.get('title','?')[:58]}  "
                    f"({ev.get('source','?')})")

    # ── NASA FIRMS live hotspots ───────────────────────────────────────────
    firms_live = load_csv(LIVE / "wildfire_live.csv")
    if not firms_live.empty:
        sub(f"NASA FIRMS Active Hotspots  [{len(firms_live):,} detections, last 7 days]")
        by_state = firms_live.groupby('state').agg(
            count   =('latitude', 'count'),
            max_frp =('fire_radiative_power_mw', 'max'),
        ).sort_values('count', ascending=False).head(12)
        print(f"  {'State':<26}  {'Hotspots':>10}  {'Max FRP (MW)':>14}")
        print(f"  {'─'*54}")
        for state, r in by_state.iterrows():
            icon = '🔴' if r['count'] > 500 else '🟠' if r['count'] > 100 else '🟡'
            print(f"  {icon} {str(state):<24}  {int(r['count']):>10,}  "
                  f"{float(r.get('max_frp', 0)):>14.1f}")

    # ── 3-month forecast ───────────────────────────────────────────────────
    alerts = load_csv(ALERTS / "wildfire_alerts.csv", "Wildfire alerts")
    if not alerts.empty:
        is_live  = alerts.get('is_live_event', pd.Series(False)).fillna(False).astype(bool)
        forecast = alerts[~is_live]
        months_list, years_list = months_ahead_list(3)

        prob_col = next((c for c in ['fire_occurrence_prob', 'combined_risk_score']
                         if c in forecast.columns), None)

        for m, y in zip(months_list, years_list):
            mdf = forecast[forecast.get('forecast_month', pd.Series()) == m] \
                  if 'forecast_month' in forecast.columns else forecast
            if mdf.empty:
                continue

            mark   = ' ◄ CURRENT' if m == TODAY.month else \
                     ' ◄ NEXT'    if m == (TODAY.month % 12) + 1 else ''
            season = ' 🔥 FIRE SEASON' if m in [2, 3, 4, 5] else \
                     ' 🔥 NE FIRE SEASON' if m in [10, 11, 12, 1] else ''

            sub(f"Wildfire Forecast: {MN[m-1]} {y}{mark}{season}")

            red = mdf[mdf.get('alert_level', pd.Series(dtype=str)).str.contains('RED',    na=False)]
            ora = mdf[mdf.get('alert_level', pd.Series(dtype=str)).str.contains('ORANGE', na=False)]
            yel = mdf[mdf.get('alert_level', pd.Series(dtype=str)).str.contains('YELLOW', na=False)]
            grn = mdf[mdf.get('alert_level', pd.Series(dtype=str)).str.contains('GREEN',  na=False)]
            log(f"RED:{len(red)}  ORANGE:{len(ora)}  YELLOW:{len(yel)}  GREEN:{len(grn)}")

            show = mdf.copy()
            if prob_col:
                show = show.sort_values(prob_col, ascending=False)
            show = show[show.get('alert_level', pd.Series(dtype=str)).str.contains('RED|ORANGE', na=False)]

            if not show.empty:
                print(f"\n  {'State':<22}  {'Alert':>13}  {'FireProb':>9}  "
                      f"{'FWI':>5}  {'ML Risk':<10}  {'Est.Ha':>8}")
                print(f"  {'─'*70}")
                for _, r in show.head(12).iterrows():
                    p   = float(r.get(prob_col, 0)) if prob_col else 0
                    fwi = float(r.get('fwi', 0))
                    arl = str(r.get('alert_level', ''))
                    rl  = ('EXTREME  🔴' if 'RED'    in arl else
                           'HIGH     🟠' if 'ORANGE' in arl else
                           'WATCH    🟡' if 'YELLOW' in arl else 'NORMAL   🟢')
                    print(f"  {str(r.get('state','?'))[:21]:<22}  {rl:>13}  "
                          f"{p:>9.0%}  {fwi:>5.0f}  "
                          f"{str(r.get('ml_pred_risk_class','?')):<10}  "
                          f"{int(r.get('estimated_area_ha', 0)):>8,}")


# ===========================================================================
# MASTER SUMMARY TABLE
# ===========================================================================
def print_summary_table():
    section("MASTER RISK SUMMARY TABLE", "📊")
    log(f"As of: {TODAY.strftime('%A, %d %B %Y  %H:%M')}  (India Standard Time)\n")

    print(f"  {'Disaster':<14}  {'Status':<22}  {'Active Alerts':>14}  "
          f"{'Live Events':>12}  {'Data Age'}")
    print(f"  {'─'*70}")

    # ── Earthquake ────────────────────────────────────────────────────────
    try:
        eq_alerts = load_csv(ALERTS / "earthquake_alerts.csv")
        eq_active = 0
        if not eq_alerts.empty and 'alert_triggered' in eq_alerts.columns:
            eq_active = int((eq_alerts['alert_triggered'] == 1).sum())
        elif not eq_alerts.empty and 'aftershock_prob_48h' in eq_alerts.columns:
            eq_active = int((eq_alerts['aftershock_prob_48h'] >= 0.20).sum())

        eq_live = load_csv(LIVE / "earthquakes_live.csv")
        eq_live_n = len(eq_live[eq_live.get('magnitude', pd.Series(0)) >= 4.5]) \
                    if not eq_live.empty and 'magnitude' in eq_live.columns else 0

        eq_meta = load_json(MODELS / "earthquake_meta.json")
        eq_age  = data_age(eq_meta.get('trained_at', '')) if eq_meta else 'N/A'
        eq_status = 'ELEVATED' if eq_active > 5 else 'WATCH' if eq_active > 0 else 'NORMAL'
        icon_eq = '🔴' if eq_active > 5 else '🟡' if eq_active > 0 else '🟢'
        print(f"  🌍 {'Earthquake':<12}  {icon_eq} {eq_status:<20}  "
              f"{eq_active:>14}  {eq_live_n:>12}  {eq_age}")
    except Exception as e:
        print(f"  🌍 {'Earthquake':<12}  ❌ Error: {str(e)[:30]}")

    # ── Cyclone ───────────────────────────────────────────────────────────
    try:
        cy_live = load_json(LIVE / "cyclone_live_status.json")
        cy_active_n = cy_live.get('active_count', 0) if cy_live else 0

        cy_alerts = load_csv(ALERTS / "cyclone_alerts.csv")
        cy_alert_n = 0
        if not cy_alerts.empty and 'storm_name' in cy_alerts.columns:
            cy_alert_n = len(cy_alerts[~cy_alerts['storm_name'].str.contains(
                'NO_ACTIVE', na=False)])

        cy_checked = cy_live.get('checked_at', '') if cy_live else ''
        cy_age     = data_age(cy_checked) if cy_checked else 'N/A'
        cy_status  = 'ACTIVE STORM' if cy_active_n else 'CLEAR'
        icon_cy    = '🔴' if cy_active_n else '🟢'
        print(f"  🌀 {'Cyclone':<12}  {icon_cy} {cy_status:<20}  "
              f"{cy_alert_n:>14}  {cy_active_n:>12}  {cy_age}")
    except Exception as e:
        print(f"  🌀 {'Cyclone':<12}  ❌ Error: {str(e)[:30]}")

    # ── Flood ─────────────────────────────────────────────────────────────
    try:
        fl_live = load_json(LIVE / "flood_live_status.json")
        fl_live_n = fl_live.get('india_count', 0) if fl_live else 0

        fl_alerts = load_csv(ALERTS / "flood_alerts.csv")
        fl_red = 0
        if not fl_alerts.empty and 'alert_level' in fl_alerts.columns:
            fl_red = int(fl_alerts['alert_level'].str.contains('RED', na=False).sum())

        fl_checked = fl_live.get('checked_at', '') if fl_live else ''
        fl_age     = data_age(fl_checked) if fl_checked else 'N/A'
        fl_status  = 'ACTIVE FLOODS' if fl_live_n else 'SEASONAL WATCH'
        icon_fl    = '🔴' if fl_live_n > 0 else '🟡' if fl_red > 3 else '🟢'
        print(f"  🌊 {'Flood':<12}  {icon_fl} {fl_status:<20}  "
              f"{fl_red:>14}  {fl_live_n:>12}  {fl_age}")
    except Exception as e:
        print(f"  🌊 {'Flood':<12}  ❌ Error: {str(e)[:30]}")

    # ── Wildfire ──────────────────────────────────────────────────────────
    try:
        wf_live = load_json(LIVE / "wildfire_live_status.json")
        wf_live_n = (wf_live.get('gdacs_count', 0) +
                     wf_live.get('reliefweb_count', 0)) if wf_live else 0

        wf_alerts = load_csv(ALERTS / "wildfire_alerts.csv")
        wf_red = 0
        if not wf_alerts.empty and 'alert_level' in wf_alerts.columns:
            wf_red = int(wf_alerts['alert_level'].str.contains('RED', na=False).sum())

        wf_checked = wf_live.get('checked_at', '') if wf_live else ''
        wf_age     = data_age(wf_checked) if wf_checked else 'N/A'
        wf_status  = 'ACTIVE FIRES' if wf_live_n else ('FIRE SEASON' if TODAY.month in [2,3,4,5] else 'WATCH')
        icon_wf    = '🔴' if wf_live_n > 0 else '🟠' if TODAY.month in [2,3,4,5] else '🟢'
        print(f"  🔥 {'Wildfire':<12}  {icon_wf} {wf_status:<20}  "
              f"{wf_red:>14}  {wf_live_n:>12}  {wf_age}")
    except Exception as e:
        print(f"  🔥 {'Wildfire':<12}  ❌ Error: {str(e)[:30]}")

    # ── Footer ────────────────────────────────────────────────────────────
    print(f"\n  {THIN}")
    print(f"  Active Alerts = alerts flagged RED or triggered by ML model")
    print(f"  Live Events   = events confirmed by GDACS/USGS/ReliefWeb TODAY")
    print(f"\n  Legend:  🔴 High/Active  🟠 Elevated  🟡 Watch  🟢 Normal/Clear")


# ===========================================================================
# QUICK COMMANDS GUIDE
# ===========================================================================
def print_commands_guide():
    sub("RECOMMENDED WORKFLOW")
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  ① DAILY  (~1 min)  — fetch live data + regenerate EQ forecast     │
  │                                                                     │
  │    python fetch_live_data.py                                        │
  │         Updates: USGS earthquakes, JTWC cyclones,                  │
  │                  GDACS/ReliefWeb floods                             │
  │    python wildfire_disaster_ai.py fetch                             │
  │         Updates: NASA FIRMS hotspots, GDACS wildfire events         │
  │    python generate_eq_forecast.py                                   │
  │         Updates: 3-month EQ zone + location forecast (5 sec)        │
  │    python disaster_alert_dashboard.py                               │
  │         View: Full dashboard with EQ future predictions             │
  │                                                                     │
  │    OR just run:  python run_daily.py  (does all 4 steps above)     │
  │                                                                     │
  │  ② VIEW ANYTIME  (~3 sec)  — no downloads needed                   │
  │                                                                     │
  │    python disaster_alert_dashboard.py                               │
  │    python disaster_alert_dashboard.py summary   Quick table only    │
  │    python disaster_alert_dashboard.py eq        Earthquake only     │
  │    python disaster_alert_dashboard.py cy        Cyclone only        │
  │    python disaster_alert_dashboard.py fl        Flood only          │
  │    python disaster_alert_dashboard.py wf        Wildfire only       │
  │                                                                     │
  │  ③ WEEKLY  (~5 min)  — retrain earthquake model on fresh USGS data │
  │                                                                     │
  │    python earthquake_model_v2.py                                    │
  │         Trains: XGBoost + location names + generates 3-month        │
  │         forecast at end: zone_forecast + location_forecast CSVs     │
  │    python disaster_alert_dashboard.py eq                            │
  │                                                                     │
  │    OR just run:  python run_weekly.py                               │
  │                                                                     │
  │  ④ MONTHLY  (~20 min)  — full retrain of all 4 models              │
  │                                                                     │
  │    python earthquake_model_v2.py      EQ  — XGBoost + locations    │
  │    python generate_eq_forecast.py     EQ  — 3-month forecast        │
  │    python cyclone_model_v2.py         Cyclone — LightGBM RI+LF     │
  │    python flood_model_v3.py           Flood — RF + NDMA enrichment  │
  │    python wildfire_disaster_ai.py all Wildfire — collect+train+dash │
  │    python fetch_live_data.py          Refresh all live data         │
  │    python disaster_alert_dashboard.py Full dashboard                │
  │                                                                     │
  │    OR just run:  python run_monthly.py                              │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================
def main():
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'

    print(SEP)
    print(f"  DISASTER AI — UNIFIED ALERT DASHBOARD")
    print(f"  {TODAY.strftime('%A, %d %B %Y  %H:%M:%S')}")
    print(f"  Mode: {cmd.upper()}")
    print(SEP)

    if cmd == 'summary':
        print_summary_table()
        print_commands_guide()

    elif cmd in ('eq', 'earthquake'):
        print_data_status()
        print_earthquake_dashboard()

    elif cmd in ('cy', 'cyclone'):
        print_data_status()
        print_cyclone_dashboard()

    elif cmd in ('fl', 'flood'):
        print_data_status()
        print_flood_dashboard()

    elif cmd in ('wf', 'wildfire', 'fire'):
        print_data_status()
        print_wildfire_dashboard()

    elif cmd in ('all', 'full', 'dashboard', 'dash'):
        print_summary_table()
        print_data_status()
        print_model_summary()
        print_earthquake_dashboard()
        print_cyclone_dashboard()
        print_flood_dashboard()
        print_wildfire_dashboard()
        print_commands_guide()

    elif cmd in ('status', 'check'):
        print_data_status()
        print_model_summary()
        print_summary_table()

    else:
        print(f"\n  Unknown command: '{cmd}'")
        print(f"\n  Valid commands:")
        print(f"    all / dashboard   Full dashboard (default)")
        print(f"    summary           Quick master risk table only")
        print(f"    eq                Earthquake section only")
        print(f"    cy                Cyclone section only")
        print(f"    fl                Flood section only")
        print(f"    wf                Wildfire section only")
        print(f"    status            Data file status + model metrics only")

    print(f"\n{SEP}")
    print(f"  Dashboard complete.  {TODAY.strftime('%H:%M:%S')}")
    print(SEP)


if __name__ == "__main__":
    main()
