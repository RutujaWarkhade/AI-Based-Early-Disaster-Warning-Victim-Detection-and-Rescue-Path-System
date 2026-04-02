"""
=============================================================================
DISASTER AI — WEEKLY UPDATE SCRIPT
=============================================================================
Save as: D:/Disaster_Prediction/run_weekly.py

Run every week (e.g. Sunday morning):
    python run_weekly.py

What it does (~5 min):
  1. Fetches fresh live data (same as run_daily.py)
  2. Retrains earthquake_model_v2.py on latest USGS data
     → generates earthquake_alerts.csv WITH exact location names:
       location_name, district, state_name, nearest_city, tectonic_zone
  3. Retrains wildfire model on any new FIRMS data
  4. Shows full dashboard with fresh alerts

Why weekly for earthquake only?
  - USGS provides real 2026 data → model improves with new events
  - Location names (Nominatim API) are resolved per alert
  - Cyclone & Flood models use historical data that doesn't change weekly
=============================================================================
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

BASE  = Path("D:/Disaster_Prediction")
TODAY = datetime.now()
SEP   = "=" * 65


def run_step(label, script, args="", required=True):
    print(f"\n{SEP}")
    print(f"  STEP: {label}")
    print(f"  Script: {script} {args}")
    print(SEP)

    script_path = BASE / script
    if not script_path.exists():
        msg = "❌ MISSING (required)" if required else "⚠️  Missing (optional — skipping)"
        print(f"  {msg}: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd += args.split()

    result = subprocess.run(cmd, cwd=str(BASE))
    ok = result.returncode == 0
    print(f"\n  {'✅' if ok else '⚠️ '} {label} {'complete' if ok else 'finished with warnings'}")
    return ok


def main():
    print(SEP)
    print(f"  DISASTER AI — WEEKLY RETRAIN")
    print(f"  {TODAY.strftime('%A, %d %B %Y  %H:%M:%S')}")
    print(SEP)
    print(f"\n  This will:")
    print(f"  1. Fetch all live data  (EQ + Cyclone + Flood + Wildfire)")
    print(f"  2. Retrain Earthquake model v2  — XGBoost + location names")
    print(f"     earthquake_alerts.csv will have: location_name, district,")
    print(f"     state_name, nearest_city, tectonic_zone for every alert")
    print(f"  3. Retrain Wildfire model  — RF + FWI + NASA FIRMS")
    print(f"  4. Show full dashboard")
    print(f"\n  Runtime: ~5 minutes\n")

    steps = [
        # Step 1: Live data refresh
        ("Live Data Fetch  (EQ + Cyclone + Flood)", "fetch_live_data.py",          "",        True),
        ("Wildfire Live Fetch  (FIRMS + GDACS)",    "wildfire_disaster_ai.py",     "fetch",   True),

        # Step 2: Weekly retrain — Earthquake (v2 with location names)
        ("Earthquake Model v2  (XGBoost + Locations)", "earthquake_model_v2.py",   "",        True),

        # Step 3: Weekly retrain — Wildfire
        ("Wildfire Model  (RF + FWI)",              "wildfire_disaster_ai.py",     "train",   True),

        # Step 4: Dashboard
        ("Full Dashboard Display",                   "disaster_alert_dashboard.py", "",        True),
    ]

    results = {}
    for label, script, args, req in steps:
        ok = run_step(label, script, args, req)
        results[label] = ok

    print(f"\n{SEP}")
    print(f"  WEEKLY RETRAIN COMPLETE  —  {TODAY.strftime('%H:%M:%S')}")
    print(SEP)
    print(f"\n  Results:")
    for label, ok in results.items():
        icon = '✅' if ok else '⚠️ '
        print(f"  {icon}  {label}")

    eq_ok = results.get("Earthquake Model v2  (XGBoost + Locations)", False)
    if eq_ok:
        print(f"\n  📍 Earthquake alerts now include exact location names!")
        print(f"     View with: python disaster_alert_dashboard.py eq")
    else:
        print(f"\n  ⚠️  Earthquake model had issues. Check earthquake_model_v2.py")

    print(f"\n  Next steps:")
    print(f"    Daily   : python run_daily.py      (live data only, no retrain)")
    print(f"    Monthly : python run_monthly.py    (retrain ALL 4 models)")


if __name__ == "__main__":
    main()
