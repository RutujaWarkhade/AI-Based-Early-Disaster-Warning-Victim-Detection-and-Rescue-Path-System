"""
=============================================================================
DISASTER AI — MONTHLY FULL RETRAIN SCRIPT
=============================================================================
Save as: D:/Disaster_Prediction/run_monthly.py

Run once a month:
    python run_monthly.py

What it does (~20 min):
  1. Fetches all live data
  2. Retrains ALL 4 models:
     - earthquake_model_v2.py    XGBoost + exact location names (~5 min)
     - cyclone_model_v2.py       LightGBM RI + Landfall (~5 min)
     - flood_model_v3.py         Random Forest + NDMA enrichment (~3 min)
     - wildfire_disaster_ai.py   RF + FWI + FIRMS collect+train (~5 min)
  3. Regenerates all alert CSVs
  4. Shows full dashboard

All 4 output alert files will be completely fresh:
  future_alerts/earthquake_alerts.csv   ← with location names
  future_alerts/cyclone_alerts.csv
  future_alerts/flood_alerts.csv
  future_alerts/wildfire_alerts.csv
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
    print(f"  Script: python {script} {args}")
    print(SEP)

    script_path = BASE / script
    if not script_path.exists():
        msg = "❌ REQUIRED script missing" if required else "⚠️  Optional script missing — skipping"
        print(f"  {msg}: {script_path}")
        if required:
            print(f"  Cannot continue without this file.")
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
    print(f"  DISASTER AI — MONTHLY FULL RETRAIN")
    print(f"  {TODAY.strftime('%A, %d %B %Y  %H:%M:%S')}")
    print(SEP)
    print(f"""
  This will run the FULL pipeline for all 4 disaster modules.
  Estimated time: ~20 minutes

  Models being retrained:
    🌍 Earthquake  — earthquake_model_v2.py
       XGBoost aftershock classifier + exact location names via
       Nominatim API (district, state, nearest city, tectonic zone)

    🌀 Cyclone     — cyclone_model_v2.py
       LightGBM Rapid Intensification + Landfall probability

    🌊 Flood       — flood_model_v3.py
       Random Forest severity + NDMA occurrence probability

    🔥 Wildfire    — wildfire_disaster_ai.py all
       RF risk model + FWI climatology + NASA FIRMS data

  Output alert files (future_alerts/):
    earthquake_alerts.csv    ← includes location_name, district,
                                nearest_city, tectonic_zone
    cyclone_alerts.csv
    flood_alerts.csv
    wildfire_alerts.csv
""")

    input("  Press Enter to start monthly retrain (Ctrl+C to cancel)...")

    steps = [
        # ── Phase 1: Live data ─────────────────────────────────────────
        ("① Live Data Fetch  (EQ + Cyclone + Flood)",
         "fetch_live_data.py", "", True),

        # ── Phase 2: All 4 models ──────────────────────────────────────
        ("② Earthquake Model v2  (XGBoost + Location Names)",
         "earthquake_model_v2.py", "", True),

        ("③ Cyclone Model v2  (LightGBM RI + Landfall)",
         "cyclone_model_v2.py", "", True),

        ("④ Flood Model v3  (Random Forest + NDMA)",
         "flood_model_v3.py", "", True),

        ("⑤ Wildfire Full Pipeline  (collect + train + fetch + alerts)",
         "wildfire_disaster_ai.py", "all", True),

        # ── Phase 3: Final live data refresh + dashboard ───────────────
        ("⑥ Final Live Data Refresh",
         "fetch_live_data.py", "", False),

        ("⑦ Full Dashboard Display",
         "disaster_alert_dashboard.py", "", True),
    ]

    results = {}
    start   = datetime.now()

    for label, script, args, req in steps:
        step_start = datetime.now()
        ok = run_step(label, script, args, req)
        elapsed = (datetime.now() - step_start).seconds
        results[label] = (ok, elapsed)

    total_sec = (datetime.now() - start).seconds
    total_min = total_sec // 60

    print(f"\n{SEP}")
    print(f"  MONTHLY RETRAIN COMPLETE  —  {TODAY.strftime('%H:%M:%S')}")
    print(f"  Total time: {total_min}m {total_sec % 60}s")
    print(SEP)

    print(f"\n  Results:")
    all_ok = True
    for label, (ok, secs) in results.items():
        icon = '✅' if ok else '⚠️ '
        print(f"  {icon}  {label:<55}  ({secs}s)")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"""
  🎉 All 4 models retrained successfully!

  Your alert files are now fully up to date:
    📁 future_alerts/earthquake_alerts.csv
         Includes: location_name, district, nearest_city, tectonic_zone
    📁 future_alerts/cyclone_alerts.csv
    📁 future_alerts/flood_alerts.csv
    📁 future_alerts/wildfire_alerts.csv

  View the dashboard anytime:
    python disaster_alert_dashboard.py           Full dashboard
    python disaster_alert_dashboard.py eq        Earthquake + locations
    python disaster_alert_dashboard.py summary   Quick risk table
""")
    else:
        print(f"\n  ⚠️  Some steps had warnings. Check logs above.")
        print(f"     You can re-run individual steps manually:")
        print(f"       python earthquake_model_v2.py")
        print(f"       python cyclone_model_v2.py")
        print(f"       python flood_model_v3.py")
        print(f"       python wildfire_disaster_ai.py all")

    print(f"\n  Workflow reminder:")
    print(f"    Daily   : python run_daily.py")
    print(f"    View    : python disaster_alert_dashboard.py")
    print(f"    Weekly  : python run_weekly.py   (EQ + Wildfire retrain)")
    print(f"    Monthly : python run_monthly.py  (all 4 models, you just ran this)")


if __name__ == "__main__":
    main()
