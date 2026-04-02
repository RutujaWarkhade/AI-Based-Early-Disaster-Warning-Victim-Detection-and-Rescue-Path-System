"""
=============================================================================
DISASTER AI — DAILY UPDATE SCRIPT
=============================================================================
Save as: D:/Disaster_Prediction/run_daily.py

Run every morning:
    python run_daily.py

What it does (~1 min):
  1. Fetches live earthquakes from USGS (last 90 days)
  2. Checks JTWC + GDACS for active cyclones
  3. Checks GDACS + ReliefWeb for active India floods
  4. Fetches NASA FIRMS wildfire hotspots + GDACS wildfire events
  5. Prints the full dashboard

NO model retraining — just live data refresh + dashboard display.

To automate on Windows Task Scheduler:
  Action: python D:/Disaster_Prediction/run_daily.py
  Trigger: Daily at 07:00 AM
=============================================================================
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

BASE   = Path("D:/Disaster_Prediction")
TODAY  = datetime.now()
SEP    = "=" * 65

def run_step(label, script, args=""):
    print(f"\n{SEP}")
    print(f"  STEP: {label}")
    print(f"  Script: {script} {args}")
    print(SEP)

    script_path = BASE / script
    if not script_path.exists():
        print(f"  ❌ Script not found: {script_path}")
        print(f"     Skipping this step.")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd += args.split()

    result = subprocess.run(cmd, cwd=str(BASE))
    if result.returncode == 0:
        print(f"\n  ✅ {label} complete")
        return True
    else:
        print(f"\n  ⚠️  {label} finished with warnings (code {result.returncode})")
        return False


def main():
    print(SEP)
    print(f"  DISASTER AI — DAILY REFRESH")
    print(f"  {TODAY.strftime('%A, %d %B %Y  %H:%M:%S')}")
    print(SEP)
    print(f"\n  This will:")
    print(f"  1. Fetch live USGS earthquakes (90 days) + score with ML model")
    print(f"  2. Add exact location names to earthquake alerts (offline, instant)")
    print(f"  3. Check JTWC + GDACS for active cyclones")
    print(f"  4. Check GDACS + ReliefWeb for active floods")
    print(f"  5. Fetch NASA FIRMS wildfire hotspots")
    print(f"  6. Show full dashboard with all forecasts")
    print(f"\n  No model retraining. Runtime: ~1 min\n")

    steps = [
        # Step 1: Fetch + score live data
        ("Live Data Fetch  (EQ + Cyclone + Flood)",
         "fetch_live_data.py", ""),

        # Step 2: Add location names to EQ alerts (fetch_live_data writes CSV without them)
        ("EQ Alert Location Enricher  (adds district / city / tectonic zone)",
         "enrich_eq_alerts_locations.py", ""),

        # Step 3: Wildfire fetch
        ("Wildfire Fetch   (FIRMS + GDACS)",
         "wildfire_disaster_ai.py", "fetch"),

        # Step 4: Dashboard
        ("Dashboard Display",
         "disaster_alert_dashboard.py", ""),
    ]

    results = {}
    for label, script, args in steps:
        ok = run_step(label, script, args)
        results[label] = ok

    print(f"\n{SEP}")
    print(f"  DAILY REFRESH COMPLETE  —  {TODAY.strftime('%H:%M:%S')}")
    print(SEP)
    print(f"\n  Results:")
    for label, ok in results.items():
        icon = '✅' if ok else '⚠️ '
        print(f"  {icon}  {label}")

    print(f"\n  Next steps:")
    print(f"    Weekly  : python run_weekly.py     (retrain earthquake model)")
    print(f"    Monthly : python run_monthly.py    (retrain all 4 models)")
    print(f"    View    : python disaster_alert_dashboard.py")


if __name__ == "__main__":
    main()
