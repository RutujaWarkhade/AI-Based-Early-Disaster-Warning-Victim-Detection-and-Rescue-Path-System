"""
=============================================================================
DISASTER AI — Add Wildfire to Main Dashboard
=============================================================================
Run this ONCE to patch disaster_alert_dashboard.py to include wildfire.

    python add_wildfire_to_dashboard.py

What it does:
  1. Adds wildfire row to the OVERALL RISK SUMMARY table
  2. Adds wildfire section at the end of the dashboard
  3. Backs up original dashboard as disaster_alert_dashboard_backup.py
=============================================================================
"""

from pathlib import Path
import shutil, re

BASE      = Path("D:/Disaster_Prediction")
DASH_FILE = BASE / "disaster_alert_dashboard.py"
BACKUP    = BASE / "disaster_alert_dashboard_backup.py"

if not DASH_FILE.exists():
    print(f"ERROR: {DASH_FILE} not found"); exit(1)

# Backup
shutil.copy(DASH_FILE, BACKUP)
print(f"Backed up to: {BACKUP}")

txt = DASH_FILE.read_text(encoding='utf-8')

# ── PATCH 1: Add wildfire import at top ──────────────────────────────────
IMPORT_LINE = "import sys"
WILDFIRE_IMPORT = """import sys
# Wildfire module (add wildfire_disaster_ai.py to same folder)
try:
    sys.path.insert(0, str(Path("D:/Disaster_Prediction")))
    from wildfire_disaster_ai import get_wildfire_summary, show_dashboard as show_wildfire
    WILDFIRE_AVAILABLE = True
except ImportError:
    WILDFIRE_AVAILABLE = False
"""
if "WILDFIRE_AVAILABLE" not in txt:
    txt = txt.replace(IMPORT_LINE, WILDFIRE_IMPORT, 1)
    print("Patch 1: Wildfire import added")
else:
    print("Patch 1: Already applied")

# ── PATCH 2: Add wildfire row to risk summary table ───────────────────────
# Find the flood row in the summary table and add wildfire after it
FLOOD_ROW_PATTERN = r"(🌊 Flood.*?Watch alerts.*?\n)"
WILDFIRE_ROW = r"""\g<1>  🔥 Wildfire        {wf_alerts:>10}  {wf_status}\n"""

# Simpler: find the total line and insert before it
INSERT_BEFORE = "TOTAL"
WILDFIRE_SUMMARY_CODE = '''
    # Wildfire row
    if WILDFIRE_AVAILABLE:
        wf = get_wildfire_summary()
        wf_n   = wf.get('high_alerts', 0)
        wf_st  = wf.get('status', 'No data')
        wf_dt  = wf.get('data_date', 'N/A')
        wf_live= wf.get('live_events', 0)
        wf_icon = "ACTIVE FIRE" if wf_live else ("Warning" if wf_n > 0 else "Seasonal watch")
        print(f"  {'Fire':.<20}{'Wildfire':.<20}{wf_n:>11}  {'FIRE DETECTED' if wf_live else wf_icon}  [data: {wf_dt[:10]}]")
    else:
        print(f"  {'Fire':.<20}{'Wildfire':.<20}{'N/A':>11}  Run wildfire_disaster_ai.py all")
'''

# Find the total line in show_summary and patch before it
if "WILDFIRE_AVAILABLE" not in txt or "wf_n" not in txt:
    # Find the TOTAL line in the summary section
    total_pattern = r'(print\(f".*?TOTAL.*?\n)'
    match = re.search(total_pattern, txt)
    if match:
        txt = txt[:match.start()] + WILDFIRE_SUMMARY_CODE + "\n" + txt[match.start():]
        print("Patch 2: Wildfire added to risk summary table")
    else:
        print("Patch 2: Could not find TOTAL line — add manually")
else:
    print("Patch 2: Already applied")

# ── PATCH 3: Add wildfire section at end of main() ───────────────────────
SHOW_WILDFIRE_SECTION = '''
    # ── Wildfire section ─────────────────────────────────────────────────
    if WILDFIRE_AVAILABLE:
        show_wildfire()
    else:
        print("\\n  WILDFIRE section: Run 'python wildfire_disaster_ai.py all' first")
'''

# Find the end of main() — look for "Dashboard complete" print
DASHBOARD_COMPLETE = 'Dashboard complete'
if DASHBOARD_COMPLETE in txt and "show_wildfire" not in txt:
    # Insert before the "Dashboard complete" line
    idx = txt.rfind(DASHBOARD_COMPLETE)
    # Find the start of that print statement
    line_start = txt.rfind('\n', 0, idx) + 1
    txt = txt[:line_start] + SHOW_WILDFIRE_SECTION + "\n" + txt[line_start:]
    print("Patch 3: Wildfire section added to main dashboard")
else:
    print("Patch 3: Already applied or Dashboard complete line not found")

# Write patched file
DASH_FILE.write_text(txt, encoding='utf-8')
print(f"\nDashboard patched: {DASH_FILE}")
print(f"\nNow run:")
print(f"  python wildfire_disaster_ai.py all")
print(f"  python disaster_alert_dashboard.py")
