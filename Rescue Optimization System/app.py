from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import numpy as np
import requests
from ultralytics import YOLO
import math
import threading
import time
import os
import csv
from datetime import datetime

app = Flask(__name__)

# ============================================================
#  CSV DATA STORAGE SYSTEM
# All data saved to  data/  folder next to app.py
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PERSONS_CSV  = os.path.join(DATA_DIR, "detected_persons.csv")
GPS_CSV      = os.path.join(DATA_DIR, "gps_history.csv")

csv_lock = threading.Lock()

# ── In-memory log (last 500 rows each) for live dashboard ──
persons_log = []   # list of dicts
gps_log     = []   # list of dicts
LOG_MAX     = 500

# Track already-logged persons to avoid duplicate rows within same detection cycle
# key = (cam_id, rounded_lat, rounded_lon, minute) → True
_person_seen = {}
_gps_seen    = {}

def _init_csv():
    """Create CSV files with headers if they don't exist yet."""
    if not os.path.isfile(PERSONS_CSV):
        with open(PERSONS_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "date", "time",
                "cam_id", "cam_name", "person_id",
                "confidence_pct",
                "latitude", "longitude", "altitude_m", "accuracy_m", "gps_source",
                "nearest_team", "road_distance_km", "eta_min"
            ])
    if not os.path.isfile(GPS_CSV):
        with open(GPS_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "date", "time",
                "cam_id", "cam_name",
                "latitude", "longitude", "altitude_m", "accuracy_m", "gps_source"
            ])

_init_csv()

def log_person(cam_id, cam_name, person_id, conf, loc, nearest, route):
    """Write one person detection row to CSV + in-memory log."""
    now = datetime.now()
    ts  = now.strftime("%Y-%m-%d %H:%M:%S")
    dt  = now.strftime("%Y-%m-%d")
    tm  = now.strftime("%H:%M:%S")

    lat = lon = alt = acc = src = ""
    if loc:
        lat = round(loc["lat"], 6)
        lon = round(loc["lon"], 6)
        alt = round(loc["alt"], 1) if loc.get("alt") is not None else ""
        acc = round(loc["accuracy"], 1) if loc.get("accuracy") is not None else ""
        src = loc.get("source", "")

    # Dedup: skip if same cam+person+rounded coord logged in same minute
    minute_key = (cam_id, person_id,
                  round(float(lat), 3) if lat != "" else 0,
                  now.strftime("%Y-%m-%d %H:%M"))
    with csv_lock:
        if minute_key in _person_seen:
            return
        _person_seen[minute_key] = True
        # Prune old keys (keep last 2000)
        if len(_person_seen) > 2000:
            for k in list(_person_seen.keys())[:500]:
                del _person_seen[k]

    nearest_name = nearest["name"] if nearest else ""
    road_km      = route["distance_km"] if route else ""
    eta          = route["duration_min"] if route else ""

    row = {
        "timestamp": ts, "date": dt, "time": tm,
        "cam_id": cam_id, "cam_name": cam_name, "person_id": person_id,
        "confidence_pct": round(conf * 100, 1),
        "latitude": lat, "longitude": lon,
        "altitude_m": alt, "accuracy_m": acc, "gps_source": src,
        "nearest_team": nearest_name,
        "road_distance_km": road_km, "eta_min": eta,
    }

    with csv_lock:
        # Append to CSV file
        with open(PERSONS_CSV, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writerow(row)
        # Keep in-memory log
        persons_log.append(row)
        if len(persons_log) > LOG_MAX:
            del persons_log[0]

def log_gps(cam_id, cam_name, lat, lon, alt, acc, source):
    """Write GPS reading to CSV every ~10 seconds per camera (deduped)."""
    now = datetime.now()
    # Dedup: same cam + same rounded coord + same 10-second window
    sec10 = now.strftime("%Y-%m-%d %H:%M:") + str(now.second // 10)
    key   = (cam_id, round(lat, 4), round(lon, 4), sec10)
    with csv_lock:
        if key in _gps_seen:
            return
        _gps_seen[key] = True
        if len(_gps_seen) > 2000:
            for k in list(_gps_seen.keys())[:500]:
                del _gps_seen[k]

    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": ts,
        "date":      now.strftime("%Y-%m-%d"),
        "time":      now.strftime("%H:%M:%S"),
        "cam_id":    cam_id,
        "cam_name":  cam_name,
        "latitude":  round(lat, 6),
        "longitude": round(lon, 6),
        "altitude_m":  round(alt, 1) if alt is not None else "",
        "accuracy_m":  round(acc, 1) if acc is not None else "",
        "gps_source":  source,
    }
    with csv_lock:
        with open(GPS_CSV, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writerow(row)
        gps_log.append(row)
        if len(gps_log) > LOG_MAX:
            del gps_log[0]

# ============================================================
#  4 MOBILE CAMERAS — set each phone's IP here
#    All phones must run IP Webcam app on port 8080
#    All must be on the same WiFi / hotspot network as PC
#    Find each phone's IP on the IP Webcam main screen
# ============================================================
CAMERAS = [
    {"id": 1, "name": "CAM-1 North Gate",   "ip": "#"},
    {"id": 2, "name": "CAM-2 East Block",   "ip": "#"},
    {"id": 3, "name": "CAM-3 Main Building","ip": "#"},
    {"id": 4, "name": "CAM-4 South Exit",   "ip": "#"},
]
# ============================================================

def cam_urls(ip):
    return {
        "video": f"http://{ip}:8080/video",
        "shot":  f"http://{ip}:8080/shot.jpg",
        "gps":   f"http://{ip}:8080/gps.json",
    }

# --------------------------------------------------------
# YOLOv8 — single shared model
# --------------------------------------------------------
model_lock = threading.Lock()

def load_yolo_model():
    candidates = ["yolov8m.pt", "yolov8s.pt", "yolov8n.pt"]
    for name in candidates:
        paths = [
            name,
            os.path.join(os.path.dirname(__file__), name),
            os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Ultralytics", name),
            os.path.join(os.path.expanduser("~"), ".config", "Ultralytics", name),
        ]
        for p in paths:
            if os.path.isfile(p) and os.path.getsize(p) > 1_000_000:
                try:
                    m = YOLO(p)
                    print(f"[YOLO] Loaded {name}")
                    return m
                except Exception as e:
                    print(f"[YOLO] Failed {p}: {e}")
    print("[YOLO] Downloading yolov8n.pt...")
    return YOLO("yolov8n.pt")

model = load_yolo_model()

CONF_THRESHOLD = 0.30
IOU_THRESHOLD  = 0.45
PERSON_CLASS   = 0

# --------------------------------------------------------
#  RESCUE TEAMS — hyper-local MIT College locations
# --------------------------------------------------------
RESCUE_TEAMS = [
    {"id": 1, "name": "Alpha Team",   "loc": "MIT College Gate No.5 (Main Entrance)",             "lat": 19.8493, "lon": 75.3219},
    {"id": 2, "name": "Bravo Team",   "loc": "United Pride Complex (Shops, Beed Bypass beside MIT)","lat": 19.8516, "lon": 75.3200},
    {"id": 3, "name": "Charlie Team", "loc": "MIT Engineering College Building (Campus Interior)", "lat": 19.8508, "lon": 75.3230},
    {"id": 4, "name": "Delta Team",   "loc": "MIT Campus Research Centre (B.Tech Building)",      "lat": 19.8496, "lon": 75.3220},
]

# --------------------------------------------------------
#  Per-camera state
# --------------------------------------------------------
cam_state = {}
for cam in CAMERAS:
    cid = cam["id"]
    cam_state[cid] = {
        "gps":          {"lat": None, "lon": None, "alt": None,
                         "accuracy": None, "valid": False, "source": "none"},
        "gps_lock":     threading.Lock(),
        "frame":        None,
        "frame_lock":   threading.Lock(),
        "persons":      [],
        "persons_lock": threading.Lock(),
        "online":       False,
    }

# --------------------------------------------------------
#  OSRM route cache
# --------------------------------------------------------
route_cache      = {}
route_cache_lock = threading.Lock()


# ========================================================
#  GPS poller — one thread per camera
# ========================================================
def make_gps_poller(cam):
    cid  = cam["id"]
    urls = cam_urls(cam["ip"])
    st   = cam_state[cid]

    def run():
        while True:
            try:
                resp = requests.get(urls["gps"], timeout=3)
                data = resp.json()
                lat = lon = alt = acc = None
                source = "none"

                gps_block = data.get("gps", {})
                if gps_block:
                    lat = gps_block.get("latitude") or gps_block.get("lat")
                    lon = gps_block.get("longitude") or gps_block.get("lon") or gps_block.get("lng")
                    alt = gps_block.get("altitude")  or gps_block.get("alt")
                    acc = gps_block.get("accuracy")  or gps_block.get("acc")
                    if lat and lon and float(lat) != 0.0 and float(lon) != 0.0:
                        source = "gps"

                if source == "none":
                    net = data.get("network", {})
                    if not net and "latitude" in data:
                        net = data
                    if net:
                        nlat = net.get("latitude")  or net.get("lat")
                        nlon = net.get("longitude") or net.get("lon") or net.get("lng")
                        nacc = net.get("accuracy")  or net.get("acc")
                        nalt = net.get("altitude")  or net.get("alt")
                        if nlat and nlon and float(nlat) != 0.0 and float(nlon) != 0.0:
                            lat, lon, alt, acc = nlat, nlon, nalt, nacc
                            source = "network"

                valid = source != "none" and lat is not None and lon is not None
                with st["gps_lock"]:
                    st["gps"].update({
                        "lat":      float(lat) if lat is not None else None,
                        "lon":      float(lon) if lon is not None else None,
                        "alt":      float(alt) if alt is not None else None,
                        "accuracy": float(acc) if acc is not None else None,
                        "valid":    valid,
                        "source":   source,
                    })
                st["online"] = True
                if valid:
                    print(f"[CAM{cid} GPS:{source}] {lat:.5f}, {lon:.5f}")
                    log_gps(cid, cam["name"],
                            float(lat), float(lon),
                            float(alt) if alt is not None else None,
                            float(acc) if acc is not None else None,
                            source)
            except Exception as e:
                st["online"] = False
                print(f"[CAM{cid} GPS ERR] {e}")
            time.sleep(1)
    return run


# ========================================================
# Frame grabber — one thread per camera
# ========================================================
def make_frame_grabber(cam):
    cid  = cam["id"]
    urls = cam_urls(cam["ip"])
    st   = cam_state[cid]

    def run():
        use_shot = True
        while True:
            try:
                if use_shot:
                    r = requests.get(urls["shot"], timeout=4)
                    if r.status_code == 200 and len(r.content) > 1000:
                        with st["frame_lock"]:
                            st["frame"] = r.content
                        st["online"] = True
                    else:
                        use_shot = False
                    time.sleep(0.07)
                else:
                    with requests.get(urls["video"], stream=True, timeout=10) as stream:
                        buf = b""
                        for chunk in stream.iter_content(chunk_size=4096):
                            buf += chunk
                            s = buf.find(b'\xff\xd8')
                            e = buf.find(b'\xff\xd9')
                            if s != -1 and e != -1 and e > s:
                                jpg = buf[s:e + 2]
                                buf = buf[e + 2:]
                                if len(jpg) > 1000:
                                    with st["frame_lock"]:
                                        st["frame"] = jpg
                                    st["online"] = True
                            if len(buf) > 200000:
                                buf = buf[-50000:]
            except Exception as e:
                st["online"] = False
                print(f"[CAM{cid} FRAME ERR] {e} — retry 2s")
                time.sleep(2)
    return run


# Start all camera threads
for cam in CAMERAS:
    threading.Thread(target=make_gps_poller(cam),    daemon=True, name=f"gps-{cam['id']}").start()
    threading.Thread(target=make_frame_grabber(cam), daemon=True, name=f"frame-{cam['id']}").start()
    print(f"[CAM{cam['id']}] Threads started — {cam['name']} @ {cam['ip']}")


# ========================================================
# DISASTER-AWARE SMART ROUTING ENGINE
# ========================================================
# Modes:
#   normal   -> driving roads (normal conditions)
#   disaster -> foot/walking  (roads blocked, rescuers through rubble)
#   flood    -> foot routing + blocked flood zones avoided
#
# Algorithm layers (applied in order):
#   1. Haversine straight-line for initial team ranking
#   2. OSRM road/foot routing for actual path
#   3. Blocked zone check -- if route passes rubble/flood zone, reroute
#   4. If OSRM offline/fails -> straight-line bearing fallback (works offline)
#   5. Triage priority score per person
# ========================================================

# -- Global routing state (controlled from dashboard) ------
routing_state = {
    "mode":          "normal",   # "normal" | "disaster" | "flood"
    "blocked_zones": [],         # [{"lat","lon","radius_m","label","type"}]
    "manual_pins":   [],         # operator victim pins [{"lat","lon","label","floor"}]
}
routing_state_lock = threading.Lock()

OSRM_PROFILES = {
    "normal":   "driving",
    "disaster": "foot",
    "flood":    "foot",
}
OSRM_BASE_TMPL = "https://router.project-osrm.org/route/v1/{profile}"

# -- Blocked zone helpers ----------------------------------
def point_in_blocked_zone(lat, lon):
    with routing_state_lock:
        zones = list(routing_state["blocked_zones"])
    for z in zones:
        d = haversine(lat, lon, z["lat"], z["lon"]) * 1000
        if d <= z.get("radius_m", 50):
            return z.get("label", "Blocked zone")
    return None

def route_passes_blocked(polyline):
    with routing_state_lock:
        zones = list(routing_state["blocked_zones"])
    if not zones:
        return None
    for pt in polyline[::3]:
        for z in zones:
            d = haversine(pt[0], pt[1], z["lat"], z["lon"]) * 1000
            if d <= z.get("radius_m", 50):
                return z.get("label", "Blocked")
    return None

# -- Straight-line fallback (fully offline) ----------------
def straight_line_path(vlat, vlon, tlat, tlon):
    dist_km = haversine(vlat, vlon, tlat, tlon)
    steps_n = max(2, int(dist_km / 0.01))
    polyline = []
    for i in range(steps_n + 1):
        t   = i / steps_n
        lat = vlat + (tlat - vlat) * t
        lon = vlon + (tlon - vlon) * t
        polyline.append([lat, lon])

    dlon_r  = math.radians(tlon - vlon)
    y       = math.sin(dlon_r) * math.cos(math.radians(tlat))
    x       = (math.cos(math.radians(vlat)) * math.sin(math.radians(tlat))
               - math.sin(math.radians(vlat)) * math.cos(math.radians(tlat)) * math.cos(dlon_r))
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
    dirs    = ["N","NE","E","SE","S","SW","W","NW"]
    compass = dirs[round(bearing / 45) % 8]

    walk_speed = 3.0  # km/h rescuer on foot through rubble
    eta_min    = round((dist_km / walk_speed) * 60, 1)

    return {
        "polyline":     polyline,
        "distance_km":  round(dist_km, 3),
        "duration_min": eta_min,
        "mode":         "straight-line",
        "bearing_deg":  round(bearing, 1),
        "compass":      compass,
        "warning":      "OFFLINE STRAIGHT-LINE PATH - roads unknown. Follow compass bearing.",
        "steps": [
            {"instruction": f"Head {compass} ({round(bearing)}deg) toward victim",
             "distance_m": round(dist_km * 1000),
             "maneuver": "depart", "modifier": compass, "location": []},
            {"instruction": f"Arrive at victim ({round(dist_km*1000)} m)",
             "distance_m": 0, "maneuver": "arrive", "modifier": "", "location": [tlat, tlon]},
        ]
    }

# -- Triage priority score (0-100) -------------------------
def triage_score(conf, loc, nearest_km):
    score = 0.0
    score += conf * 35
    if loc:
        acc = loc.get("accuracy") or 999
        if   acc < 20:  score += 30
        elif acc < 50:  score += 22
        elif acc < 150: score += 12
        elif acc < 300: score += 5
    if nearest_km is not None:
        score += max(0, 35 - nearest_km * 70)
    return round(min(score, 100), 1)

# -- Main routing function ---------------------------------
def get_road_route(vlat, vlon, tlat, tlon, mode_override=None):
    with routing_state_lock:
        mode = mode_override or routing_state["mode"]
    profile = OSRM_PROFILES.get(mode, "driving")
    key     = (round(vlat,5), round(vlon,5), round(tlat,5), round(tlon,5), profile)

    with route_cache_lock:
        cached = route_cache.get(key)
    if cached and not route_passes_blocked(cached.get("polyline",[])):
        return cached

    try:
        base = OSRM_BASE_TMPL.format(profile=profile)
        url  = f"{base}/{vlon},{vlat};{tlon},{tlat}?overview=full&geometries=geojson&steps=true"
        resp = requests.get(url, timeout=8)
        data = resp.json()

        if data.get("code") != "Ok" or not data.get("routes"):
            print(f"[ROUTE] OSRM no result ({profile}), straight-line fallback")
            return straight_line_path(vlat, vlon, tlat, tlon)

        route    = data["routes"][0]
        leg      = route["legs"][0]
        polyline = [[c[1], c[0]] for c in route["geometry"]["coordinates"]]

        blocked_by = route_passes_blocked(polyline)
        if blocked_by:
            print(f"[ROUTE] Blocked by '{blocked_by}', trying foot reroute")
            if profile != "foot":
                alt = get_road_route(vlat, vlon, tlat, tlon, mode_override="disaster")
                if alt:
                    alt["warning"] = f"REROUTED on foot - original path blocked: {blocked_by}"
                    return alt
            sl = straight_line_path(vlat, vlon, tlat, tlon)
            sl["warning"] = f"ALL ROUTES BLOCKED: {blocked_by} - Follow compass bearing"
            return sl

        steps = []
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            mtype    = maneuver.get("type", "")
            modifier = maneuver.get("modifier", "")
            name     = step.get("name", "")
            dist_m   = step.get("distance", 0)

            if   mtype == "depart":                instr = f"Head {modifier}" + (f" on {name}" if name else "")
            elif mtype == "arrive":                instr = "Arrive at destination"
            elif mtype == "turn":                  instr = f"Turn {modifier}" + (f" onto {name}" if name else "")
            elif mtype in ("roundabout","rotary"): instr = f"Take exit {maneuver.get('exit','')} at roundabout" + (f" onto {name}" if name else "")
            elif mtype == "fork":                  instr = f"Keep {modifier}" + (f" on {name}" if name else "")
            elif mtype == "merge":                 instr = f"Merge {modifier}" + (f" onto {name}" if name else "")
            elif mtype == "new name":              instr = f"Continue onto {name}" if name else "Continue straight"
            else:
                instr = mtype.replace("-"," ").title()
                if modifier: instr += f" {modifier}"
                if name:     instr += f" {name}"

            if dist_m > 5:
                steps.append({"instruction": instr, "distance_m": round(dist_m),
                               "maneuver": mtype, "modifier": modifier,
                               "location": maneuver.get("location", [])})

        result = {
            "polyline":     polyline,
            "distance_km":  round(route["distance"] / 1000, 3),
            "duration_min": round(route["duration"]  / 60,  1),
            "mode":         profile,
            "warning":      None,
            "steps":        steps,
        }
        with route_cache_lock:
            route_cache[key] = result
        return result

    except Exception as e:
        print(f"[ROUTE ERR] {e} - straight-line fallback")
        return straight_line_path(vlat, vlon, tlat, tlon)


# ========================================================
#  Haversine + team ranking
# ========================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def rank_teams(vlat, vlon):
    results = [{**t, "distance_km": round(haversine(vlat, vlon, t["lat"], t["lon"]), 4)}
               for t in RESCUE_TEAMS]
    return sorted(results, key=lambda x: x["distance_km"])


# ========================================================
#  MJPEG generator per camera
# ========================================================
def make_frame_generator(cam):
    cid = cam["id"]
    st  = cam_state[cid]

    def offline_img():
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (0,0),(640,360),(0,20,10),-1)
        cv2.putText(img, f"CAM {cid} — OFFLINE",    (150,150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,60,180), 2)
        cv2.putText(img, cam["name"],                (160,195), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,50,140), 1)
        cv2.putText(img, f"IP: {cam['ip']}:8080",   (180,230), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(40,40,40), 1)
        cv2.putText(img, "Start IP Webcam on phone", (155,265), cv2.FONT_HERSHEY_SIMPLEX, 0.50,(30,30,30), 1)
        _, b = cv2.imencode('.jpg', img)
        return b.tobytes()

    def generate():
        # Wait up to 10s for first frame
        for _ in range(100):
            with st["frame_lock"]:
                if st["frame"] is not None:
                    break
            time.sleep(0.1)

        while True:
            with st["frame_lock"]:
                raw = st["frame"]

            if raw is None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + offline_img() + b'\r\n')
                time.sleep(0.5)
                continue

            nparr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                time.sleep(0.05)
                continue

            with st["gps_lock"]:
                gps = dict(st["gps"])

            with model_lock:
                results = model(frame, classes=[PERSON_CLASS],
                                conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                                verbose=False, agnostic_nms=True)

            persons = []
            pid = 1
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) != PERSON_CLASS:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    loc = nearest = route = None
                    ranked = []

                    if gps["valid"]:
                        loc     = {"lat": gps["lat"], "lon": gps["lon"],
                                   "alt": gps["alt"], "accuracy": gps["accuracy"], "source": gps["source"]}
                        ranked  = rank_teams(gps["lat"], gps["lon"])
                        nearest = ranked[0] if ranked else None
                        if nearest:
                            # Shortest road route to nearest team
                            route = get_road_route(gps["lat"], gps["lon"],
                                                   nearest["lat"], nearest["lon"])
                        # All 4 team routes for this person
                        all_person_routes = []
                        for i, team in enumerate(ranked):
                            rt = get_road_route(gps["lat"], gps["lon"],
                                                team["lat"], team["lon"])
                            all_person_routes.append({
                                "team_id":   team["id"],
                                "team_name": team["name"],
                                "rank":      i + 1,
                                "is_nearest": (i == 0),
                                "distance_km": team["distance_km"],
                                "route": rt,
                            })
                    else:
                        all_person_routes = []

                    # Triage priority score
                    nearest_km = nearest["distance_km"] if nearest else None
                    priority   = triage_score(conf, loc, nearest_km)

                    # Route warning (blocked zone / offline fallback)
                    route_warn = route.get("warning") if route else None
                    route_mode = route.get("mode", "driving") if route else "none"

                    persons.append({"id": pid, "cam_id": cid, "conf": round(conf, 2),
                                    "bbox": [x1,y1,x2,y2], "loc": loc,
                                    "ranked": ranked, "nearest": nearest, "route": route,
                                    "all_routes": all_person_routes,
                                    "priority": priority,
                                    "route_mode": route_mode,
                                    "route_warning": route_warn})
                    # Log to CSV
                    log_person(cid, cam["name"], pid, conf, loc, nearest, route)

                    col = (0,255,80) if loc else (0,200,255)
                    cv2.rectangle(frame, (x1,y1),(x2,y2), col, 2)
                    lbl = f"P{pid} {conf:.0%}"
                    ly  = max(y1-18, 0)
                    cv2.rectangle(frame, (x1,ly),(x1+len(lbl)*8,y1), col,-1)
                    cv2.putText(frame, lbl, (x1+2, max(y1-4,10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0,0,0), 1)
                    if loc:
                        cv2.putText(frame, f"{loc['lat']:.5f},{loc['lon']:.5f}",
                                    (x1, min(y2+13, frame.shape[0]-24)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0,220,255), 1)
                    if nearest:
                        rd = f"{route['distance_km']}km" if route else f"{nearest['distance_km']}km"
                        cv2.putText(frame, f"->{nearest['name'].split()[0]} {rd}",
                                    (x1, min(y2+24, frame.shape[0]-12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,140,0), 1)
                    pid += 1

            # HUD bar
            h, w = frame.shape[:2]
            cv2.rectangle(frame,(0,0),(w,18),(0,0,0),-1)
            cv2.putText(frame, f"[{cam['name']}]",
                        (4,13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,(0,255,65),1)
            gps_txt = (f"{gps['lat']:.5f},{gps['lon']:.5f}" if gps["valid"] else "GPS:ACQUIRING")
            cv2.putText(frame, gps_txt, (w//2-60,13), cv2.FONT_HERSHEY_SIMPLEX, 0.32,(0,200,255),1)
            pc = (0,255,65) if persons else (80,80,80)
            cv2.putText(frame, f"P:{len(persons)}", (w-38,13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, pc, 1)

            with st["persons_lock"]:
                st["persons"] = persons

            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    return generate


# ========================================================
# Flask routes
# ========================================================
@app.route("/")
def home():
    # Serve as plain static file — bypasses Jinja2 template engine
    # This avoids Jinja2 misinterpreting JS {{ }} syntax as template variables
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    return send_from_directory(templates_dir, "index.html")

@app.route("/video_feed/<int:cam_id>")
def video_feed(cam_id):
    cam = next((c for c in CAMERAS if c["id"] == cam_id), None)
    if not cam:
        return "Not found", 404
    return Response(make_frame_generator(cam)(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/rescue_teams")
def rescue_teams_api():
    all_persons  = []
    cam_statuses = []

    for cam in CAMERAS:
        cid = cam["id"]
        st  = cam_state[cid]
        with st["gps_lock"]:
            gps = dict(st["gps"])
        with st["persons_lock"]:
            persons = list(st["persons"])
        cam_statuses.append({
            "id": cid, "name": cam["name"], "ip": cam["ip"],
            "online": st["online"], "gps": gps,
            "person_count": len(persons),
        })
        all_persons.extend(persons)

    # Best GPS = first camera with a valid fix
    global_gps = {"valid": False}
    for cam in CAMERAS:
        with cam_state[cam["id"]]["gps_lock"]:
            g = dict(cam_state[cam["id"]]["gps"])
        if g["valid"]:
            global_gps = g
            break

    global_ranked  = rank_teams(global_gps["lat"], global_gps["lon"]) if global_gps["valid"] else []
    global_nearest = global_ranked[0] if global_ranked else None

    all_routes = []
    if global_gps["valid"]:
        for i, team in enumerate(global_ranked):
            route = get_road_route(global_gps["lat"], global_gps["lon"],
                                   team["lat"], team["lon"])
            all_routes.append({
                "team_id":    team["id"],
                "team_name":  team["name"],
                "team_loc":   team.get("loc",""),
                "rank":       i + 1,
                "is_nearest": (i == 0),
                "distance_km": team["distance_km"],
                "route":      route,
            })

    global_route = all_routes[0]["route"] if all_routes else None

    return jsonify({
        "gps":            global_gps,
        "cameras":        cam_statuses,
        "persons":        all_persons,
        "person_count":   len(all_persons),
        "rescue_teams":   RESCUE_TEAMS,
        "global_ranked":  global_ranked,
        "global_nearest": global_nearest,
        "global_route":   global_route,
        "all_routes":     all_routes,
    })

@app.route("/cam_status")
def cam_status():
    out = []
    for cam in CAMERAS:
        cid = cam["id"]
        st  = cam_state[cid]
        with st["gps_lock"]:    gps = dict(st["gps"])
        with st["persons_lock"]: pc = len(st["persons"])
        out.append({"id":cid,"name":cam["name"],"ip":cam["ip"],
                    "online":st["online"],"gps":gps,"person_count":pc})
    return jsonify(out)


# ========================================================
#  DATA / HISTORY API ENDPOINTS
# ========================================================

@app.route("/data/log")
def data_log():
    """Return last 500 rows of both logs for live dashboard table."""
    with csv_lock:
        p = list(persons_log)
        g = list(gps_log)
    return jsonify({
        "persons_log": list(reversed(p)),   # newest first
        "gps_log":     list(reversed(g)),
        "persons_total": len(p),
        "gps_total":     len(g),
    })

@app.route("/data/download/persons")
def download_persons():
    """Download detected_persons.csv file directly."""
    if os.path.isfile(PERSONS_CSV):
        return send_file(PERSONS_CSV, as_attachment=True,
                         download_name=f"detected_persons_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                         mimetype="text/csv")
    # Empty file fallback
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp","cam_id","cam_name","person_id","confidence_pct",
                "latitude","longitude","nearest_team","road_distance_km","eta_min"])
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     as_attachment=True,
                     download_name="detected_persons_empty.csv",
                     mimetype="text/csv")

@app.route("/data/download/gps")
def download_gps():
    """Download gps_history.csv file directly."""
    if os.path.isfile(GPS_CSV):
        return send_file(GPS_CSV, as_attachment=True,
                         download_name=f"gps_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                         mimetype="text/csv")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp","cam_id","cam_name","latitude","longitude",
                "altitude_m","accuracy_m","gps_source"])
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     as_attachment=True,
                     download_name="gps_history_empty.csv",
                     mimetype="text/csv")

@app.route("/data/clear", methods=["POST"])
def data_clear():
    """Clear in-memory logs + reset CSV files (keeps headers)."""
    global persons_log, gps_log, _person_seen, _gps_seen
    with csv_lock:
        persons_log  = []
        gps_log      = []
        _person_seen = {}
        _gps_seen    = {}
        # Rewrite CSV files with just the header
        with open(PERSONS_CSV, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                "timestamp","date","time","cam_id","cam_name","person_id",
                "confidence_pct","latitude","longitude","altitude_m",
                "accuracy_m","gps_source","nearest_team","road_distance_km","eta_min"
            ])
        with open(GPS_CSV, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                "timestamp","date","time","cam_id","cam_name",
                "latitude","longitude","altitude_m","accuracy_m","gps_source"
            ])
    return jsonify({"status": "cleared"})

@app.route("/data/stats")
def data_stats():
    """Quick stats: total rows, first/last timestamp in each CSV."""
    def csv_info(path):
        if not os.path.isfile(path):
            return {"rows": 0, "first": None, "last": None}
        rows = 0
        first = last = None
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows += 1
                ts = row.get("timestamp","")
                if ts:
                    if first is None: first = ts
                    last = ts
        return {"rows": rows, "first": first, "last": last}
    return jsonify({
        "persons": csv_info(PERSONS_CSV),
        "gps":     csv_info(GPS_CSV),
        "data_dir": DATA_DIR,
    })


# ========================================================
# DISASTER CONTROL API ROUTES
# ========================================================

@app.route("/disaster/mode", methods=["GET","POST"])
def disaster_mode():
    from flask import request
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        new_mode = data.get("mode","normal")
        if new_mode in ("normal","disaster","flood"):
            with routing_state_lock:
                routing_state["mode"] = new_mode
            # Clear route cache so routes recalculate with new mode
            with route_cache_lock:
                route_cache.clear()
            print(f"[DISASTER] Mode set to: {new_mode}")
    with routing_state_lock:
        return jsonify({"mode": routing_state["mode"],
                        "blocked_zones": routing_state["blocked_zones"],
                        "manual_pins":   routing_state["manual_pins"]})

@app.route("/disaster/block", methods=["POST"])
def add_blocked_zone():
    from flask import request
    data = request.get_json(force=True) or {}
    zone = {
        "lat":       float(data.get("lat", 0)),
        "lon":       float(data.get("lon", 0)),
        "radius_m":  float(data.get("radius_m", 50)),
        "label":     data.get("label", "Blocked"),
        "type":      data.get("type", "rubble"),  # rubble | flood | fire | collapse
        "id":        data.get("id", str(time.time())),
    }
    with routing_state_lock:
        routing_state["blocked_zones"].append(zone)
    with route_cache_lock:
        route_cache.clear()
    print(f"[DISASTER] Blocked zone added: {zone['label']} @ {zone['lat']:.5f},{zone['lon']:.5f}")
    return jsonify({"status":"ok","zone":zone,
                    "total_zones": len(routing_state["blocked_zones"])})

@app.route("/disaster/block/<zone_id>", methods=["DELETE"])
def remove_blocked_zone(zone_id):
    with routing_state_lock:
        before = len(routing_state["blocked_zones"])
        routing_state["blocked_zones"] = [
            z for z in routing_state["blocked_zones"] if z.get("id") != zone_id
        ]
        after = len(routing_state["blocked_zones"])
    with route_cache_lock:
        route_cache.clear()
    return jsonify({"status":"ok","removed": before - after})

@app.route("/disaster/block/clear", methods=["POST"])
def clear_blocked_zones():
    with routing_state_lock:
        routing_state["blocked_zones"] = []
    with route_cache_lock:
        route_cache.clear()
    return jsonify({"status":"ok"})

@app.route("/disaster/pin", methods=["POST"])
def add_manual_pin():
    """Operator places manual victim pin on map (for indoor/GPS-unavailable victims)."""
    from flask import request
    data = request.get_json(force=True) or {}
    pin = {
        "lat":   float(data.get("lat", 0)),
        "lon":   float(data.get("lon", 0)),
        "label": data.get("label", "Manual Victim"),
        "floor": data.get("floor", "Unknown"),
        "id":    data.get("id", str(time.time())),
    }
    with routing_state_lock:
        routing_state["manual_pins"].append(pin)
    print(f"[DISASTER] Manual pin added: {pin['label']} floor={pin['floor']}")
    return jsonify({"status":"ok","pin":pin})

@app.route("/disaster/pin/clear", methods=["POST"])
def clear_manual_pins():
    with routing_state_lock:
        routing_state["manual_pins"] = []
    return jsonify({"status":"ok"})

@app.route("/disaster/status")
def disaster_status():
    with routing_state_lock:
        return jsonify({
            "mode":          routing_state["mode"],
            "blocked_zones": routing_state["blocked_zones"],
            "manual_pins":   routing_state["manual_pins"],
            "route_cache_size": len(route_cache),
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
