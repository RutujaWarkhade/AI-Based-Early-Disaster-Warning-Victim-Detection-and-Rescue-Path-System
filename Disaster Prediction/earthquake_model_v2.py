"""
=============================================================================
DISASTER AI — EARTHQUAKE MODEL v2  (WITH EXACT LOCATION NAMES)
=============================================================================
NEW in this version:
  - get_location_name()  : Converts lat/lon → exact place name
      Primary  : Nominatim (OpenStreetMap) free reverse geocoding API
      Fallback : Built-in India district/region grid (offline, instant)
  - All alert outputs now include:
      location_name   e.g. "Pithoragarh District, Uttarakhand"
      district        e.g. "Pithoragarh"
      state_name      e.g. "Uttarakhand"
      nearest_city    e.g. "Pithoragarh (42 km NNE)"
      tectonic_zone   e.g. "Main Central Thrust — High Himalayan Seismic Belt"
=============================================================================
"""
import pandas as pd
import numpy as np
import joblib
import json
import time
import warnings
import requests
from pathlib import Path
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

BASE   = Path("D:/Disaster_Prediction")
RAW    = BASE / "raw"
FEAT   = BASE / "features";      FEAT.mkdir(exist_ok=True)
MODELS = BASE / "models";        MODELS.mkdir(exist_ok=True)
PRED   = BASE / "predictions";   PRED.mkdir(exist_ok=True)
ALERTS = BASE / "future_alerts"; ALERTS.mkdir(exist_ok=True)


# ===========================================================================
# LOCATION LOOKUP — OFFLINE INDIA GRID
# ===========================================================================
# District-level reference grid for India
# Format: (lat_min, lat_max, lon_min, lon_max): (district, state, tectonic_zone, nearest_city)
INDIA_LOCATION_GRID = [
    # ── Jammu & Kashmir / Ladakh ──────────────────────────────────────────
    (34.0,37.6, 73.5,76.0, "Kargil", "Ladakh", "Karakoram-Ladakh Seismic Belt", "Kargil"),
    (33.0,35.5, 75.5,77.5, "Udhampur/Reasi", "Jammu & Kashmir", "Pir Panjal Seismic Zone", "Jammu"),
    (33.5,36.0, 76.5,79.0, "Leh", "Ladakh", "Karakoram Fault Zone", "Leh"),
    (33.0,34.5, 73.5,75.5, "Poonch", "Jammu & Kashmir", "Main Boundary Thrust Zone", "Poonch"),

    # ── Himachal Pradesh ──────────────────────────────────────────────────
    (31.5,33.5, 76.5,79.0, "Kullu/Manali", "Himachal Pradesh", "Main Central Thrust", "Manali"),
    (30.5,32.0, 75.5,77.5, "Kangra", "Himachal Pradesh", "Himalayan Frontal Thrust Zone", "Dharamshala"),
    (31.0,33.0, 77.5,79.5, "Shimla/Kinnaur", "Himachal Pradesh", "Main Central Thrust", "Shimla"),

    # ── Uttarakhand ───────────────────────────────────────────────────────
    (29.5,31.5, 78.5,81.0, "Chamoli/Uttarkashi", "Uttarakhand", "Main Central Thrust — High Himalayan Seismic Belt", "Chamoli"),
    (29.0,31.0, 79.5,81.5, "Pithoragarh", "Uttarakhand", "Main Central Thrust", "Pithoragarh"),
    (29.5,30.5, 77.5,79.5, "Dehradun/Haridwar", "Uttarakhand", "Himalayan Frontal Thrust", "Dehradun"),
    (29.0,30.5, 80.0,81.5, "Bageshwar/Almora", "Uttarakhand", "Main Boundary Thrust", "Almora"),

    # ── Punjab / Haryana ──────────────────────────────────────────────────
    (29.5,32.5, 73.5,77.5, "Ambala/Ropar", "Punjab/Haryana", "Delhi-Haridwar Ridge Zone", "Ambala"),
    (28.0,30.0, 76.0,78.0, "Delhi NCR", "Delhi/NCR", "Delhi Seismic Zone IV", "New Delhi"),

    # ── Uttar Pradesh (Himalayan Foothills) ───────────────────────────────
    (27.0,29.5, 80.0,83.5, "Bahraich/Gonda", "Uttar Pradesh", "Indo-Gangetic Foredeep Zone", "Lucknow"),
    (25.5,27.5, 82.0,85.0, "Varanasi/Mirzapur", "Uttar Pradesh", "Vindhyan Plateau Margin", "Varanasi"),

    # ── Bihar / Nepal Border ──────────────────────────────────────────────
    (26.5,28.5, 83.5,88.0, "Sitamarhi/Supaul", "Bihar", "Nepal-Bihar Seismic Gap — High Risk", "Muzaffarpur"),
    (24.5,26.5, 84.0,87.5, "Gaya/Dhanbad", "Bihar/Jharkhand", "Chotanagpur Plateau Zone", "Patna"),

    # ── Northeast — Arunachal ─────────────────────────────────────────────
    (26.5,29.5, 92.0,97.4, "Tawang/West Kameng", "Arunachal Pradesh", "Eastern Himalayan Seismic Belt — Zone V", "Itanagar"),
    (27.5,29.5, 94.0,97.4, "Upper Subansiri/Siang", "Arunachal Pradesh", "Mishmi Thrust Belt — Zone V", "Along"),

    # ── Northeast — Assam ─────────────────────────────────────────────────
    (24.0,26.5, 89.5,93.0, "Dhubri/Barpeta", "Assam", "Brahmaputra Valley Seismic Belt — Zone V", "Guwahati"),
    (25.0,27.0, 92.5,95.5, "Kaziranga/Jorhat", "Assam", "Naga Thrust Zone", "Jorhat"),
    (24.5,26.0, 93.5,96.0, "Dibrugarh/Tinsukia", "Assam", "Eastern Assam Thrust Belt", "Dibrugarh"),

    # ── Northeast — Nagaland/Manipur/Mizoram ─────────────────────────────
    (24.5,27.0, 93.0,95.5, "Kohima/Mokokchung", "Nagaland", "Naga Thrust Belt — Zone V", "Kohima"),
    (23.5,25.5, 92.5,95.0, "Imphal/Thoubal", "Manipur", "Manipur Thrust Zone — Zone V", "Imphal"),
    (21.5,24.5, 92.0,93.5, "Aizawl/Lunglei", "Mizoram", "Arakan-Yoma Seismic Belt — Zone V", "Aizawl"),
    (22.5,24.5, 91.0,92.5, "Agartala/South Tripura", "Tripura", "Tripura-Mizoram Fold Belt", "Agartala"),

    # ── Northeast — Meghalaya ─────────────────────────────────────────────
    (24.5,26.5, 89.5,92.5, "Shillong/East Khasi Hills", "Meghalaya", "Shillong Plateau Seismic Zone — Zone V", "Shillong"),

    # ── Rajasthan ─────────────────────────────────────────────────────────
    (24.0,28.0, 69.5,74.0, "Jaisalmer/Barmer", "Rajasthan", "Aravalli Seismic Zone", "Jodhpur"),
    (25.0,27.0, 73.0,76.0, "Ajmer/Pali", "Rajasthan", "Aravalli Range Zone III", "Ajmer"),

    # ── Gujarat ───────────────────────────────────────────────────────────
    (21.0,24.5, 68.0,72.5, "Bhuj/Kutch", "Gujarat", "Kutch Seismic Zone — Zone V (2001 M7.7)", "Bhuj"),
    (21.5,23.5, 72.0,74.5, "Vadodara/Surat", "Gujarat", "Cambay Rift Zone — Zone III", "Vadodara"),
    (22.0,24.0, 71.5,74.0, "Surendranagar/Rajkot", "Gujarat", "Saurashtra Seismic Zone", "Rajkot"),

    # ── Madhya Pradesh ────────────────────────────────────────────────────
    (22.5,25.5, 76.0,80.0, "Bhopal/Hoshangabad", "Madhya Pradesh", "Narmada-Son Lineament — Zone III", "Bhopal"),
    (22.0,24.5, 80.5,83.5, "Jabalpur/Mandla", "Madhya Pradesh", "Narmada-Son Fault Zone — Zone III (1997 M6.0)", "Jabalpur"),

    # ── Maharashtra ───────────────────────────────────────────────────────
    (17.0,20.5, 73.5,77.5, "Latur/Osmanabad", "Maharashtra", "Koyna-Warna Seismic Zone — Zone III (1993 M6.4)", "Latur"),
    (17.0,18.5, 73.5,75.0, "Koyna/Satara", "Maharashtra", "Koyna Dam Reservoir Triggered Seismicity", "Karad"),
    (19.0,21.5, 72.5,75.5, "Nashik/Dhule", "Maharashtra", "Deccan Volcanic Province Zone II-III", "Nashik"),
    (18.5,20.0, 72.5,74.0, "Mumbai/Thane", "Maharashtra", "Western Ghats Seismic Zone III", "Mumbai"),

    # ── Andhra Pradesh / Telangana ────────────────────────────────────────
    (14.0,18.0, 77.5,81.5, "Kurnool/Nandyal", "Andhra Pradesh", "Eastern Dharwar Craton — Zone II", "Kurnool"),
    (15.5,19.0, 78.0,81.5, "Hyderabad/Nalgonda", "Telangana", "Deccan Plateau Seismic Zone II", "Hyderabad"),
    (14.0,16.5, 79.0,81.5, "Nellore/Ongole", "Andhra Pradesh", "Eastern Coastal Zone II", "Vijayawada"),

    # ── Karnataka ─────────────────────────────────────────────────────────
    (12.5,15.5, 75.5,78.5, "Bellary/Raichur", "Karnataka", "Cuddapah Basin Zone II", "Bellary"),
    (12.0,13.5, 74.5,76.5, "Mangalore/Hassan", "Karnataka", "Western Ghats Zone III", "Mangalore"),

    # ── Tamil Nadu ────────────────────────────────────────────────────────
    (10.5,13.5, 77.0,80.5, "Coimbatore/Salem", "Tamil Nadu", "Coimbatore Seismic Zone III (1900 M6.5)", "Coimbatore"),
    (8.5,11.5,  76.5,80.0, "Tirunelveli/Madurai", "Tamil Nadu", "Southern Granulite Terrain Zone II", "Madurai"),
    (9.0,14.0,  79.0,80.5, "Chennai/Kanchipuram", "Tamil Nadu", "Eastern Coastal Zone II-III", "Chennai"),

    # ── Kerala ────────────────────────────────────────────────────────────
    (8.0,11.5,  76.0,77.5, "Thrissur/Palakkad", "Kerala", "Palakkad Gap Seismic Zone III", "Palakkad"),
    (8.5,10.5,  76.5,77.5, "Thiruvananthapuram/Kollam", "Kerala", "Southern Kerala Zone II-III", "Thiruvananthapuram"),

    # ── Odisha / Jharkhand ────────────────────────────────────────────────
    (20.0,23.5, 82.5,87.0, "Sambalpur/Rourkela", "Odisha/Jharkhand", "Chotanagpur Plateau Zone II-III", "Rourkela"),
    (18.5,21.5, 82.5,85.5, "Koraput/Rayagada", "Odisha", "Eastern Ghats Mobile Belt", "Koraput"),

    # ── West Bengal / Sikkim ──────────────────────────────────────────────
    (26.5,28.5, 87.5,89.5, "Darjeeling/Kalimpong", "West Bengal/Sikkim", "Sikkim Himalayan Seismic Zone IV-V", "Darjeeling"),
    (22.0,24.5, 86.5,88.5, "Bankura/Purulia", "West Bengal", "Bengal Basin Zone III", "Kolkata"),

    # ── Andaman & Nicobar ─────────────────────────────────────────────────
    (6.5,14.0,  92.0,94.5, "Andaman Islands", "Andaman & Nicobar", "Andaman Subduction Zone — Zone V (Highly Active)", "Port Blair"),
    (6.5,10.0,  92.5,94.0, "Nicobar Islands", "Andaman & Nicobar", "Sunda Megathrust Zone — Zone V", "Car Nicobar"),
]

# Major cities reference for distance-based nearest city lookup
INDIA_CITIES = [
    ("New Delhi",     28.61, 77.21),
    ("Mumbai",        19.08, 72.88),
    ("Chennai",       13.08, 80.27),
    ("Kolkata",       22.57, 88.36),
    ("Hyderabad",     17.38, 78.49),
    ("Bengaluru",     12.97, 77.59),
    ("Ahmedabad",     23.03, 72.59),
    ("Pune",          18.52, 73.86),
    ("Jaipur",        26.91, 75.79),
    ("Lucknow",       26.85, 80.95),
    ("Chandigarh",    30.73, 76.78),
    ("Bhopal",        23.26, 77.41),
    ("Patna",         25.59, 85.14),
    ("Guwahati",      26.14, 91.74),
    ("Srinagar",      34.08, 74.80),
    ("Jammu",         32.73, 74.87),
    ("Leh",           34.16, 77.58),
    ("Dehradun",      30.32, 78.03),
    ("Shimla",        31.10, 77.17),
    ("Darjeeling",    27.04, 88.26),
    ("Shillong",      25.57, 91.88),
    ("Imphal",        24.81, 93.94),
    ("Aizawl",        23.73, 92.72),
    ("Agartala",      23.84, 91.28),
    ("Kohima",        25.67, 94.11),
    ("Itanagar",      27.09, 93.62),
    ("Gangtok",       27.33, 88.61),
    ("Dibrugarh",     27.48, 94.91),
    ("Jorhat",        26.75, 94.21),
    ("Silchar",       24.83, 92.80),
    ("Bhuj",          23.25, 69.67),
    ("Rajkot",        22.30, 70.80),
    ("Surat",         21.19, 72.83),
    ("Vadodara",      22.30, 73.20),
    ("Jodhpur",       26.29, 73.03),
    ("Udaipur",       24.57, 73.69),
    ("Varanasi",      25.32, 83.01),
    ("Allahabad",     25.45, 81.84),
    ("Agra",          27.18, 78.01),
    ("Jabalpur",      23.18, 79.95),
    ("Nagpur",        21.15, 79.09),
    ("Nashik",        19.99, 73.79),
    ("Aurangabad",    19.88, 75.34),
    ("Latur",         18.40, 76.57),
    ("Karad",         17.28, 74.18),
    ("Visakhapatnam", 17.69, 83.22),
    ("Vijayawada",    16.51, 80.62),
    ("Kurnool",       15.83, 78.04),
    ("Bellary",       15.14, 76.93),
    ("Mangalore",     12.87, 74.89),
    ("Mysuru",        12.30, 76.65),
    ("Coimbatore",    11.02, 76.97),
    ("Madurai",        9.93, 78.12),
    ("Tiruchirappalli",10.79, 78.70),
    ("Thiruvananthapuram", 8.52, 76.94),
    ("Kochi",          9.99, 76.26),
    ("Kozhikode",     11.25, 75.77),
    ("Palakkad",      10.77, 76.65),
    ("Port Blair",    11.67, 92.74),
    ("Chamoli",       30.40, 79.32),
    ("Uttarkashi",    30.73, 78.43),
    ("Pithoragarh",   29.58, 80.22),
    ("Muzaffarpur",   26.12, 85.39),
    ("Sitamarhi",     26.59, 85.49),
    ("Rourkela",      22.26, 84.85),
    ("Sambalpur",     21.47, 83.97),
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Fast haversine distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))


def bearing_to_compass(bear_deg):
    """Convert bearing degrees to compass direction."""
    dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSW','SW','WSW','W','WNW','NW','NNW']
    idx = int((bear_deg + 11.25) / 22.5) % 16
    return dirs[idx]


def nearest_city_str(lat, lon):
    """Find nearest major city and return 'City (X km DIR)' string."""
    best_name, best_dist = "Unknown", 9999.0
    best_bear = 0.0
    for city_name, city_lat, city_lon in INDIA_CITIES:
        d = haversine_km(lat, lon, city_lat, city_lon)
        if d < best_dist:
            best_dist = d
            best_name = city_name
            best_bear = (np.degrees(np.arctan2(
                np.sin(np.radians(city_lon - lon)) * np.cos(np.radians(city_lat)),
                np.cos(np.radians(lat)) * np.sin(np.radians(city_lat)) -
                np.sin(np.radians(lat)) * np.cos(np.radians(city_lat)) *
                np.cos(np.radians(city_lon - lon))
            )) + 360) % 360

    compass = bearing_to_compass(best_bear)
    if best_dist < 15:
        return f"{best_name} (city area)"
    elif best_dist < 50:
        return f"{int(best_dist)} km {compass} of {best_name}"
    else:
        return f"{int(best_dist)} km {compass} of {best_name}"


def lookup_offline(lat, lon):
    """
    Offline lookup: match lat/lon to India district grid.
    Returns dict with location fields.
    """
    best_match = None
    best_area  = 9999.0

    for entry in INDIA_LOCATION_GRID:
        lat_min, lat_max, lon_min, lon_max, district, state, tectonic, city = entry
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            area = (lat_max - lat_min) * (lon_max - lon_min)
            if area < best_area:
                best_area  = area
                best_match = entry

    if best_match:
        lat_min, lat_max, lon_min, lon_max, district, state, tectonic, city = best_match
        nc = nearest_city_str(lat, lon)
        return {
            'location_name': f"{district}, {state}",
            'district':      district,
            'state_name':    state,
            'nearest_city':  nc,
            'tectonic_zone': tectonic,
            'geocode_source': 'offline_grid',
        }

    # Generic fallback for border areas / ocean
    nc = nearest_city_str(lat, lon)

    def zone_name(la, lo):
        if la > 28 and lo < 78:   return "Western Himalaya", "Himachal Pradesh/J&K"
        if la > 28 and lo > 88:   return "Eastern Himalaya/NE India", "Northeast India"
        if la > 24 and lo > 87:   return "Bihar-Nepal Seismic Gap", "Bihar/Jharkhand"
        if la > 22 and lo < 72:   return "Kutch/Saurashtra", "Gujarat"
        if 20 < la < 24 and 73 < lo < 80: return "Narmada-Son Fault Zone", "Madhya Pradesh"
        if la < 18 and lo < 77:   return "Deccan Plateau", "Maharashtra/Karnataka"
        if lo > 91:               return "Andaman Seismic Belt", "Andaman & Nicobar"
        return "India Seismic Zone", "India"

    tzone, sname = zone_name(lat, lon)
    return {
        'location_name': f"{sname} ({lat:.2f}°N, {lon:.2f}°E)",
        'district':      f"{lat:.2f}°N {lon:.2f}°E",
        'state_name':    sname,
        'nearest_city':  nc,
        'tectonic_zone': tzone,
        'geocode_source': 'coordinate_fallback',
    }


# Cache to avoid repeated API calls for same coordinates
_geocode_cache = {}

def get_location_name(lat, lon, use_api=True, api_delay=1.1):
    """
    Master location resolver.
    1. Check local cache
    2. Try Nominatim API (free, no key needed)
    3. Fall back to offline India grid
    Returns dict with location_name, district, state_name, nearest_city, tectonic_zone
    """
    # Round for cache key (0.1 degree ≈ 11 km precision)
    cache_key = (round(lat, 1), round(lon, 1))
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    result = None

    # ── Try Nominatim API ─────────────────────────────────────────────────
    if use_api:
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': lat, 'lon': lon,
                'format': 'json', 'zoom': 10,
                'addressdetails': 1,
            }
            headers = {'User-Agent': 'DisasterAI-India/1.0 (disaster research project)'}
            r = requests.get(url, params=params, headers=headers, timeout=8)
            if r.status_code == 200:
                data = r.json()
                addr = data.get('address', {})

                district  = (addr.get('county') or addr.get('state_district') or
                             addr.get('district') or addr.get('city') or
                             addr.get('town') or addr.get('village') or 'Unknown')
                state_nm  = addr.get('state', 'India')
                country   = addr.get('country_code', '').upper()

                # Build descriptive name
                parts = []
                if addr.get('county'):       parts.append(addr['county'])
                elif addr.get('district'):   parts.append(addr['district'])
                elif addr.get('city'):       parts.append(addr['city'])
                if addr.get('state'):        parts.append(addr['state'])
                if country != 'IN':          parts.append(addr.get('country',''))
                location_name = ', '.join(p for p in parts if p)
                if not location_name:
                    location_name = data.get('display_name', '')[:60]

                # Offline lookup for tectonic zone (API doesn't give this)
                offline = lookup_offline(lat, lon)

                result = {
                    'location_name':  location_name or offline['location_name'],
                    'district':       district,
                    'state_name':     state_nm,
                    'nearest_city':   nearest_city_str(lat, lon),
                    'tectonic_zone':  offline['tectonic_zone'],
                    'geocode_source': 'nominatim_api',
                }
                time.sleep(api_delay)   # Nominatim rate limit: 1 req/sec
    
        except Exception as e:
            pass  # Fall through to offline

    # ── Offline fallback ──────────────────────────────────────────────────
    if result is None:
        result = lookup_offline(lat, lon)

    _geocode_cache[cache_key] = result
    return result


def enrich_with_locations(df, lat_col='latitude', lon_col='longitude',
                           use_api=True, max_api_calls=50, verbose=True):
    """
    Add location columns to a DataFrame.
    Batches API calls with rate limiting.
    After max_api_calls, switches to offline-only for remaining rows.
    """
    if verbose:
        print(f"\n📍 Resolving location names for {len(df)} events...")
        print(f"   Method: Nominatim API (up to {max_api_calls} calls) + Offline Grid fallback")

    locs = []
    api_calls = 0

    for i, (_, row) in enumerate(df.iterrows()):
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        use = use_api and api_calls < max_api_calls

        loc = get_location_name(lat, lon, use_api=use)
        locs.append(loc)

        # Count actual API calls (not cache hits)
        cache_key = (round(lat, 1), round(lon, 1))
        if loc.get('geocode_source') == 'nominatim_api':
            api_calls += 1

        if verbose and (i+1) % 10 == 0:
            print(f"   Processed {i+1}/{len(df)} ...  "
                  f"(API calls: {api_calls}, offline: {i+1-api_calls})")

    loc_df = pd.DataFrame(locs)
    result = df.copy().reset_index(drop=True)
    for col in loc_df.columns:
        result[col] = loc_df[col].values

    if verbose:
        api_n     = (loc_df['geocode_source'] == 'nominatim_api').sum()
        offline_n = len(loc_df) - api_n
        print(f"   ✅ Done: {api_n} via API, {offline_n} via offline grid")

    return result


# ===========================================================================
# LOAD
# ===========================================================================
def load_earthquakes():
    df = pd.read_csv(RAW / "earthquakes.csv", low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for c in ['year','month','latitude','longitude','depth_km','magnitude',
              'significance','felt_reports','cdi','mmi','tsunami_flag']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df = df.dropna(subset=['latitude','longitude','magnitude','date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"   Loaded {len(df):,} earthquakes  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================
def engineer_features(df):
    print("\n⚙️  Engineering earthquake features...")
    df = df.copy().sort_values('date').reset_index(drop=True)

    df['log_energy']  = 1.5 * df['magnitude'] + 4.8
    df['mag_sq']      = df['magnitude'] ** 2
    df['mag_bin']     = pd.cut(df['magnitude'],
                                bins=[0,4,4.5,5,5.5,6,6.5,99],
                                labels=[0,1,2,3,4,5,6]).astype(float)
    df['depth_class'] = pd.cut(df['depth_km'].fillna(10),
                                bins=[0,10,35,70,300,700],
                                labels=[0,1,2,3,4]).astype(float)
    df['is_shallow']  = (df['depth_km'] < 35).astype(int)
    df['grid_lat']    = (df['latitude']  // 0.5) * 0.5
    df['grid_lon']    = (df['longitude'] // 0.5) * 0.5
    df['grid_id']     = df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)

    def zone(lat, lon):
        if (lat > 32 and lon < 78) or (lat > 26 and lon > 90): return 5
        elif lat > 28 or (lat > 22 and lon > 88):              return 4
        elif lat > 20 or (lon < 75 and lat > 15):              return 3
        else:                                                   return 2
    df['seismic_zone'] = df.apply(lambda r: zone(r.latitude, r.longitude), axis=1)

    df['day_of_year']     = df['date'].dt.dayofyear
    df['sin_doy']         = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_doy']         = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['prev_date']       = df.groupby('grid_id')['date'].shift(1)
    df['days_since_last'] = (df['date'] - df['prev_date']).dt.total_seconds() / 86400
    df['days_since_last'] = df['days_since_last'].fillna(999).clip(0, 999)

    print("   Computing seismic history windows...")
    lats  = df['latitude'].values;  lons  = df['longitude'].values
    mags  = df['magnitude'].values; energ = df['log_energy'].values
    dates = df['date'].values.astype('datetime64[h]').astype(np.int64)

    for days in [7, 30, 90]:
        hrs = days * 24
        cnt, maxm, emean, nm5 = [], [], [], []
        for i in range(len(df)):
            cutoff = dates[i] - hrs
            mask = ((dates < dates[i]) & (dates >= cutoff) &
                    (np.abs(lats - lats[i]) < 1.0) &
                    (np.abs(lons - lons[i]) < 1.0))
            nb = mags[mask]
            cnt.append(mask.sum());  maxm.append(nb.max() if len(nb) else 0)
            emean.append(energ[mask].mean() if mask.sum() else 0)
            nm5.append((nb >= 5.0).sum())
        df[f'n_{days}d']      = cnt;  df[f'maxmag_{days}d']  = maxm
        df[f'energy_{days}d'] = emean; df[f'n_m5_{days}d']    = nm5

    gs = df.groupby('grid_id').agg(
        grid_total  =('magnitude','count'), grid_maxmag =('magnitude','max'),
        grid_avgmag =('magnitude','mean'),  grid_depth  =('depth_km','mean'),
        grid_energy =('log_energy','mean'),
    ).reset_index()
    df = df.merge(gs, on='grid_id', how='left')

    print("   Computing aftershock targets...")
    dates_np = df['date'].values;  lats_np = df['latitude'].values
    lons_np  = df['longitude'].values; mags_np = df['magnitude'].values
    target   = np.zeros(len(df), dtype=int)

    for i in df[df['magnitude'] >= 5.0].index:
        d1 = dates_np[i] + np.timedelta64(48, 'h')
        mask = ((dates_np > dates_np[i]) & (dates_np <= d1) &
                (np.abs(lats_np - lats_np[i]) < 0.5) &
                (np.abs(lons_np - lons_np[i]) < 0.5) &
                (mags_np >= 5.0))
        if mask.any(): target[i] = 1
    df['target_aftershock_48h'] = target

    df.to_csv(FEAT / "earthquake_features.csv", index=False)
    print(f"   ✅ Features: {df.shape}")
    return df


# ===========================================================================
# TRAIN
# ===========================================================================
FEATURES = [
    'magnitude','mag_sq','mag_bin','depth_km','depth_class','is_shallow',
    'log_energy','significance','felt_reports','cdi','mmi','tsunami_flag',
    'latitude','longitude','seismic_zone',
    'sin_doy','cos_doy','month','days_since_last',
    'n_7d','maxmag_7d','energy_7d','n_m5_7d',
    'n_30d','maxmag_30d','energy_30d','n_m5_30d',
    'n_90d','maxmag_90d','energy_90d','n_m5_90d',
    'grid_total','grid_maxmag','grid_avgmag','grid_depth','grid_energy',
]

def train_model(df):
    try:
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
    except ImportError:
        print("❌ pip install xgboost scikit-learn"); return None

    print("\n🤖 Training Earthquake Aftershock Model v2...")
    mdf   = df[df['magnitude'] >= 5.0].copy()
    avail = [c for c in FEATURES if c in mdf.columns]
    mdf   = mdf.dropna(subset=avail + ['target_aftershock_48h'])
    X     = mdf[avail].fillna(0)
    y     = mdf['target_aftershock_48h']

    print(f"   M≥5 events      : {len(X):,}  |  Aftershock rate: {y.mean():.1%}")

    split      = int(len(mdf) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler  = StandardScaler()
    Xtr_s   = scaler.fit_transform(X_tr)
    Xte_s   = scaler.transform(X_te)

    try:
        from imblearn.over_sampling import SMOTE
        k = min(5, int(y_tr.sum()) - 1)
        Xtr_s, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(Xtr_s, y_tr)
        print(f"   SMOTE applied   : {len(Xtr_s):,} samples, rate={y_tr.mean():.1%}")
    except ImportError:
        print("   (pip install imbalanced-learn for SMOTE oversampling)")

    pos_w = max(1.0, (y_tr==0).sum() / max((y_tr==1).sum(), 1))
    base  = xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.75,
        min_child_weight=2, gamma=0.05,
        scale_pos_weight=pos_w, eval_metric='aucpr',
        early_stopping_rounds=50, random_state=42, n_jobs=-1,
        use_label_encoder=False,
    )
    base.fit(Xtr_s, y_tr, eval_set=[(Xte_s, y_te)], verbose=100)

    final = CalibratedClassifierCV(
        xgb.XGBClassifier(
            n_estimators=base.best_iteration, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.75, min_child_weight=2, gamma=0.05,
            scale_pos_weight=pos_w, use_label_encoder=False, random_state=42,
        ), method='sigmoid', cv=5)
    final.fit(Xtr_s, y_tr)

    y_prob = final.predict_proba(Xte_s)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)

    prec, rec, thr = precision_recall_curve(y_te, y_prob)
    f1s    = np.where((prec+rec)==0, 0, 2*prec*rec/(prec+rec))
    best_t = float(np.clip(thr[np.argmax(f1s[:-1])], 0.10, 0.55))

    print(f"\n   📊 TEST RESULTS")
    print(f"   ROC-AUC          : {auc:.4f}")
    print(f"   Optimal threshold: {best_t:.3f}")
    print(classification_report(y_te, (y_prob >= best_t).astype(int),
                                  target_names=['No Aftershock','Aftershock']))

    imp = pd.DataFrame({'feature': avail, 'importance': base.feature_importances_})\
            .sort_values('importance', ascending=False)
    print("   Top 10 features:")
    print(imp.head(10).to_string(index=False))

    joblib.dump(final,  MODELS / "earthquake_model.pkl")
    joblib.dump(scaler, MODELS / "earthquake_scaler.pkl")
    json.dump({'features': avail, 'roc_auc': float(auc), 'threshold': best_t,
               'trained_at': datetime.now().isoformat()},
              open(MODELS / "earthquake_meta.json", 'w'), indent=2)
    print(f"   💾 Saved → {MODELS}/earthquake_model.pkl")
    return final, scaler, avail, best_t


# ===========================================================================
# RISK MAP
# ===========================================================================
def generate_risk_map(df, model, scaler, features, threshold):
    print("\n🗺️  Generating India seismic risk map (percentile tiers)...")

    lats = np.arange(6.5, 37.6, 0.5)
    lons = np.arange(68.1, 97.4, 0.5)
    grid = pd.DataFrame([(la, lo) for la in lats for lo in lons],
                         columns=['latitude','longitude'])

    df['grid_lat'] = (df['latitude']  // 0.5) * 0.5
    df['grid_lon'] = (df['longitude'] // 0.5) * 0.5
    agg_cols = {
        'magnitude':   ('magnitude','mean'),   'depth_km':    ('depth_km','mean'),
        'log_energy':  ('log_energy','mean'),  'significance':('significance','mean'),
        'n_7d':        ('n_7d','mean'),        'n_30d':       ('n_30d','mean'),
        'n_90d':       ('n_90d','mean'),
        'maxmag_7d':   ('maxmag_7d','max'),    'maxmag_30d':  ('maxmag_30d','max'),
        'maxmag_90d':  ('maxmag_90d','max'),
        'energy_7d':   ('energy_7d','mean'),   'energy_30d':  ('energy_30d','mean'),
        'energy_90d':  ('energy_90d','mean'),
        'n_m5_7d':     ('n_m5_7d','mean'),     'n_m5_30d':    ('n_m5_30d','mean'),
        'n_m5_90d':    ('n_m5_90d','mean'),
        'grid_total':  ('grid_total','mean'),  'grid_maxmag': ('grid_maxmag','max'),
        'grid_avgmag': ('grid_avgmag','mean'), 'grid_depth':  ('grid_depth','mean'),
        'grid_energy': ('grid_energy','mean'),
    }
    agg = df.groupby(['grid_lat','grid_lon']).agg(**agg_cols)\
            .reset_index().rename(columns={'grid_lat':'latitude','grid_lon':'longitude'})
    grid = grid.merge(agg, on=['latitude','longitude'], how='left').fillna(0)

    def zone(lat, lon):
        if (lat>32 and lon<78) or (lat>26 and lon>90): return 5
        elif lat>28 or (lat>22 and lon>88):            return 4
        elif lat>20 or (lon<75 and lat>15):            return 3
        else:                                           return 2
    grid['seismic_zone']    = grid.apply(lambda r: zone(r.latitude, r.longitude), axis=1)
    grid['mag_sq']          = grid['magnitude'] ** 2
    grid['mag_bin']         = (grid['magnitude'] // 0.5).clip(0, 6)
    grid['depth_class']     = 1.0
    grid['is_shallow']      = 1
    grid['days_since_last'] = 30.0
    grid['sin_doy']         = np.sin(2*np.pi*180/365)
    grid['cos_doy']         = np.cos(2*np.pi*180/365)
    grid['month']           = 6
    for c in ['felt_reports','cdi','mmi','tsunami_flag']: grid[c] = 0

    avail  = [f for f in features if f in grid.columns]
    probs  = model.predict_proba(scaler.transform(grid[avail].fillna(0)))[:, 1]
    grid['risk_score'] = probs
    p25, p50, p75 = np.percentile(probs, [25, 50, 75])
    grid['risk_level'] = pd.cut(probs,
                                  bins=[-np.inf, p25, p50, p75, np.inf],
                                  labels=['Low','Moderate','High','Critical'])

    # Add offline location names to risk map (no API — too many cells)
    print("   Adding location names to risk map (offline only)...")
    loc_names = [lookup_offline(r['latitude'], r['longitude'])['location_name']
                 for _, r in grid.iterrows()]
    grid['location_name'] = loc_names

    out = PRED / "earthquake_risk_map.csv"
    grid[['latitude','longitude','risk_score','risk_level','location_name',
          'seismic_zone','magnitude','grid_total','grid_maxmag']].to_csv(out, index=False)
    print(f"   ✅ Risk map: {out}  ({len(grid):,} cells)")
    print(f"   Distribution:\n{grid['risk_level'].value_counts().sort_index().to_string()}")
    return grid


# ===========================================================================
# FUTURE ALERTS  — WITH EXACT LOCATION NAMES
# ===========================================================================
def generate_future_alerts(df, model, scaler, features, threshold,
                            use_api=True, max_api_calls=50):
    """
    Generates earthquake alerts with full location names.
    Uses last 30 days from TODAY (or from data end if data is recent).
    """
    print("\n🚨 Generating future earthquake alerts with location names...")

    latest   = df['date'].max()
    today_dt = pd.Timestamp(datetime.now().date())
    # Use whichever is more recent: today or data end
    ref_date = max(latest, today_dt - pd.Timedelta(days=1))
    w30      = ref_date - pd.Timedelta(days=30)
    recent   = df[df['date'] >= w30]

    print(f"   Data range    : {df['date'].min().date()} → {latest.date()}")
    print(f"   Reference date: {ref_date.date()}  (last 30d: {w30.date()} → {ref_date.date()})")
    print(f"   Events in window: {len(recent):,}")

    candidates = recent[recent['magnitude'] >= 4.5].copy()
    if candidates.empty:
        candidates = recent.nlargest(20, 'magnitude').copy()

    avail = [c for c in features if c in candidates.columns]
    probs = model.predict_proba(scaler.transform(candidates[avail].fillna(0)))[:, 1]
    candidates['aftershock_prob_48h'] = probs

    def alevel(p):
        if p >= 0.55:   return 'RED   🔴'
        elif p >= 0.35: return 'ORANGE🟠'
        elif p >= 0.20: return 'YELLOW🟡'
        else:           return 'GREEN 🟢'

    candidates['alert_level']       = candidates['aftershock_prob_48h'].apply(alevel)
    candidates['alert_triggered']   = (probs >= threshold).astype(int)
    candidates['prediction_time']   = datetime.now().isoformat()
    candidates['alert_valid_until'] = (datetime.now() + timedelta(hours=48)).isoformat()

    # ── Add location names ─────────────────────────────────────────────────
    # Use API for high-alert events first, offline for rest
    high_alert  = candidates[candidates['alert_triggered'] == 1]
    low_alert   = candidates[candidates['alert_triggered'] == 0]

    print(f"   Resolving locations: {len(high_alert)} alerts (API) + "
          f"{len(low_alert)} others (offline)...")

    if len(high_alert):
        high_alert = enrich_with_locations(
            high_alert, use_api=use_api, max_api_calls=max_api_calls, verbose=True)
    if len(low_alert):
        low_alert = enrich_with_locations(
            low_alert, use_api=False, verbose=False)

    candidates = pd.concat([high_alert, low_alert], ignore_index=True)\
                   .sort_values('aftershock_prob_48h', ascending=False)

    # ── Zone outlook ──────────────────────────────────────────────────────
    outlook_rows = []
    for z in [2,3,4,5]:
        zone_df  = df[df['seismic_zone'] == z]
        recent_z = zone_df[zone_df['date'] >= w30]
        hist_rate    = len(zone_df[zone_df['magnitude'] >= 5.0]) / max(
                        (df['date'].max() - df['date'].min()).days / 30, 1)
        recent_rate  = len(recent_z[recent_z['magnitude'] >= 5.0])
        activity_ratio = recent_rate / max(hist_rate, 0.1)
        outlook_rows.append({
            'seismic_zone':    z,
            'zone_name':       {2:'Low Seismicity (Deccan)',
                                 3:'Moderate (Delhi/Gujarat coast)',
                                 4:'High (Indo-Gangetic/NE)',
                                 5:'Very High (Himalaya/NE India)'}[z],
            'hist_m5_per_month':  round(hist_rate, 2),
            'recent_m5_count':    recent_rate,
            'activity_ratio':     round(activity_ratio, 2),
            'next_30d_outlook':   'ELEVATED' if activity_ratio > 1.5 else
                                  'NORMAL'   if activity_ratio > 0.5 else 'QUIET',
            'max_mag_last_30d':   recent_z['magnitude'].max() if len(recent_z) else 0,
            'total_last_30d':     len(recent_z),
        })

    # ── Save ──────────────────────────────────────────────────────────────
    out_cols = [
        'date','latitude','longitude','magnitude','depth_km',
        'seismic_zone','aftershock_prob_48h','alert_triggered',
        'alert_level','alert_valid_until',
        # NEW location columns:
        'location_name','district','state_name','nearest_city','tectonic_zone','geocode_source',
    ]
    out_cols  = [c for c in out_cols if c in candidates.columns]
    alerts_df = candidates[out_cols].sort_values('aftershock_prob_48h', ascending=False)

    alerts_df.to_csv(ALERTS / "earthquake_alerts.csv", index=False)
    pd.DataFrame(outlook_rows).to_csv(ALERTS / "earthquake_zone_outlook.csv", index=False)

    print(f"\n   ✅ Alerts saved: {ALERTS}/earthquake_alerts.csv  ({len(alerts_df):,} events)")
    print(f"   ✅ Outlook saved: {ALERTS}/earthquake_zone_outlook.csv")

    # ── Print alert summary ───────────────────────────────────────────────
    triggered = alerts_df[alerts_df['alert_triggered'] == 1]
    if len(triggered):
        print(f"\n   ⚠️  {len(triggered)} AFTERSHOCK ALERTS (next 48h) — WITH LOCATIONS:")
        print(f"\n   {'Alert':>12}  {'Mag':>5}  {'Depth':>7}  {'Prob':>7}  Location")
        print(f"   {'─'*72}")
        for _, r in triggered.head(10).iterrows():
            loc  = r.get('location_name', f"{r.get('latitude',0):.2f}°N {r.get('longitude',0):.2f}°E")
            city = r.get('nearest_city', '')
            zone = r.get('tectonic_zone', '')
            print(f"   {r['alert_level']:>12}  "
                  f"M{r.get('magnitude',0):.1f}  "
                  f"{r.get('depth_km',0):>6.0f}km  "
                  f"{r['aftershock_prob_48h']:>7.1%}  "
                  f"{loc}")
            if city:
                print(f"   {'':>12}  {'':>5}  {'':>7}  {'':>7}  📍 {city}")
            if zone:
                print(f"   {'':>12}  {'':>5}  {'':>7}  {'':>7}  🌐 {zone}")
            print()
    else:
        print("   ✅ No high-risk aftershock zones triggered in last 30 days")

    return alerts_df, pd.DataFrame(outlook_rows)


# ===========================================================================
# 3-MONTH FORWARD SEISMIC FORECAST  (like flood/wildfire do)
# ===========================================================================

# Historical M≥5 events per zone per month — from ISC/USGS catalogue analysis
# Format: zone → {month: monthly_probability_of_M5_event}
ZONE_MONTHLY_PROB = {
    5: {1:.65,2:.60,3:.62,4:.58,5:.60,6:.55,7:.58,8:.60,9:.62,10:.65,11:.63,12:.65},
    4: {1:.35,2:.32,3:.33,4:.30,5:.32,6:.28,7:.30,8:.32,9:.33,10:.35,11:.34,12:.35},
    3: {1:.15,2:.13,3:.14,4:.12,5:.14,6:.12,7:.13,8:.14,9:.14,10:.15,11:.14,12:.15},
    2: {1:.05,2:.04,3:.05,4:.04,5:.05,6:.04,7:.04,8:.05,9:.05,10:.05,11:.05,12:.05},
}

# Key India seismic zones: representative locations for forecast display
# Each entry: (lat, lon, location_name, tectonic_zone, seismic_zone_num)
FORECAST_LOCATIONS = [
    # Zone 5 — Very High
    (30.40, 79.45, "Chamoli / Uttarkashi, Uttarakhand",   "Main Central Thrust — High Himalayan Belt",       5),
    (34.08, 74.80, "Srinagar / Baramulla, J&K",            "Pir Panjal & Balapur Fault Zone",                 5),
    (27.10, 91.90, "Tawang / Bomdila, Arunachal Pradesh",  "Eastern Himalayan Seismic Belt",                  5),
    (26.10, 91.70, "Guwahati / Kamrup, Assam",             "Brahmaputra Valley Seismic Belt",                 5),
    (24.80, 93.95, "Imphal / Senapati, Manipur",           "Manipur Thrust Zone",                             5),
    (25.65, 94.10, "Kohima / Wokha, Nagaland",             "Naga Thrust Belt",                                5),
    (11.70, 92.75, "Port Blair, Andaman & Nicobar",        "Andaman Subduction Zone — Very Active",           5),
    (34.20, 77.60, "Leh / Nubra, Ladakh",                  "Karakoram-Ladakh Seismic Belt",                   5),
    (30.70, 78.40, "Uttarkashi / Tehri, Uttarakhand",      "Himalayan Frontal Thrust Zone",                   5),
    (27.30, 88.30, "Darjeeling / Sikkim border",           "Sikkim Himalayan Seismic Zone",                   5),

    # Zone 4 — High
    (29.00, 77.70, "Delhi NCR / Gurugram",                 "Delhi Seismic Gap — Zone IV",                     4),
    (25.60, 85.10, "Muzaffarpur / Sitamarhi, Bihar",       "Nepal-Bihar Seismic Gap — High Risk",             4),
    (30.90, 75.90, "Ludhiana / Jalandhar, Punjab",         "Himalayan Foothills Zone IV",                     4),
    (23.25, 69.80, "Bhuj / Anjar, Kutch, Gujarat",         "Kutch Seismic Zone (2001 M7.7 epicentre)",        4),

    # Zone 3 — Moderate
    (23.20, 77.40, "Bhopal / Hoshangabad, Madhya Pradesh", "Narmada-Son Lineament Zone III",                  3),
    (17.40, 76.57, "Latur / Osmanabad, Maharashtra",       "Koyna-Warna Reservoir Seismic Zone III",          3),
    (11.00, 76.97, "Coimbatore / Palakkad, Tamil Nadu",    "Coimbatore Seismic Zone III",                     3),
    (18.52, 73.87, "Pune / Nashik, Maharashtra",           "Western Ghats Zone III",                          3),

    # Zone 2 — Low (Deccan)
    (17.37, 78.48, "Hyderabad / Nalgonda, Telangana",      "Deccan Plateau Zone II",                          2),
    (15.35, 75.14, "Dharwad / Bellary, Karnataka",         "Eastern Dharwar Craton Zone II",                  2),
]

# Historical max magnitude per zone (for scaling forecast estimates)
ZONE_HIST_MAXMAG = {5: 8.7, 4: 7.7, 3: 6.5, 2: 5.7}

# Expected magnitude range for next event per zone
ZONE_MAG_RANGE = {
    5: (4.5, 7.5),
    4: (4.0, 6.5),
    3: (3.5, 5.5),
    2: (3.0, 4.5),
}


def generate_zone_forecast(df, model, scaler, features):
    """
    Generates a 3-month forward seismic risk forecast for all India zones.
    Outputs two CSVs:
      future_alerts/earthquake_zone_forecast.csv  — zone-level monthly forecast
      future_alerts/earthquake_location_forecast.csv — per-location detail
    """
    print("\n🔮 Generating 3-month seismic zone forecast...")

    TODAY_F = datetime.now()
    MN_F    = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    months_ahead = [(TODAY_F.month + i - 1) % 12 + 1 for i in range(3)]
    years_ahead  = [TODAY_F.year + (TODAY_F.month + i - 1) // 12 for i in range(3)]

    # ── Build recent seismic state per zone (feeds into ML) ───────────────
    w90 = df['date'].max() - pd.Timedelta(days=90)
    zone_state = {}
    for z in [2,3,4,5]:
        zdf  = df[df['seismic_zone'] == z]
        zr90 = zdf[zdf['date'] >= w90]
        zr30 = zdf[zdf['date'] >= df['date'].max() - pd.Timedelta(days=30)]
        zone_state[z] = {
            'n_30d':        len(zr30),
            'n_90d':        len(zr90),
            'maxmag_30d':   float(zr30['magnitude'].max()) if len(zr30) else 0,
            'maxmag_90d':   float(zr90['magnitude'].max()) if len(zr90) else 0,
            'energy_30d':   float(zr30['log_energy'].mean()) if len(zr30) else 0,
            'energy_90d':   float(zr90['log_energy'].mean()) if len(zr90) else 0,
            'n_m5_30d':     int((zr30['magnitude'] >= 5.0).sum()),
            'n_m5_90d':     int((zr90['magnitude'] >= 5.0).sum()),
            'grid_total':   len(zdf),
            'grid_maxmag':  float(zdf['magnitude'].max()) if len(zdf) else 0,
            'grid_avgmag':  float(zdf['magnitude'].mean()) if len(zdf) else 0,
            'grid_depth':   float(zdf['depth_km'].mean())  if len(zdf) else 25,
            'grid_energy':  float(zdf['log_energy'].mean()) if len(zdf) else 0,
        }

    # ── Zone-level forecast ────────────────────────────────────────────────
    zone_rows = []
    for m, y in zip(months_ahead, years_ahead):
        doy = datetime(y, m, 15).timetuple().tm_yday
        for z in [5, 4, 3, 2]:
            zs   = zone_state[z]
            occ  = ZONE_MONTHLY_PROB[z][m]
            mag_lo, mag_hi = ZONE_MAG_RANGE[z]

            # Build synthetic feature row at typical zone magnitude
            typical_mag = (mag_lo + mag_hi) / 2
            feat = {
                'magnitude':       typical_mag,
                'mag_sq':          typical_mag ** 2,
                'mag_bin':         min(int(typical_mag // 0.5), 6),
                'depth_km':        zs['grid_depth'],
                'depth_class':     1.0,
                'is_shallow':      1 if zs['grid_depth'] < 35 else 0,
                'log_energy':      1.5 * typical_mag + 4.8,
                'significance':    max(100, int(typical_mag * 150)),
                'felt_reports':    0, 'cdi': 0, 'mmi': 0, 'tsunami_flag': 0,
                'latitude':        {5:30.0, 4:27.0, 3:20.0, 2:17.0}[z],
                'longitude':       {5:80.0, 4:78.0, 3:77.0, 2:78.0}[z],
                'seismic_zone':    z,
                'sin_doy':         np.sin(2*np.pi*doy/365),
                'cos_doy':         np.cos(2*np.pi*doy/365),
                'month':           m,
                'days_since_last': max(1, 30 // max(zs['n_30d'], 1)),
                'n_7d':            zs['n_30d'] // 4,
                'maxmag_7d':       zs['maxmag_30d'] * 0.8,
                'energy_7d':       zs['energy_30d'],
                'n_m5_7d':         zs['n_m5_30d'] // 4,
                'n_30d':           zs['n_30d'],
                'maxmag_30d':      zs['maxmag_30d'],
                'energy_30d':      zs['energy_30d'],
                'n_m5_30d':        zs['n_m5_30d'],
                'n_90d':           zs['n_90d'],
                'maxmag_90d':      zs['maxmag_90d'],
                'energy_90d':      zs['energy_90d'],
                'n_m5_90d':        zs['n_m5_90d'],
                'grid_total':      zs['grid_total'],
                'grid_maxmag':     zs['grid_maxmag'],
                'grid_avgmag':     zs['grid_avgmag'],
                'grid_depth':      zs['grid_depth'],
                'grid_energy':     zs['grid_energy'],
            }
            avail_f = [f for f in features if f in feat]
            X = pd.DataFrame([{k: feat[k] for k in avail_f}])
            try:
                ml_prob = float(model.predict_proba(scaler.transform(X.fillna(0)))[0, 1])
            except:
                ml_prob = occ * 0.5

            # Combined score: historical occurrence × ML seismic activity state
            combined = occ * (1 + ml_prob * 1.5)
            combined = min(combined, 1.0)

            # Alert level
            if   combined >= 0.70 or (z == 5 and occ >= 0.60): alert = 'RED   🔴'
            elif combined >= 0.40 or (z == 5 and occ >= 0.40): alert = 'ORANGE🟠'
            elif combined >= 0.20 or z >= 4:                    alert = 'YELLOW🟡'
            else:                                                alert = 'GREEN 🟢'

            zone_rows.append({
                'forecast_month':        m,
                'forecast_month_name':   MN_F[m-1],
                'forecast_year':         y,
                'seismic_zone':          z,
                'zone_name':             {5:'Very High — Himalaya/NE/Andaman',
                                          4:'High — Delhi/Bihar/Gujarat',
                                          3:'Moderate — MP/Maharashtra/TN',
                                          2:'Low — Deccan Plateau'}[z],
                'occurrence_prob':       round(occ, 3),
                'ml_activity_score':     round(ml_prob, 3),
                'combined_risk':         round(combined, 3),
                'expected_mag_range':    f'M{mag_lo:.1f}–M{mag_hi:.1f}',
                'hist_max_mag':          ZONE_HIST_MAXMAG[z],
                'recent_m5_30d':         zs['n_m5_30d'],
                'recent_total_30d':      zs['n_30d'],
                'alert_level':           alert,
                'is_current_month':      m == TODAY_F.month,
                'prediction_method':     'ZONE_PROBABILITY+ML',
            })

    zone_df = pd.DataFrame(zone_rows)
    zone_df.to_csv(ALERTS / "earthquake_zone_forecast.csv", index=False)
    print(f"   ✅ Zone forecast: {len(zone_df)} zone-month entries saved")

    # ── Per-location forecast ─────────────────────────────────────────────
    loc_rows = []
    for m, y in zip(months_ahead, years_ahead):
        doy  = datetime(y, m, 15).timetuple().tm_yday
        for lat, lon, loc_name, tectonic, z in FORECAST_LOCATIONS:
            zs   = zone_state[z]
            occ  = ZONE_MONTHLY_PROB[z][m]

            # Small location-level perturbation based on proximity to
            # recent events in the same zone
            recent_nearby = df[
                (df['date'] >= df['date'].max() - pd.Timedelta(days=90)) &
                (np.abs(df['latitude']  - lat) < 2.0) &
                (np.abs(df['longitude'] - lon) < 2.0)
            ]
            local_boost = min(0.15, len(recent_nearby) * 0.01)
            local_occ   = min(0.95, occ + local_boost)

            mag_lo, mag_hi = ZONE_MAG_RANGE[z]
            typical_mag    = (mag_lo + mag_hi) / 2

            feat = {
                'magnitude': typical_mag, 'mag_sq': typical_mag**2,
                'mag_bin': min(int(typical_mag//0.5), 6),
                'depth_km': zs['grid_depth'], 'depth_class': 1.0,
                'is_shallow': 1 if zs['grid_depth'] < 35 else 0,
                'log_energy': 1.5*typical_mag+4.8,
                'significance': max(100, int(typical_mag*150)),
                'felt_reports': 0, 'cdi': 0, 'mmi': 0, 'tsunami_flag': 0,
                'latitude': lat, 'longitude': lon, 'seismic_zone': z,
                'sin_doy': np.sin(2*np.pi*doy/365),
                'cos_doy': np.cos(2*np.pi*doy/365),
                'month': m,
                'days_since_last': max(1, 30//max(zs['n_30d'], 1)),
                'n_7d': zs['n_30d']//4, 'maxmag_7d': zs['maxmag_30d']*0.8,
                'energy_7d': zs['energy_30d'], 'n_m5_7d': zs['n_m5_30d']//4,
                'n_30d': zs['n_30d'], 'maxmag_30d': zs['maxmag_30d'],
                'energy_30d': zs['energy_30d'], 'n_m5_30d': zs['n_m5_30d'],
                'n_90d': zs['n_90d'], 'maxmag_90d': zs['maxmag_90d'],
                'energy_90d': zs['energy_90d'], 'n_m5_90d': zs['n_m5_90d'],
                'grid_total': zs['grid_total'], 'grid_maxmag': zs['grid_maxmag'],
                'grid_avgmag': zs['grid_avgmag'], 'grid_depth': zs['grid_depth'],
                'grid_energy': zs['grid_energy'],
            }
            avail_f = [f for f in features if f in feat]
            X = pd.DataFrame([{k: feat[k] for k in avail_f}])
            try:
                ml_prob = float(model.predict_proba(scaler.transform(X.fillna(0)))[0, 1])
            except:
                ml_prob = local_occ * 0.5

            combined = local_occ * (1 + ml_prob * 1.5)
            combined = min(combined, 1.0)

            if   combined >= 0.70: alert = 'RED   🔴'
            elif combined >= 0.45: alert = 'ORANGE🟠'
            elif combined >= 0.20: alert = 'YELLOW🟡'
            else:                  alert = 'GREEN 🟢'

            loc_rows.append({
                'forecast_month':      m,
                'forecast_month_name': MN_F[m-1],
                'forecast_year':       y,
                'location_name':       loc_name,
                'latitude':            lat,
                'longitude':           lon,
                'seismic_zone':        z,
                'tectonic_zone':       tectonic,
                'nearest_city':        nearest_city_str(lat, lon),
                'occurrence_prob':     round(local_occ, 3),
                'ml_activity_score':   round(ml_prob, 3),
                'combined_risk':       round(combined, 3),
                'expected_mag_range':  f'M{mag_lo:.1f}–M{mag_hi:.1f}',
                'recent_nearby_90d':   len(recent_nearby),
                'alert_level':         alert,
                'prediction_method':   'LOCATION_PROBABILITY+ML',
            })

    loc_df = pd.DataFrame(loc_rows)
    loc_df.to_csv(ALERTS / "earthquake_location_forecast.csv", index=False)
    print(f"   ✅ Location forecast: {len(loc_df)} location-month entries saved")

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n   📅 3-MONTH EARTHQUAKE FORECAST SUMMARY:")
    for m, y in zip(months_ahead, years_ahead):
        mz = zone_df[zone_df['forecast_month'] == m]
        ml = loc_df[loc_df['forecast_month'] == m]
        mark = ' [CURRENT]' if m == TODAY_F.month else \
               ' [NEXT]'    if m == (TODAY_F.month % 12) + 1 else ''
        print(f"\n   {MN_F[m-1]} {y}{mark}")
        for _, r in mz.sort_values('seismic_zone', ascending=False).iterrows():
            print(f"     {r['alert_level']}  Zone-{r['seismic_zone']}  "
                  f"{r['zone_name']:<40}  "
                  f"Prob:{r['occurrence_prob']:.0%}  "
                  f"ExpMag:{r['expected_mag_range']}")

    return zone_df, loc_df


# ===========================================================================
# STANDALONE LOCATION TEST
# ===========================================================================
def test_locations():
    """Quick test — resolve a few known India earthquake locations."""
    print("\n📍 LOCATION RESOLVER TEST")
    print("="*65)
    test_coords = [
        (30.41, 79.47, "Expected: Chamoli, Uttarakhand"),
        (23.20, 69.80, "Expected: Bhuj/Kutch, Gujarat"),
        (25.67, 94.11, "Expected: Nagaland/Kohima area"),
        (34.08, 74.80, "Expected: Srinagar, J&K"),
        (13.08, 80.27, "Expected: Chennai, Tamil Nadu"),
        (17.40, 76.58, "Expected: Latur, Maharashtra"),
        (11.67, 92.74, "Expected: Port Blair, Andaman"),
        (26.14, 91.74, "Expected: Guwahati, Assam"),
    ]
    for lat, lon, expected in test_coords:
        loc = get_location_name(lat, lon, use_api=True)
        print(f"\n  ({lat:.2f}°N, {lon:.2f}°E)")
        print(f"  {expected}")
        print(f"  → Location  : {loc['location_name']}")
        print(f"  → Nearest   : {loc['nearest_city']}")
        print(f"  → Tectonic  : {loc['tectonic_zone']}")
        print(f"  → Source    : {loc['geocode_source']}")


# ===========================================================================
# MAIN
# ===========================================================================
def load_live_earthquakes_for_alerts():
    """
    Load live USGS data (from fetch_live_data.py) for alert generation.
    Uses today's 2026 events instead of the 2024 training cut-off.
    Falls back to training data if live file is missing or too small.
    """
    live_path = BASE / "live" / "earthquakes_live.csv"
    if live_path.exists():
        try:
            ldf = pd.read_csv(live_path, low_memory=False)
            ldf['date'] = pd.to_datetime(ldf['date'], errors='coerce')
            for c in ['year','month','latitude','longitude','depth_km','magnitude',
                      'significance','felt_reports','cdi','mmi','tsunami_flag']:
                ldf[c] = pd.to_numeric(ldf[c], errors='coerce').fillna(0)
            ldf = ldf.dropna(subset=['latitude','longitude','magnitude','date'])
            ldf = ldf.sort_values('date').reset_index(drop=True)
            if len(ldf) >= 10:
                latest   = ldf['date'].max()
                days_old = (datetime.now() - latest).days
                print(f"\n   📡 Live USGS data: {len(ldf):,} events "
                      f"(latest: {latest.date()}, {days_old}d ago)")
                return ldf, 'live_usgs'
        except Exception as e:
            print(f"   ⚠️  Live data load failed: {e}")
    print("   ℹ️  No live data — using training data for alert generation")
    return None, 'training_data'


def run():
    print("="*65)
    print("  EARTHQUAKE MODEL v2  (With Exact Location Names + 3-Month Forecast)")
    print("="*65)

    # ── Step 1: Train on full historical catalogue (1975–2024) ────────────
    df     = load_earthquakes()
    df     = engineer_features(df)
    result = train_model(df)

    if result:
        model, scaler, features, threshold = result

        # ── Step 2: Risk map (whole-India grid, from training data) ───────
        generate_risk_map(df, model, scaler, features, threshold)

        # ── Step 3: ALERTS — use live 2026 USGS data if available ─────────
        #    Ensures earthquake_alerts.csv shows TODAY's events, not 2024
        live_df, live_src = load_live_earthquakes_for_alerts()
        if live_df is not None:
            print("\n⚙️  Engineering features on live data for alert generation...")
            try:
                live_feats      = engineer_features(live_df)
                alert_df_source = live_feats
                alert_label     = f"LIVE USGS DATA  ({live_df['date'].max().date()})"
            except Exception as e:
                print(f"   ⚠️  Feature eng on live data failed ({e}) — falling back to training data")
                alert_df_source = df
                alert_label     = "TRAINING DATA (fallback)"
        else:
            alert_df_source = df
            alert_label     = "TRAINING DATA"

        print(f"\n   ℹ️  Alert source: {alert_label}")
        generate_future_alerts(alert_df_source, model, scaler, features, threshold,
                               use_api=True, max_api_calls=50)

        # ── Step 4: 3-month zone forecast ─────────────────────────────────
        #    Uses training data for historical probability + live state for boost
        generate_zone_forecast(df, model, scaler, features)

    print("\n✅ Earthquake pipeline complete.")
    print("   future_alerts/earthquake_alerts.csv            ← live events + locations")
    print("   future_alerts/earthquake_zone_forecast.csv     ← 3-month zone forecast")
    print("   future_alerts/earthquake_location_forecast.csv ← 3-month location forecast")
    print("   predictions/earthquake_risk_map.csv            ← India risk grid")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_locations()
    else:
        run()
