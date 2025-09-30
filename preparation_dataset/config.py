from dotenv import load_dotenv
import os
import json
from pathlib import Path


load_dotenv()

def get_env_var(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Missing environment variable: {key}")
    return value


# --- Customer-aware paths ---

PROJECT_ROOT= Path(__file__).resolve().parent
CUSTOMER = os.getenv("CUSTOMER", "Unox")  
CUSTOMER_DIR = PROJECT_ROOT / "customers" / CUSTOMER
if not CUSTOMER_DIR.exists():
    raise ValueError(f"Customer directory does not exist: {CUSTOMER_DIR}")


INPUT_GEP_DIR= CUSTOMER_DIR / "input" / "geojson"
DF_DIR= CUSTOMER_DIR / "artifacts" / "df"
OUTPUT_DIR= CUSTOMER_DIR / "artifacts" / "output"
DF_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- GeoJSON helper ----------------------
# def _load_geojson_geometry(path: str) -> str:
#     with open(path, "r", encoding="utf-8") as f:
#         obj = json.load(f)

#     t = obj.get("type")
#     if t == "FeatureCollection":
#         feats = obj.get("features", [])
#         if not feats:
#             raise ValueError("GeoJSON FeatureCollection has no features.")
#         geom = feats[0].get("geometry")
#         if not geom:
#             raise ValueError("Feature has no geometry.")
#     elif t == "Feature":
#         geom = obj.get("geometry") or {}
#         if not geom:
#             raise ValueError("Feature has no geometry.")
#     else:
#         geom = obj  # already a Geometry

#     return json.dumps(geom)


# # THAT PART SHOULD BE CHANGED TO CUSTOMER
# GEOJSON_FILENAME= os.getenv("GEOJSON_FILENAME", "nordland.geojson")
# NORDLAND_GEOJSON_PARAM= _load_geojson_geometry(INPUT_GEP_DIR / GEOJSON_FILENAME)



def _extract_geometries(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    geoms = []
    t = obj.get("type")
    if t == "FeatureCollection":
        feats = obj.get("features", [])
        if not feats:
            raise ValueError(f"FeatureCollection has no features: {path}")
        for feat in feats:
            geom = (feat or {}).get("geometry")
            if geom:
                geoms.append(json.dumps(geom))
    elif t == "Feature":
        geom = obj.get("geometry") or {}
        if not geom:
            raise ValueError(f"Feature has no geometry: {path}")
        geoms.append(json.dumps(geom))
    else:
        # Already a Geometry object
        geoms.append(json.dumps(obj))

    if not geoms:
        raise ValueError(f"No geometries found in: {path}")
    return geoms

# Read comma-separated list from .env (fallback to single file)
_names = os.getenv("GEOJSON_FILENAMES")
if not _names:
    raise ValueError(
        "Set GEOJSON_FILENAMES in .env, e.g. GEOJSON_FILENAMES=area1.geojson,area2.geojson"
    )
GEOJSON_FILENAMES = [s.strip() for s in _names.split(",") if s.strip()]


# Load ALL geometries across ALL files -> ARRAY<STRING> for BigQuery
REGION_GEOJSON_PARAMS = []
_missing = []
for fname in GEOJSON_FILENAMES:
    p = INPUT_GEP_DIR / fname
    if not p.exists():
        _missing.append(str(p))
        continue
    REGION_GEOJSON_PARAMS.extend(_extract_geometries(p))
if _missing:
    raise FileNotFoundError(f"GeoJSON file(s) not found: {_missing}")


# ---------------------- BigQuery Config ----------------------
PROJECT_ID = get_env_var("PROJECT_ID")
DATASET_ID = get_env_var("DATASET_ID")
STATIC_TABLE = get_env_var("STATIC_TABLE")
OCCUPATION_TABLE = get_env_var("OCCUPATION_TABLE")
POI_TABLE = get_env_var("POI_TABLE")
CATEGORIES_TABLE = get_env_var("CATEGORIES_TABLE")
ROADS_TABLE = get_env_var("ROADS_TABLE")

# FILTERED_STATION_CONDITION_NORDLAND = """
#     source = 'nobil'
#     AND is_approved = TRUE
#     AND capacity_kw >= 50
#     And country_name= 'Norway'
#     AND ST_WITHIN(
#         geometry,
#         ST_GEOGFROMGEOJSON(@nordland_geojson))
# """

# FILTERED_STATION_CONDITION = FILTERED_STATION_CONDITION_NORDLAND

# DATE_FILTER_CONDITION = """
#     DATE(u.hour) BETWEEN DATE '2024-09-01' AND LAST_DAY(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH))
# """


FILTERED_STATION_CONDITION_REGIONS = """
    source = 'nobil'
    AND is_approved = TRUE
    AND capacity_kw >= 50
    AND country_name = 'Norway'
    AND ST_WITHIN(
        geometry,
        (SELECT ST_UNION_AGG(ST_GEOGFROMGEOJSON(g))
         FROM UNNEST(@regions_geojson) AS g)
    )
"""

# choose the multi-region condition by default
FILTERED_STATION_CONDITION = FILTERED_STATION_CONDITION_REGIONS

DATE_FILTER_CONDITION = """
    DATE(u.hour) BETWEEN DATE '2024-09-01'
                     AND LAST_DAY(DATE_SUB(CURRENT_DATE('Europe/Oslo'), INTERVAL 1 MONTH))
"""


# ---------------------- Customers ----------------------

FILTERED_STATION_CONDITION_nobil = """
    source = 'nobil'
    AND is_approved = TRUE
    AND capacity_kw >= 50
    And country_name= 'Norway'
"""



FILTERED_STATION_CONDITION_sw = """
    source = 'sw'
    AND is_approved = TRUE
    AND capacity_kw > 100
"""


# AND capacity_kw >= 50 AND capacity_kw < 100
# AND country_name IN ('Norway', 'Sweden', 'Denmark', 'Finland')




