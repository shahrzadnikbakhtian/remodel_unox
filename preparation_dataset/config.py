from dotenv import load_dotenv
import os
import json
from pathlib import Path



# BQ_LOCATION = os.getenv("BQ_LOCATION", "europe-west4")  # default is fine if you want


load_dotenv()

def get_env_var(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Missing environment variable: {key}")
    return value


# ---------------------- GeoJSON helper ----------------------
def _load_geojson_geometry(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    t = obj.get("type")
    if t == "FeatureCollection":
        feats = obj.get("features", [])
        if not feats:
            raise ValueError("GeoJSON FeatureCollection has no features.")
        geom = feats[0].get("geometry")
        if not geom:
            raise ValueError("Feature has no geometry.")
    elif t == "Feature":
        geom = obj.get("geometry") or {}
        if not geom:
            raise ValueError("Feature has no geometry.")
    else:
        geom = obj  # already a Geometry

    return json.dumps(geom)

BASE_DIR = Path(__file__).resolve().parent
NORDLAND_GEOJSON_PATH = "nordland.geojson"
NORDLAND_GEOJSON_PARAM = _load_geojson_geometry(NORDLAND_GEOJSON_PATH) 



PROJECT_ID = get_env_var("PROJECT_ID")
DATASET_ID = get_env_var("DATASET_ID")
STATIC_TABLE = get_env_var("STATIC_TABLE")
OCCUPATION_TABLE = get_env_var("OCCUPATION_TABLE")
POI_TABLE = get_env_var("POI_TABLE")
CATEGORIES_TABLE = get_env_var("CATEGORIES_TABLE")
ROADS_TABLE = get_env_var("ROADS_TABLE")

FILTERED_STATION_CONDITION_NORDLAND = """
    source = 'nobil'
    AND is_approved = TRUE
    AND capacity_kw >= 50
    And country_name= 'Norway'
    AND ST_WITHIN(
        geometry,
        ST_GEOGFROMGEOJSON(@nordland_geojson))
"""

FILTERED_STATION_CONDITION = FILTERED_STATION_CONDITION_NORDLAND


DATE_FILTER_CONDITION = """
    DATE(u.hour) BETWEEN DATE '2024-09-01' AND LAST_DAY(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH))
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




