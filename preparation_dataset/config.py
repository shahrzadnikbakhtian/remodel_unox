from dotenv import load_dotenv
import os

load_dotenv()

def get_env_var(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Missing environment variable: {key}")
    return value

PROJECT_ID = get_env_var("PROJECT_ID")
DATASET_ID = get_env_var("DATASET_ID")

STATIC_TABLE = get_env_var("STATIC_TABLE")
OCCUPATION_TABLE = get_env_var("OCCUPATION_TABLE")
POI_TABLE = get_env_var("POI_TABLE")
CATEGORIES_TABLE = get_env_var("CATEGORIES_TABLE")
ROADS_TABLE = get_env_var("ROADS_TABLE")

FILTERED_STATION_CONDITION = """
    source = 'nobil'
    AND is_approved = TRUE
    AND capacity_kw >= 100
    AND country_name IN ('Norway', 'Sweden', 'Denmark', 'Finland')
"""

DATE_FILTER_CONDITION = """
    DATE(u.hour) BETWEEN DATE '2024-09-01' AND LAST_DAY(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH))
"""


FILTERED_STATION_CONDITION_sw = """
    source = 'sw'
    AND is_approved = TRUE
    AND capacity_kw > 100
"""


# AND capacity_kw >= 50 AND capacity_kw < 100