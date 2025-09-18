import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any
from shapely import wkt
import geopandas as gpd
from google.cloud import bigquery

# -------------------- Config & Setup --------------------

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
client = bigquery.Client(project=PROJECT_ID)


POI_CLASSES = [
    "cafe_and_restaurant",
    "hobby",
    "home",
    "service",
]

POI_TIME_PER_VISIT = {
    'activities_count': 90,
    'beauty_count': 60,
    'cafe_and_restaurant_count': 67.5,
    'clothing_count': 40,
    'grocery_count': 30,
    'shopping_center_count': 90,
    'accommodation_count': 120,
    'electronics_count': 30,
    'home_count': 30,
    'food_count': 15,
    'public_service_count': 30,
    'service_count': 30,
    'religious_count': 60,
    'medical_count': 60,
    'hobby_count': 30,
}


# POI_CATEGORY_MAP = { 
#     'Supermarket': supermarket_count + shopping_center_count,
#     'Shopping' : electronics_count +hobby_count +home_count +clothing_count + shopping_center_count,
#     'Services': medical_count +service_count +public_service_count + beauty_count,
#     'Hotel': accommodation_count,
#     'Restaurant' : cafe_and_restaurant_count + food_count,
#     'Leisure': activities_count +religious_count ,
# }





# CONCAT(
#   CAST(
#     (CASE WHEN TRIM(IFNULL(CAST(Supermarket AS TEXT), '')) = '' OR TRIM(IFNULL(CAST(Supermarket AS TEXT), '')) = '0' THEN 0 ELSE 1 END) +
#     (CASE WHEN TRIM(IFNULL(CAST(Shopping   AS TEXT), '')) = '' OR TRIM(IFNULL(CAST(Shopping   AS TEXT), '')) = '0' THEN 0 ELSE 1 END) +
#     (CASE WHEN TRIM(IFNULL(CAST(Services   AS TEXT), '')) = '' OR TRIM(IFNULL(CAST(Services   AS TEXT), '')) = '0' THEN 0 ELSE 1 END) +
#     (CASE WHEN TRIM(IFNULL(CAST(Hotel      AS TEXT), '')) = '' OR TRIM(IFNULL(CAST(Hotel      AS TEXT), '')) = '0' THEN 0 ELSE 1 END) +
#     (CASE WHEN TRIM(IFNULL(CAST(Restaurant AS TEXT), '')) = '' OR TRIM(IFNULL(CAST(Restaurant AS TEXT), '')) = '0' THEN 0 ELSE 1 END) +
#     (CASE WHEN TRIM(IFNULL(CAST(Leisure    AS TEXT), '')) = '' OR TRIM(IFNULL(CAST(Leisure    AS TEXT), '')) = '0' THEN 0 ELSE 1 END)
#   AS TEXT),
#   '/6'
# )

# ROAD_CATEGORY_MAP = {
#     IF(
#   motorway_dist <= trunk_dist AND motorway_dist <= primary_dist,
#   motorway_dist,
#   IF(
#     trunk_dist <= motorway_dist AND trunk_dist <= primary_dist,
#     trunk_dist,
#     primary_dist
#   )
# )
# }



# -------------------- Loaders --------------------

def load_segmented_data() -> pd.DataFrame:
    return pd.read_csv("data/station_with_poi_df.csv")
    

def load_classified_roads() -> pd.DataFrame:
    df = pd.read_csv("data/station_with_road_df.csv")
    return df.drop(columns=["road_count_nan", "road_distance_nan"], errors="ignore")

def load_population_data() -> gpd.GeoDataFrame:
    query = """
        SELECT * FROM `surplusmap-393908.dev_sherry.nordic_population`
        WHERE last_updated > '2019-01-01'
        # SELECT * FROM `surplusmap-393908.dev_sherry.nordic_pop_city`
    """
    df = client.query(query).to_dataframe()

    df['geometry'] = df['geog'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    return gdf[['population', 'geometry']]

# -------------------- Cleaners --------------------

def clean_missing_segmented(df: pd.DataFrame) -> pd.DataFrame:
    poi_cols = [f"{c}_count" for c in POI_CLASSES]

    missing_ids = df[df[poi_cols].sum(axis=1) == 0]["id"].tolist()
    return df[~df["id"].isin(missing_ids)]

def clean_missing_roads(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_check = [
        'road_distance_motorway', 'road_distance_primary',
        'road_distance_secondary', 'road_distance_trunk',
        'road_distance_living_street', 'road_distance_residential', 'road_distance_service',
        'road_distance_tertiary',  'road_distance_unclassified',
        ]
    missing_ids = df[df[cols_to_check].sum(axis=1) == 0]["id"].tolist()
    return df[~df["id"].isin(missing_ids)]

# -------------------- Feature Engineering --------------------

def merge_data(segmented: pd.DataFrame, roads: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(roads, segmented, on='id', how='left')
    drop_cols = [  # Keep this updated based on your needs
        'calendar_day_count_x_y', 'calendar_day_count_y_y',
        'size_total_y', 'start_day_y', 'end_day_y', 'end_day_x', 'start_day_x',
        'percentage_average_y', 'capacity_kw_y', 'average_daily_minutes_y' ,
        'calendar_day_count_x_x', 'address_y', 'title_y', 'country_name_y',
        'geometry_y', 'operator_x', 'address_x', 'active_days_count_y'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return df.rename(columns={
        'calendar_day_count_y_x': 'calendar_day_count',
        'active_days_count_x': 'active_days_count',
        'size_total_x': 'size_total',
        'operator_y': 'operator',
        'percentage_average_x': 'percentage_average',
        'average_daily_minutes_x': 'average_daily_minutes',
        'capacity_kw_x': 'capacity_kw',
        'geometry_x': 'geometry',
        'title_x': 'station_name', 
        'country_name_x': 'country_name',
        'road_distance_motorway': 'motorway_dist',
        'road_distance_primary': 'primary_dist',
        'road_distance_secondary': 'secondary_dist',
        'road_distance_service': 'road_service_dist',
        'road_distance_tertiary': 'tertiary_dist',
        'road_distance_trunk': 'trunk_dist',
        'road_distance_unclassified': 'unclassified_dist',
        'road_count_motorway': 'motorway_count',
        'road_count_primary' : 'primary_count',    
        'road_count_secondary': 'secondary_count',
        'road_count_service' : 'road_service_count',
        'road_count_tertiary': 'tertiary_count',
        'road_count_trunk': 'trunk_count',
        'road_count_unclassified': 'unclassified_count',
        'operator_x': 'operator',
        'grocery_count': 'supermarket_count',
        'road_distance_living_street' : 'living_street_dist',
        'road_distance_residential': 'residential_dist', 
    })

def bin_occupation_rate(df: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(0, df['percentage_average'].max(), 11)
    df['occupation_rate'] = pd.cut(df['percentage_average'], bins=bins)
    return df


def enrich_with_population(df: gpd.GeoDataFrame, population: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df['geometry'] = df['geometry'].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    
    population = population.to_crs(df.crs)
    population = population[['population', 'geometry']]

    # joined = gpd.sjoin_nearest(df, population, how="left", max_distance=0.1, distance_col="pop_distance")
    joined = gpd.sjoin_nearest(df, population, how="left")


    joined = joined.drop_duplicates(subset=["id"])

    return joined


def estimate_time_spent(df: pd.DataFrame) -> pd.DataFrame:
    df['total_time_spent'] = sum(
        df[col].gt(0).astype(int) * time
        for col, time in POI_TIME_PER_VISIT.items()
        if col in df.columns
    )
    return df

def extract_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df[['lng', 'lat']] = df['geometry'].astype(str).str.extract(r'POINT \(([-\d\.]+) ([-\d\.]+)\)').astype(float)
    return df





# -------------------- Looker --------------------

def _preview(df: pd.DataFrame, cols: list[str], title: str, n: int = 5) -> None:
    print(f"\n== {title} ==")
    print("shape:", df.shape)
    cols = [c for c in cols if c in df.columns]
    if cols:
        print(df[cols].head(n).to_string(index=False))
    else:
        print("(no matching columns to preview)")

def ensure_cols(df: pd.DataFrame, cols: list[str], verbose: bool = False) -> list[str]:
    added = []
    for c in cols:
        if c not in df.columns:
            df[c] = 0
            added.append(c)
    if verbose:
        if added:
            print(f"[ensure_cols] Added {len(added)} missing columns: {added[:10]}{' …' if len(added) > 10 else ''}")
        else:
            print("[ensure_cols] No columns needed to be added.")
    return added

def add_poi_category_columns(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    needed = [
        "supermarket_count", "shopping_center_count",
        "electronics_count", "hobby_count", "home_count", "clothing_count",
        "medical_count", "service_count", "public_service_count", "beauty_count",
        "accommodation_count",
        "cafe_and_restaurant_count", "food_count",
        "activities_count", "religious_count",
    ]
    ensure_cols(df, needed, verbose=verbose)

    before = set(df.columns)

    df["Supermarket"] = df["supermarket_count"] + df["shopping_center_count"]
    df["Shopping"]    = (
        df["electronics_count"] + df["hobby_count"] + df["home_count"] +
        df["clothing_count"] + df["shopping_center_count"]
    )
    df["Services"]    = (
        df["medical_count"] + df["service_count"] +
        df["public_service_count"] + df["beauty_count"]
    )
    df["Hotel"]       = df["accommodation_count"]
    df["Restaurant"]  = df["cafe_and_restaurant_count"] + df["food_count"]
    df["Leisure"]     = df["activities_count"] + df["religious_count"]

    if verbose:
        created = sorted(list(set(df.columns) - before))
        print(f"[add_poi_category_columns] Created columns: {created}")
        _preview(df, ["Supermarket","Shopping","Services","Hotel","Restaurant","Leisure"], "POI category sums (sample)")

    return df


def add_poi_presence_ratio(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    cats = ["Supermarket", "Shopping", "Services", "Hotel", "Restaurant", "Leisure"]
    ensure_cols(df, cats, verbose=verbose)

    df["poi_categories_present"] = sum((df[c].fillna(0) > 0).astype(int) for c in cats).astype(int)
    df["poi_categories_present_str"] = df["poi_categories_present"].astype(str) + "/6"

    if verbose:
        print("[add_poi_presence_ratio] Distribution of categories present:")
        print(df["poi_categories_present"].value_counts(dropna=False).sort_index())
        _preview(df, cats + ["poi_categories_present","poi_categories_present_str"], "POI presence (sample)")

    return df


def add_nearest_major_road(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    cols = ["motorway_dist", "trunk_dist", "primary_dist"]
    ensure_cols(df, cols, verbose=verbose)

    df["nearest_major_road_dist"] = df[cols].min(axis=1)
    df["nearest_major_road_type"] = df[cols].idxmin(axis=1).str.replace("_dist$", "", regex=True)

    if verbose:
        print("[add_nearest_major_road] Type counts:")
        print(df["nearest_major_road_type"].value_counts(dropna=False))
        print("[add_nearest_major_road] Distance summary:")
        print(df["nearest_major_road_dist"].describe())
        _preview(df, cols + ["nearest_major_road_type","nearest_major_road_dist"], "Nearest major road (sample)")

    return df




# def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
#     for c in cols:
#         if c not in df.columns:
#             df[c] = 0

# def add_poi_category_columns(df: pd.DataFrame) -> pd.DataFrame:
#     needed = [
#         "supermarket_count", "shopping_center_count",
#         "electronics_count", "hobby_count", "home_count", "clothing_count",
#         "medical_count", "service_count", "public_service_count", "beauty_count",
#         "accommodation_count",
#         "cafe_and_restaurant_count", "food_count",
#         "activities_count", "religious_count",
#     ]
#     ensure_cols(df, needed)

#     df["Supermarket"] = df["supermarket_count"] + df["shopping_center_count"]
#     df["Shopping"]    = (
#         df["electronics_count"] + df["hobby_count"] + df["home_count"] +
#         df["clothing_count"] + df["shopping_center_count"]
#     )
#     df["Services"]    = (
#         df["medical_count"] + df["service_count"] +
#         df["public_service_count"] + df["beauty_count"]
#     )
#     df["Hotel"]       = df["accommodation_count"]
#     df["Restaurant"]  = df["cafe_and_restaurant_count"] + df["food_count"]
#     df["Leisure"]     = df["activities_count"] + df["religious_count"]

#     return df






# def add_poi_presence_ratio(df: pd.DataFrame) -> pd.DataFrame:
#     cats = ["Supermarket", "Shopping", "Services", "Hotel", "Restaurant", "Leisure"]
#     # ensure the columns exist even if the previous step was skipped
#     ensure_cols(df, cats)

#     present_count = sum((df[c].fillna(0) > 0).astype(int) for c in cats)
#     df["poi_categories_present"] = present_count.astype(int)  # numeric count 0..6
#     df["poi_categories_present_str"] = df["poi_categories_present"].astype(str) + "/6"
#     return df


# def add_nearest_major_road(df: pd.DataFrame) -> pd.DataFrame:
#     cols = ["motorway_dist", "trunk_dist", "primary_dist"]
#     ensure_cols(df, cols)

#     # distance to the nearest among these
#     df["nearest_major_road_dist"] = df[cols].min(axis=1)

#     # which one is the minimum (label)
#     df["nearest_major_road_type"] = df[cols].idxmin(axis=1)  # e.g., "motorway_dist"
#     # Optional: cleaner label without "_dist"
#     df["nearest_major_road_type"] = df["nearest_major_road_type"].str.replace("_dist$", "", regex=True)

#     return df










# -------------------- Scoring --------------------
# https://www.sciencedirect.com/science/article/pii/S0966692325002194?utm_source=chatgpt.com
# https://www.mdpi.com/2071-1050/13/4/2298?utm_source=chatgpt.com

def compute_thresholds(df: pd.DataFrame) -> tuple[Dict[str, float], Dict[str, float]]:
    high_counts = {col: df[col].quantile(0.7) for col in df.columns if col.endswith("_count")}
    low_dists = {col: df[col].quantile(0.3) for col in df.columns if col.endswith("_dist") or col.startswith("road_distance_")}
    return high_counts, low_dists

def score_row(row: pd.Series, high_counts: Dict[str, float], low_dists: Dict[str, float]) -> float:
    score = 0
    score += sum(row.get(col, 0) >= threshold for col, threshold in high_counts.items())
    score += sum(row.get(col, 0) <= threshold for col, threshold in low_dists.items())

    score += int(row.get("is_motorway", 0)) + int(row.get("is_local_center", 0))

    # Normalize score to 0–10
    max_possible_score = len(high_counts) + len(low_dists) + 2 
    normalized_score = max(0, min((score / max_possible_score) * 10, 10)) 

    return normalized_score


def add_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    high, low = compute_thresholds(df)
    df["good_feature_score"] = df.apply(lambda row: score_row(row, high, low), axis=1)

    total_stations = len(df)
    df["score_performance"] = df["percentage_average"].rank(method="min", ascending=True).astype(int)
    df["score_performance"] = total_stations - df["score_performance"] + 1

    bin_order = sorted(df["occupation_rate"].unique(), reverse=True)
    bin_rank_map = {bin_label: rank for rank, bin_label in enumerate(bin_order, start=1)}
    df["occupation_rank"] = df["occupation_rate"].map(bin_rank_map)


    return df

# -------------------- Runner --------------------

def main():
    segmented = load_segmented_data()
    print("After loading segmented data:", segmented.shape)
    segmented = clean_missing_segmented(segmented)
    print("After cleaning segmented data:", segmented.shape)

    roads = load_classified_roads()
    print("After loading road data:", roads.shape)
    roads = clean_missing_roads(roads)
    print("After cleaning road data:", roads.shape)


    merged = merge_data(segmented, roads)
    print("After merging road + segmented:", merged.shape)

    merged = bin_occupation_rate(merged)
    print("After binning occupation rate:", merged.shape)
    merged.dropna(inplace=True)
    print("After dropping NaNs:", merged.shape)

    population = load_population_data()
    print("Population data loaded:", population.shape)
    merged = enrich_with_population(merged, population)
    print("Duplicate station IDs:", merged['id'].duplicated().sum())
    print("After enriching with population:", merged.shape)

    merged = estimate_time_spent(merged)
    print("After estimating time spent:", merged.shape)

    merged = extract_coordinates(merged)
    print("After extracting coordinates:", merged.shape)

    missing_counts = [
        dist_col.replace("_dist", "_count")
        for dist_col in merged.columns if dist_col.endswith("_dist")
        if dist_col.replace("_dist", "_count") not in merged.columns
    ]
    if missing_counts:
        print("⚠️ Missing matching _count columns for:", missing_counts)

    dupes = merged.columns[merged.columns.duplicated()].tolist()
    if dupes:
        print("⚠️ Duplicate columns found:", dupes)


    merged = add_score_columns(merged)
    print("After adding score columns:", merged.shape)

    merged.drop(columns=['near_shops', 'index_right'], inplace=True, errors='ignore')
    print("After dropping unneeded columns:", merged.shape)
    count_cols = [col for col in merged.columns if col.endswith("_count")]
    merged[count_cols] = merged[count_cols].fillna(0)


    # ----- NEW: derive POI category sums and presence -----
    merged = add_poi_category_columns(merged, verbose=True)
    merged = add_poi_presence_ratio(merged, verbose=True)

    # ----- NEW: derive nearest major road category & distance -----
    merged = add_nearest_major_road(merged, verbose=True)

    print("POI category sample:")
    print(merged[["Supermarket","Shopping","Services","Hotel","Restaurant","Leisure",
                "poi_categories_present","poi_categories_present_str"]].head())

    print("Road category sample:")
    print(merged[["motorway_dist","trunk_dist","primary_dist",
                "nearest_major_road_dist","nearest_major_road_type"]].head())


    print("Final dataset shape:", merged.shape)
    print("Unique station IDs:", merged['id'].nunique())

    cols_to_round = [col for col in merged.columns if col.endswith('_dist') or col.endswith('_count') or col in ['total_time_spent', 'population', 'good_feature_score']]
    merged[cols_to_round] = merged[cols_to_round].round(2)


    merged.to_csv("nordic_stations_df_new.csv", index=False)
    print("Saved to nordic_stations_df_new.csv")
    print("POI distance columns:", [col for col in merged.columns])



if __name__ == "__main__":
    main()
