import pandas as pd
from typing import Dict, Any
from tqdm import tqdm
import logging
import geopandas as gpd
import osmnx as ox
from osmnx._errors import InsufficientResponseError
from shapely.geometry import Point
from shapely import wkt


logging.basicConfig(level=logging.INFO)

stations_df = pd.read_csv("data/merged_station_df.csv")
roads_df = pd.read_csv("data/merged_road_df.csv")


poi_list = [
    "accommodation", "activities", "beauty", "cafe_and_restaurant", "clothing",
    "electronics", "food", "grocery", "hobby", "home", "medical",
    "public_service", "service", "shopping_center", "religious"
]


shop_categories = ["clothing", "electronics", "food", "grocery", "hobby", "home"]
stations_df["shop_count"] = stations_df[[f"poi_count_{c}" for c in shop_categories]].sum(axis=1)
stations_df["distance_to_shopping_center"] = stations_df["poi_distance_shopping_center"]



def is_near_shops(
    station: Dict[str, Any],
    poi_counts: Dict[str, float],
    shop_count_threshold: float = 0.2,
    distance_to_shopping_center_threshold: float = 500,
) -> bool:
    """
    Determine whether a station is considered near shops based on its shop count relative to the city's
    total shopping POIs or its proximity to a shopping center.

    A station is deemed near shops if either:
      - The ratio of its 'shop_count' to the city's total shopping POI count (from `poi_counts`) exceeds
        the specified threshold, or
      - Its 'distance_to_shopping_center' is below the defined distance threshold.

    Args:
        station (Dict[str, Any]): A dictionary of station attributes, expected to include:
            - 'shop_count': the number of shops at or near the station.
            - 'distance_to_shopping_center': the distance from the station to the nearest shopping center.
        poi_counts (Dict[str, float]): A dictionary containing the total count of shopping-related POIs for
            the station's city. The count is used as the reference value for normalizing the station's shop count.
        shop_count_threshold (float, optional): The minimum ratio of the station's shop count to the city's
            shopping POI total required to consider the station as near shops. Defaults to 0.2.
        distance_to_shopping_center_threshold (float, optional): The maximum distance (in the same units as
            the station's data) from the station to a shopping center for it to be considered near shops.
            Defaults to 500.

    Returns:
        bool: True if the station meets at least one of the criteria for being near shops; otherwise, False.
    """
    shop_count = station["shop_count"]
    distance_to_shopping_center = station["distance_to_shopping_center"]


    # add to fix error
    total_city_shop_count = poi_counts.get("shop_count", 0)

    # Safeguard against zero or missing city shop count
    shop_ratio = (shop_count / total_city_shop_count) if total_city_shop_count > 0 else 0

    return (shop_ratio > shop_count_threshold) or (
        distance_to_shopping_center < distance_to_shopping_center_threshold
    )

    # return (shop_count / poi_counts["shop_count"] > shop_count_threshold) or (
    #     distance_to_shopping_center < distance_to_shopping_center_threshold
    # )


def is_local_center(
    station: Dict[str, Any],
    service_threshold: int = 10,
    shop_threshold: int = 10,
    poi_types_required: int = 5,
) -> int:
    
    poi_categories = [
        "poi_count_accommodation", "poi_count_activities", "poi_count_beauty",
        "poi_count_cafe_and_restaurant", "poi_count_clothing", "poi_count_electronics",
        "poi_count_food", "poi_count_grocery", "poi_count_hobby", "poi_count_home",
        "poi_count_medical", "poi_count_public_service", "poi_count_service",
        "poi_count_shopping_center", "poi_count_religious"
    ]
    poi_diversity_count = sum(1 for key in poi_categories if station.get(key, 0) > 0)
    has_shopping_center = station.get("poi_count_shopping_center", 0) != 0


    return int(
    has_shopping_center and
    station.get("poi_count_service", 0) >= service_threshold and
    station.get("shop_count", 0) >= shop_threshold and
    poi_diversity_count >= poi_types_required
    )


def segment_stations(
    stations: pd.DataFrame, poi_df: pd.DataFrame, poi_list: list[str]
) -> pd.DataFrame:
    
    stations["near_shops"] = 0
    stations["is_local_center"] = stations.apply(
        lambda row: is_local_center(row), axis=1
    )

    return stations


segmented_df = segment_stations(stations_df, stations_df, poi_list)
segmented_df.to_csv("data/segmented_stations.csv", index=False)



def classify_road_context(road_df: pd.DataFrame) -> pd.DataFrame:
    highway_types = ["motorway", "primary", "secondary"]

    for hwy in highway_types + ["residential"]:

        dist_col= f"road_distance_{hwy}"
        count_col = f"road_count_{hwy}"

        if dist_col not in road_df.columns:
            road_df[dist_col] = 0
        else:
            road_df[dist_col] = road_df[dist_col].fillna(0)

        if count_col not in road_df.columns:
            road_df[count_col] = 0
        else:
            road_df[count_col] = road_df[count_col].fillna(0)

    road_df["is_motorway"] = (
        (road_df["road_distance_motorway"] > 0) & 
        (road_df["road_distance_motorway"] < 100)
    ).astype(int)

    road_df["is_highway_area"] = (
        ((road_df["road_distance_motorway"] > 0) & (road_df["road_distance_motorway"] < 200)) |
        # ((road_df["road_distance_trunk"] > 0) & (road_df["road_distance_trunk"] < 200)) |
        ((road_df["road_distance_primary"] > 0) & (road_df["road_distance_primary"] < 200)) |
        ((road_df["road_distance_secondary"] > 0) & (road_df["road_distance_secondary"] < 200))
    ).astype(int)

    road_df["is_residential_area"] = (
        (road_df["road_distance_residential"] > 0) & 
        (road_df["road_distance_residential"] < 350) &
        (road_df["road_count_residential"] >= 1) &
        (road_df["road_distance_motorway"] == 0) & (road_df["road_count_motorway"] == 0) &
        # (road_df["road_distance_trunk"] == 0) & (road_df["road_count_trunk"] == 0) &
        (road_df["road_distance_primary"] == 0) & (road_df["road_count_primary"] == 0) &
        (road_df["road_distance_secondary"] == 0) & (road_df["road_count_secondary"] == 0)
    ).astype(int)

    return road_df  ### it should be change to not confused with previous name in aprevoius def

road_features_classified = classify_road_context(roads_df)
road_features_classified.to_csv("data/classified_roads.csv", index=False)









