import os
import logging

import pandas as pd
from google.cloud import bigquery

from config import (
    PROJECT_ID,
    STATIC_TABLE,
    OCCUPATION_TABLE,
    POI_TABLE,
    CATEGORIES_TABLE,
    ROADS_TABLE,
    FILTERED_STATION_CONDITION,
    DATE_FILTER_CONDITION,
    NORDLAND_GEOJSON_PARAM,
)

# ---------------------- Setup ----------------------
client = bigquery.Client(project=PROJECT_ID)
logging.basicConfig(level=logging.INFO)

# Output paths
DATA_DIR = "data"
OCCUPATION_PATH = f"{DATA_DIR}/occupation_df.csv"
STATIC_OCCUPATION_PATH = f"{DATA_DIR}/static_occupation_df.csv"
STATION_FEATURES_PATH = f"{DATA_DIR}/poi_features_df.csv"
MERGED_POI_PATH = f"{DATA_DIR}/station_with_poi_df.csv"
ROAD_FEATURES_PATH = f"{DATA_DIR}/roads_df.csv"
MERGED_ROAD_PATH = f"{DATA_DIR}/station_with_road_df.csv"


# ---------------------- Config ----------------------
SAVE_TO_DISK = True  # Set True to save intermediate outputs

# ---------------------- Logging Helpers ----------------------
def log_df_info(df: pd.DataFrame, label: str) -> None:
    logging.info(f"{label} shape: {df.shape}")


def save_dataframe(df: pd.DataFrame, path: str, label: str) -> None:
    log_df_info(df, label)
    if SAVE_TO_DISK:
        df.to_csv(path, index=False)
        logging.info(f"{label} saved to: {path}")

# ---------------------- Query Loaders ----------------------
def _run_query(query: str) -> pd.DataFrame:
    params = []
    if "@nordland_geojson" in query:
        params.append(
            bigquery.ScalarQueryParameter("nordland_geojson", "STRING", NORDLAND_GEOJSON_PARAM)
        )
    job_config = bigquery.QueryJobConfig(query_parameters=params, use_legacy_sql=False)
    return client.query(query, job_config=job_config).to_dataframe()

run_query = _run_query



def load_static_data() -> pd.DataFrame:
    query = f"""
        WITH filtered_stations AS (
            SELECT id, capacity_kw, size_total, geometry, operator
            FROM `{STATIC_TABLE}`
            WHERE {FILTERED_STATION_CONDITION}
        ),
        base_data AS (
            SELECT u.id, DATE(u.hour) AS day
            FROM `{OCCUPATION_TABLE}` u
            JOIN filtered_stations s ON u.id = s.id
            WHERE {DATE_FILTER_CONDITION}
        ),
        utilization_summary AS (
            SELECT id, MIN(day) AS start_day, MAX(day) AS end_day,
                   DATE_DIFF(MAX(day), MIN(day), DAY) + 1 AS calendar_day_count
            FROM base_data
            GROUP BY id
        ),
        stations_with_utilization AS (
            SELECT s.id, s.capacity_kw, s.size_total, s.geometry, s.operator, u.calendar_day_count
            FROM utilization_summary u
            JOIN filtered_stations s ON u.id = s.id
        )
        SELECT
            s.id,
            ANY_VALUE(s.operator) AS operator,
            s.capacity_kw,
            st.address,
            st.title,
            st.country_name,
            s.calendar_day_count,
            ANY_VALUE(s.geometry) AS geometry
        FROM stations_with_utilization s
        JOIN `{STATIC_TABLE}` st ON s.id = st.id
        GROUP BY s.id, s.capacity_kw, st.address, st.title, st.country_name, s.calendar_day_count
        ORDER BY s.calendar_day_count DESC
    """
    # return client.query(query).to_dataframe()
    return run_query(query)


def load_occupation_data() -> pd.DataFrame:
    query = f"""
        WITH filtered_stations AS (
            SELECT id, capacity_kw, geometry
            FROM `{STATIC_TABLE}`
            WHERE {FILTERED_STATION_CONDITION}
        ),
        base_data AS (
            SELECT u.id, DATE(u.hour) AS day, u.charging_duration, u.size_total
            FROM `{OCCUPATION_TABLE}` u
            JOIN filtered_stations s ON u.id = s.id
            WHERE {DATE_FILTER_CONDITION}
        ),
        daily_percentage AS (
            SELECT id, day, ANY_VALUE(size_total) AS size_total,
                SAFE_DIVIDE(SUM(charging_duration), ANY_VALUE(size_total) * 24 * 60) * 100 AS daily_utilization,
                SAFE_DIVIDE(SUM(charging_duration), ANY_VALUE(size_total)) AS daily_avg_minutes
            FROM base_data
            GROUP BY id, day
        ),

        utilization_summary AS (
            SELECT 
                id,
                MIN(day) AS start_day,
                MAX(day) AS end_day,
                DATE_DIFF(MAX(day), MIN(day), DAY) + 1 AS calendar_day_count,
                SUM(daily_utilization) AS total_utilization_percent,
                SUM(daily_avg_minutes) AS total_avg_minutes,
                COUNT(DISTINCT day) AS active_days_count,
                ANY_VALUE(size_total) AS size_total
            FROM daily_percentage
            GROUP BY id
        ),

        final_output AS (
            SELECT 
                s.id,
                u.size_total,
                u.start_day,
                u.end_day,
                u.calendar_day_count,
                u.active_days_count,
                SAFE_DIVIDE(u.total_utilization_percent, u.calendar_day_count) AS percentage_average,
                SAFE_DIVIDE(u.total_avg_minutes, u.calendar_day_count) AS average_daily_minutes
            FROM utilization_summary u
            JOIN filtered_stations s ON u.id = s.id
        )

        SELECT *
        FROM final_output
        ORDER BY percentage_average DESC;

    """
    # return client.query(query).to_dataframe()
    return run_query(query)

# ---------------------- Timeseries Loaders ----------------------
    
def load_timeseries_data() -> pd.DataFrame:
    query = f"""
        -- ===== BASE FILTERS (STATIC SIZE FROM STATION TABLE) =====
        WITH filtered_stations AS (
        SELECT id, size_total AS size_total_static
        FROM `{STATIC_TABLE}`
        WHERE {FILTERED_STATION_CONDITION}
        ),

        -- ===== HOURLY -> DAILY (restrict by date window) =====
        base_data AS (
        SELECT
            u.id,
            DATE(u.hour, 'Europe/Oslo') AS day,
            u.charging_duration
        FROM `{OCCUPATION_TABLE}` u
        JOIN filtered_stations s USING (id)
        WHERE DATE(u.hour, 'Europe/Oslo')
                BETWEEN DATE '2024-09-01'
                    AND LAST_DAY(DATE_SUB(CURRENT_DATE('Europe/Oslo'), INTERVAL 1 MONTH))
        ),

        daily AS (
        SELECT
            b.id,
            b.day,
            s.size_total_static,
            SUM(b.charging_duration) AS minutes_charged_day,
            -- daily metrics
            ROUND(SAFE_DIVIDE(SUM(b.charging_duration), s.size_total_static * 24 * 60) * 100, 2) AS daily_utilization_pct,
            ROUND(SAFE_DIVIDE(SUM(b.charging_duration), s.size_total_static), 2)                  AS daily_avg_minutes
        FROM base_data b
        JOIN filtered_stations s USING (id)
        GROUP BY b.id, b.day, s.size_total_static
        ),

        -- ===== SUMMARY (per station) =====
        utilization_summary AS (
        SELECT 
            id,
            MIN(day) AS start_day,
            MAX(day) AS end_day,
            DATE_DIFF(MAX(day), MIN(day), DAY) + 1 AS calendar_day_count,
            SUM(daily_utilization_pct) AS total_utilization_percent,
            SUM(daily_avg_minutes)     AS total_avg_minutes,
            COUNT(*)                   AS active_days_count,
            ANY_VALUE(size_total_static) AS size_total_static
        FROM daily
        GROUP BY id
        ),

        -- ===== MONTHLY (formula: sum(minutes) / (size_total * 1440 * days_in_month)) =====
        monthly_base AS (
        SELECT
            id,
            DATE_TRUNC(day, MONTH) AS month_start,
            ANY_VALUE(size_total_static) AS size_total_static,
            SUM(minutes_charged_day)     AS minutes_in_month
        FROM daily
        GROUP BY id, month_start
        ),
        monthly AS (
        SELECT
            id,
            month_start,
            ROUND(
            COALESCE(
                SAFE_DIVIDE(
                minutes_in_month,
                size_total_static * 1440 * DATE_DIFF(DATE_ADD(month_start, INTERVAL 1 MONTH), month_start, DAY)
                ) * 100,
                0
            ), 2
            ) AS avg_utilization_pct,
            ROUND(SAFE_DIVIDE(minutes_in_month, size_total_static), 2) AS total_minutes_month,
            SAFE_DIVIDE(minutes_in_month, size_total_static)           AS total_minutes_month_raw
        FROM monthly_base
        ),

        -- ===== MONTHLY ARRAYS (per-month perf + active flag) =====
        monthly_arrays AS (
        SELECT
            m.id,
            ARRAY_AGG(STRUCT(
            m.month_start,
            m.avg_utilization_pct,
            (CAST(m.total_minutes_month AS FLOAT64) >= CAST(0 AS FLOAT64)) AS active_flag  -- min_active_minutes_month = 0
            )
            ORDER BY m.month_start) AS monthly_perf,
            ARRAY_AGG(
            IF(CAST(m.total_minutes_month AS FLOAT64) >= CAST(0 AS FLOAT64),
                FORMAT_DATE('%b', m.month_start), NULL)
            IGNORE NULLS
            ORDER BY m.month_start
            ) AS active_months
        FROM monthly m
        GROUP BY m.id
        ),

        -- ===== MONTH-OF-YEAR UTIL AVERAGES (from avg_utilization_pct) =====
        month_util_avg AS (
        SELECT
            id,
            EXTRACT(MONTH FROM month_start) AS month_num,
            AVG(avg_utilization_pct)        AS month_avg_utilization_pct
        FROM monthly
        GROUP BY id, month_num
        ),

        -- >>> Peak month by utilization (NOT by minutes share) <<<
        month_util_arrays AS (
        SELECT
            id,
            ARRAY_AGG(STRUCT(
            month_num,
            month_avg_utilization_pct
            ) ORDER BY month_num) AS month_util,
            ARRAY_AGG(
            FORMAT_DATE('%b', DATE(2000, month_num, 1))
            ORDER BY month_avg_utilization_pct DESC, month_num
            LIMIT 1
            )[OFFSET(0)] AS peak_month
        FROM month_util_avg
        GROUP BY id
        ),

        -- ===== (Keep) Seasonality by minutes share for season_share_avg =====
        month_shares_base AS (
        SELECT
            id,
            month_start,
            EXTRACT(YEAR  FROM month_start) AS yr,
            EXTRACT(MONTH FROM month_start) AS month_of_year,
            minutes_in_month,
            SAFE_DIVIDE(
            minutes_in_month,
            NULLIF(SUM(minutes_in_month) OVER (
                PARTITION BY id, EXTRACT(YEAR FROM month_start)
            ), 0)
            ) AS month_share
        FROM monthly_base
        ),
        seasonality_shares_avg AS (
        SELECT
            id,
            month_of_year,
            AVG(month_share) AS month_share_avg
        FROM month_shares_base
        GROUP BY id, month_of_year
        ),

        -- ===== SEASONS =====
        season_map AS (
        SELECT 1 AS month_num, 'Winter' AS season UNION ALL
        SELECT 2, 'Winter' UNION ALL SELECT 12, 'Winter' UNION ALL
        SELECT 3, 'Spring' UNION ALL SELECT 4, 'Spring' UNION ALL SELECT 5, 'Spring' UNION ALL
        SELECT 6, 'Summer' UNION ALL SELECT 7, 'Summer' UNION ALL SELECT 8, 'Summer' UNION ALL
        SELECT 9, 'Fall'   UNION ALL SELECT 10,'Fall'   UNION ALL SELECT 11,'Fall'
        ),
        season_totals_avg AS (
        SELECT
            m.id,
            sm.season,
            ROUND(SUM(COALESCE(ssa.month_share_avg, 0.0)), 2) AS season_share_avg,
            -- Sum the monthly utilization % across months in the season
            ROUND(SUM(m.avg_utilization_pct), 2) AS season_avg_utilization_pct
        FROM monthly m
        JOIN season_map sm
            ON EXTRACT(MONTH FROM m.month_start) = sm.month_num
        LEFT JOIN seasonality_shares_avg ssa
            ON ssa.id = m.id AND ssa.month_of_year = EXTRACT(MONTH FROM m.month_start)
        GROUP BY m.id, sm.season
        ),
        season_arrays AS (
        SELECT
            id,
            ARRAY_AGG(STRUCT(
            season,
            season_share_avg,
            season_avg_utilization_pct
            )
            ORDER BY CASE season
                    WHEN 'Winter' THEN 1 WHEN 'Spring' THEN 2
                    WHEN 'Summer' THEN 3 WHEN 'Fall'   THEN 4
                    END) AS season_perf,
            ARRAY_AGG(season ORDER BY season_avg_utilization_pct DESC, season LIMIT 1)[OFFSET(0)] AS peak_season
        FROM season_totals_avg
        GROUP BY id
        ),

        -- ===== WEEKLY =====
        weekly AS (
        SELECT
            id,
            DATE_TRUNC(day, WEEK(MONDAY)) AS week_start,
            ROUND(AVG(daily_utilization_pct), 2) AS avg_utilization_pct,
            ROUND(SUM(daily_avg_minutes), 2)     AS total_minutes_week
        FROM daily
        GROUP BY id, week_start
        ),
        weekly_arrays AS (
        SELECT
            id,
            ARRAY_AGG(STRUCT(
            week_start,
            avg_utilization_pct,
            total_minutes_week
            )
            ORDER BY week_start) AS weekly_perf
        FROM weekly
        GROUP BY id
        )

        -- ===== FINAL OUTPUT =====
        SELECT
        u.id,
        u.size_total_static AS size_total,
        u.start_day,
        u.end_day,
        u.calendar_day_count,
        u.active_days_count,
        (u.calendar_day_count - u.active_days_count) AS diff_day,
        ROUND(SAFE_DIVIDE(u.total_utilization_percent, u.calendar_day_count), 2) AS percentage_average,
        ROUND(SAFE_DIVIDE(u.total_avg_minutes,         u.calendar_day_count), 2) AS average_daily_minutes,
        ma.monthly_perf,
        ma.active_months,
        mu.peak_month,
        se.season_perf,
        se.peak_season,
        wa.weekly_perf
        FROM utilization_summary u
        LEFT JOIN monthly_arrays     ma USING (id)
        LEFT JOIN month_util_arrays  mu USING (id)
        LEFT JOIN season_arrays      se USING (id)
        LEFT JOIN weekly_arrays      wa USING (id)
        ORDER BY u.start_day ASC;
    """
    # return client.query(query).to_dataframe()
    return run_query(query)



def load_hourly_data() -> pd.DataFrame:
    query = f"""

        WITH filtered_stations AS (
        SELECT id, capacity_kw, size_total, operator, title, geometry
        FROM `{STATIC_TABLE}` AS s
        WHERE {FILTERED_STATION_CONDITION}
    )

    SELECT 
        u.id,
        u.hour,
        u.charging_duration,
        u.capacity_kw,
        s.operator,
        s.geometry,
        s.title,
        DATE(u.hour) AS day,
        FORMAT_DATE('%Y-%m', DATE(u.hour)) AS month,
        EXTRACT(HOUR FROM u.hour) AS hour_of_day,
        s.size_total,
    FROM `{OCCUPATION_TABLE}` u
    JOIN filtered_stations s ON u.id = s.id

    WHERE DATE(u.hour) BETWEEN DATE '2024-09-01' AND DATE '2025-09-15'
    ORDER BY u.id, u.hour;
    """
    
    return run_query(query)


# ---------------------- POI Loaders ----------------------

def load_poi_data() -> pd.DataFrame:
    query = f"""
        WITH filtered_stations AS (
            SELECT id, capacity_kw, size_total, geometry
            FROM `{STATIC_TABLE}`
            WHERE {FILTERED_STATION_CONDITION}
        ),
        base_data AS (
            SELECT u.id, DATE(u.hour) AS day
            FROM `{OCCUPATION_TABLE}` u
            JOIN filtered_stations s ON u.id = s.id
            WHERE {DATE_FILTER_CONDITION}
        ),
        utilization_summary AS (
            SELECT id, MIN(day) AS start_day, MAX(day) AS end_day,
                   DATE_DIFF(MAX(day), MIN(day), DAY) + 1 AS calendar_day_count
            FROM base_data
            GROUP BY id
        ),
        stations_with_utilization AS (
            SELECT s.id, s.capacity_kw, s.size_total, s.geometry
            FROM utilization_summary u
            JOIN filtered_stations s ON u.id = s.id
        )
        SELECT
            s.id AS id,
            cat.superclass,
            ANY_VALUE(s.geometry) AS geometry,
            COUNT(DISTINCT pois.id) AS poi_count,
            MIN(ST_DISTANCE(s.geometry, ST_GEOGFROMTEXT(pois.geometry))) AS closest_distance_meters,
            ARRAY_AGG(DISTINCT pois.name IGNORE NULLS) AS poi_names
        FROM stations_with_utilization s
        LEFT JOIN `{POI_TABLE}` pois
            ON ST_DWITHIN(s.geometry, ST_GEOGFROMTEXT(pois.geometry), 500)
        LEFT JOIN `{CATEGORIES_TABLE}` cat
            ON pois.fclass IN UNNEST(cat.subclass)
        GROUP BY s.id, cat.superclass
        ORDER BY s.id, poi_count DESC
    """
    # return client.query(query).to_dataframe()
    return run_query(query)

def load_road_data() -> pd.DataFrame:
    query = f"""
    select * from {ROADS_TABLE}
    """
    # return client.query(query).to_dataframe()
    return run_query(query)

# ---------------------- Feature Builders ----------------------

def generate_poi_features(poi_counts_df: pd.DataFrame, station_ids_df: pd.DataFrame) -> pd.DataFrame:
    poi_count_df = poi_counts_df.pivot_table(
        index="id", columns="superclass", values="poi_count", fill_value=0
    ).reset_index()

    poi_dist_df = poi_counts_df.pivot_table(
        index="id", columns="superclass", values="closest_distance_meters", fill_value=0
    ).reset_index()

    poi_count_df = poi_count_df.rename(columns={
        k: f"{k.lower().replace(' ', '_')}_count" for k in poi_count_df.columns if k != "id"
    })
    poi_dist_df = poi_dist_df.rename(columns={
        k: f"{k.lower().replace(' ', '_')}_dist" for k in poi_dist_df.columns if k != "id"
    })

    merged = pd.merge(pd.merge(station_ids_df, poi_count_df, on="id", how="left"), poi_dist_df, on="id", how="left")
    merged = merged.fillna(0)

    for col in poi_count_df.columns:
        if col == "id":
            continue
        category = col.replace("_count", "")
        dist_col = f"{category}_dist"
        if dist_col in merged.columns:
            mask = (merged[col] == 0) & (merged[dist_col] == 0)
            merged.loc[mask, dist_col] = 1000

    return merged



def build_road_features(road_df: pd.DataFrame) -> pd.DataFrame:
    road_classes = ['motorway', 'trunk', 'unclassified', 'primary', 'residential', 'living_street', 'secondary']

    dist = road_df.pivot(index='id', columns='type_road', values='closest_distance_meters').fillna(0)
    dist.columns = [f"road_distance_{col}" for col in dist.columns]

    dist = dist.applymap(lambda x: 1000 if x == 0 else x)


    for cls in road_classes:
        col = f"road_distance_{cls}"
        if col not in dist.columns:
            dist[col] = 1000

    count = road_df.pivot(index='id', columns='type_road', values='roads_count').fillna(0)
    count.columns = [f"road_count_{col}" for col in count.columns]

    for cls in road_classes:
        col = f"road_count_{cls}"
        if col not in count.columns:
            count[col] = 0

    return pd.concat([dist, count], axis=1).reset_index()

# ---------------------- Main ----------------------
def main():
    if SAVE_TO_DISK:
        os.makedirs(DATA_DIR, exist_ok=True)


    # Occupation
    occupation_df = load_occupation_data()
    save_dataframe(occupation_df, OCCUPATION_PATH, "Occupation data")

    # Static + Occupation
    static_df = load_static_data()
    static_occupation_df = pd.merge(occupation_df, static_df, on="id", how="left")
    save_dataframe(static_occupation_df, STATIC_OCCUPATION_PATH, "Static + Occupation data")

    # Timeseries
    timeseries_df = load_timeseries_data()
    save_dataframe(timeseries_df, f"{DATA_DIR}/timeseries_df.csv", "Timeseries data")

    # Hourly
    hourly_df = load_hourly_data()
    save_dataframe(hourly_df, f"{DATA_DIR}/hourly_df.csv", "Hourly data")


    # POI
    poi_counts_df = load_poi_data()
    log_df_info(poi_counts_df, "Raw POI data")

    station_ids_df = occupation_df[["id"]]
    poi_features_df = generate_poi_features(poi_counts_df, station_ids_df)
    save_dataframe(poi_features_df, STATION_FEATURES_PATH, "POI features (count & distance)")

    station_with_poi_df = pd.merge(static_occupation_df, poi_features_df, on="id", how="left")
    save_dataframe(station_with_poi_df, MERGED_POI_PATH, "Final merged POI features")

    # Road
    road_df = load_road_data()
    log_df_info(road_df, "Raw Road data")



    road_features = build_road_features(road_df)
    save_dataframe(road_features, ROAD_FEATURES_PATH, "Road features")

    station_with_road_df = pd.merge(static_occupation_df, road_features, on="id", how="left")
    save_dataframe(station_with_road_df, MERGED_ROAD_PATH, "Final merged station + road data")


if __name__ == "__main__":
    main()



