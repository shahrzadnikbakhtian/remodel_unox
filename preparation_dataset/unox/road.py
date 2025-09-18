CREATE OR REPLACE TABLE `dev_sherry.sw_charging_stations` AS
WITH filtered_stations AS (
    SELECT id, capacity_kw, size_total, geometry
    FROM `surplusmap-393908.utilization_rate_aggregated.charging_stations`
    WHERE source = 'sw'
    #   AND is_approved = TRUE
      AND capacity_kw > 100
),

base_data AS (
    SELECT
        u.id,
        DATE(u.hour) AS day
    FROM `surplusmap-393908.utilization_rate_aggregated.charging_stations_rate_of_utilization_aggregated` u
    JOIN filtered_stations s ON u.id = s.id
    WHERE DATE(u.hour) BETWEEN DATE '2024-07-01' AND LAST_DAY(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH))
),

utilization_summary AS (
    SELECT
        id,
        MIN(day) AS start_day,
        MAX(day) AS end_day,
        DATE_DIFF(MAX(day), MIN(day), DAY) + 1 AS calendar_day_count,
        COUNT(DISTINCT day) AS active_days_count,
    FROM base_data
    GROUP BY id
)

SELECT
    s.id,
    s.capacity_kw,
    s.size_total,
    s.geometry
FROM utilization_summary u
JOIN filtered_stations s ON u.id = s.id
WHERE u.active_days_count > 100;






CREATE OR REPLACE TABLE `dev_sherry.sw_roads_raw` AS
SELECT *
FROM `surplusmap-393908.overture.overture_roads`
WHERE country = 'Switzerland';





CREATE OR REPLACE TABLE `dev_sherry.sw_roads` AS
SELECT
  id,
  ST_GEOGFROMTEXT(geometry) AS geometry,  -- convert to GEOGRAPHY
  class,
  name,
  country
FROM `dev_sherry.sw_roads_raw`;





CREATE OR REPLACE TABLE `dev_sherry.raw_road_sw` AS
SELECT
    s.id,
    roads.class AS type_road,
    ANY_VALUE(s.geometry) AS geometry,
    COUNT(DISTINCT roads.id) AS roads_count,
    MIN(ST_DISTANCE(s.geometry, roads.geometry)) AS closest_distance_meters,
    ARRAY_AGG(DISTINCT roads.name IGNORE NULLS) AS roads_names
FROM 
    `dev_sherry.sw_charging_stations` s
LEFT JOIN 
    `dev_sherry.sw_roads` roads
    ON ST_DWITHIN(s.geometry, roads.geometry, 500)
GROUP BY 
    s.id, roads.class
ORDER BY 
    s.id, roads_count DESC;






