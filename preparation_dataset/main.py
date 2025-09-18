import subprocess

def run_script(script_name):
    result = subprocess.run(["python", script_name])
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed")

def main():
    print("🔄 Running read_table.py...")
    run_script("read_table.py")

    print("🔄 Running prepare_dataset.py...")
    run_script("prepare_dataset.py")

if __name__ == "__main__":
    main()



# # # TODO: Why 500 meter (), finding paper, map checking
# # # TODO: ineractive from bq
# # # TODO: Make it more than 500 for 1000 when we have no data   




# # # TODO: Check stations that have multiple chargers for different capacity

# # # TODO: superclass in categories table does not show all (Null values)
# # # TODO: Push to DVC
# # # TODO: Read data from DVC
# # # TODO: 
# # # TODO: https://download.geofabrik.de/europe.html
# # # TODO: https://github.com/SurplusMap/notebooks/blob/master/fr/pois_fr.ipynb
# # # TODO: https://console.cloud.google.com/bigquery?inv=1&invt=Ab2ScA&project=surpl%5B%E2%80%A6%5D8&resourceName=projects%2F270451612072%2Flocations%2Fus%2FdataExchanges%2Foverture_maps_data%2Flistings%2Foverture_maps_data&ws=!1m5!1m4!4m3!1sbigquery-public-data!2soverture_maps!3splace
# # # TODO: Add taxi hub, supermarket or parking
# # # TODO: 
# # # TODO: Visualization here
# # # TODO: Reasech for binary column which are not in feature importance
# # # TODO: 
# # # TODO: apply shap
# # # TODO: XGboosting
# # # TODO: NN
# # # TODO: 
# # # TODO: name and level of city instead of value for population
# # # TODO: Time spending * possiblte (matters for dc not hpc) binary value instead of count
# # # TODO: 
# # # TODO: Rank the station based on better logic that consider level of population, poi density
# # # TODO: 
# # # TODO: City
# # # TODO: 
# # # TODO: Save it automatically to the BQ
# # # TODO: Population and road query should be cheked add and for population for country
# # # TODO: 
# # # TODO: 
# # # TODO: Add perfomance increased 
# # # TODO: Add peak plot






# # # TODO: Test some location (rosos unox)
# # # TODO: mean average, exploring time populare in google maps













