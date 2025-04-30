# %matplotlib inline
import glob
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import dask.dataframe as dd
import matplotlib.pyplot as plt
import openpyxl

from pandana.loaders import osm
from pandana.loaders.pandash5 import network_to_pandas_hdf5
import pandana as pdna
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import osmnx as ox
from random import sample
from tqdm import tqdm

import networkx as nx

# from scipy.stats import mstats
# from shapely.geometry import Polygon
# import seaborn as sns
# from urban_access.data.urban_access import create_hexgrid, create_hex_access
#directory = os.chdir(r'C:\Users\Leonardo\OneDrive\Documents\TU_Delft\CodingProjects\UrbanAccessibilityWorld')

print("loading data")
# download Eindhoven City data extract from general planet.osm.pbf at https://download.bbbike.org/osm/bbbike/Eindhoven/
# download osmpois.jar at https://github.com/MorbZ/OsmPoisPbf/ using uac_filter.txt, in terminal run:
# java -jar osmpois.jar --filterFile uac_filter.txt --printHeader planet.osm.pbf

df = pd.read_csv("dataPreparation/Eindhoven.csv", sep='|')
df['category'] = df['category'].astype(str)
df = df[df.category.str.isnumeric()==True]
df['category'] = df['category'].astype(float)
df = df.rename(columns={"category": "poi_type_id"})

poi_types = pd.read_excel("dataPreparation/poi_code_name_mapper.xlsx")
poi_types = poi_types.replace(" ", np.nan).dropna()

df = df.merge(poi_types, on="poi_type_id")

# OUTDATED (download this data at http://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_STAT_UCDB2015MT_GLOBE_R2019A/V1-2/)
# An updated release of the UCDB is now available: Urban Centre Database R2024A. Updated edition, built using the GHS-SMOD R2023A dataset, split by country.
# (released on: 05/11/2024) https://human-settlement.emergency.copernicus.eu/ghs_ucdb_2024.php
# download data only about Eindhoven City at https://human-settlement.emergency.copernicus.eu/data/ucdb2024/single/GHS_UCDB_GLOBE_2627_R2024A_V1_0.zip
# uc = pd.read_csv("dataPreparation/GHS_UCDB_GLOBE_2627_R2024A_V1_0/GHS_UCDB_GLOBE_2627_R2024A.xlsx", encoding = "ISO-8859-1", engine='python')

print("preparing data")
# merge df with uc data
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon, df.lat)).set_crs(4326)

original_count = len(gdf)

geo_uc = gpd.read_file("dataPreparation/GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A_V1_0/GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A.gpkg", layer="GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A")

# Changes in column naming:
# ID_HDC_G0 -> ID_UC_G0
# CTR_MN_NM -> GC_CNT_GAD_2025
# UC_NM_MN -> GC_UCN_MAI_2025
# P15 -> GC_POP_TOT_2025
# AREA -> GC_UCA_KM2_2025
# geometry -> geometry

geo_uc = geo_uc[['ID_UC_G0', "GC_CNT_GAD_2025", "GC_UCN_MAI_2025", "GC_POP_TOT_2025", "GC_UCA_KM2_2025", "geometry"]].dropna()
geo_uc["UC_NM_CTR"] = geo_uc["GC_UCN_MAI_2025"] + ", " +  geo_uc["GC_CNT_GAD_2025"]

geo_uc = geo_uc.to_crs(gdf.crs)

# joining to exclude any POIs that do not lie within the exact borders of Eindhoven
# since bbbike Eindhoven extract is bounding-box based, not administratively clipped,
# they include everything that falls withing a rectangle that roughly contains Eindhoven
gdf = gdf.sjoin(geo_uc, how="inner")

df = gdf

joined_count = len(df)
truncated_count = original_count - joined_count
#print(f"Number of POIs excluded (truncated) during spatial join: {truncated_count}, original count: {original_count}")

# make df of ratio poi/pop to filter data
df["count"] = 1
df_poi_per_pop = df.groupby(["ID_UC_G0", "UC_NM_CTR"]).agg({"GC_POP_TOT_2025":"mean", "count":"sum", "GC_UCA_KM2_2025":"mean"}).reset_index()
df_poi_per_pop["poi_per_pop"] = df_poi_per_pop["count"]/df_poi_per_pop["GC_POP_TOT_2025"]
df_poi_per_pop["poi_per_km2"] = df_poi_per_pop["count"]/df_poi_per_pop["GC_UCA_KM2_2025"]

# list of urban centers to keep (at least 1 POI per km2)
uc_keep = df_poi_per_pop[(df_poi_per_pop.poi_per_km2>=1)&(df_poi_per_pop["count"]>=20)].ID_UC_G0.to_list()
df_keep = df[df['ID_UC_G0'].isin(uc_keep)]

ratio_filtered_count = len(df_keep)

# function to compute accessibility indicator for each category with regards to each node on the network
def create_access_gdf(pois = None, network = None, maxdist = 1000):

    '''Computes walking distances from each street intersection to each of the seven categories of urban amenities'''

    # # precomputes the shrotest-path distances from every node in the network to all other nodes within 1500 meters
    network.precompute(maxdist + 1) # 1500 meters = ~15-minute walk

    # Maps each POI to its nearest street intersection (or node) in the walkable street network
    #df_keep["node_ids"] = network.get_node_ids(df_keep["lat"], df_keep["lon"])

    cat_list_str = pois['category'].unique()
    print("cat_list_str: ", cat_list_str)

    # Dummy run for initialization
    for cat in cat_list_str:
        pois_subset = pois[pois['category'] == cat]
        network.set_pois(
            category=cat,
            maxdist=1,
            maxitems=len(pois_subset),
            x_col=pois_subset['lon'],
            y_col=pois_subset['lat']
        )
        accessibility = network.nearest_pois(distance=1, category=cat)

    accessibility = pd.DataFrame()

    print("Calculating distances for each category...")
    for cat in cat_list_str:
        pois_subset = pois[pois['category'] == cat]
        network.set_pois(
            category=cat,
            maxdist=maxdist,
            maxitems=len(pois_subset),
            x_col=pois_subset['lon'],
            y_col=pois_subset['lat']
        )
        accessibility[str(cat)] = network.nearest_pois(distance=maxdist, category=cat)

    print("Merging accessibility results...")
    print(accessibility.reset_index().columns)
    print(network.nodes_df.reset_index().columns)
    #merge accessibility values with walk nodes ids geodataframe
    access = pd.merge(
        accessibility.reset_index().drop(columns=[1], errors="ignore"),  # drop extra col if exists
        network.nodes_df.reset_index(),
        #on ='osmid'
        on = 'id'
    )
    # add metadata
    for col in ['ID_UC_G0', 'GC_CNT_GAD_2025', 'GC_UCN_MAI_2025', 'GC_POP_TOT_2025', 'GC_UCA_KM2_2025', 'UC_NM_CTR']:
        if col in pois.columns:
            access[col] = pois[col].unique()[0]
    
    #convert to geodataframe
    access = gpd.GeoDataFrame(access, geometry=gpd.points_from_xy(access.x, access.y), crs="EPSG:4326")
    
    #drop NaNs
    access = access.dropna()
    # access = access[~(access == 10000).any(1)].reset_index().drop('index', axis=1)

    return access

# city = 'Eindhoven'

# print(f"calculating accessibility for {city}")

# # subset pois for specific urban center
# pois = df_keep
# pois = gpd.GeoDataFrame(pois, geometry=gpd.points_from_xy(pois.lon, pois.lat))
# # # get boundary coords of urban center
# # lng_min = pois.total_bounds[0] #lng_min
# # lat_min = pois.total_bounds[1] #lat_min
# # lng_max = pois.total_bounds[2] #lng_max
# # lat_max = pois.total_bounds[3] #lat_max

# # # get pedestrian network
# # network = osm.pdna_network_from_bbox(lat_min, lng_min, lat_max, lng_max, network_type='walk')

# print("initialize network pois")
# # Use osmnx to get walkable street network for Eindhoven
# G = ox.graph_from_place("Eindhoven, Netherlands", network_type = "walk")
# G = ox.project_graph(G)

# # Convert to Pandana network
# # splits G into two gdf: nodes (intersections, dead ends) and edges (street segments)
# gdf_nodes, gdf_edges = ox.convert.graph_to_gdfs(G, nodes = True, edges = True, node_geometry = True, fill_edge_geometry = True) 

# gdf_nodes = gdf_nodes.reset_index()
# gdf_edges = gdf_edges.reset_index()

# gdf_nodes['osmid'] = gdf_nodes['osmid'].astype(np.int64)
# gdf_nodes['x'] = gdf_nodes['x'].astype(np.float64)
# gdf_nodes['y'] = gdf_nodes['y'].astype(np.float64)

# gdf_edges['u'] = gdf_edges['u'].astype(np.int64)
# gdf_edges['v'] = gdf_edges['v'].astype(np.int64)
# gdf_edges['length'] = gdf_edges['length'].astype(np.float64)

# gdf_edges = gdf_edges[gdf_edges['u'].isin(gdf_nodes['osmid']) & gdf_edges['v'].isin(gdf_nodes['osmid'])]

# gdf_nodes.set_index('osmid', inplace = True)

# # Then pass directly to pandana
# # IMPORTANT: to run, downgrade numpy to 1.23.5
# net = pdna.Network(
#     gdf_nodes['x'],
#     gdf_nodes['y'],
#     gdf_edges['u'],
#     gdf_edges['v'],
#     gdf_edges[['length']]
# )

# lng_min = pois.total_bounds[0] #lng_min
# lat_min = pois.total_bounds[1] #lat_min
# lng_max = pois.total_bounds[2] #lng_max
# lat_max = pois.total_bounds[3] #lat_max
    
# # get pedestrian network
# network = osm.pdna_network_from_bbox(lat_min, lng_min, lat_max, lng_max, network_type='walk')

# access = create_access_gdf(pois = pois, network = network, maxdist = 1500)

# access.to_csv(f"{city}.csv")

# print(access.head())

# Changes in column naming:
# ID_HDC_G0 -> ID_UC_G0
# CTR_MN_NM -> GC_CNT_GAD_2025
# UC_NM_MN -> GC_UCN_MAI_2025
# P15 -> GC_POP_TOT_2025
# AREA -> GC_UCA_KM2_2025
# geometry -> geometry


# Filter for Eindhoven
eindhoven_df = df_keep[df_keep["UC_NM_CTR"].str.contains("Eindhoven", case=False)]

if not eindhoven_df.empty:
    city = eindhoven_df["ID_UC_G0"].unique()[0]
    city_name = eindhoven_df["UC_NM_CTR"].unique()[0]
    print(f"Calculating accessibility for {city_name}")

    pois = gpd.GeoDataFrame(
        eindhoven_df, geometry=gpd.points_from_xy(eindhoven_df.lon, eindhoven_df.lat)
    )

    lng_min, lat_min, lng_max, lat_max = pois.total_bounds

    network = osm.pdna_network_from_bbox(lat_min, lng_min, lat_max, lng_max, network_type='walk')
    access = create_access_gdf(pois=pois, network=network, maxdist=1500)

    access.to_csv(f"{city}.csv")
else:
    print("Eindhoven data not found in df_keep.")