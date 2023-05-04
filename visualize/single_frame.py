import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from dataload.Datasets import ERA5


class VisualFrame:
    def __init__(self) -> None:
        pass

    def visual_wind(self, u, v, lon, lat):
        # print(lon)
        # print(lat)
        lon2D, lat2D = np.meshgrid(lon, lat)
        fig = plt.figure()
        fig.quiver(lon2D, lat2D, u, v)
        plt.show()
        return 


    def visual_pcolor(self, lon, lat, data):
        """
        :param lon: 经度 1D-ndarray 从小到大
        :param lat: 纬度 1D-ndarray 从大到小
        :param data: 2D-ndarray

        """
        lon2D, lat2D = np.meshgrid(lon, lat)
        plt.figure()
        plt.pcolor(lon2D, lat2D, data)
        plt.show()
        return 0



if __name__ == '__main__':
    level = "Ground"
    ll = ERA5()
    u_file = "../data/Ground/10m_u_component_of_wind/era5.10m_u_component_of_wind.20140630.nc"
    data_dict = ll.read_era5_data(u_file, level)
    u = data_dict["data"][6]
    lon = data_dict["lon"]
    lat = data_dict["lat"]
    visual = VisualFrame()
    visual.visual_pcolor(lon, lat, u)
