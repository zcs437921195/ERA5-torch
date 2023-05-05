# import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from dataload.Datasets import ERA5


class VisualFrame:
    def __init__(self, save_fig: bool=True, save_path: str=None) -> None:
        self.save_fig = save_fig
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        self.save_path = save_path

    def visual_wind(self, u, v, lon, lat):
        # print(lon)
        # print(lat)
        lon2D, lat2D = np.meshgrid(lon, lat)
        fig = plt.figure()
        fig.quiver(lon2D, lat2D, u, v)
        plt.show()
        return 


    def visual_pcolor(self, lon, lat, preds, tgts=None, show_fig: bool=True):
        """
        :param lon: 经度 2D-ndarray 从小到大 [num, lon]
        :param lat: 纬度 2D-ndarray 从大到小 [num, lat]
        :param preds: 3D-ndarray [num, data: 2D]

        """
        n = len(preds)
        for i in range(n):
            lon2D, lat2D = np.meshgrid(lon[i], lat[i])
            if tgts is None:
                plt.figure()
                plt.pcolor(lon2D, lat2D, preds[i])
                plt.title("Pred")
                if self.save_fig:
                    plt.savefig(os.path.join(self.save_path, '%d.jpg' % i))
            else:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.pcolor(lon2D, lat2D, preds[i])
                ax1.set_title("Pred")
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.pcolor(lon2D, lat2D, tgts[i])
                ax2.set_title("Ground Truth")
                if self.save_fig:
                    plt.savefig(os.path.join(self.save_path, '%d.jpg' % i))
        if show_fig:
            plt.show()
        return 0



if __name__ == '__main__':
    level = "Ground"
    ll = ERA5()
    u_file = "/home/chuansai/ERA5_torch/data/Ground/10m_u_component_of_wind/era5.10m_u_component_of_wind.20140630.nc"
    data_dict = ll.read_era5_data(u_file, level)
    u = [data_dict["data"][6][:80, :80]]
    lon = [data_dict["lon"][:80]]
    lat = [data_dict["lat"][:80]]
    visual = VisualFrame(save_path="/home/chuansai/ERA5_torch/outputs/visual")
    visual.visual_pcolor(lon, lat, u, u)
