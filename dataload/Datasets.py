import numpy as np
import datetime
import os
from netCDF4 import Dataset

import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from constants import ERA5_LON_LAT_INFO
from constants import ERA5_NAME_TRANS
from constants import ERA5_PRESSURE_LEVEL
        

class ERA5:
    def __init__(self, data_cfg: dict=None) -> None:
        self.data_cfg = data_cfg


    def get_era5_file(self, data_dir: str, level: str, date: datetime.datetime, element: str) -> str:
        return os.path.join(data_dir, level, element, "era5.%s.%s.nc" % (ERA5_NAME_TRANS[element], date.strftime("%Y%m%d")))


    def normalize_ground(self, inputs, element):
        if element == "10m_u_component_of_wind":
            return 1. / (1 + np.exp(-0.3 * inputs))


    def read_era5_data(self, file_path: str, level: str) -> np.ndarray:
        """
        用来读取ERA5数据, ERA5数据应该包含有time, latitude, longitude, 变量名 一共4个dimensions
        :param file_path: 数据路径
        :param level: 是气压层(Pressure)还是地面层(Ground)数据, 气压层多了一个level变量, 需要考虑其维度。
        :return: np.ndarray 
        [时效, 纬度, 经度] -- Ground
        [时效, 气压层, 纬度, 经度] -- Pressure

        """
        f = Dataset(file_path)
        # print(f)
        # t = f.variables['time'][:]
        # for x in t:
        #     print(datetime.datetime(1900, 1, 1, 0, 0) + datetime.timedelta(hours=int(x)))
        # print(f.variables['level'][:])
        # print(list(f.variables['level'][:]).index(300))
        variable_name = list(f.variables.keys())[-1]
        # print(variable_name)
        # ERA5经度从小到大，纬度从大到小排列
        era5_lon = list(f.variables["longitude"][:])
        era5_lat = list(f.variables["latitude"][:])
        # cut_lon1, cut_lon2 = ERA5_LON_LAT_INFO['cut_lon']  # cut_lon1 < cut_lon2
        # cut_lat1, cut_lat2 = ERA5_LON_LAT_INFO['cut_lat']  # cut_lat1 < cut_lat2
        # cut_lon_index1, cut_lon_index2 = era5_lon.index(cut_lon1), era5_lon.index(cut_lon2)
        # cut_lat_index1, cut_lat_index2 = era5_lat.index(cut_lat1), era5_lat.index(cut_lat2)
        # print(era5_lon[cut_lon_index1:cut_lon_index2+1])
        # print(era5_lat[cut_lat_index2:cut_lat_index1+1])
        data = f.variables[variable_name][:]
        # if level == "Pressure":
        #     data = data[:, :, cut_lat_index2:cut_lat_index1+1, cut_lon_index1:cut_lon_index2+1]
        #     level_index = [list(f.variables['level'][:]).index(i) for i in ERA5_PRESSURE_LEVEL]
        #     data = data[:, level_index]
        # else:
        #     data = data[:, cut_lat_index2:cut_lat_index1+1, cut_lon_index1:cut_lon_index2+1]
        #     # data = np.expand_dims(data, axis=0)
        return {"data": data, "lon": era5_lon, "lat":era5_lat}

    
    def load(self) -> np.ndarray:
        """
        :return: 5-dim np.ndarray [N, E, T, H, W] --- [天数, 时效数, 要素数, 高度, 宽度]
        """
        level = self.data_cfg.get("level", None)
        data_dir = self.data_cfg.get("data_dir", "")
        sql_len = self.data_cfg.get("total_sql_len")
        height, width = self.data_cfg.get("height"), self.data_cfg.get("width")
        load_data = []
        if level is not None:
            dates = self.data_cfg.get("dates", [])
            elements = self.data_cfg.get("elements", [])
            for d in dates:
                one_sample = []
                for e in elements:
                    data_file = self.get_era5_file(data_dir, level, d, e)
                    data = self.read_era5_data(data_file, level)
                    lon, lat = data["lon"][:width], data["lat"][:height]
                    data = data["data"][:sql_len, :height, :width]
                    if level == "Ground":
                        data = self.normalize_ground(data, e)
                    one_sample.append(np.expand_dims(data, axis=1))
                load_data.append(np.expand_dims(np.concatenate(one_sample, axis=1), axis=0))
            load_data = np.concatenate(load_data, axis=0)
        return {"data": load_data, "lon": lon, "lat": lat}

def test():
    return 0
        

if __name__ == '__main__':  
    cfg = dict(
        data_dir="/Users/zhouchuansai/Desktop/code/transformer_ERA5/data/",
        level="Ground",
        dates=[datetime.datetime(2014, 6, 30, 0)],
        elements=["10m_u_component_of_wind"],
    )
    ll = ERA5(cfg)
    era5 = ll.load()
    # print(x.shape)
    # d = datetime.datetime(2014, 6, 30, 0)
    # level = "Ground"
    # element = "10m_u_component_of_wind"
    # era5 = get_era5_file(os.path.join("/Users/zhouchuansai/Desktop/code/transformer_ERA5/data", level), d, element)
    # era5 = read_era5_data(era5, level)
    # print(isinstance(era5, np.ndarray))
    print(era5.shape)
    print(era5.max(), era5.min())