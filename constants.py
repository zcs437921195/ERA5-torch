ERA5_DATA_ELEMENTS = {
    'Ground': [
        'ptype',
        'zero_degree_level',  # zero degree level
        '100m_u_component_of_wind',
        '100m_v_component_of_wind',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2_metre_dewpoint_temperature',
        '2m_Temperature',
        'convective_available_potential_energy',
        'forecast_albedo',
        'total_column_water',
        'total_column_water_vapour',
        'Total_Precipitation',
    ],
    'Pressure': [
        'geopotential',
        'relative_humidity',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'vertical_velocity',
    ],
}

ERA5_PRESSURE_LEVEL = [
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    800,
    850,
    900,
    950,
    1000
]

ERA5_NAME_TRANS = {
    '100m_u_component_of_wind': '100m_u_component_of_wind',
    '100m_v_component_of_wind': '100m_v_component_of_wind',
    '10m_u_component_of_wind': '10m_u_component_of_wind',
    '10m_v_component_of_wind': '10m_v_component_of_wind',
    '2_metre_dewpoint_temperature': '2m_dewpoint_temperature',
    '2m_Temperature': '2mt',
    'convective_available_potential_energy': 'convective_available_potential_energy',
    'forecast_albedo': 'forecast_albedo',
    'total_column_water': 'total_column_water',
    'total_column_water_vapour': 'total_column_water_vapour',
    'Total_Precipitation': 'tp',
    'zero_degree_level': 'zdl',
    'ptype': 'ptype',
    'geopotential': 'geopotential',
    'relative_humidity': 'relative_humidity',
    'temperature': 'temperature',
    'u_component_of_wind': 'u_component_of_wind',
    'v_component_of_wind': 'v_component_of_wind',
    'vertical_velocity': 'vertical_velocity'
}

ERA5_LON_LAT_INFO = {
    'lon': [70., 140.],
    'lat': [0., 60.],
    'resolution': 0.25,
    'cut_lon': [97., 125.],
    'cut_lat': [18., 40.]
}

ERA5_OFFSETS = [i for i in range(24)]