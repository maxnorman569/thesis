# data imports
import xarray as xr

# misc imports
import os
from typing import List, Tuple

class MissionData:

    def __init__(self, 
                 root_folder : str, 
                 mission_name : str, 
                 years : List[str], 
                 months : List[str], 
                 latitude_range : Tuple[float, float] = (-90., 90.),
                 longitude_range : Tuple[float, float] = (-180., 180.)
                 ):
        """
        Class to load and store mission data.

        Arguments:
            root_folder (str)       : The folder path where the mission data is located.
            mission_name (str)      : The name of the mission. Should be one of the acceptable mission names. (i.e in ['e1', 'e1g', 'e2', 'tp', 'tpn', 'g2', 'j1', 'j1n', 'j1g', 'j2', 'j2n', 'j2g', 'j3', 'j3n', 'en', 'enn', 'c2', 'c2n', 'al', 'alg', 'h2a', 'h2ag', 'h2b', 'h2c', 's3a', 's3b', 's6a-hr', 's6a-lr'])
            year (str)              : The year of the mission as a string. (i.e 'YYYY')
            month (str)             : The month of the mission as a string. (i.e 'MM', '01' for January)
            latitude_range (tuple)  : The latitude range of the mission data. (min, max) from -90 to 90.
            longitude_range (tuple) : The longitude range of the mission data. (min, max) from -180 to 180.

        Returns:
            None

        Raises:
            ValueError              : If the provided mission name is not in the list of acceptable mission names.
        """

        # check that provided mission name is valid
        available_missions = ['e1', 'e1g', 'e2', 'tp', 'tpn', 
                              'g2', 'j1', 'j1n', 'j1g', 'j2', 
                              'j2n', 'j2g', 'j3', 'j3n', 'en', 
                              'enn', 'c2', 'c2n', 'al', 'alg', 
                              'h2a', 'h2ag', 'h2b', 'h2c', 's3a', 
                              's3b', 's6a-hr', 's6a-lr']
        
        # raise error if mission name not in list of available missions
        if mission_name not in available_missions:
            raise ValueError("Invalid mission name provided. Mission name must be one of the following: {}".format(available_missions))
        
        # set class attributes
        self.mission_name = mission_name
        self.mission_folder = os.path.join(root_folder, f'cmems_obs-sl_eur_phy-ssh_my_{self.mission_name}-l3-duacs_PT1S')
        self.years = years
        self.months = months
        self.min_latitude = latitude_range[0]
        self.max_latitude = latitude_range[1]
        self.min_longitude = longitude_range[0]
        self.max_longitude = longitude_range[1]
        self.mission_data = self.load_data()


    def load_data(self, ):
        """
        Loades the mission data from the provided arguments.

        Arguments:
            None

        Returns:
            month_dataset (xarray.Dataset)  : The mission data for the provided arguments.

        Raises:
            ValueError                      : If no data files are found in the provided directory.
        """
        
        # print mission name
        print("\nLoading data for mission: '{}'".format(self.mission_name))

        # load mission data
        datasets = []
        for year in self.years:
            for month in self.months:

                # get data directory
                data_dir = os.path.join(self.mission_folder, year, month)

                # check data directory exists
                if os.path.isdir(data_dir):
                    pass
                else:
                    print('> {}-{} | Directory does not exist: {}'.format(year, month, data_dir))
                    continue

                # get data files
                files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

                # check data files exist
                if len(files) == 0:
                    print('> {}-{} | Data files do not exist: {}'.format(year, month, data_dir))
                    continue
                
                # load data files as xarray datasets
                for file in files:
                    try:
                        data = xr.open_dataset(file)

                        # convert longitude from 0-360 to -180-180
                        data['longitude'] = xr.where(data['longitude'] > 180., data['longitude'] - 360., data['longitude'])

                        # filter data by latitude and longitude
                        longitude_mask = xr.where((data['longitude'] > self.min_longitude) & (data['longitude'] < self.max_longitude), True, False)
                        latitude_mask = xr.where((data['latitude'] > self.min_latitude) & (data['latitude'] < self.max_latitude), True, False)
                        data = data.where(longitude_mask & latitude_mask, drop = True)

                        # append data to list of datasets
                        datasets.append(data)

                    except:
                        pass
                
                # print completion message
                print('> {}-{} | completed'.format(year, month))

        # concatenate the datasets along the time dimension
        if len(datasets) == 0:
            return xr.Dataset(
                coords=dict(
                        time=None, 
                        longitude=None, 
                        latitude=None),
                data_vars=dict(
                        cycle = None, 
                        track = None, 
                        sla_unfiltered = None, 
                        sla_filtered = None,
                        dac = None, 
                        ocean_tide = None, 
                        internal_tide = None, 
                        lwe = None, 
                        mdt = None, 
                        tpa_correction = None,),
                attrs=dict(
                        description="Empty dataset, no data found."),
            )
        
        elif len(datasets) == 1:
            mission_data = datasets[0]

        else:
            mission_data = xr.concat(datasets, dim = 'time')

        return mission_data
    

class MissionAgnosticData:
    
    def __init__(self, 
                root_folder : str,
                mission_names : List[str], 
                years : List[str], 
                months : List[str],
                latitude_range : Tuple[float, float] = (-90., 90.),
                longitude_range : Tuple[float, float] = (-180., 180.)
                 ):
        """
        Class to load and store mission agnostic data. (i.e data that is not specific to a single mission.)

        Arguments:
            root_folder (str)    : The folder path where the mission data is located.
            mission_names (str)      : The name of the mission. Should be one of the acceptable mission names.
            years (str)              : The year of the mission.
            months (str)             : The month of the mission.
            latitude_range (tuple)  : The latitude range of the mission data. (min, max) from -90 to 90.
            longitude_range (tuple) : The longitude range of the mission data. (min, max) from -180 to 180.

        Returns:
            None

        Raises:
            ValueError              : If the provided mission names are not in the list of acceptable mission names.
        """

        # check that provided mission name is valid
        available_missions = ['e1', 'e1g', 'e2', 'tp', 'tpn', 
                              'g2', 'j1', 'j1n', 'j1g', 'j2', 
                              'j2n', 'j2g', 'j3', 'j3n', 'en', 
                              'enn', 'c2', 'c2n', 'al', 'alg', 
                              'h2a', 'h2ag', 'h2b', 'h2c', 's3a', 
                              's3b', 's6a-hr', 's6a-lr']

        if set(mission_names).issubset(set(available_missions)):
            pass
        else:
            raise ValueError("Invalid mission name provided. Mission names must be in: {}".format(available_missions))

        # set class attributes
        self.data = xr.concat([MissionData(root_folder, mission_name, years, months, latitude_range, longitude_range).mission_data for mission_name in mission_names], dim = 'time')