# data imports
import xarray as xr

# misc imports
import os
from typing import List

class MissionData:

    def __init__(self, root_folder : str, mission_name : str, years : List[str], months : List[str]):
        """
        Class to load and store mission data.

        Arguments:
            mission_folder (str)    : The folder path where the mission data is located.
            mission_name (str)      : The name of the mission. Should be one of the acceptable mission names.
            year (str)              : The year of the mission.
            month (str)             : The month of the mission.

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
        datasets = []
        for year in self.years:
            for month in self.months:

                # get data directory
                data_dir = os.path.join(self.mission_folder, year, month)

                # check data directory exists
                if os.path.isdir(data_dir):
                    pass
                else:
                    raise ValueError('Directory does not exist: {}'.format(data_dir))

                # get data files
                files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

                # raise error if no data files found
                if len(files) == 0:
                    raise ValueError('No data files found in directory: {}'.format(data_dir))
                
                # load data files as xarray datasets
                for file in files:
                    try:
                        datasets.append(xr.open_dataset(file))
                    except:
                        pass

        # concatenate the datasets along the time dimension
        mission_data = xr.concat(datasets, dim='time')

        return mission_data