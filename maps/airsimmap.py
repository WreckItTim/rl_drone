from maps.map import Map
import utils
import subprocess
import os
from component import _init_wrapper

# handles airsim release executables
class AirSimMap(Map):

    # constructor, pass in a dictionary for settings and/or file paths to merge multiple settings .json files
    @_init_wrapper
    def __init__(self,
                 # can define json-structured settings
                 settings:dict = None,
                 # or define setting files to read in from given directory
                 settings_directory:str = 'maps/airsim_settings/',
                 # will aggregate passed in json settings and all files
                 # update priority is given to the settings argument and last listed files
                 # amoung the settings must be information for which sensors to use
                 # below arg is a list of file names, see the maps/airsim_settings for examples
                 setting_files:list = ['vanilla'],
                 # directory to release .exe file to be launched
                 release_directory:str = None,
                 # name of release .exe file to be launched, inside relase_directory
                 release_name:str = None, 
                 ):
        super().__init__()
        # get path to release executable file to launch
        self._release_path = os.path.join(release_directory, release_name)
        # create setting dictionary
        self.settings = {}
        if settings is not None:
            self.settings = settings
        # read in any other settings files
        other_settings = {}
        if setting_files is not None:
            other_settings = self.read_settings(settings_directory, setting_files)
        # merge all settings
        self.settings.update(other_settings)
        # write to temp file to be read in when launching realease executable
        self._settings_path = os.getcwd() + '/temp/overwrite_settings.json'
        self.write_settings(self.settings, self._settings_path)

    # launch airsim map
    def connect(self):
        super().connect()
        # send command to terminal to launch the relase executable
        terminal_command = f'{self._release_path} -settings=\"{self._settings_path}"'
        subprocess.Popen(terminal_command, close_fds=True)
        # prompt user to confirm when launch is successful (can launch manually if needs be)
        print(f'Send any key when AirSim {self._release_path} is fully launched, this make take several minutes....')
        x = input()

    # close airsim map
    def disconnect(self):
        # send command to terminal to kill the relase executable
        terminal_command = 'taskkill /f /im ' + self.release_name
        subprocess.call(terminal_command)
    
    # read several json files with Airsim settings and merge
    def read_settings(self, settings_directory, setting_files):
        merged_settings = {}
        for setting_file in setting_files:
            setting_path = os.path.join(settings_directory, setting_file) + '.json'
            setting = utils.read_json(setting_path)
            merged_settings.update(setting)
        return merged_settings

    # write a json settings dictionary to file
    def write_settings(self, merged_settings, settings_path):
        utils.write_json(merged_settings, settings_path)