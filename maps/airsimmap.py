# handles airsim releases
from maps.map import Map
from utils import read_json, write_json
import subprocess
from os import getcwd
from component import _init_wrapper

class AirSimMap(Map):

    # constructor
    @_init_wrapper
    def __init__(self, settings=None, setting_files=['base'], release_file='Blocks',
                    release_directory='D:/airsim_releases/', settings_directory='maps/airsim_settings/',
                    _name=None):
        super().__init__()
        self._release_path = release_directory + release_file + '/' + release_file + '.exe'
        if settings is None:
            self.settings = self._read_settings(settings_directory, setting_files)
        else:
            self.settings = settings
        self._settings_path = getcwd() + '/settings.json'
        self._write_settings(self.settings, self._settings_path)

    # launch airsim map
    def connect(self):
        terminal_command = f'{self._release_path} -settings=\"{self._settings_path}"'
        print(terminal_command)
        subprocess.Popen(terminal_command, close_fds=True)
        print(f'Send any key when AirSim {self.release_file}.exe is fully launched, this make take several minutes....')
        x = input()

    # close airsim app
    def disconnect(self):
        terminal_command = 'taskkill /f /im ' + self.release_file + '.exe'
        subprocess.call(terminal_command)

    @staticmethod
    def _read_settings(settings_directory, setting_files):
        merged_settings = {}
        for setting_component in setting_files:
            setting_path = settings_directory + setting_component + '.json'
            setting = read_json(setting_path)
            merged_settings.update(setting)
        return merged_settings

    @staticmethod
    def _write_settings(merged_settings, settings_path):
        write_json(merged_settings, settings_path)