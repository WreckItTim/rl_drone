import rl_utils as utils
import os

def mkdir_cmd(p):
    if not os.path.exists(p):
        os.mkdir(p)

mkdir_cmd('local/')
mkdir_cmd('local/airsim_maps/')
mkdir_cmd('local/models/')
mkdir_cmd('local/runs/')
mkdir_cmd('local/temp/')

instance_name = 'MyPC'
render_screen = True

utils.set_global_parameter('instance_name', instance_name)
utils.set_global_parameter('render_screen', render_screen)

utils.write_global_parameters()


# ignore local Dropbox files -- same as those in .gitignore for GitHub

import platform
OS = platform.system()

windows = OS in ['Windows']

def ignore_cmd(p):
    command = f'attr -s com.dropbox.ignored -V 1 \"{p}\"'
    os.system(command)

if windows:
    import subprocess
    def ignore_cmd(p):
        command = f'Set-Content -Path \"{p}\" -Stream com.dropbox.ignored -Value 1'
        subprocess.call('%SystemRoot%\\system32\\WindowsPowerShell\\v1.0\\powershell.exe ' + command, shell=True)


ignore_cmd('local')
mkdir_cmd('__pycache__/')
ignore_cmd('__pycache__')
mkdir_cmd('.vs/')
ignore_cmd('.vs')
    
dirs = [d for d in os.listdir() if os.path.isdir(d) and d[0] not in ['.','_']]
for d in dirs:
    pycache = d + '/__pycache__'
    mkdir_cmd(pycache + '/')
    ignore_cmd(pycache)