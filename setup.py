import utils
import os

if not os.path.exists('local'):
    os.mkdir('local')
if not os.path.exists('local/airsim_maps'):
    os.mkdir('local/arisim_maps')
if not os.path.exists('local/runs'):
    os.mkdir('local/runs')

instance_name = 'MyPC'
render_screen = True
airsim_release = 'Blocks'

utils.set_global_parameter('instance_name', instance_name)
utils.set_global_parameter('render_screen', render_screen)
utils.set_global_parameter('airsim_release', airsim_release)

utils.write_global_parameters()
