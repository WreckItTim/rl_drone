import utils
import os

instance_name = 'hacknet4'
render_screen = True
airsim_release = 'Hood'

utils.set_global_parameter('instance_name', instance_name)
utils.set_global_parameter('render_screen', render_screen)
utils.set_global_parameter('airsim_release', airsim_release)

utils.write_global_parameters()
