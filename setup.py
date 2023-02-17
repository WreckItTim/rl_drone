import utils
import os

instance_name = 'hackfest4'
render_screen = True

utils.set_global_parameter('instance_name', instance_name)
utils.set_global_parameter('render_screen', render_screen)

utils.write_global_parameters()