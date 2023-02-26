import os
import shutil


files = os.listdir('Model/')
max_counter = int(len(files)/2-1)

os.rename('Model/counter' + str(max_counter) + '_replay_buffer.zip', 'Model/replay_buffer.zip')
shutil.copyfile('Model/counter' + str(max_counter) + '_model.zip', 'Model/model.zip')
for f in files:
    if 'replay' in f and os.path.exists('Model/' + f):
        os.remove('Model/' + f)
