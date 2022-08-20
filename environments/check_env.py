from gym import spaces
import numpy as np
import cv2
from stable_baselines3.common.env_checker import check_env
from environments.dummy import Dummy


observation_spaces = {}
# continuous/discrete observation spaces
observation_spaces['discrete'] = spaces.Discrete(4) # discrete 1 of 4 states
observation_spaces['continuous'] = spaces.Box(-1, 1, shape=(4,)) # continous 4 features
# observation images must be of atleast size 36x36 to use CNN feature extractor
observation_spaces['gray_image'] = spaces.Box(0, 255, shape=(256,256,1), dtype=np.uint8)
observation_spaces['color_image'] = spaces.Box(0, 255, shape=(256,256,3), dtype=np.uint8)
observation_spaces['multimodal'] = spaces.Box(0, 255, shape=(256,256,5), dtype=np.uint8)

action_spaces = {}
# continuous/discrete action spaces
action_spaces['discrete'] = spaces.Discrete(4) # deterministic one of 4 actions
action_spaces['continous'] = spaces.Box(-1, 1, shape=(4,)) # stochastic continuous values of 4 actions
action_spaces['gray_image'] = spaces.Box(-1, 1, shape=(256,256,1))
action_spaces['color_image'] = spaces.Box(-1, 1, shape=(256,256,3))
action_spaces['multimodal'] = spaces.Box(-1, 1, shape=(256,256,5))


for observation_component in observation_spaces:
	observation_space = observation_spaces[observation_component]

	for action_component in action_spaces:
		action_space = action_spaces[action_component]

		print('checking', observation_component + '_observation_space', action_component + '_action_space')
			
		print('Observation Sample:')
		observation = observation_space.sample()
		if 'image' in observation_component:
			print('see opencv window')
			cv2.imshow('observation', observation)
			cv2.waitKey(0)
		elif 'multimodal' in observation_component:
			print('multimodal of size', observation.shape)
		else:
			print('value', observation)

		print('Action Sample:')
		action = action_space.sample()
		if 'image' in action_component:
			print('see opencv window')
			cv2.imshow('action', (255 * (action+1)/2).astype(np.uint8))
			cv2.waitKey(0)
		elif 'multimodal' in action_component:
			print('multimodal of size', action.shape)
		else:
			print('value', action)
			
		print('Any Stable_Baselines3 warnings???')
		env = Dummy(observation_space, action_space)
		check_env(env)
		print()
		print()
