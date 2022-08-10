# discrete move in one direction
from actions.action import Action
from component import _init_wrapper

class FixedMove(Action):
    # constructor takes 4d array where first 3 components are direction and 4th component is speed
    # note that speed is an arbitrary unit that is defined by the drone
    @_init_wrapper
    def __init__(self, drone_name='', x_distance=0, y_distance=0, z_distance=0, speed=10, front_facing=True, name=None):
        super().__init__()

    def act(self):
        self._drone.move(self.x_distance, self.y_distance, self.z_distance, self.speed, self.front_facing)
        
    def test(self):
        print(f'taking action:{self._name}, moving distance vector of <{self.x_distance}, {self.y_distance}, {self.z_distance}> at speed {self.speed}')
        self.act()

    # uses a string to fetch a preset movement (left, right, ...)
    # movements are addititve with underscores in move_name (to make diagnols)
    @staticmethod
    def get_move(drone_name, move_type, step_size, speed, front_facing=True):
        x_distance, y_distance, z_distance = 0, 0, 0
        moves = move_type.split('_')
        for move in moves:
            if 'left' in move: x_distance += step_size
            if 'right' in move: x_distance -= step_size
            if 'up' in move: y_distance += step_size
            if 'down' in move: y_distance -= step_size
            if 'forward' in move: z_distance += step_size
            if 'backward' in move: z_distance -= step_size
        return FixedMove(
            drone_name=drone_name, 
            x_distance=x_distance, 
            y_distance=y_distance, 
            z_distance=z_distance,
            speed=speed, 
            front_facing=front_facing
        )
