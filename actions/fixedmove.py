# discrete move in one direction
from actions.action import Action
from component import _init_wrapper

class FixedMove(Action):
    # constructor takes 4d array where first 3 components are direction and 4th component is speed
    # note that speed is an arbitrary unit that is defined by the drone
    @_init_wrapper
    def __init__(self, drone_component, x_distance, y_distance, z_distance, speed, front_facing=True):
        super().__init__()

    def act(self):
        self._drone.move(self.x_distance, self.y_distance, self.z_distance, self.speed, self.front_facing)

    # uses a string to fetch a preset movement (left, right, ...)
    # movements are addititve with underscores in move_component (to make diagnols)
    @staticmethod
    def get_move(drone_component, move_type, step_size, speed, front_facing=True):
        x_distance, y_distance, z_distance = 0, 0, 0
        moves = move_type.split('_')
        for move in moves:
            if 'left' in move: y_distance -= step_size
            if 'right' in move: y_distance += step_size
            if 'up' in move: z_distance += step_size
            if 'down' in move: z_distance -= step_size
            if 'forward' in move: x_distance += step_size
            if 'backward' in move: x_distance -= step_size
        return FixedMove(
            drone_component=drone_component, 
            x_distance=x_distance, 
            y_distance=y_distance, 
            z_distance=z_distance,
            speed=speed, 
            front_facing=front_facing,
            name=move_type,
        )
