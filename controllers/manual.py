# abstract class used to handle all components
from controllers.controller import Controller
from component import get_component, component_list, _init_wrapper

class Manual(Controller):
    # constructor
    @_init_wrapper
    def __init__(self):
        super().__init__()

    # runs control on components
    def run(self):
        component_names = list(component_list.keys())
        while(True):
            print('Enter component _name to activate')
            user_input = input().lower()
            if user_input == 'quit':
                break
            elif user_input == 'list':
                component_names = list(component_list.keys())
                for idx, component_name in enumerate(component_names):
                    print(idx, ':', component_name)
            elif user_input == 'reset':
                for component_name in component_names:
                    get_component(component_name).reset()
            else:
                if user_input in component_list: 
                    print(get_component(user_input).activate())
                else:
                    try:
                        idx = int(user_input)
                        component_name = component_names[idx]
                        print(get_component(component_name).activate())
                    except Exception as e:
                        print('invalid entry')