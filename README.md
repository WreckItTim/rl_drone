USING BASE REPO WITH AIRSIM

Install Microsoft AirSim for Unreal Engine following the tutorial in their GitHub: https://github.com/microsoft/AirSim
, Set path to the folder containing any AirSim release executables (see global_parameters.json)
, Run main.py to either read a configuration file or create new configuration (see main.py)


MAKING A NEW DRONE CLASS

1. overwrite all class methods for your NewDrone class as defined in the Drone base class, found at drones/drone.py
2. derive the Drone base class, such as: class NewDrone(Drone)
3. decorate your NewDrone.__init__() method with the @_init_wrapper found in the component.py file (see below sections)  


THE COMPONENT CLASS

The Component base-class can be found in the component.py file in master. All classes derive from Component. This will allow components to be serializable for loading/saving configuration files, able to be seen by other components, benchmarkable, and have all class attributes automagically set. See bottom for a full example of making a new component class.


THE INIT WRAPPER

The @_init_wrapper can be found in the component.py file in master.
The @_init_wrapper will automagically set all public arguments whose names do not start with a leading underscore.
The @_init_wrapper will save the self instance on creation to the global dictionary of components, and assign a unique name to it if not given during construction.
The @_init_wrapper will fetch any _component(s) arguments and set them private variables.
The @_init_wrapper will also wrap all public class methods in a timer method for benchmarking.
The @_init_wrapper will also add the two arguments connect_priority=None and disconnect_priority=None used to handle load orders


MAKING A NEW COMPONENT CLASS

# README - for all new classes that you make (I suggest everyone read this to understand how the repo works anyways):
# all classes are children of this Component class 
# this uses overlapping logic for serialization, connecting, running, logging, conflict management...
# after you know how this works you can copy and paste new child classes and have them integrated into the repo, working in seconds with little extra coding required!
# 1. if you are making a new parent class named {parent}: ---- note that it's probably easier to just make a new "Other" child class ----
	# a. add a folder named '{parent}s' and within it add a python file named '{parent}.py'
	# b. create a new class named '{Parent}' in the {parent}.py file
	# c. have the {Parent} class inherit Component, such as 'class {Parent}(Component):' - note the capital P
	# d. optionally, define any class methods as declared in the Component class, seen below
	# e. define a {Parent}.connect() method that calls super, such as 'def connect(self): super().connect()'
# 2. if you are making a new child class named {child} from the parent class named {parent}:
	# a. add a python file named '{child}.py' inside the existing folder named '{parent}s'
	# b. create a new class named '{Child}' in the {child}.py file
	# c. have the {Child} class inherit {Parent}, such as 'class {Child}({Parent}):' - note the capital C
	# d. decorate {Child}.__init__() with @_init_wrapper - see definition below
	# e. define any necesary (abstract) class methods as declared in {Parent}
	# f. optionally, define any class methods as declared in the Component class, seen below
	# g. if you redefine {Child}.connect() make sure to call super, such as 'def connect(self): super().connect() ...'
# 3. if you want to have a Component class named {Component} as a member named {member} in another Class named {Class}:
	# a. define an argument named {member}_component in the {Class}.__init__() method
		# NOTE, so that order does not matter, for component creation, the following is done:
		# (otherwise order can create conflicts) (this also allows for automatic serialization)
		# let a global Class instance belonging to {Class} be {class_instance}
		# let another global Component instance belonging to {Component} be {member_instance}
		# such that we want: {class_instance}._{member}={member_instance} - note that member will be private (necesary for serialization)
		# all global Component instances, needed for a given configuration, are first created before any Component members are properly set
		# you can pass a string argument {member_name} when creating {member_instance}, such as {member_instance}={Component}.__init__(..., name={member_name})
		# {member_instance} will set the unique string ID upon creation, such as {member_instance}._name={member_name}, and save it to the global component_list.
		# This way you can define a {class_instance}.{member} before {member_instance} is even created!
		# upon creation of {class_instance}, this is done: {class_instance}.{member}_component = {member_name}
		# all class Component members are properly set during Component.connect(), such as {class_instance}._{member}=get_component({class_instance}.{member}_component) 
		# the argument {member}_component passed into {Class}.__init__() can be the exact {member_instance} if already created, or {member_name} if otherwise not yet created
		# all {member}_component arguments to {Class}.__init__() will automatically set a private class member after connect() is called, such as self._{member}={member_instance}
		# deserialization, reading from a configuation file, also leverages the above methodology so that your component classes can automatically be serialized/deserialized
		# if you want a component to connect first or last, set {class_instance}.connect_priority={x}. I typicaly set this from {Class}.__init__()
		# priority will load lowest-to-highest positive {x} before all 0-priority (default) components, and highest-to-lowest negative {x} after all 0-priority (default) components
		# priority load example: (1) (2) (3) (0) (0) (0) (0) (-1) (-2)
		# ties in priority will run in order of instance creation
		# similarily you can set {class_instance}.disconnect_priority={y}
# 4. if you want to have a list of Component members named {members} in another Class named {Class}:
	# a. add an argument named {member}_components to the {Class}.__init__() method - note the s at the end
		# read secton 3, defining a list of components follows the same logic
		# the {member}_components argument to {Class}.__init__() is a mixed list of {member_instance} and {member_name} like variables
		# let {member_instances} be the hypothetical list of all the desired {member_instance} variables
		# all {member}_components arguments to {Class}.__init__() will automatically set a private class member after connect() is called, such as self._{member}={member_instances}
# IF you followed steps 1-5 correctly, than your new component class will:
	# a. be serializable (for configuration files)
	# b. will avoid conflicts so the order of creation does not matter
	# c. can be time/memory benchmarked
	# d. can be used with other components
	# e. can be used in the same manner as any parent-base class, for example you can make a new reward or action or sensor or whatever