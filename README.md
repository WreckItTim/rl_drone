USING BASE REPO WITH AIRSIM

Install Microsoft AirSim for Unreal Engine following the tutorial in their GitHub: https://github.com/microsoft/AirSim
, Set path(s) to the AirSim release executable and/or settings file(s) as needed (see main.py or any configuration file)
, Run main.py to either read a configuration file or create new configuration (see main.py or any configuration file)


MAKING A NEW DRONE CLASS

1. overwrite all class methods for your NewDrone class as defined in the Drone base class, found at drones/drone.py
2. derive the Drone base class, such as: class NewDrone(Drone)
3. decorate your NewDrone.__init__() method with the @_init_wrapper found in the component.py file (see below sections)  


THE COMPONENT CLASS

The Component base-class can be found in the component.py file in master. All classes derive from Component. This will allow components to be serializable for loading/saving configuration files, able to be seen by other components, benchmarkable, and have all class attributes automagically set. See bottom for a full example of making a new component class.


THE INIT WRAPPER

The @_init_wrapper can be found in the component.py file in master.
The @_init_wrapper will automagically set all public arguments whose names do not start with a leading '_'.
The @_init_wrapper will save the self instance on creation to the global dictionary of components, and assign a unique name to it if not given during construction.
The @_init_wrapper will fetch any _component(s) arguments and set them private variables.
The @_init_wrapper will also wrap all public class methods in a timer method for benchmarking.
Any of the above can be turned off by setting a bool in the constructor (see component.py), though it is not advised.


MAKING A NEW COMPONENT CLASS

See example at bottom.

Create a new BaseComponent base-class by creating a new folder with the base-class name followed by an 's', a baseclass.py file, and inherit the Component class (see example)

Create a new SubComponent sub-class by creating a subcomponent.py file in the parent folder, inheriting the parent-component, overloading any parent class-methods as needed, and decorating SubComponent.__init__() with @_init_wrapper (see example)

Any component can be passed as either a string or instance to a contructor method, by trailing the variable name with '_component' (see example)

Similarily, any list of components can be passed as either a list of strings or list of instances to a contructor method, by trailing the variable name with '_components' (see example)


EXAMPLE

You want to create a new component base-class callsed "Base"...
1. Add a new folder named bases to master
2. Add a new file base.py to the bases folder
3. Have the Base class derive the Component class, such as: class Base(Component)

You want to now create a component sub-class called "Sub" to your new Base class, or to any other base-class...
1. Add the file sub.py in the bases folder
2. Have the Sub class derive the Base, such as: class Sub(Base)
3. Wrap the Sub.__init__() method with @_init_wrapper

You want to add a component class-attribute to your SubClass, let's use drone as an example...
1. pass the argument drone_component to the Sub.__init__() method, such as: Sub.__init__(drone_component)
The @_init_wrapper will handle the rest...
1. The class attribute Sub._drone will be automattically set which will point to the drone instance
2. The Sub._drone instance will be automattically serialized by simply being saved as Sub._drone._name in any configuration files
3. The Sub._drone instance will be automattically deserialized by the same name, passed as a string during construction and fetched from the global list
WARNING: order matters, just like declaring any object

You want to add a list-of-components class-attribute to your SubClass, let's use transformers as an example...
1. pass the argument transformer_components to the Sub.__init__() method, such as: Sub.__init__(transformer_components)
The @_init_wrapper will handle the rest...
1. The class attribute Sub._transformers will be automattically set which will point to the list of transformer instances
2. The Sub._transformers list will be serialized, by each of the transformer names in the list, in any configuration files
3. The Sub._transformers list will be deserialized by the same names (passed in as a list of trings during construction)
WARNING: order matters, just like declaring any object
