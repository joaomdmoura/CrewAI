import asyncio
import inspect
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from uuid import uuid4

from blinker import Signal
from pydantic import BaseModel, Field, ValidationError

from crewai.flow.flow_events import (
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.flow.flow_visualizer import plot_flow
from crewai.flow.persistence import FlowPersistence
from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.utils import get_possible_return_constants
from crewai.telemetry import Telemetry


class FlowState(BaseModel):
    """Base model for all flow states, ensuring each state has a unique ID."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the flow state")

# Type variables with explicit bounds
T = TypeVar("T", bound=Union[Dict[str, Any], BaseModel])  # Generic flow state type parameter
StateT = TypeVar("StateT", bound=Union[Dict[str, Any], BaseModel])  # State validation type parameter

def ensure_state_type(state: Any, expected_type: Type[StateT]) -> StateT:
    """Ensure state matches expected type with proper validation.
    
    Args:
        state: State instance to validate
        expected_type: Expected type for the state
        
    Returns:
        Validated state instance
        
    Raises:
        TypeError: If state doesn't match expected type
        ValueError: If state validation fails
    """
    """Ensure state matches expected type with proper validation.
    
    Args:
        state: State instance to validate
        expected_type: Expected type for the state
        
    Returns:
        Validated state instance
        
    Raises:
        TypeError: If state doesn't match expected type
        ValueError: If state validation fails
    """
    if expected_type == dict:
        if not isinstance(state, dict):
            raise TypeError(f"Expected dict, got {type(state).__name__}")
        return cast(StateT, state)
    if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
        if not isinstance(state, expected_type):
            raise TypeError(f"Expected {expected_type.__name__}, got {type(state).__name__}")
        return cast(StateT, state)
    raise TypeError(f"Invalid expected_type: {expected_type}")


def start(condition: Optional[Union[str, dict, Callable]] = None) -> Callable:
    """
    Marks a method as a flow's starting point.

    This decorator designates a method as an entry point for the flow execution.
    It can optionally specify conditions that trigger the start based on other
    method executions.

    Parameters
    ----------
    condition : Optional[Union[str, dict, Callable]], optional
        Defines when the start method should execute. Can be:
        - str: Name of a method that triggers this start
        - dict: Contains "type" ("AND"/"OR") and "methods" (list of triggers)
        - Callable: A method reference that triggers this start
        Default is None, meaning unconditional start.

    Returns
    -------
    Callable
        A decorator function that marks the method as a flow start point.

    Raises
    ------
    ValueError
        If the condition format is invalid.

    Examples
    --------
    >>> @start()  # Unconditional start
    >>> def begin_flow(self):
    ...     pass

    >>> @start("method_name")  # Start after specific method
    >>> def conditional_start(self):
    ...     pass

    >>> @start(and_("method1", "method2"))  # Start after multiple methods
    >>> def complex_start(self):
    ...     pass
    """
    def decorator(func):
        func.__is_start_method__ = True
        if condition is not None:
            if isinstance(condition, str):
                func.__trigger_methods__ = [condition]
                func.__condition_type__ = "OR"
            elif (
                isinstance(condition, dict)
                and "type" in condition
                and "methods" in condition
            ):
                func.__trigger_methods__ = condition["methods"]
                func.__condition_type__ = condition["type"]
            elif callable(condition) and hasattr(condition, "__name__"):
                func.__trigger_methods__ = [condition.__name__]
                func.__condition_type__ = "OR"
            else:
                raise ValueError(
                    "Condition must be a method, string, or a result of or_() or and_()"
                )
        return func

    return decorator

def listen(condition: Union[str, dict, Callable]) -> Callable:
    """
    Creates a listener that executes when specified conditions are met.

    This decorator sets up a method to execute in response to other method
    executions in the flow. It supports both simple and complex triggering
    conditions.

    Parameters
    ----------
    condition : Union[str, dict, Callable]
        Specifies when the listener should execute. Can be:
        - str: Name of a method that triggers this listener
        - dict: Contains "type" ("AND"/"OR") and "methods" (list of triggers)
        - Callable: A method reference that triggers this listener

    Returns
    -------
    Callable
        A decorator function that sets up the method as a listener.

    Raises
    ------
    ValueError
        If the condition format is invalid.

    Examples
    --------
    >>> @listen("process_data")  # Listen to single method
    >>> def handle_processed_data(self):
    ...     pass

    >>> @listen(or_("success", "failure"))  # Listen to multiple methods
    >>> def handle_completion(self):
    ...     pass
    """
    def decorator(func):
        if isinstance(condition, str):
            func.__trigger_methods__ = [condition]
            func.__condition_type__ = "OR"
        elif (
            isinstance(condition, dict)
            and "type" in condition
            and "methods" in condition
        ):
            func.__trigger_methods__ = condition["methods"]
            func.__condition_type__ = condition["type"]
        elif callable(condition) and hasattr(condition, "__name__"):
            func.__trigger_methods__ = [condition.__name__]
            func.__condition_type__ = "OR"
        else:
            raise ValueError(
                "Condition must be a method, string, or a result of or_() or and_()"
            )
        return func

    return decorator


def router(condition: Union[str, dict, Callable]) -> Callable:
    """
    Creates a routing method that directs flow execution based on conditions.

    This decorator marks a method as a router, which can dynamically determine
    the next steps in the flow based on its return value. Routers are triggered
    by specified conditions and can return constants that determine which path
    the flow should take.

    Parameters
    ----------
    condition : Union[str, dict, Callable]
        Specifies when the router should execute. Can be:
        - str: Name of a method that triggers this router
        - dict: Contains "type" ("AND"/"OR") and "methods" (list of triggers)
        - Callable: A method reference that triggers this router

    Returns
    -------
    Callable
        A decorator function that sets up the method as a router.

    Raises
    ------
    ValueError
        If the condition format is invalid.

    Examples
    --------
    >>> @router("check_status")
    >>> def route_based_on_status(self):
    ...     if self.state.status == "success":
    ...         return SUCCESS
    ...     return FAILURE

    >>> @router(and_("validate", "process"))
    >>> def complex_routing(self):
    ...     if all([self.state.valid, self.state.processed]):
    ...         return CONTINUE
    ...     return STOP
    """
    def decorator(func):
        func.__is_router__ = True
        if isinstance(condition, str):
            func.__trigger_methods__ = [condition]
            func.__condition_type__ = "OR"
        elif (
            isinstance(condition, dict)
            and "type" in condition
            and "methods" in condition
        ):
            func.__trigger_methods__ = condition["methods"]
            func.__condition_type__ = condition["type"]
        elif callable(condition) and hasattr(condition, "__name__"):
            func.__trigger_methods__ = [condition.__name__]
            func.__condition_type__ = "OR"
        else:
            raise ValueError(
                "Condition must be a method, string, or a result of or_() or and_()"
            )
        return func

    return decorator

def or_(*conditions: Union[str, dict, Callable]) -> dict:
    """
    Combines multiple conditions with OR logic for flow control.

    Creates a condition that is satisfied when any of the specified conditions
    are met. This is used with @start, @listen, or @router decorators to create
    complex triggering conditions.

    Parameters
    ----------
    *conditions : Union[str, dict, Callable]
        Variable number of conditions that can be:
        - str: Method names
        - dict: Existing condition dictionaries
        - Callable: Method references

    Returns
    -------
    dict
        A condition dictionary with format:
        {"type": "OR", "methods": list_of_method_names}

    Raises
    ------
    ValueError
        If any condition is invalid.

    Examples
    --------
    >>> @listen(or_("success", "timeout"))
    >>> def handle_completion(self):
    ...     pass
    """
    methods = []
    for condition in conditions:
        if isinstance(condition, dict) and "methods" in condition:
            methods.extend(condition["methods"])
        elif isinstance(condition, str):
            methods.append(condition)
        elif callable(condition):
            methods.append(getattr(condition, "__name__", repr(condition)))
        else:
            raise ValueError("Invalid condition in or_()")
    return {"type": "OR", "methods": methods}


def and_(*conditions: Union[str, dict, Callable]) -> dict:
    """
    Combines multiple conditions with AND logic for flow control.

    Creates a condition that is satisfied only when all specified conditions
    are met. This is used with @start, @listen, or @router decorators to create
    complex triggering conditions.

    Parameters
    ----------
    *conditions : Union[str, dict, Callable]
        Variable number of conditions that can be:
        - str: Method names
        - dict: Existing condition dictionaries
        - Callable: Method references

    Returns
    -------
    dict
        A condition dictionary with format:
        {"type": "AND", "methods": list_of_method_names}

    Raises
    ------
    ValueError
        If any condition is invalid.

    Examples
    --------
    >>> @listen(and_("validated", "processed"))
    >>> def handle_complete_data(self):
    ...     pass
    """
    methods = []
    for condition in conditions:
        if isinstance(condition, dict) and "methods" in condition:
            methods.extend(condition["methods"])
        elif isinstance(condition, str):
            methods.append(condition)
        elif callable(condition):
            methods.append(getattr(condition, "__name__", repr(condition)))
        else:
            raise ValueError("Invalid condition in and_()")
    return {"type": "AND", "methods": methods}


class FlowMeta(type):
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        start_methods = []
        listeners = {}
        router_paths = {}
        routers = set()

        for attr_name, attr_value in dct.items():
            # Check for any flow-related attributes
            if (hasattr(attr_value, "__is_flow_method__") or
                hasattr(attr_value, "__is_start_method__") or
                hasattr(attr_value, "__trigger_methods__") or
                hasattr(attr_value, "__is_router__")):
                
                # Register start methods
                if hasattr(attr_value, "__is_start_method__"):
                    start_methods.append(attr_name)
                
                # Register listeners and routers
                if hasattr(attr_value, "__trigger_methods__"):
                    methods = attr_value.__trigger_methods__
                    condition_type = getattr(attr_value, "__condition_type__", "OR")
                    listeners[attr_name] = (condition_type, methods)
                    
                    if hasattr(attr_value, "__is_router__") and attr_value.__is_router__:
                        routers.add(attr_name)
                        possible_returns = get_possible_return_constants(attr_value)
                        if possible_returns:
                            router_paths[attr_name] = possible_returns

        setattr(cls, "_start_methods", start_methods)
        setattr(cls, "_listeners", listeners)
        setattr(cls, "_routers", routers)
        setattr(cls, "_router_paths", router_paths)

        return cls


class Flow(Generic[T], metaclass=FlowMeta):
    """Base class for all flows.
    
    Type parameter T must be either Dict[str, Any] or a subclass of BaseModel."""
    _telemetry = Telemetry()

    _start_methods: List[str] = []
    _listeners: Dict[str, tuple[str, List[str]]] = {}
    _routers: Set[str] = set()
    _router_paths: Dict[str, List[str]] = {}
    initial_state: Union[Type[T], T, None] = None
    event_emitter = Signal("event_emitter")

    def __class_getitem__(cls: Type["Flow"], item: Type[T]) -> Type["Flow"]:
        class _FlowGeneric(cls):  # type: ignore
            _initial_state_T = item  # type: ignore

        _FlowGeneric.__name__ = f"{cls.__name__}[{item.__name__}]"
        return _FlowGeneric

    def __init__(
        self,
        persistence: Optional[FlowPersistence] = None,
        restore_uuid: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Flow instance.
        
        Args:
            persistence: Optional persistence backend for storing flow states
            restore_uuid: Optional UUID to restore state from persistence
            **kwargs: Additional state values to initialize or override
        """
        # Initialize basic instance attributes
        self._methods: Dict[str, Callable] = {}
        self._method_execution_counts: Dict[str, int] = {}
        self._pending_and_listeners: Dict[str, Set[str]] = {}
        self._method_outputs: List[Any] = []  # List to store all method outputs
        self._persistence: Optional[FlowPersistence] = persistence
        
        # Validate state model before initialization
        if isinstance(self.initial_state, type):
            if issubclass(self.initial_state, BaseModel) and not issubclass(self.initial_state, FlowState):
                # Check if model has id field
                model_fields = getattr(self.initial_state, "model_fields", None)
                if not model_fields or "id" not in model_fields:
                    raise ValueError("Flow state model must have an 'id' field")
        
        # Handle persistence and potential ID conflicts
        stored_state = None
        if self._persistence is not None:
            if restore_uuid and kwargs and "id" in kwargs and restore_uuid != kwargs["id"]:
                raise ValueError(
                    f"Conflicting IDs provided: restore_uuid='{restore_uuid}' "
                    f"vs kwargs['id']='{kwargs['id']}'. Use only one ID for restoration."
                )
            
            # Attempt to load state, prioritizing restore_uuid
            if restore_uuid:
                stored_state = self._persistence.load_state(restore_uuid)
                if not stored_state:
                    raise ValueError(f"No state found for restore_uuid='{restore_uuid}'")
            elif kwargs and "id" in kwargs:
                stored_state = self._persistence.load_state(kwargs["id"])
                if not stored_state:
                    # For kwargs["id"], we allow creating new state if not found
                    self._state = self._create_initial_state()
                    if kwargs:
                        self._initialize_state(kwargs)
                    return
        
        # Initialize state based on persistence and kwargs
        if stored_state:
            # Create initial state and restore from persistence
            self._state = self._create_initial_state()
            self._restore_state(stored_state)
            # Apply any additional kwargs to override specific fields
            if kwargs:
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != "id"}
                if filtered_kwargs:
                    self._initialize_state(filtered_kwargs)
        else:
            # No stored state, create new state with initial values
            self._state = self._create_initial_state()
            # Apply any additional kwargs
            if kwargs:
                self._initialize_state(kwargs)

        self._telemetry.flow_creation_span(self.__class__.__name__)

        # Register all flow-related methods
        for method_name in dir(self):
            if not method_name.startswith("_"):
                method = getattr(self, method_name)
                # Check for any flow-related attributes
                if (hasattr(method, "__is_flow_method__") or
                    hasattr(method, "__is_start_method__") or
                    hasattr(method, "__trigger_methods__") or
                    hasattr(method, "__is_router__")):
                    # Ensure method is bound to this instance
                    if not hasattr(method, "__self__"):
                        method = method.__get__(self, self.__class__)
                    self._methods[method_name] = method


    
    def _create_initial_state(self) -> T:
        """Create and initialize flow state with UUID and default values.
        
        Returns:
            New state instance with UUID and default values initialized
            
        Raises:
            ValueError: If structured state model lacks 'id' field
            TypeError: If state is neither BaseModel nor dictionary
        """
        # Handle case where initial_state is None but we have a type parameter
        if self.initial_state is None and hasattr(self, "_initial_state_T"):
            state_type = getattr(self, "_initial_state_T")
            if isinstance(state_type, type):
                if issubclass(state_type, FlowState):
                    # Create instance without id, then set it
                    instance = state_type()
                    if not hasattr(instance, 'id'):
                        setattr(instance, 'id', str(uuid4()))
                    return cast(T, instance)
                elif issubclass(state_type, BaseModel):
                    # Create a new type that includes the ID field
                    class StateWithId(state_type, FlowState):  # type: ignore
                        pass
                    instance = StateWithId()
                    if not hasattr(instance, 'id'):
                        setattr(instance, 'id', str(uuid4()))
                    return cast(T, instance)
                elif state_type == dict:
                    return cast(T, {"id": str(uuid4())})  # Minimal dict state
        
        # Handle case where no initial state is provided
        if self.initial_state is None:
            return cast(T, {"id": str(uuid4())})
        
        # Handle case where initial_state is a type (class)
        if isinstance(self.initial_state, type):
            if issubclass(self.initial_state, FlowState):
                return cast(T, self.initial_state())  # Uses model defaults
            elif issubclass(self.initial_state, BaseModel):
                # Validate that the model has an id field
                model_fields = getattr(self.initial_state, "model_fields", None)
                if not model_fields or "id" not in model_fields:
                    raise ValueError("Flow state model must have an 'id' field")
                return cast(T, self.initial_state())  # Uses model defaults
            elif self.initial_state == dict:
                return cast(T, {"id": str(uuid4())})
        
        # Handle dictionary instance case
        if isinstance(self.initial_state, dict):
            new_state = dict(self.initial_state)  # Copy to avoid mutations
            if "id" not in new_state:
                new_state["id"] = str(uuid4())
            return cast(T, new_state)
        
        # Handle BaseModel instance case
        if isinstance(self.initial_state, BaseModel):
            model = cast(BaseModel, self.initial_state)
            if not hasattr(model, "id"):
                raise ValueError("Flow state model must have an 'id' field")
            
            # Create new instance with same values to avoid mutations
            if hasattr(model, "model_dump"):
                # Pydantic v2
                state_dict = model.model_dump()
            elif hasattr(model, "dict"):
                # Pydantic v1
                state_dict = model.dict()
            else:
                # Fallback for other BaseModel implementations
                state_dict = {
                    k: v for k, v in model.__dict__.items()
                    if not k.startswith("_")
                }
            
            # Create new instance of the same class
            model_class = type(model)
            return cast(T, model_class(**state_dict))
            
        raise TypeError(
            f"Initial state must be dict or BaseModel, got {type(self.initial_state)}"
        )
        # Handle case where initial_state is None but we have a type parameter
        if self.initial_state is None and hasattr(self, "_initial_state_T"):
            state_type = getattr(self, "_initial_state_T")
            if isinstance(state_type, type):
                if issubclass(state_type, FlowState):
                    return cast(T, state_type())
                elif issubclass(state_type, BaseModel):
                    # Create a new type that includes the ID field
                    class StateWithId(state_type, FlowState):  # type: ignore
                        pass
                    return cast(T, StateWithId())
                elif state_type == dict:
                    return cast(T, {"id": str(uuid4())})

        # Handle case where no initial state is provided
        if self.initial_state is None:
            return cast(T, {"id": str(uuid4())})

        # Handle case where initial_state is a type (class)
        if isinstance(self.initial_state, type):
            if issubclass(self.initial_state, FlowState):
                return cast(T, self.initial_state())
            elif issubclass(self.initial_state, BaseModel):
                # Validate that the model has an id field
                model_fields = getattr(self.initial_state, "model_fields", None)
                if not model_fields or "id" not in model_fields:
                    raise ValueError("Flow state model must have an 'id' field")
                return cast(T, self.initial_state())
            elif self.initial_state == dict:
                return cast(T, {"id": str(uuid4())})

        # Handle dictionary instance case
        if isinstance(self.initial_state, dict):
            if "id" not in self.initial_state:
                self.initial_state["id"] = str(uuid4())
            return cast(T, dict(self.initial_state))  # Create new dict to avoid mutations

        # Handle BaseModel instance case
        if isinstance(self.initial_state, BaseModel):
            if not hasattr(self.initial_state, "id"):
                raise ValueError("Flow state model must have an 'id' field")
            return cast(T, self.initial_state)
            
        raise TypeError(
            f"Initial state must be dict or BaseModel, got {type(self.initial_state)}"
        )

    @property
    def state(self) -> T:
        return self._state

    @property
    def method_outputs(self) -> List[Any]:
        """Returns the list of all outputs from executed methods."""
        return self._method_outputs

    def _initialize_state(self, inputs: Dict[str, Any]) -> None:
        """Initialize or update flow state with new inputs.
        
        Args:
            inputs: Dictionary of state values to set/update
            
        Raises:
            ValueError: If validation fails for structured state
            TypeError: If state is neither BaseModel nor dictionary
        """
        if isinstance(self._state, dict):
            # For dict states, preserve existing fields unless overridden
            current_id = self._state.get("id")
            # Only update specified fields
            for k, v in inputs.items():
                self._state[k] = v
            # Ensure ID is preserved or generated
            if current_id:
                self._state["id"] = current_id
            elif "id" not in self._state:
                self._state["id"] = str(uuid4())
        elif isinstance(self._state, BaseModel):
            # For BaseModel states, preserve existing fields unless overridden
            try:
                model = cast(BaseModel, self._state)
                # Get current state as dict
                if hasattr(model, "model_dump"):
                    current_state = model.model_dump()
                elif hasattr(model, "dict"):
                    current_state = model.dict()
                else:
                    current_state = {
                        k: v for k, v in model.__dict__.items()
                        if not k.startswith("_")
                    }
                
                # Create new state with preserved fields and updates
                new_state = {**current_state, **inputs}
                
                # Create new instance with merged state
                model_class = type(model)
                if hasattr(model_class, "model_validate"):
                    # Pydantic v2
                    self._state = cast(T, model_class.model_validate(new_state))
                elif hasattr(model_class, "parse_obj"):
                    # Pydantic v1
                    self._state = cast(T, model_class.parse_obj(new_state))
                else:
                    # Fallback for other BaseModel implementations
                    self._state = cast(T, model_class(**new_state))
            except ValidationError as e:
                raise ValueError(f"Invalid inputs for structured state: {e}") from e
        else:
            raise TypeError("State must be a BaseModel instance or a dictionary.")
            
    def _restore_state(self, stored_state: Dict[str, Any]) -> None:
        """Restore flow state from persistence.
        
        Args:
            stored_state: Previously stored state to restore
            
        Raises:
            ValueError: If validation fails for structured state
            TypeError: If state is neither BaseModel nor dictionary
        """
        # When restoring from persistence, use the stored ID
        stored_id = stored_state.get("id")
        if not stored_id:
            raise ValueError("Stored state must have an 'id' field")
        
        if isinstance(self._state, dict):
            # For dict states, update all fields from stored state
            self._state.clear()
            self._state.update(stored_state)
        elif isinstance(self._state, BaseModel):
            # For BaseModel states, create new instance with stored values
            model = cast(BaseModel, self._state)
            if hasattr(model, "model_validate"):
                # Pydantic v2
                self._state = cast(T, type(model).model_validate(stored_state))
            elif hasattr(model, "parse_obj"):
                # Pydantic v1
                self._state = cast(T, type(model).parse_obj(stored_state))
            else:
                # Fallback for other BaseModel implementations
                self._state = cast(T, type(model)(**stored_state))
        else:
            raise TypeError(
                f"State must be dict or BaseModel, got {type(self._state)}"
            )

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        self.event_emitter.send(
            self,
            event=FlowStartedEvent(
                type="flow_started",
                flow_name=self.__class__.__name__,
            ),
        )

        if inputs is not None:
            self._initialize_state(inputs)
        return asyncio.run(self.kickoff_async())

    async def kickoff_async(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        if not self._start_methods:
            raise ValueError("No start method defined")

        self._telemetry.flow_execution_span(
            self.__class__.__name__, list(self._methods.keys())
        )

        tasks = [
            self._execute_start_method(start_method)
            for start_method in self._start_methods
        ]
        await asyncio.gather(*tasks)

        final_output = self._method_outputs[-1] if self._method_outputs else None

        self.event_emitter.send(
            self,
            event=FlowFinishedEvent(
                type="flow_finished",
                flow_name=self.__class__.__name__,
                result=final_output,
            ),
        )
        return final_output

    async def _execute_start_method(self, start_method_name: str) -> None:
        """
        Executes a flow's start method and its triggered listeners.

        This internal method handles the execution of methods marked with @start
        decorator and manages the subsequent chain of listener executions.

        Parameters
        ----------
        start_method_name : str
            The name of the start method to execute.

        Notes
        -----
        - Executes the start method and captures its result
        - Triggers execution of any listeners waiting on this start method
        - Part of the flow's initialization sequence
        """
        result = await self._execute_method(
            start_method_name, self._methods[start_method_name]
        )
        await self._execute_listeners(start_method_name, result)

    async def _execute_method(
        self, method_name: str, method: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        result = (
            await method(*args, **kwargs)
            if asyncio.iscoroutinefunction(method)
            else method(*args, **kwargs)
        )
        self._method_outputs.append(result)
        self._method_execution_counts[method_name] = (
            self._method_execution_counts.get(method_name, 0) + 1
        )
        return result

    async def _execute_listeners(self, trigger_method: str, result: Any) -> None:
        """
        Executes all listeners and routers triggered by a method completion.

        This internal method manages the execution flow by:
        1. First executing all triggered routers sequentially
        2. Then executing all triggered listeners in parallel

        Parameters
        ----------
        trigger_method : str
            The name of the method that triggered these listeners.
        result : Any
            The result from the triggering method, passed to listeners
            that accept parameters.

        Notes
        -----
        - Routers are executed sequentially to maintain flow control
        - Each router's result becomes the new trigger_method
        - Normal listeners are executed in parallel for efficiency
        - Listeners can receive the trigger method's result as a parameter
        """
        # First, handle routers repeatedly until no router triggers anymore
        while True:
            routers_triggered = self._find_triggered_methods(
                trigger_method, router_only=True
            )
            if not routers_triggered:
                break
            for router_name in routers_triggered:
                await self._execute_single_listener(router_name, result)
                # After executing router, the router's result is the path
                # The last router executed sets the trigger_method
                # The router result is the last element in self._method_outputs
                trigger_method = self._method_outputs[-1]

        # Now that no more routers are triggered by current trigger_method,
        # execute normal listeners
        listeners_triggered = self._find_triggered_methods(
            trigger_method, router_only=False
        )
        if listeners_triggered:
            tasks = [
                self._execute_single_listener(listener_name, result)
                for listener_name in listeners_triggered
            ]
            await asyncio.gather(*tasks)

    def _find_triggered_methods(
        self, trigger_method: str, router_only: bool
    ) -> List[str]:
        """
        Finds all methods that should be triggered based on conditions.

        This internal method evaluates both OR and AND conditions to determine
        which methods should be executed next in the flow.

        Parameters
        ----------
        trigger_method : str
            The name of the method that just completed execution.
        router_only : bool
            If True, only consider router methods.
            If False, only consider non-router methods.

        Returns
        -------
        List[str]
            Names of methods that should be triggered.

        Notes
        -----
        - Handles both OR and AND conditions:
          * OR: Triggers if any condition is met
          * AND: Triggers only when all conditions are met
        - Maintains state for AND conditions using _pending_and_listeners
        - Separates router and normal listener evaluation
        """
        triggered = []
        for listener_name, (condition_type, methods) in self._listeners.items():
            is_router = listener_name in self._routers

            if router_only != is_router:
                continue

            if condition_type == "OR":
                # If the trigger_method matches any in methods, run this
                if trigger_method in methods:
                    triggered.append(listener_name)
            elif condition_type == "AND":
                # Initialize pending methods for this listener if not already done
                if listener_name not in self._pending_and_listeners:
                    self._pending_and_listeners[listener_name] = set(methods)
                # Remove the trigger method from pending methods
                if trigger_method in self._pending_and_listeners[listener_name]:
                    self._pending_and_listeners[listener_name].discard(trigger_method)

                if not self._pending_and_listeners[listener_name]:
                    # All required methods have been executed
                    triggered.append(listener_name)
                    # Reset pending methods for this listener
                    self._pending_and_listeners.pop(listener_name, None)

        return triggered

    async def _execute_single_listener(self, listener_name: str, result: Any) -> None:
        """
        Executes a single listener method with proper event handling.

        This internal method manages the execution of an individual listener,
        including parameter inspection, event emission, and error handling.

        Parameters
        ----------
        listener_name : str
            The name of the listener method to execute.
        result : Any
            The result from the triggering method, which may be passed
            to the listener if it accepts parameters.

        Notes
        -----
        - Inspects method signature to determine if it accepts the trigger result
        - Emits events for method execution start and finish
        - Handles errors gracefully with detailed logging
        - Recursively triggers listeners of this listener
        - Supports both parameterized and parameter-less listeners

        Error Handling
        -------------
        Catches and logs any exceptions during execution, preventing
        individual listener failures from breaking the entire flow.
        """
        try:
            method = self._methods[listener_name]

            self.event_emitter.send(
                self,
                event=MethodExecutionStartedEvent(
                    type="method_execution_started",
                    method_name=listener_name,
                    flow_name=self.__class__.__name__,
                ),
            )

            sig = inspect.signature(method)
            params = list(sig.parameters.values())
            method_params = [p for p in params if p.name != "self"]

            if method_params:
                listener_result = await self._execute_method(
                    listener_name, method, result
                )
            else:
                listener_result = await self._execute_method(listener_name, method)

            self.event_emitter.send(
                self,
                event=MethodExecutionFinishedEvent(
                    type="method_execution_finished",
                    method_name=listener_name,
                    flow_name=self.__class__.__name__,
                ),
            )

            # Execute listeners (and possibly routers) of this listener
            await self._execute_listeners(listener_name, listener_result)

        except Exception as e:
            print(
                f"[Flow._execute_single_listener] Error in method {listener_name}: {e}"
            )
            import traceback

            traceback.print_exc()

    def plot(self, filename: str = "crewai_flow") -> None:
        self._telemetry.flow_plotting_span(
            self.__class__.__name__, list(self._methods.keys())
        )
        plot_flow(self, filename)
