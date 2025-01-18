import ast
import datetime
import json
import re
import time
from difflib import SequenceMatcher
from textwrap import dedent
from typing import Any, Dict, List, Union

from json_repair import repair_json

import crewai.utilities.events as events
from crewai.agents.tools_handler import ToolsHandler
from crewai.task import Task
from crewai.telemetry import Telemetry
from crewai.tools import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling
from crewai.tools.tool_usage_events import ToolUsageError, ToolUsageFinished
from crewai.utilities import I18N, Converter, ConverterError, Printer

try:
    import agentops  # type: ignore
except ImportError:
    agentops = None
OPENAI_BIGGER_MODELS = [
    "gpt-4",
    "gpt-4o",
    "o1-preview",
    "o1-mini",
    "o1",
    "o3",
    "o3-mini",
]


class ToolUsageErrorException(Exception):
    """Exception raised for errors in the tool usage."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ToolUsage:
    """
    Class that represents the usage of a tool by an agent.

    Attributes:
      task: Task being executed.
      tools_handler: Tools handler that will manage the tool usage.
      tools: List of tools available for the agent.
      original_tools: Original tools available for the agent before being converted to BaseTool.
      tools_description: Description of the tools available for the agent.
      tools_names: Names of the tools available for the agent.
      function_calling_llm: Language model to be used for the tool usage.
    """

    def __init__(
        self,
        tools_handler: ToolsHandler,
        tools: List[BaseTool],
        original_tools: List[Any],
        tools_description: str,
        tools_names: str,
        task: Task,
        function_calling_llm: Any,
        agent: Any,
        action: Any,
    ) -> None:
        self._i18n: I18N = agent.i18n
        self._printer: Printer = Printer()
        self._telemetry: Telemetry = Telemetry()
        self._run_attempts: int = 1
        self._max_parsing_attempts: int = 3
        self._remember_format_after_usages: int = 3
        self.agent = agent
        self.tools_description = tools_description
        self.tools_names = tools_names
        self.tools_handler = tools_handler
        self.original_tools = original_tools
        self.tools = tools
        self.task = task
        self.action = action
        self.function_calling_llm = function_calling_llm

        # Set the maximum parsing attempts for bigger models
        if (
            self.function_calling_llm
            and self.function_calling_llm in OPENAI_BIGGER_MODELS
        ):
            self._max_parsing_attempts = 2
            self._remember_format_after_usages = 4

    def parse_tool_calling(self, tool_string: str):
        """Parse the tool string and return the tool calling."""
        return self._tool_calling(tool_string)

    def use(
        self, calling: Union[ToolCalling, InstructorToolCalling], tool_string: str
    ) -> str:
        if isinstance(calling, ToolUsageErrorException):
            error = calling.message
            if self.agent.verbose:
                self._printer.print(content=f"\n\n{error}\n", color="red")
            self.task.increment_tools_errors()
            return error

        try:
            tool = self._select_tool(calling.tool_name)
        except Exception as e:
            error = getattr(e, "message", str(e))
            self.task.increment_tools_errors()
            if self.agent.verbose:
                self._printer.print(content=f"\n\n{error}\n", color="red")
            return error

        if (
            isinstance(tool, CrewStructuredTool)
            and tool.name == self._i18n.tools("add_image")["name"]
        ):  # type: ignore
            try:
                result = self._use(tool_string=tool_string, tool=tool, calling=calling)
                return result

            except Exception as e:
                error = getattr(e, "message", str(e))
                self.task.increment_tools_errors()
                if self.agent.verbose:
                    self._printer.print(content=f"\n\n{error}\n", color="red")
                return error

        return f"{self._use(tool_string=tool_string, tool=tool, calling=calling)}"

    def _use(
        self,
        tool_string: str,
        tool: Any,
        calling: Union[ToolCalling, InstructorToolCalling],
    ) -> str:  # TODO: Fix this return type
        tool_event = agentops.ToolEvent(name=calling.tool_name) if agentops else None  # type: ignore
        if self._check_tool_repeated_usage(calling=calling):  # type: ignore # _check_tool_repeated_usage of "ToolUsage" does not return a value (it only ever returns None)
            try:
                result = self._i18n.errors("task_repeated_usage").format(
                    tool_names=self.tools_names
                )
                self._telemetry.tool_repeated_usage(
                    llm=self.function_calling_llm,
                    tool_name=tool.name,
                    attempts=self._run_attempts,
                )
                result = self._format_result(result=result)  # type: ignore #  "_format_result" of "ToolUsage" does not return a value (it only ever returns None)
                return result  # type: ignore # Fix the return type of this function

            except Exception:
                self.task.increment_tools_errors()

        started_at = time.time()
        from_cache = False

        result = None  # type: ignore # Incompatible types in assignment (expression has type "None", variable has type "str")
        # check if cache is available
        if self.tools_handler.cache:
            result = self.tools_handler.cache.read(  # type: ignore # Incompatible types in assignment (expression has type "str | None", variable has type "str")
                tool=calling.tool_name, input=calling.arguments
            )
            from_cache = result is not None

        original_tool = next(
            (ot for ot in self.original_tools if ot.name == tool.name), None
        )

        if result is None:  #! finecwg: if not result --> if result is None
            try:
                if calling.tool_name in [
                    "Delegate work to coworker",
                    "Ask question to coworker",
                ]:
                    coworker = (
                        calling.arguments.get("coworker") if calling.arguments else None
                    )
                    self.task.increment_delegations(coworker)

                if calling.arguments:
                    try:
                        acceptable_args = tool.args_schema.model_json_schema()[
                            "properties"
                        ].keys()  # type: ignore # Item "None" of "type[BaseModel] | None" has no attribute "schema"
                        arguments = {
                            k: v
                            for k, v in calling.arguments.items()
                            if k in acceptable_args
                        }
                        result = tool.invoke(input=arguments)
                    except Exception:
                        arguments = calling.arguments
                        result = tool.invoke(input=arguments)
                else:
                    result = tool.invoke(input={})
            except Exception as e:
                self.on_tool_error(tool=tool, tool_calling=calling, e=e)
                self._run_attempts += 1
                if self._run_attempts > self._max_parsing_attempts:
                    self._telemetry.tool_usage_error(llm=self.function_calling_llm)
                    error_message = self._i18n.errors("tool_usage_exception").format(
                        error=e, tool=tool.name, tool_inputs=tool.description
                    )
                    error = ToolUsageErrorException(
                        f'\n{error_message}.\nMoving on then. {self._i18n.slice("format").format(tool_names=self.tools_names)}'
                    ).message
                    self.task.increment_tools_errors()
                    if self.agent.verbose:
                        self._printer.print(
                            content=f"\n\n{error_message}\n", color="red"
                        )
                    return error  # type: ignore # No return value expected

                self.task.increment_tools_errors()
                if agentops:
                    agentops.record(
                        agentops.ErrorEvent(exception=e, trigger_event=tool_event)
                    )
                return self.use(calling=calling, tool_string=tool_string)  # type: ignore # No return value expected

            if self.tools_handler:
                should_cache = True
                if (
                    hasattr(original_tool, "cache_function")
                    and original_tool.cache_function  # type: ignore # Item "None" of "Any | None" has no attribute "cache_function"
                ):
                    should_cache = original_tool.cache_function(  # type: ignore # Item "None" of "Any | None" has no attribute "cache_function"
                        calling.arguments, result
                    )

                self.tools_handler.on_tool_use(
                    calling=calling, output=result, should_cache=should_cache
                )

        if agentops:
            agentops.record(tool_event)
        self._telemetry.tool_usage(
            llm=self.function_calling_llm,
            tool_name=tool.name,
            attempts=self._run_attempts,
        )
        result = self._format_result(result=result)  # type: ignore # "_format_result" of "ToolUsage" does not return a value (it only ever returns None)
        data = {
            "result": result,
            "tool_name": tool.name,
            "tool_args": calling.arguments,
        }

        self.on_tool_use_finished(
            tool=tool,
            tool_calling=calling,
            from_cache=from_cache,
            started_at=started_at,
        )

        if (
            hasattr(original_tool, "result_as_answer")
            and original_tool.result_as_answer  # type: ignore # Item "None" of "Any | None" has no attribute "cache_function"
        ):
            result_as_answer = original_tool.result_as_answer  # type: ignore # Item "None" of "Any | None" has no attribute "result_as_answer"
            data["result_as_answer"] = result_as_answer

        self.agent.tools_results.append(data)

        return result  # type: ignore # No return value expected

    def _format_result(self, result: Any) -> None:
        self.task.used_tools += 1
        if self._should_remember_format():  # type: ignore # "_should_remember_format" of "ToolUsage" does not return a value (it only ever returns None)
            result = self._remember_format(result=result)  # type: ignore # "_remember_format" of "ToolUsage" does not return a value (it only ever returns None)
        return result

    def _should_remember_format(self) -> bool:
        return self.task.used_tools % self._remember_format_after_usages == 0

    def _remember_format(self, result: str) -> None:
        result = str(result)
        result += "\n\n" + self._i18n.slice("tools").format(
            tools=self.tools_description, tool_names=self.tools_names
        )
        return result  # type: ignore # No return value expected

    def _check_tool_repeated_usage(
        self, calling: Union[ToolCalling, InstructorToolCalling]
    ) -> None:
        if not self.tools_handler:
            return False  # type: ignore # No return value expected
        if last_tool_usage := self.tools_handler.last_used_tool:
            return (calling.tool_name == last_tool_usage.tool_name) and (  # type: ignore # No return value expected
                calling.arguments == last_tool_usage.arguments
            )

    def _select_tool(self, tool_name: str) -> Any:
        order_tools = sorted(
            self.tools,
            key=lambda tool: SequenceMatcher(
                None, tool.name.lower().strip(), tool_name.lower().strip()
            ).ratio(),
            reverse=True,
        )
        for tool in order_tools:
            if (
                tool.name.lower().strip() == tool_name.lower().strip()
                or SequenceMatcher(
                    None, tool.name.lower().strip(), tool_name.lower().strip()
                ).ratio()
                > 0.85
            ):
                return tool
        self.task.increment_tools_errors()
        if tool_name and tool_name != "":
            raise Exception(
                f"Action '{tool_name}' don't exist, these are the only available Actions:\n{self.tools_description}"
            )
        else:
            raise Exception(
                f"I forgot the Action name, these are the only available Actions: {self.tools_description}"
            )

    def _render(self) -> str:
        """Render the tool name and description in plain text."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(tool.description)
        return "\n--\n".join(descriptions)

    def _function_calling(self, tool_string: str):
        model = (
            InstructorToolCalling
            if self.function_calling_llm.supports_function_calling()
            else ToolCalling
        )
        converter = Converter(
            text=f"Only tools available:\n###\n{self._render()}\n\nReturn a valid schema for the tool, the tool name must be exactly equal one of the options, use this text to inform the valid output schema:\n\n### TEXT \n{tool_string}",
            llm=self.function_calling_llm,
            model=model,
            instructions=dedent(
                """\
        The schema should have the following structure, only two keys:
        - tool_name: str
        - arguments: dict (always a dictionary, with all arguments being passed)

        Example:
        {"tool_name": "tool name", "arguments": {"arg_name1": "value", "arg_name2": 2}}""",
            ),
            max_attempts=1,
        )
        tool_object = converter.to_pydantic()
        calling = ToolCalling(
            tool_name=tool_object["tool_name"],
            arguments=tool_object["arguments"],
            log=tool_string,  # type: ignore
        )

        if isinstance(calling, ConverterError):
            raise calling

        return calling

    def _original_tool_calling(self, tool_string: str, raise_error: bool = False):
        tool_name = self.action.tool
        tool = self._select_tool(tool_name)
        try:
            arguments = self._validate_tool_input(self.action.tool_input)

        except Exception:
            if raise_error:
                raise
            else:
                return ToolUsageErrorException(
                    f'{self._i18n.errors("tool_arguments_error")}'
                )

        if not isinstance(arguments, dict):
            if raise_error:
                raise
            else:
                return ToolUsageErrorException(
                    f'{self._i18n.errors("tool_arguments_error")}'
                )

        return ToolCalling(
            tool_name=tool.name,
            arguments=arguments,
            log=tool_string,
        )

    def _tool_calling(
        self, tool_string: str
    ) -> Union[ToolCalling, InstructorToolCalling]:
        try:
            try:
                return self._original_tool_calling(tool_string, raise_error=True)
            except Exception:
                if self.function_calling_llm:
                    return self._function_calling(tool_string)
                else:
                    return self._original_tool_calling(tool_string)
        except Exception as e:
            self._run_attempts += 1
            if self._run_attempts > self._max_parsing_attempts:
                self._telemetry.tool_usage_error(llm=self.function_calling_llm)
                self.task.increment_tools_errors()
                if self.agent.verbose:
                    self._printer.print(content=f"\n\n{e}\n", color="red")
                return ToolUsageErrorException(  # type: ignore # Incompatible return value type (got "ToolUsageErrorException", expected "ToolCalling | InstructorToolCalling")
                    f'{self._i18n.errors("tool_usage_error").format(error=e)}\nMoving on then. {self._i18n.slice("format").format(tool_names=self.tools_names)}'
                )
            return self._tool_calling(tool_string)

    def _validate_tool_input(self, tool_input: str) -> Dict[str, Any]:
        try:
            # Replace Python literals with JSON equivalents
            replacements = {
                r"'": '"',
                r"None": "null",
                r"True": "true",
                r"False": "false",
            }
            for pattern, replacement in replacements.items():
                tool_input = re.sub(pattern, replacement, tool_input)

            arguments = json.loads(tool_input)
        except json.JSONDecodeError:
            # Attempt to repair JSON string
            repaired_input = repair_json(tool_input)
            try:
                arguments = json.loads(repaired_input)
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid tool input JSON: {e}")

        return arguments

    def on_tool_error(self, tool: Any, tool_calling: ToolCalling, e: Exception) -> None:
        event_data = self._prepare_event_data(tool, tool_calling)
        events.emit(
            source=self, event=ToolUsageError(**{**event_data, "error": str(e)})
        )

    def on_tool_use_finished(
        self, tool: Any, tool_calling: ToolCalling, from_cache: bool, started_at: float
    ) -> None:
        finished_at = time.time()
        event_data = self._prepare_event_data(tool, tool_calling)
        event_data.update(
            {
                "started_at": datetime.datetime.fromtimestamp(started_at),
                "finished_at": datetime.datetime.fromtimestamp(finished_at),
                "from_cache": from_cache,
            }
        )
        events.emit(source=self, event=ToolUsageFinished(**event_data))

    def _prepare_event_data(self, tool: Any, tool_calling: ToolCalling) -> dict:
        return {
            "agent_key": self.agent.key,
            "agent_role": (self.agent._original_role or self.agent.role),
            "run_attempts": self._run_attempts,
            "delegations": self.task.delegations,
            "tool_name": tool.name,
            "tool_args": tool_calling.arguments,
            "tool_class": tool.__class__.__name__,
        }
