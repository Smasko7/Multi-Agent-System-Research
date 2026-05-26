"""Code execution tool — safe Python sandbox with restricted globals and timeout."""

import threading
from langchain_core.tools import Tool


def _run_python(code: str) -> str:
    """Execute Python code in a restricted sandbox with a 5-second timeout."""
    result_container = {"output": "", "error": ""}

    def _execute():
        safe_globals = {
            "__builtins__": {
                "print": print,
                "range": range,
                "len": len,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "isinstance": isinstance,
                "type": type,
                "repr": repr,
            }
        }
        output_lines = []
        original_print = __builtins__["print"] if isinstance(__builtins__, dict) else print  # noqa

        def capture_print(*args, **kwargs):
            output_lines.append(" ".join(str(a) for a in args))

        safe_globals["__builtins__"]["print"] = capture_print
        local_vars = {}
        try:
            exec(code, safe_globals, local_vars)  # noqa: S102
            if output_lines:
                result_container["output"] = "\n".join(output_lines)
            elif local_vars:
                # Return last assigned variable if nothing was printed
                last_val = list(local_vars.values())[-1]
                result_container["output"] = repr(last_val)
            else:
                result_container["output"] = "Code executed successfully (no output)."
        except Exception as exc:
            result_container["error"] = f"ExecutionError: {exc}"

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()
    thread.join(timeout=5)

    if thread.is_alive():
        return "Error: Code execution timed out after 5 seconds."
    if result_container["error"]:
        return result_container["error"]
    return result_container["output"] or "Code executed successfully (no output)."


def get_code_exec_tool() -> Tool:
    """Return a LangChain Tool wrapping the Python sandbox."""
    return Tool(
        name="python_repl",
        func=_run_python,
        description=(
            "Execute Python code for calculations or data analysis. "
            "Input should be valid Python code."
        ),
    )
