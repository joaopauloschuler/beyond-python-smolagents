from .remote_executors import RemotePythonExecutor
from typing import Any
from .tools import Tool
from .monitoring import LogLevel
import io
import contextlib
import base64
from io import BytesIO
import PIL.Image
from .utils import AgentError
from .local_python_executor import CodeOutput

class LocalExecExecutor(RemotePythonExecutor):
    """
    Executes Python code using the built-in exec() function.
    
    This executor runs code in the local Python environment without sandboxing,
    capturing output and handling various return types.
    
    WARNING: This executor is NOT SECURE for untrusted code. It executes code directly
    in the current Python process without any isolation. Use only with trusted code.
    
    Args:
        additional_imports (`list[str]`): Additional packages to install.
        logger (`Logger`): Logger to use.
        capture_graphics (`bool`): Whether to capture matplotlib and other graphics output.
        restricted_modules (`List[str]`, optional): List of modules to restrict from importing.
        **kwargs: Additional configuration parameters.
    """

    def __init__(self, additional_imports: list[str], logger, capture_graphics: bool = True, 
                 restricted_modules: list[str] = None, **kwargs):
        super().__init__(additional_imports, logger)
        self.installed_packages = self.install_packages(additional_imports)
        self.globals_dict = {
            '_output_data': None,
            '_output_type': None,
            '_last_value': None,
            '_final_answer': None,
            'display': self._display_func
        }
        self.capture_graphics = capture_graphics
        self.restricted_modules = restricted_modules or []
        self.tools = {}

    def __call__(self, code_action: str) -> CodeOutput:
        """Check if code is a final answer and run it accordingly"""
        does_not_care = False
        output = self.run_code_raise_errors(code_action, return_final_answer=does_not_care)
        is_final_answer = output.is_final_answer
        return output
        
    def send_tools(self, tools: dict[str, Tool]):
        self.tools = tools
    
    def _setup_security_restrictions(self):
        """
        Set up basic security restrictions for executing code.
        Not foolproof, but provides some basic protection.
        """
        original_import = __import__
        
        def restricted_import(name, *args, **kwargs):
            # Check if the module is restricted
            for restricted in self.restricted_modules:
                if name == restricted or name.startswith(f"{restricted}."):
                    raise ImportError(f"Import of '{name}' is restricted for security reasons")
            return original_import(name, *args, **kwargs)
        
        # Add the restricted import to the globals dict
        self.globals_dict['__builtins__'] = dict(__import__=restricted_import, **__builtins__.__dict__)
        
        # Add a note about the restrictions
        modules_list = ", ".join(self.restricted_modules)
        self.logger.log(f"Basic import restrictions applied for: {modules_list}", level=LogLevel.INFO)
    
    def install_packages(self, packages: list[str]) -> list[str]:
        """Install Python packages using pip."""
        installed = []
    
    def _display_func(self, obj, mime_type=None):
        """
        Function to handle display of various object types.
        Similar to IPython's display function.
        """
        # Determine the MIME type if not specified
        if mime_type is None:
            if hasattr(obj, '_repr_png_'):
                mime_type = 'image/png'
                obj = obj._repr_png_()
            elif hasattr(obj, '_repr_jpeg_'):
                mime_type = 'image/jpeg'
                obj = obj._repr_jpeg_()
            elif hasattr(obj, '_repr_html_'):
                mime_type = 'text/html'
                obj = obj._repr_html_()
            elif hasattr(obj, '_repr_markdown_'):
                mime_type = 'text/markdown'
                obj = obj._repr_markdown_()
            elif hasattr(obj, '_repr_json_'):
                mime_type = 'application/json'
                obj = obj._repr_json_()
            elif isinstance(obj, PIL.Image.Image):
                mime_type = 'image/png'
                buffer = BytesIO()
                obj.save(buffer, format='PNG')
                buffer.seek(0)
                obj = base64.b64encode(buffer.read()).decode('utf-8')
            else:
                mime_type = 'text/plain'
                obj = str(obj)
        
        # Store the output
        self.globals_dict['_output_data'] = obj
        self.globals_dict['_output_type'] = mime_type
    
    def _setup_matplotlib_hook(self):
        """Set up hooks to capture matplotlib output."""
        if not self.capture_graphics:
            return
        
        # This code will be executed in the globals context
        setup_code = """
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io, base64

    # Save the original show method
    _original_show = plt.show

    # Define a custom show method to capture figures
    def _custom_show(*args, **kwargs):
        global _output_data, _output_type
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG')
        buf.seek(0)
        _output_data = base64.b64encode(buf.read()).decode('utf-8')
        _output_type = 'image/png'
        plt.close()
        return None

    # Replace plt.show with our custom version
    plt.show = _custom_show
except ImportError:
    # Matplotlib is not available, skip setup
    pass
"""
        try:
            exec(setup_code, self.globals_dict)
            # self.logger.log("Matplotlib hooks configured for graphics capture", level=LogLevel.INFO)
        except Exception as e:
            self.logger.log(f"Failed to configure matplotlib hooks: {e}", level=LogLevel.WARNING)
    
    def run_code_raise_errors(self, code: str, return_final_answer: bool = False) -> tuple[Any, str]:
        """
        Execute Python code and return the result and output logs.
        
        Args:
            code (str): The Python code to execute
            return_final_answer (bool): Whether to raise an error if no result is returned
            
        Returns:
            tuple[Any, str]: A tuple containing (result, logs)
        """
        # Reset output data
        self.globals_dict['_output_data'] = None
        self.globals_dict['_output_type'] = None
        self.globals_dict['_last_value'] = None
        self.globals_dict['_final_answer'] = None
        self.globals_dict.update(self.tools)
        is_final_answer = False
        
        # Set up matplotlib if needed
        if self.capture_graphics:
            self._setup_matplotlib_hook()
        
        # Create buffers for stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        locals_dict = None
        wrapped_code = f"""
_final_answer = None
_last_value = None
def final_answer(pfinal):
    global _final_answer
    _final_answer = pfinal
{code}
if '_' in locals():
    _last_value = _
"""
        # print(wrapped_code)
        
        # Execute the code, capturing stdout and stderr
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                # Add special handling for Jupyter-style last expression value
                # Wrap the code to capture the last expression value
                # Execute the code
                exec(wrapped_code, self.globals_dict, locals_dict)                
            except Exception as e:
                # Get the traceback
                import traceback
                tb = traceback.format_exc()
                error_msg = f"Error executing code: {str(e)}\n{tb}"
                stderr_buffer.write(error_msg)
                raise AgentError(error_msg, self.logger)
        
        # print('Global dict', self.globals_dict)
        # print('Local dict', locals_dict)

        is_final_answer = (self.globals_dict['_final_answer'] is not None)
        # Collect logs
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()
        
        logs = stdout_content
        logs += "\nLast value:\n" + str(self.globals_dict['_last_value'])
        if stderr_content:
            logs += "\nStderr:\n" + stderr_content
        print('Logs', logs)
        
        return CodeOutput(output=self.globals_dict['_final_answer'], logs=logs, is_final_answer=is_final_answer)
