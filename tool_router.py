from typing import Tuple, Optional
from tools import calculator, datetime_tool, system_tool, app_launcher

def route_tool(user_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Evaluates the user's text against available deterministic tools.
    Returns (tool_name, response) if a match is found.
    Returns (None, None) if no tools match.
    """
    
    # 1. System Tool
    sys_resp = system_tool.execute(user_text)
    if sys_resp is not None:
        return "system", sys_resp
        
    # 2. DateTime Tool
    dt_resp = datetime_tool.execute(user_text)
    if dt_resp is not None:
        return "datetime", dt_resp
        
    # 3. App Launcher Tool
    app_resp = app_launcher.execute(user_text)
    if app_resp is not None:
        return "app_launcher", app_resp
        
    # 4. Calculator Tool
    # Evaluate math last to avoid intercepting legitimate sentences that happen to have numbers/dashes
    calc_resp = calculator.execute(user_text)
    if calc_resp is not None:
        return "calculator", calc_resp
        
    return None, None
