import ast
import operator
import re

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_eval(node):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    else:
        raise TypeError(type(node))

def execute(user_text: str) -> str:
    """
    Attempts to parse and calculate a math expression.
    Returns the string result if successful, None otherwise.
    """
    clean = user_text.lower().replace("calculate", "").replace("what is", "").replace("=", "").strip()
    
    # Must contain at least one math operator to be considered a math query
    # (otherwise just saying "12" might trigger the calculator).
    if not any(op in clean for op in "+-*/"):
        return None
        
    # Strip everything except numbers and operators
    clean = re.sub(r'[^0-9\+\-\*\/\(\)\.]', '', clean)
    if not clean:
        return None
        
    try:
        tree = ast.parse(clean, mode='eval').body
        val = _safe_eval(tree)
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        elif isinstance(val, float):
            return f"{val:.4f}".rstrip('0').rstrip('.')
        return str(val)
    except Exception:
        return None
