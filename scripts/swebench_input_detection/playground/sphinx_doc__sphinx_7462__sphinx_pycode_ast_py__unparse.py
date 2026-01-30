from sphinx.pycode.ast import unparse

import ast

def node_from_dict(d):
    if d is None:
        return None
    if isinstance(d, list):
        return [node_from_dict(x) for x in d]

    if isinstance(d, dict):
        node_type = d.get("_type")
        if node_type is None:
            # Not an AST node, just a regular dict
            return {k: node_from_dict(v) for k, v in d.items()}

        cls = getattr(ast, node_type, None)
        if cls is None:
            raise ValueError(f"Unknown AST node type: {node_type}")

        # Build kwargs excluding _type
        kwargs = {k: node_from_dict(v) for k, v in d.items() if k != "_type"}
        return cls(**kwargs)

    # primitives
    return d

def test_input(pred_input):
    a = {
    "_type": "List",
    "elts": [],
    "ctx": {},
    "lineno": 1,
    "col_offset": 0,
    "end_lineno": 1,
    "end_col_offset": 9
    }
    pred_input['args']['node']['_type'] = 'List'
    assert unparse(node = node_from_dict(a))==unparse(node = node_from_dict(pred_input['args']['node'])), 'Prediction failed!'

 



