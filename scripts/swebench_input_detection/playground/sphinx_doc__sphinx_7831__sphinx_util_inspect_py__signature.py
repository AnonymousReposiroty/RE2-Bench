from sphinx.util.inspect import signature

def make_callable_from_signature(sig_str: str):
    """
    Create a dummy function with the given signature string, e.g. "(arg, kwarg=None)".
    """
    namespace = {}
    src = f"""
def _generated{sig_str}:
    # Do whatever you want here
    print("Called with:", arg, kwarg)
"""
    exec(src, namespace)
    return namespace["_generated"]

def test_input(pred_input):
    assert signature(subject = make_callable_from_signature('(arg, kwarg=None)'))==signature(subject =make_callable_from_signature(pred_input['args']['subject']['__signature__'])), 'Prediction failed!'