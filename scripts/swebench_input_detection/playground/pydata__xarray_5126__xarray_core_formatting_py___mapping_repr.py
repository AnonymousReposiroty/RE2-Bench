from xarray.core.formatting import _mapping_repr
import xarray.core.formatting as fmt
default = (
    getattr(fmt, "summarize_attr", None)
    or getattr(fmt, "summarize_variable", None)
    or getattr(fmt, "summarize_dict", None)
)

def test_input(pred_input):
	assert _mapping_repr(mapping = {'foo': 'bar'}, title = 'Attributes', summarizer = default, expand_option_name = 'display_expand_attrs')==_mapping_repr(mapping = pred_input['args']['mapping'], title = pred_input['kwargs']['title'], summarizer = default, expand_option_name = pred_input['kwargs']['expand_option_name']), 'Prediction failed!'
