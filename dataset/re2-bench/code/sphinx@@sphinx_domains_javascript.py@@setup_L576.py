from types import NoneType
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata, OptionSpec

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_domain(JavaScriptDomain)
    app.add_config_value(
        'javascript_maximum_signature_line_length',
        None,
        'env',
        types=frozenset({int, NoneType}),
    )
    app.add_config_value(
        'javascript_trailing_comma_in_multi_line_signatures',
        True,
        'env',
        types=frozenset({bool}),
    )
    return {
        'version': 'builtin',
        'env_version': 3,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
