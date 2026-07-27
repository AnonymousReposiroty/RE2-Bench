from types import NoneType
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata, OptionSpec

def setup(app: Sphinx) -> ExtensionMetadata:
    app.setup_extension('sphinx.directives')

    app.add_domain(PythonDomain)
    app.add_config_value(
        'python_use_unqualified_type_names', False, 'env', types=frozenset({bool})
    )
    app.add_config_value(
        'python_maximum_signature_line_length',
        None,
        'env',
        types=frozenset({int, NoneType}),
    )
    app.add_config_value(
        'python_trailing_comma_in_multi_line_signatures',
        True,
        'env',
        types=frozenset({bool}),
    )
    app.add_config_value(
        'python_display_short_literal_types', False, 'env', types=frozenset({bool})
    )
    app.connect('object-description-transform', filter_meta_fields)
    app.connect('missing-reference', builtin_resolver, priority=900)

    return {
        'version': 'builtin',
        'env_version': 4,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
