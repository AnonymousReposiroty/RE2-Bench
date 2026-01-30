from types import NoneType
from sphinx.domains.c._ids import _macro_keywords, _max_id
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata, OptionSpec

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_domain(CDomain)
    app.add_config_value('c_id_attributes', [], 'env', types=frozenset({list, tuple}))
    app.add_config_value(
        'c_paren_attributes', [], 'env', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'c_extra_keywords',
        _macro_keywords,
        'env',
        types=frozenset({frozenset, list, set, tuple}),
    )
    app.add_config_value(
        'c_maximum_signature_line_length',
        None,
        'env',
        types=frozenset({int, NoneType}),
    )
    app.add_post_transform(AliasTransform)

    return {
        'version': 'builtin',
        'env_version': 3,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
