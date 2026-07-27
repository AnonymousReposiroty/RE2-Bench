from types import NoneType
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata, OptionSpec

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_domain(CPPDomain)
    app.add_config_value('cpp_index_common_prefix', [], 'env', types=frozenset({list}))
    app.add_config_value('cpp_id_attributes', [], 'env', types=frozenset({list, tuple}))
    app.add_config_value(
        'cpp_paren_attributes', [], 'env', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'cpp_maximum_signature_line_length',
        None,
        'env',
        types=frozenset({int, NoneType}),
    )
    app.add_post_transform(AliasTransform)

    # debug stuff
    app.add_config_value('cpp_debug_lookup', False, '', types=frozenset({bool}))
    app.add_config_value('cpp_debug_show_tree', False, '', types=frozenset({bool}))

    app.connect('builder-inited', _init_stuff)

    return {
        'version': 'builtin',
        'env_version': 9,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
