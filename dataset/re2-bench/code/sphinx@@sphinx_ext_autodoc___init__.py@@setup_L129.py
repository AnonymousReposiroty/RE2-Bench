import sphinx
from sphinx.config import ENUM
from sphinx.ext.autodoc.typehints import _merge_typehints
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_config_value(
        'autoclass_content',
        'class',
        'env',
        types=ENUM('both', 'class', 'init'),
    )
    app.add_config_value(
        'autodoc_member_order',
        'alphabetical',
        'env',
        types=ENUM('alphabetical', 'bysource', 'groupwise'),
    )
    app.add_config_value(
        'autodoc_class_signature',
        'mixed',
        'env',
        types=ENUM('mixed', 'separated'),
    )
    app.add_config_value('autodoc_default_options', {}, 'env', types=frozenset({dict}))
    app.add_config_value(
        'autodoc_docstring_signature', True, 'env', types=frozenset({bool})
    )
    app.add_config_value(
        'autodoc_mock_imports', [], 'env', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'autodoc_typehints',
        'signature',
        'env',
        types=ENUM('signature', 'description', 'none', 'both'),
    )
    app.add_config_value(
        'autodoc_typehints_description_target',
        'all',
        'env',
        types=ENUM('all', 'documented', 'documented_params'),
    )
    app.add_config_value('autodoc_type_aliases', {}, 'env', types=frozenset({dict}))
    app.add_config_value(
        'autodoc_typehints_format',
        'short',
        'env',
        types=ENUM('fully-qualified', 'short'),
    )
    app.add_config_value('autodoc_warningiserror', True, 'env', types=frozenset({bool}))
    app.add_config_value(
        'autodoc_inherit_docstrings', True, 'env', types=frozenset({bool})
    )
    app.add_config_value(
        'autodoc_preserve_defaults', False, 'env', types=frozenset({bool})
    )
    app.add_config_value(
        'autodoc_use_type_comments', True, 'env', types=frozenset({bool})
    )
    app.add_config_value(
        'autodoc_use_legacy_class_based', False, 'env', types=frozenset({bool})
    )

    app.add_event('autodoc-before-process-signature')
    app.add_event('autodoc-process-docstring')
    app.add_event('autodoc-process-signature')
    app.add_event('autodoc-skip-member')
    app.add_event('autodoc-process-bases')

    app.connect('object-description-transform', _merge_typehints)

    app.connect('config-inited', _register_directives)

    return {
        'version': sphinx.__display_version__,
        'parallel_read_safe': True,
    }
