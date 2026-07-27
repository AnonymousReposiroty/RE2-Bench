from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_builder(MessageCatalogBuilder)

    app.add_config_value(
        'gettext_compact', True, 'gettext', types=frozenset({bool, str})
    )
    app.add_config_value('gettext_location', True, 'gettext', types=frozenset({bool}))
    app.add_config_value('gettext_uuid', False, 'gettext', types=frozenset({bool}))
    app.add_config_value('gettext_auto_build', True, 'env', types=frozenset({bool}))
    app.add_config_value(
        'gettext_additional_targets',
        [],
        'env',
        types=frozenset({frozenset, list, set, tuple}),
    )
    app.add_config_value(
        'gettext_last_translator',
        'FULL NAME <EMAIL@ADDRESS>',
        'gettext',
        types=frozenset({str}),
    )
    app.add_config_value(
        'gettext_language_team',
        'LANGUAGE <LL@li.org>',
        'gettext',
        types=frozenset({str}),
    )

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
