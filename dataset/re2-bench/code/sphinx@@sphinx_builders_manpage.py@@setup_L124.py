from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_builder(ManualPageBuilder)

    app.add_config_value(
        'man_pages', default_man_pages, '', types=frozenset({list, tuple})
    )
    app.add_config_value('man_show_urls', False, '', types=frozenset({bool}))
    app.add_config_value(
        'man_make_section_directory', False, '', types=frozenset({bool})
    )

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
