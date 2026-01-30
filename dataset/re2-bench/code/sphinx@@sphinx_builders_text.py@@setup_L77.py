from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_builder(TextBuilder)

    app.add_config_value('text_sectionchars', '*=-~"+`', 'env', types=frozenset({str}))
    app.add_config_value('text_newlines', 'unix', 'env', types=frozenset({str}))
    app.add_config_value('text_add_secnumbers', True, 'env', types=frozenset({bool}))
    app.add_config_value('text_secnumber_suffix', '. ', 'env', types=frozenset({str}))

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
