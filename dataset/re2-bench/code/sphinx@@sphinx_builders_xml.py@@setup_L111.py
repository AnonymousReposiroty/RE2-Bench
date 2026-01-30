from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_builder(XMLBuilder)
    app.add_builder(PseudoXMLBuilder)

    app.add_config_value('xml_pretty', True, 'env', types=frozenset({bool}))

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
