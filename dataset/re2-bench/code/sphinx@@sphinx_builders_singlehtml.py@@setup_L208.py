from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.setup_extension('sphinx.builders.html')

    app.add_builder(SingleFileHTMLBuilder)
    app.add_config_value(
        'singlehtml_sidebars',
        lambda self: self.html_sidebars,
        'html',
        types=frozenset({dict}),
    )

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
