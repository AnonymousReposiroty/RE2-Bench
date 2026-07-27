from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata, OptionSpec

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_domain(ReSTDomain)

    return {
        'version': 'builtin',
        'env_version': 2,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
