from docutils.parsers.rst import directives, roles
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata, OptionSpec

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_config_value(
        'strip_signature_backslash', False, 'env', types=frozenset({bool})
    )
    directives.register_directive('default-role', DefaultRole)
    directives.register_directive('default-domain', DefaultDomain)
    directives.register_directive('describe', ObjectDescription)
    # new, more consistent, name
    directives.register_directive('object', ObjectDescription)

    app.add_event('object-description-transform')

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
