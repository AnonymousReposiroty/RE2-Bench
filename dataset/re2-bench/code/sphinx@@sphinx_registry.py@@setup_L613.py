from sphinx.application import Sphinx
from sphinx.util.typing import (
    ExtensionMetadata,
    RoleFunction,
    TitleGetter,
    _ExtensionSetupFunc,
)

def setup(app: Sphinx) -> ExtensionMetadata:
    app.connect('config-inited', merge_source_suffix, priority=800)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
