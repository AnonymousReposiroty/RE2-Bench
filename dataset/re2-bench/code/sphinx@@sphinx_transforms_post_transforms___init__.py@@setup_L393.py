from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_post_transform(ReferencesResolver)
    app.add_post_transform(OnlyNodeTransform)
    app.add_post_transform(SigElementFallbackTransform)
    app.add_post_transform(PropagateDescDomain)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
