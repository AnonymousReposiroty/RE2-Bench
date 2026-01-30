from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata, OptionSpec

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_domain(ChangeSetDomain)
    app.add_directive('version-deprecated', VersionChange)
    app.add_directive('deprecated', VersionChange)
    app.add_directive('version-added', VersionChange)
    app.add_directive('versionadded', VersionChange)
    app.add_directive('version-changed', VersionChange)
    app.add_directive('versionchanged', VersionChange)
    app.add_directive('version-removed', VersionChange)
    app.add_directive('versionremoved', VersionChange)

    return {
        'version': 'builtin',
        'env_version': 1,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
