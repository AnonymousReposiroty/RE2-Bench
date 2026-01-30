import sphinx
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata, RoleFunction

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_config_value('extlinks', {}, 'env', types=frozenset({dict}))
    app.add_config_value(
        'extlinks_detect_hardcoded_links', False, 'env', types=frozenset({bool})
    )

    app.connect('builder-inited', setup_link_roles)
    app.add_post_transform(ExternalLinksChecker)
    return {
        'version': sphinx.__display_version__,
        'parallel_read_safe': True,
    }
