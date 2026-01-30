from types import NoneType
import sphinx
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_html_math_renderer(
        'mathjax',
        inline_renderers=(html_visit_math, None),
        block_renderers=(html_visit_displaymath, None),
    )

    app.add_config_value('mathjax_path', MATHJAX_URL, 'html', types=frozenset({str}))
    app.add_config_value('mathjax_options', {}, 'html', types=frozenset({dict}))
    app.add_config_value(
        'mathjax_inline', [r'\(', r'\)'], 'html', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'mathjax_display', [r'\[', r'\]'], 'html', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'mathjax_config', None, 'html', types=frozenset({dict, NoneType})
    )
    app.add_config_value(
        'mathjax2_config',
        lambda c: c.mathjax_config,
        'html',
        types=frozenset({dict, NoneType}),
    )
    app.add_config_value(
        'mathjax3_config', None, 'html', types=frozenset({dict, NoneType})
    )
    app.add_config_value(
        'mathjax4_config', None, 'html', types=frozenset({dict, NoneType})
    )
    app.add_config_value('mathjax_config_path', '', 'html', types=frozenset({str}))
    app.connect('html-page-context', install_mathjax)

    return {
        'version': sphinx.__display_version__,
        'parallel_read_safe': True,
    }
