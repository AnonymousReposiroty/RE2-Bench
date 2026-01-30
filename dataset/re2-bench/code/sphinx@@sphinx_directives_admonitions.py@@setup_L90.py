from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata, OptionSpec

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive('admonition', Admonition, override=True)
    app.add_directive('attention', Attention, override=True)
    app.add_directive('caution', Caution, override=True)
    app.add_directive('danger', Danger, override=True)
    app.add_directive('error', Error, override=True)
    app.add_directive('hint', Hint, override=True)
    app.add_directive('important', Important, override=True)
    app.add_directive('note', Note, override=True)
    app.add_directive('tip', Tip, override=True)
    app.add_directive('warning', Warning, override=True)
    app.add_directive('seealso', SeeAlso, override=True)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
