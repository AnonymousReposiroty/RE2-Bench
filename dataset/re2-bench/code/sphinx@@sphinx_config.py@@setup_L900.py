from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata, _ExtensionSetupFunc

def setup(app: Sphinx) -> ExtensionMetadata:
    app.connect('config-inited', deprecate_source_encoding, priority=790)
    app.connect('config-inited', convert_source_suffix, priority=800)
    app.connect('config-inited', convert_highlight_options, priority=800)
    app.connect('config-inited', init_numfig_format, priority=800)
    app.connect('config-inited', evaluate_copyright_placeholders, priority=795)
    app.connect('config-inited', correct_copyright_year, priority=800)
    app.connect('config-inited', check_confval_types, priority=800)
    app.connect('config-inited', check_primary_domain, priority=800)
    app.connect('env-get-outdated', check_master_doc)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
