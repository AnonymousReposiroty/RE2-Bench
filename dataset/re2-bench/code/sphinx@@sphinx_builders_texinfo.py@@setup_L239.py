from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_builder(TexinfoBuilder)

    app.add_config_value(
        'texinfo_documents',
        default_texinfo_documents,
        '',
        types=frozenset({list, tuple}),
    )
    app.add_config_value('texinfo_appendices', [], '', types=frozenset({list, tuple}))
    app.add_config_value('texinfo_elements', {}, '', types=frozenset({dict}))
    app.add_config_value(
        'texinfo_domain_indices',
        True,
        '',
        types=frozenset({frozenset, list, set, tuple}),
    )
    app.add_config_value('texinfo_show_urls', 'footnote', '', types=frozenset({str}))
    app.add_config_value('texinfo_no_detailmenu', False, '', types=frozenset({bool}))
    app.add_config_value('texinfo_cross_references', True, '', types=frozenset({bool}))

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
