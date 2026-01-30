from sphinx.config import ENUM
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.setup_extension('sphinx.builders.latex.transforms')

    app.add_builder(LaTeXBuilder)
    app.connect('config-inited', validate_config_values, priority=800)
    app.connect('config-inited', validate_latex_theme_options, priority=800)
    app.connect('builder-inited', install_packages_for_ja)

    app.add_config_value(
        'latex_engine',
        default_latex_engine,
        '',
        types=ENUM('pdflatex', 'xelatex', 'lualatex', 'platex', 'uplatex'),
    )
    app.add_config_value(
        'latex_documents', default_latex_documents, '', types=frozenset({list, tuple})
    )
    app.add_config_value('latex_logo', None, '', types=frozenset({str}))
    app.add_config_value('latex_appendices', [], '', types=frozenset({list, tuple}))
    app.add_config_value(
        'latex_use_latex_multicolumn', False, '', types=frozenset({bool})
    )
    app.add_config_value(
        'latex_use_xindy', default_latex_use_xindy, '', types=frozenset({bool})
    )
    app.add_config_value(
        'latex_toplevel_sectioning',
        None,
        '',
        types=ENUM(None, 'part', 'chapter', 'section'),
    )
    app.add_config_value(
        'latex_domain_indices', True, '', types=frozenset({frozenset, list, set, tuple})
    )
    app.add_config_value('latex_show_urls', 'no', '', types=frozenset({str}))
    app.add_config_value('latex_show_pagerefs', False, '', types=frozenset({bool}))
    app.add_config_value('latex_elements', {}, '', types=frozenset({dict}))
    app.add_config_value(
        'latex_additional_files', [], '', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'latex_table_style', ['booktabs', 'colorrows'], '', types=frozenset({list})
    )
    app.add_config_value('latex_theme', 'manual', '', types=frozenset({str}))
    app.add_config_value('latex_theme_options', {}, '', types=frozenset({dict}))
    app.add_config_value('latex_theme_path', [], '', types=frozenset({list}))

    app.add_config_value(
        'latex_docclass', default_latex_docclass, '', types=frozenset({dict})
    )

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
