import sphinx
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_builder(CoverageBuilder)
    app.add_config_value('coverage_modules', (), '', types=frozenset({list, tuple}))
    app.add_config_value(
        'coverage_ignore_modules', [], '', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'coverage_ignore_functions', [], '', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'coverage_ignore_classes', [], '', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'coverage_ignore_pyobjects', [], '', types=frozenset({list, tuple})
    )
    app.add_config_value('coverage_c_path', [], '', types=frozenset({list, tuple}))
    app.add_config_value('coverage_c_regexes', {}, '', types=frozenset({dict}))
    app.add_config_value('coverage_ignore_c_items', {}, '', types=frozenset({dict}))
    app.add_config_value('coverage_write_headline', True, '', types=frozenset({bool}))
    app.add_config_value(
        'coverage_statistics_to_report', True, '', types=frozenset({bool})
    )
    app.add_config_value(
        'coverage_statistics_to_stdout', True, '', types=frozenset({bool})
    )
    app.add_config_value(
        'coverage_skip_undoc_in_source', False, '', types=frozenset({bool})
    )
    app.add_config_value(
        'coverage_show_missing_items', False, '', types=frozenset({bool})
    )
    return {
        'version': sphinx.__display_version__,
        'parallel_read_safe': True,
    }
