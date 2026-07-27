from types import NoneType
from sphinx.config import ENUM
from sphinx.locale import _, __
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    # builders
    app.add_builder(StandaloneHTMLBuilder)

    # config values
    app.add_config_value('html_theme', 'alabaster', 'html', types=frozenset({str}))
    app.add_config_value('html_theme_path', [], 'html', types=frozenset({list, tuple}))
    app.add_config_value('html_theme_options', {}, 'html', types=frozenset({dict}))
    app.add_config_value(
        'html_title',
        lambda c: _('%s %s documentation') % (c.project, c.release),
        'html',
        types=frozenset({str}),
    )
    app.add_config_value(
        'html_short_title', lambda self: self.html_title, 'html', types=frozenset({str})
    )
    app.add_config_value(
        'html_style', None, 'html', types=frozenset({list, str, tuple})
    )
    app.add_config_value('html_logo', None, 'html', types=frozenset({str}))
    app.add_config_value('html_favicon', None, 'html', types=frozenset({str}))
    app.add_config_value('html_css_files', [], 'html', types=frozenset({list, tuple}))
    app.add_config_value('html_js_files', [], 'html', types=frozenset({list, tuple}))
    app.add_config_value('html_static_path', [], 'html', types=frozenset({list, tuple}))
    app.add_config_value('html_extra_path', [], 'html', types=frozenset({list, tuple}))
    app.add_config_value('html_last_updated_fmt', None, 'html', types=frozenset({str}))
    app.add_config_value(
        'html_last_updated_use_utc', False, 'html', types=frozenset({bool})
    )
    app.add_config_value('html_sidebars', {}, 'html', types=frozenset({dict}))
    app.add_config_value('html_additional_pages', {}, 'html', types=frozenset({dict}))
    app.add_config_value(
        'html_domain_indices',
        True,
        'html',
        types=frozenset({frozenset, list, set, tuple}),
    )
    app.add_config_value('html_permalinks', True, 'html', types=frozenset({bool}))
    app.add_config_value('html_permalinks_icon', '¶', 'html', types=frozenset({str}))
    app.add_config_value('html_use_index', True, 'html', types=frozenset({bool}))
    app.add_config_value('html_split_index', False, 'html', types=frozenset({bool}))
    app.add_config_value('html_copy_source', True, 'html', types=frozenset({bool}))
    app.add_config_value('html_show_sourcelink', True, 'html', types=frozenset({bool}))
    app.add_config_value(
        'html_sourcelink_suffix', '.txt', 'html', types=frozenset({str})
    )
    app.add_config_value('html_use_opensearch', '', 'html', types=frozenset({str}))
    app.add_config_value('html_file_suffix', None, 'html', types=frozenset({str}))
    app.add_config_value('html_link_suffix', None, 'html', types=frozenset({str}))
    app.add_config_value('html_show_copyright', True, 'html', types=frozenset({bool}))
    app.add_config_value(
        'html_show_search_summary', True, 'html', types=frozenset({bool})
    )
    app.add_config_value('html_show_sphinx', True, 'html', types=frozenset({bool}))
    app.add_config_value('html_context', {}, 'html', types=frozenset({dict}))
    app.add_config_value(
        'html_output_encoding', 'utf-8', 'html', types=frozenset({str})
    )
    app.add_config_value('html_compact_lists', True, 'html', types=frozenset({bool}))
    app.add_config_value('html_secnumber_suffix', '. ', 'html', types=frozenset({str}))
    app.add_config_value('html_search_language', None, 'html', types=frozenset({str}))
    app.add_config_value('html_search_options', {}, 'html', types=frozenset({dict}))
    app.add_config_value('html_search_scorer', '', '', types=frozenset({str}))
    app.add_config_value(
        'html_scaled_image_link', True, 'html', types=frozenset({bool})
    )
    app.add_config_value('html_baseurl', '', 'html', types=frozenset({str}))
    # removal is indefinitely on hold
    # See: https://github.com/sphinx-doc/sphinx/issues/10265
    app.add_config_value(
        'html_codeblock_linenos_style', 'inline', 'html', types=ENUM('table', 'inline')
    )
    app.add_config_value(
        'html_math_renderer', None, 'env', types=frozenset({str, NoneType})
    )
    app.add_config_value('html4_writer', False, 'html', types=frozenset({bool}))

    # events
    app.add_event('html-collect-pages')
    app.add_event('html-page-context')

    # event handlers
    app.connect('config-inited', convert_html_css_files, priority=800)
    app.connect('config-inited', convert_html_js_files, priority=800)
    app.connect('config-inited', validate_html_extra_path, priority=800)
    app.connect('config-inited', validate_html_static_path, priority=800)
    app.connect('config-inited', validate_html_logo, priority=800)
    app.connect('config-inited', validate_html_favicon, priority=800)
    app.connect('config-inited', error_on_html_sidebars_string_values, priority=800)
    app.connect('config-inited', error_on_html_4, priority=800)
    app.connect('builder-inited', validate_math_renderer)
    app.connect('html-page-context', setup_resource_paths)

    # load default math renderer
    app.setup_extension('sphinx.ext.mathjax')

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
