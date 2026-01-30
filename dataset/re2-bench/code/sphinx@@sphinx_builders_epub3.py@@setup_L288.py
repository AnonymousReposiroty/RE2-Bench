from sphinx.config import ENUM
from sphinx.util.osutil import make_filename
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_builder(Epub3Builder)

    # config values
    app.add_config_value(
        'epub_basename',
        lambda self: make_filename(self.project),
        '',
        types=frozenset({str}),
    )
    app.add_config_value(
        'epub_version', 3.0, 'epub', types=frozenset({float})
    )  # experimental
    app.add_config_value('epub_theme', 'epub', 'epub', types=frozenset({str}))
    app.add_config_value('epub_theme_options', {}, 'epub', types=frozenset({dict}))
    app.add_config_value(
        'epub_title', lambda self: self.project, 'epub', types=frozenset({str})
    )
    app.add_config_value(
        'epub_author', lambda self: self.author, 'epub', types=frozenset({str})
    )
    app.add_config_value(
        'epub_language',
        lambda self: self.language or 'en',
        'epub',
        types=frozenset({str}),
    )
    app.add_config_value(
        'epub_publisher', lambda self: self.author, 'epub', types=frozenset({str})
    )
    app.add_config_value(
        'epub_copyright', lambda self: self.copyright, 'epub', types=frozenset({str})
    )
    app.add_config_value('epub_identifier', 'unknown', 'epub', types=frozenset({str}))
    app.add_config_value('epub_scheme', 'unknown', 'epub', types=frozenset({str}))
    app.add_config_value('epub_uid', 'unknown', 'env', types=frozenset({str}))
    app.add_config_value('epub_cover', (), 'env', types=frozenset({list, tuple}))
    app.add_config_value('epub_guide', (), 'env', types=frozenset({list, tuple}))
    app.add_config_value('epub_pre_files', [], 'env', types=frozenset({list, tuple}))
    app.add_config_value('epub_post_files', [], 'env', types=frozenset({list, tuple}))
    app.add_config_value(
        'epub_css_files',
        lambda config: config.html_css_files,
        'epub',
        types=frozenset({list, tuple}),
    )
    app.add_config_value(
        'epub_exclude_files', [], 'env', types=frozenset({list, tuple})
    )
    app.add_config_value('epub_tocdepth', 3, 'env', types=frozenset({int}))
    app.add_config_value('epub_tocdup', True, 'env', types=frozenset({bool}))
    app.add_config_value('epub_tocscope', 'default', 'env', types=frozenset({str}))
    app.add_config_value('epub_fix_images', False, 'env', types=frozenset({bool}))
    app.add_config_value('epub_max_image_width', 0, 'env', types=frozenset({int}))
    app.add_config_value('epub_show_urls', 'inline', 'epub', types=frozenset({str}))
    app.add_config_value(
        'epub_use_index',
        lambda self: self.html_use_index,
        'epub',
        types=frozenset({bool}),
    )
    app.add_config_value('epub_description', 'unknown', 'epub', types=frozenset({str}))
    app.add_config_value('epub_contributor', 'unknown', 'epub', types=frozenset({str}))
    app.add_config_value(
        'epub_writing_mode', 'horizontal', 'epub', types=ENUM('horizontal', 'vertical')
    )

    # event handlers
    app.connect('config-inited', convert_epub_css_files, priority=800)
    app.connect('builder-inited', validate_config_values)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
