from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_builder(CheckExternalLinksBuilder)
    app.add_post_transform(HyperlinkCollector)

    app.add_config_value('linkcheck_ignore', [], '', types=frozenset({list, tuple}))
    app.add_config_value(
        'linkcheck_exclude_documents', [], '', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'linkcheck_allowed_redirects', _SENTINEL_LAR, '', types=frozenset({dict})
    )
    app.add_config_value('linkcheck_auth', [], '', types=frozenset({list, tuple}))
    app.add_config_value('linkcheck_request_headers', {}, '', types=frozenset({dict}))
    app.add_config_value('linkcheck_retries', 1, '', types=frozenset({int}))
    app.add_config_value('linkcheck_timeout', 30, '', types=frozenset({float, int}))
    app.add_config_value('linkcheck_workers', 5, '', types=frozenset({int}))
    app.add_config_value('linkcheck_anchors', True, '', types=frozenset({bool}))
    # Anchors starting with ! are ignored since they are
    # commonly used for dynamic pages
    app.add_config_value(
        'linkcheck_anchors_ignore', ['^!'], '', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'linkcheck_anchors_ignore_for_url', (), '', types=frozenset({list, tuple})
    )
    app.add_config_value(
        'linkcheck_rate_limit_timeout', 300.0, '', types=frozenset({float, int})
    )
    app.add_config_value(
        'linkcheck_allow_unauthorized', False, '', types=frozenset({bool})
    )
    app.add_config_value(
        'linkcheck_report_timeouts_as_broken', False, '', types=frozenset({bool})
    )
    app.add_config_value(
        'linkcheck_case_insensitive_urls',
        (),
        '',
        types=frozenset({frozenset, list, set, tuple}),
    )

    app.add_event('linkcheck-process-uri')

    # priority 900 to happen after ``check_confval_types()``
    app.connect('config-inited', compile_linkcheck_allowed_redirects, priority=900)

    # FIXME: Disable URL rewrite handler for github.com temporarily.
    # See: https://github.com/sphinx-doc/sphinx/issues/9435
    # app.connect('linkcheck-process-uri', rewrite_github_anchor)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
