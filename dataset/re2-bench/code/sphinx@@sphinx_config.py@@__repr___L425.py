import types
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
from sphinx.util.typing import ExtensionMetadata, _ExtensionSetupFunc

class Config:
    """Configuration file abstraction.

    The Config object makes the values of all config options available as
    attributes.

    It is exposed via the :py:class:`~sphinx.application.Sphinx`\\ ``.config``
    and :py:class:`sphinx.environment.BuildEnvironment`\\ ``.config`` attributes.
    For example, to get the value of :confval:`language`, use either
    ``app.config.language`` or ``env.config.language``.
    """
    config_values: dict[str, _Opt] = {'project': _Opt('Project name not set', 'env', frozenset((str,))), 'author': _Opt('Author name not set', 'env', frozenset((str,))), 'project_copyright': _Opt('', 'html', frozenset((str, tuple, list))), 'copyright': _Opt(lambda config: config.project_copyright, 'html', frozenset((str, tuple, list))), 'version': _Opt('', 'env', frozenset((str,))), 'release': _Opt('', 'env', frozenset((str,))), 'today': _Opt('', 'env', frozenset((str,))), 'today_fmt': _Opt(None, 'env', frozenset((str,))), 'language': _Opt('en', 'env', frozenset((str,))), 'locale_dirs': _Opt(['locales'], 'env', frozenset((list, tuple))), 'figure_language_filename': _Opt('{root}.{language}{ext}', 'env', frozenset((str,))), 'gettext_allow_fuzzy_translations': _Opt(False, 'gettext', frozenset((bool,))), 'translation_progress_classes': _Opt(False, 'env', ENUM(True, False, 'translated', 'untranslated')), 'master_doc': _Opt('index', 'env', frozenset((str,))), 'root_doc': _Opt(lambda config: config.master_doc, 'env', frozenset((str,))), 'source_suffix': _Opt({'.rst': 'restructuredtext'}, 'env', Any), 'source_encoding': _Opt('utf-8-sig', 'env', frozenset((str,))), 'exclude_patterns': _Opt([], 'env', frozenset((str,))), 'include_patterns': _Opt(['**'], 'env', frozenset((str,))), 'default_role': _Opt(None, 'env', frozenset((str,))), 'add_function_parentheses': _Opt(True, 'env', frozenset((bool,))), 'add_module_names': _Opt(True, 'env', frozenset((bool,))), 'toc_object_entries': _Opt(True, 'env', frozenset((bool,))), 'toc_object_entries_show_parents': _Opt('domain', 'env', ENUM('domain', 'all', 'hide')), 'trim_footnote_reference_space': _Opt(False, 'env', frozenset((bool,))), 'show_authors': _Opt(False, 'env', frozenset((bool,))), 'pygments_style': _Opt(None, 'html', frozenset((str,))), 'highlight_language': _Opt('default', 'env', frozenset((str,))), 'highlight_options': _Opt({}, 'env', frozenset((dict,))), 'templates_path': _Opt([], 'html', frozenset((list,))), 'template_bridge': _Opt(None, 'html', frozenset((str,))), 'keep_warnings': _Opt(False, 'env', frozenset((bool,))), 'suppress_warnings': _Opt([], 'env', frozenset((list, tuple))), 'show_warning_types': _Opt(True, 'env', frozenset((bool,))), 'modindex_common_prefix': _Opt([], 'html', frozenset((list, tuple))), 'rst_epilog': _Opt(None, 'env', frozenset((str,))), 'rst_prolog': _Opt(None, 'env', frozenset((str,))), 'trim_doctest_flags': _Opt(True, 'env', frozenset((bool,))), 'primary_domain': _Opt('py', 'env', frozenset((types.NoneType,))), 'needs_sphinx': _Opt(None, '', frozenset((str,))), 'needs_extensions': _Opt({}, '', frozenset((dict,))), 'manpages_url': _Opt(None, 'env', frozenset((str, types.NoneType))), 'nitpicky': _Opt(False, '', frozenset((bool,))), 'nitpick_ignore': _Opt([], '', frozenset((set, list, tuple))), 'nitpick_ignore_regex': _Opt([], '', frozenset((set, list, tuple))), 'numfig': _Opt(False, 'env', frozenset((bool,))), 'numfig_secnum_depth': _Opt(1, 'env', frozenset((int, types.NoneType))), 'numfig_format': _Opt({}, 'env', frozenset((dict,))), 'maximum_signature_line_length': _Opt(None, 'env', frozenset((int, types.NoneType))), 'math_number_all': _Opt(False, 'env', frozenset((bool,))), 'math_eqref_format': _Opt(None, 'env', frozenset((str,))), 'math_numfig': _Opt(True, 'env', frozenset((bool,))), 'math_numsep': _Opt('.', 'env', frozenset((str,))), 'tls_verify': _Opt(True, 'env', frozenset((bool,))), 'tls_cacerts': _Opt(None, 'env', frozenset((str, dict, types.NoneType))), 'user_agent': _Opt(None, 'env', frozenset((str,))), 'smartquotes': _Opt(True, 'env', frozenset((bool,))), 'smartquotes_action': _Opt('qDe', 'env', frozenset((str,))), 'smartquotes_excludes': _Opt({'languages': ['ja', 'zh_CN', 'zh_TW'], 'builders': ['man', 'text']}, 'env', frozenset((dict,))), 'option_emphasise_placeholders': _Opt(False, 'env', frozenset((bool,)))}

    def __init__(self, config: dict[str, Any] | None=None, overrides: dict[str, Any] | None=None) -> None:
        raw_config: dict[str, Any] = config or {}
        self._overrides = dict(overrides) if overrides is not None else {}
        self._options = Config.config_values.copy()
        self._raw_config = raw_config
        for name in list(self._overrides.keys()):
            if '.' in name:
                real_name, _, key = name.partition('.')
                raw_config.setdefault(real_name, {})[key] = self._overrides.pop(name)
        self.setup: _ExtensionSetupFunc | None = raw_config.get('setup')
        if 'extensions' in self._overrides:
            extensions = self._overrides.pop('extensions')
            if isinstance(extensions, str):
                raw_config['extensions'] = extensions.split(',')
            else:
                raw_config['extensions'] = extensions
        self.extensions: list[str] = raw_config.get('extensions', [])
        self._verbosity: int = 0

    def __repr__(self) -> str:
        values = []
        for opt_name in self._options:
            try:
                opt_value = getattr(self, opt_name)
            except Exception:
                opt_value = '<error!>'
            values.append(f'{opt_name}={opt_value!r}')
        return self.__class__.__qualname__ + '(' + ', '.join(values) + ')'
