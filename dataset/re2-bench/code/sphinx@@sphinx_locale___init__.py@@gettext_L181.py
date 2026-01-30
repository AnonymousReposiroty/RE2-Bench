from collections.abc import Callable, Iterable

def get_translation(catalog: str, namespace: str = 'general') -> Callable[[str], str]:
    """Get a translation function based on the *catalog* and *namespace*.

    The extension can use this API to translate the messages on the
    extension::

        from pathlib import Path
        from sphinx.locale import get_translation

        MESSAGE_CATALOG_NAME = 'myextension'  # name of *.pot, *.po and *.mo files
        _ = get_translation(MESSAGE_CATALOG_NAME)
        text = _('Hello Sphinx!')


        def setup(app):
            package_dir = Path(__file__).resolve().parent
            locale_dir = package_dir / 'locales'
            app.add_message_catalog(MESSAGE_CATALOG_NAME, locale_dir)

    With this code, sphinx searches a message catalog from
    ``${package_dir}/locales/${language}/LC_MESSAGES/myextension.mo``.
    The :confval:`language` is used for the searching.

    .. versionadded:: 1.8
    """

    def gettext(message: str) -> str:
        if not is_translator_registered(catalog, namespace):
            # not initialized yet
            return _TranslationProxy(catalog, namespace, message)  # type: ignore[return-value]
        else:
            translator = get_translator(catalog, namespace)
            return translator.gettext(message)

    return gettext
