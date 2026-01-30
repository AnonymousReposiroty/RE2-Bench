from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_transform(PreserveTranslatableMessages)
    app.add_transform(Locale)
    app.add_transform(TranslationProgressTotaliser)
    app.add_transform(AddTranslationClasses)
    app.add_transform(RemoveTranslatableInline)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
