from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_transform(FootnoteDocnameUpdater)
    app.add_post_transform(SubstitutionDefinitionsRemover)
    app.add_post_transform(BibliographyTransform)
    app.add_post_transform(CitationReferenceTransform)
    app.add_post_transform(DocumentTargetTransform)
    app.add_post_transform(IndexInSectionTitleTransform)
    app.add_post_transform(LaTeXFootnoteTransform)
    app.add_post_transform(LiteralBlockTransform)
    app.add_post_transform(MathReferenceTransform)
    app.add_post_transform(ShowUrlsTransform)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
