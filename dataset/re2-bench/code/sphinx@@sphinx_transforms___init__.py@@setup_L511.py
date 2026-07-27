from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_transform(ApplySourceWorkaround)
    app.add_transform(ExtraTranslatableNodes)
    app.add_transform(DefaultSubstitutions)
    app.add_transform(MoveModuleTargets)
    app.add_transform(HandleCodeBlocks)
    app.add_transform(SortIds)
    app.add_transform(DoctestTransform)
    app.add_transform(AutoNumbering)
    app.add_transform(AutoIndexUpgrader)
    app.add_transform(FilterSystemMessages)
    app.add_transform(UnreferencedFootnotesDetector)
    app.add_transform(SphinxSmartQuotes)
    app.add_transform(DoctreeReadEvent)
    app.add_transform(GlossarySorter)
    app.add_transform(ReorderConsecutiveTargetAndIndexNodes)

    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
