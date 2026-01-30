import logging
import logging.handlers
from docutils import nodes
from sphinx.application import Sphinx

class SphinxLogRecordTranslator(logging.Filter):
    """Converts a log record to one Sphinx expects

    * Make a instance of SphinxLogRecord
    * docname to path if location given
    * append warning type/subtype to message if :confval:`show_warning_types` is ``True``
    """
    LogRecordClass: type[logging.LogRecord]

    def __init__(self, app: Sphinx) -> None:
        self._app = app
        super().__init__()

    def filter(self, record: SphinxWarningLogRecord) -> bool:
        if isinstance(record, logging.LogRecord):
            record.__class__ = self.LogRecordClass
        location = getattr(record, 'location', None)
        if isinstance(location, tuple):
            docname, lineno = location
            if docname:
                if lineno:
                    record.location = f'{self._app.env.doc2path(docname)}:{lineno}'
                else:
                    record.location = f'{self._app.env.doc2path(docname)}'
            else:
                record.location = None
        elif isinstance(location, nodes.Node):
            record.location = get_node_location(location)
        elif location and ':' not in location:
            record.location = f'{self._app.env.doc2path(location)}'
        return True
