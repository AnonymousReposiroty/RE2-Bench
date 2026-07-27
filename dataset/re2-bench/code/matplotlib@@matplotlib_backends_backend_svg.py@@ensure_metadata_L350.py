import datetime
import itertools
import os
import numpy as np
import matplotlib as mpl
from matplotlib.backend_bases import (
     _Backend, FigureCanvasBase, FigureManagerBase, RendererBase)
from matplotlib.dates import UTC

class RendererSVG(RendererBase):

    def __init__(self, width, height, svgwriter, basename=None, image_dpi=72, *, metadata=None):
        self.width = width
        self.height = height
        self.writer = XMLWriter(svgwriter)
        self.image_dpi = image_dpi
        if basename is None:
            basename = getattr(svgwriter, 'name', '')
            if not isinstance(basename, str):
                basename = ''
        self.basename = basename
        self._groupd = {}
        self._image_counter = itertools.count()
        self._clip_path_ids = {}
        self._clipd = {}
        self._markers = {}
        self._path_collection_id = 0
        self._hatchd = {}
        self._has_gouraud = False
        self._n_gradients = 0
        super().__init__()
        self._glyph_map = dict()
        str_height = _short_float_fmt(height)
        str_width = _short_float_fmt(width)
        svgwriter.write(svgProlog)
        self._start_id = self.writer.start('svg', width=f'{str_width}pt', height=f'{str_height}pt', viewBox=f'0 0 {str_width} {str_height}', xmlns='http://www.w3.org/2000/svg', version='1.1', id=mpl.rcParams['svg.id'], attrib={'xmlns:xlink': 'http://www.w3.org/1999/xlink'})
        self._write_metadata(metadata)
        self._write_default_style()

    def _write_metadata(self, metadata):
        if metadata is None:
            metadata = {}
        metadata = {'Format': 'image/svg+xml', 'Type': 'http://purl.org/dc/dcmitype/StillImage', 'Creator': f'Matplotlib v{mpl.__version__}, https://matplotlib.org/', **metadata}
        writer = self.writer
        if 'Title' in metadata:
            title = metadata['Title']
            _check_is_str(title, 'Title')
            writer.element('title', text=title)
        date = metadata.get('Date', None)
        if date is not None:
            if isinstance(date, str):
                dates = [date]
            elif isinstance(date, (datetime.datetime, datetime.date)):
                dates = [date.isoformat()]
            elif np.iterable(date):
                dates = []
                for d in date:
                    if isinstance(d, str):
                        dates.append(d)
                    elif isinstance(d, (datetime.datetime, datetime.date)):
                        dates.append(d.isoformat())
                    else:
                        raise TypeError(f'Invalid type for Date metadata. Expected iterable of str, date, or datetime, not {type(d)}.')
            else:
                raise TypeError(f'Invalid type for Date metadata. Expected str, date, datetime, or iterable of the same, not {type(date)}.')
            metadata['Date'] = '/'.join(dates)
        elif 'Date' not in metadata:
            date = os.getenv('SOURCE_DATE_EPOCH')
            if date:
                date = datetime.datetime.fromtimestamp(int(date), datetime.timezone.utc)
                metadata['Date'] = date.replace(tzinfo=UTC).isoformat()
            else:
                metadata['Date'] = datetime.datetime.today().isoformat()
        mid = None

        def ensure_metadata(mid):
            if mid is not None:
                return mid
            mid = writer.start('metadata')
            writer.start('rdf:RDF', attrib={'xmlns:dc': 'http://purl.org/dc/elements/1.1/', 'xmlns:cc': 'http://creativecommons.org/ns#', 'xmlns:rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'})
            writer.start('cc:Work')
            return mid
        uri = metadata.pop('Type', None)
        if uri is not None:
            mid = ensure_metadata(mid)
            writer.element('dc:type', attrib={'rdf:resource': uri})
        for key in ['Title', 'Coverage', 'Date', 'Description', 'Format', 'Identifier', 'Language', 'Relation', 'Source']:
            info = metadata.pop(key, None)
            if info is not None:
                mid = ensure_metadata(mid)
                _check_is_str(info, key)
                writer.element(f'dc:{key.lower()}', text=info)
        for key in ['Creator', 'Contributor', 'Publisher', 'Rights']:
            agents = metadata.pop(key, None)
            if agents is None:
                continue
            if isinstance(agents, str):
                agents = [agents]
            _check_is_iterable_of_str(agents, key)
            mid = ensure_metadata(mid)
            writer.start(f'dc:{key.lower()}')
            for agent in agents:
                writer.start('cc:Agent')
                writer.element('dc:title', text=agent)
                writer.end('cc:Agent')
            writer.end(f'dc:{key.lower()}')
        keywords = metadata.pop('Keywords', None)
        if keywords is not None:
            if isinstance(keywords, str):
                keywords = [keywords]
            _check_is_iterable_of_str(keywords, 'Keywords')
            mid = ensure_metadata(mid)
            writer.start('dc:subject')
            writer.start('rdf:Bag')
            for keyword in keywords:
                writer.element('rdf:li', text=keyword)
            writer.end('rdf:Bag')
            writer.end('dc:subject')
        if mid is not None:
            writer.close(mid)
        if metadata:
            raise ValueError('Unknown metadata key(s) passed to SVG writer: ' + ','.join(metadata))

    def _write_default_style(self):
        writer = self.writer
        default_style = _generate_css({'stroke-linejoin': 'round', 'stroke-linecap': 'butt'})
        writer.start('defs')
        writer.element('style', type='text/css', text='*{%s}' % default_style)
        writer.end('defs')
