import json
from pathlib import Path
import matplotlib as mpl

class _JSONEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, FontManager):
            return dict(o.__dict__, __class__='FontManager')
        elif isinstance(o, FontEntry):
            d = dict(o.__dict__, __class__='FontEntry')
            try:
                d['fname'] = str(Path(d['fname']).relative_to(mpl.get_data_path()))
            except ValueError:
                pass
            return d
        else:
            return super().default(o)
