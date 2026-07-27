class Stat:

    def __init__(self, image_or_list, mask=None):
        try:
            if mask:
                self.h = image_or_list.histogram(mask)
            else:
                self.h = image_or_list.histogram()
        except AttributeError:
            self.h = image_or_list
        if not isinstance(self.h, list):
            msg = 'first argument must be image or list'
            raise TypeError(msg)
        self.bands = list(range(len(self.h) // 256))

    def __getattr__(self, id):
        """Calculate missing attribute"""
        if id[:4] == '_get':
            raise AttributeError(id)
        v = getattr(self, '_get' + id)()
        setattr(self, id, v)
        return v
