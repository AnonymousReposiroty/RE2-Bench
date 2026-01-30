import calendar
import time

class PdfDict(_DictBase):

    def __getattr__(self, key):
        try:
            value = self[key.encode('us-ascii')]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(value, bytes):
            value = decode_text(value)
        if key.endswith('Date'):
            if value.startswith('D:'):
                value = value[2:]
            relationship = 'Z'
            if len(value) > 17:
                relationship = value[14]
                offset = int(value[15:17]) * 60
                if len(value) > 20:
                    offset += int(value[18:20])
            format = '%Y%m%d%H%M%S'[:len(value) - 2]
            value = time.strptime(value[:len(format) + 2], format)
            if relationship in ['+', '-']:
                offset *= 60
                if relationship == '+':
                    offset *= -1
                value = time.gmtime(calendar.timegm(value) + offset)
        return value
