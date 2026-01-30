from . import Image, ImageFile

class JpegImageFile(ImageFile.ImageFile):
    format = 'JPEG'
    format_description = 'JPEG (ISO 10918)'

    def getxmp(self):
        """
        Returns a dictionary containing the XMP tags.
        Requires defusedxml to be installed.

        :returns: XMP tags in a dictionary.
        """
        for segment, content in self.applist:
            if segment == 'APP1':
                marker, xmp_tags = content.split(b'\x00')[:2]
                if marker == b'http://ns.adobe.com/xap/1.0/':
                    return self._getxmp(xmp_tags)
        return {}
