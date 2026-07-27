import logging

@writers.register('ffmpeg')
class FFMpegWriter(FFMpegBase, MovieWriter):
    """
    Pipe-based ffmpeg writer.

    Frames are streamed directly to ffmpeg via a pipe and written in a single pass.

    This effectively works as a slideshow input to ffmpeg with the fps passed as
    ``-framerate``, so see also `their notes on frame rates`_ for further details.

    .. _their notes on frame rates: https://trac.ffmpeg.org/wiki/Slideshow#Framerates
    """

    def _args(self):
        args = [self.bin_path(), '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '%dx%d' % self.frame_size, '-pix_fmt', self.frame_format, '-framerate', str(self.fps)]
        if _log.getEffectiveLevel() > logging.DEBUG:
            args += ['-loglevel', 'error']
        args += ['-i', 'pipe:'] + self.output_args
        return args
