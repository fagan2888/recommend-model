#coding=utf8

import os
import sys
import time

DEFAULT_COLUMNS = 80

if os.name == 'nt':
    BEFORE_BAR = '\r'
    AFTER_BAR = '\n'
else:
    BEFORE_BAR = '\r\033[?25l'
    AFTER_BAR = '\033[?25h\n'

def get_terminal_size():
    """Returns the current size of the terminal as tuple in the form
    ``(width, height)`` in columns and rows.
    """
    # If shutil has get_terminal_size() (Python 3.3 and later) use that
    if sys.version_info >= (3, 3):
        import shutil
        shutil_get_terminal_size = getattr(shutil, 'get_terminal_size', None)
        if shutil_get_terminal_size:
            sz = shutil_get_terminal_size()
            return sz.columns, sz.lines

    # if get_winterm_size is not None:
    #     return get_winterm_size()

    def ioctl_gwinsz(fd):
        try:
            import fcntl
            import termios
            cr = struct.unpack(
                'hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
        except Exception:
            return
        return cr

    cr = ioctl_gwinsz(0) or ioctl_gwinsz(1) or ioctl_gwinsz(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            try:
                cr = ioctl_gwinsz(fd)
            finally:
                os.close(fd)
        except Exception:
            pass
    if not cr or not cr[0] or not cr[1]:
        cr = (os.environ.get('LINES', 25),
              os.environ.get('COLUMNS', DEFAULT_COLUMNS))
    return int(cr[1]), int(cr[0])


class ProgressBar(object):

    def __init__(self, iterable, length=None, fill_char='#', empty_char=' ',
                 bar_template='%(bar)s', info_sep='  ', show_eta=True,
                 show_percent=None, show_pos=False, item_show_func=None,
                 label=None, file=None, color=None, width=30):
        self.fill_char = fill_char
        self.empty_char = empty_char
        self.bar_template = bar_template
        self.info_sep = info_sep
        self.show_eta = show_eta
        self.show_percent = show_percent
        self.show_pos = show_pos
        self.item_show_func = item_show_func
        self.label = label or ''
        if file is None:
            file = sys.stdout
        self.file = file
        self.color = color
        self.width = width
        self.autowidth = width == 0

        if length is None:
            length = len(iterable)
        if iterable is None:
            if length is None:
                raise TypeError('iterable or length is required')
            iterable = range(0, length)
        self.iter = iter(iterable)
        self.length = length
        self.length_known = length is not None
        self.pos = 0
        self.avg = []
        self.start = self.last_eta = time.time()
        self.eta_known = False
        self.finished = False
        self.max_width = None
        self.entered = False
        self.current_item = None
        self.is_hidden = not self.file.isatty()
        self._last_line = None

    def __enter__(self):
        self.entered = True
        self.render_progress()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.render_finish()

    def __iter__(self):
        if not self.entered:
            raise RuntimeError('You need to use progress bars in a with block.')
        self.render_progress()
        return self

    def render_finish(self):
        if self.is_hidden:
            return
        self.file.write(AFTER_BAR)
        self.file.flush()

    @property
    def pct(self):
        if self.finished:
            return 1.0
        return min(self.pos / (float(self.length) or 1), 1.0)

    @property
    def time_per_iteration(self):
        if not self.avg:
            return 0.0
        return sum(self.avg) / float(len(self.avg))

    @property
    def eta(self):
        if self.length_known and not self.finished:
            return self.time_per_iteration * (self.length - self.pos)
        return 0.0

    def format_eta(self):
        if self.eta_known:
            t = self.eta + 1
            seconds = t % 60
            t /= 60
            minutes = t % 60
            t /= 60
            hours = t % 24
            t /= 24
            if int(t) > 0:
                days = t
                return '%dd %02d:%02d:%02d' % (days, hours, minutes, seconds)
            else:
                return '%02d:%02d:%02d' % (hours, minutes, seconds)
        return ''

    def format_pos(self):
        pos = str(self.pos)
        if self.length_known:
            pos += '/%s' % self.length
        return pos

    def format_pct(self):
        return ('% 4d%%' % int(self.pct * 100))[1:]

    def format_progress_line(self):
        show_percent = self.show_percent

        info_bits = []
        if self.length_known:
            bar_length = int(self.pct * self.width)
            bar = self.fill_char * bar_length
            bar += self.empty_char * (self.width - bar_length)
            if show_percent is None:
                show_percent = not self.show_pos
        else:
            if self.finished:
                bar = self.fill_char * self.width
            else:
                bar = list(self.empty_char * (self.width or 1))
                if self.time_per_iteration != 0:
                    bar[int((math.cos(self.pos * self.time_per_iteration)
                        / 2.0 + 0.5) * self.width)] = self.fill_char
                bar = ''.join(bar)

        if self.show_pos:
            info_bits.append(self.format_pos())
        if show_percent:
            info_bits.append(self.format_pct())
        if self.show_eta and self.eta_known and not self.finished:
            info_bits.append(self.format_eta())

        label = self.label
        if self.item_show_func is not None:
            item_info = self.item_show_func(self.current_item)
            if item_info is not None:
                label += " " + item_info

        return (self.bar_template % {
            'label': label,
            'bar': bar,
            'info': self.info_sep.join(info_bits)
        }).rstrip()

    def render_progress(self):
        nl = False

        if self.is_hidden:
            buf = [self.label]
            nl = True
        else:
            buf = []
            # Update width in case the terminal has been resized
            if self.autowidth:
                old_width = self.width
                self.width = 0
                clutter_length = len(self.format_progress_line())
                new_width = max(0, get_terminal_size()[0] - clutter_length)
                if new_width < old_width:
                    buf.append(BEFORE_BAR)
                    buf.append(' ' * self.max_width)
                    self.max_width = new_width
                self.width = new_width

            clear_width = self.width
            if self.max_width is not None:
                clear_width = self.max_width

            buf.append(BEFORE_BAR)
            line = self.format_progress_line()
            line_len = len(line)
            if self.max_width is None or self.max_width < line_len:
                self.max_width = line_len
            buf.append(line)

            buf.append(' ' * (clear_width - line_len))
        line = ''.join(buf)

        # Render the line only if it changed.
        if line != self._last_line:
            self._last_line = line
            # echo(line, file=self.file, color=self.color, nl=nl)
            self.file.write(line)
            self.file.flush()

    def make_step(self, n_steps):
        self.pos += n_steps
        if self.length_known and self.pos >= self.length:
            self.finished = True

        if (time.time() - self.last_eta) < 1.0:
            return

        self.last_eta = time.time()
        self.avg = self.avg[-6:] + [-(self.start - time.time()) / (self.pos)]

        self.eta_known = self.length_known

    def update(self, n_steps):
        self.make_step(n_steps)
        self.render_progress()

    def finish(self):
        self.eta_known = 0
        self.current_item = None
        self.finished = True

    def next(self):
        if self.is_hidden:
            return next(self.iter)
        try:
            rv = next(self.iter)
            self.current_item = rv
        except StopIteration:
            self.finish()
            self.render_progress()
            raise StopIteration()
        else:
            self.update(1)
            return rv

