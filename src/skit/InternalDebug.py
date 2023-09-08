class InternalDebug:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

    def log(self, *args):
        if self.debug_mode:
            print(*args)

    def separator(self, character="=", length=50):
        separator = character * length
        self.log(separator)

    def info(self, *args):
        if self.debug_mode:
            prefix = "[INFO]"
            self.log(prefix, *args)

    def warning(self, *args):
        if self.debug_mode:
            prefix = "[WARNING]"
            self.log(prefix, *args)

    def error(self, *args):
        if self.debug_mode:
            prefix = "[ERROR]"
            self.log(prefix, *args)

    def set_debug_mode(self, debug_mode):
        self.debug_mode = debug_mode
