class Debug:
    debug_mode = False  # Par défaut en mode non débogage

    @staticmethod
    def log(*args):
        if Debug.debug_mode:
            print(*args)

    @classmethod
    def set_debug_mode(cls, debug_mode):
        cls.debug_mode = debug_mode

    @staticmethod
    def separator(character="=", length=50):
        separator = character * length
        Debug.log(separator)

    @staticmethod
    def info(*args):
        if Debug.debug_mode:
            prefix = "[INFO]"
            Debug.log(prefix, *args)

    @staticmethod
    def warning(*args):
        if Debug.debug_mode:
            prefix = "[WARNING]"
            Debug.log(prefix, *args)

    @staticmethod
    def error(*args):
        if Debug.debug_mode:
            prefix = "[ERROR]"
            Debug.log(prefix, *args)
