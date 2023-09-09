# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 YanSte

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

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
