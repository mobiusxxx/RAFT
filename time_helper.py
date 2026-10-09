#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Time: 2023/2/28 10:06
@Author: SPZ
@File: time_helper.py
@Software: pycharm
"""
import time

from functools import wraps

debug = True


def open_debug():
    global debug
    debug = False


def cost_time(description='', show=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            if debug and show:
                print(f"{description} is done. && cost {(time.time() - start) * 1000}ms")
            return result

        return wrapper

    return decorator