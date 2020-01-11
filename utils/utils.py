import timeit


def time(number, activated=True, print_func_call=False):
    """Decorator that can be used to time a function.
    It prints the function call as a string and returns the time spend
    inside the function, according to timeit module.

    Arguments:
        number {int} -- Number of calls to the function.
                        Passed as an argument to `timeit.timeit`

    Keyword Arguments:
        activated {bool} -- Whether to actually activate the decorator
                            or not. It is useful since this decorator
                            totally change the behavior of the decorated
                            function. (default: {False})

    Returns:
        float -- The duration returned by `timeit.timeit`
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not activated:
                return func(*args, **kwargs)

            def closure():
                return func(*args, **kwargs)
            if print_func_call:
                print(str_func_call(func.__name__, *args, **kwargs))
            return timeit.timeit(closure, number=number)
        return wrapper
    return decorator


def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return int(h), int(m), int(s)


def str_func_call(funcname, *args, **kwargs):

    if not args and not kwargs:
        return "{}()".format(funcname)

    if args and not kwargs:
        return "{}({})".format(funcname, args2str(args))

    if args and kwargs:
        return "{}({}, {})".format(funcname, args2str(args), kwargs2str(kwargs))

    if kwargs and not args:
        return "{}({})".format(funcname, kwargs2str(kwargs))


def args2str(args):
    return ', '.join([repr(a) for a in args])


def kwargs2str(kwargs):
    return ', '.join(['='.join([repr(i[0]), repr(i[1])]) for i in kwargs.items()])
