import inspect

def print_func_name():
    print(inspect.stack()[1].code_context[0].strip('\n'))
