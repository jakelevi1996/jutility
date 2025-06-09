class FunctionList:
    def __init__(self, *functions):
        self._function_list = []
        self.add_functions(*functions)

    def add_functions(self, *functions):
        for f in functions:
            self._function_list.append(f)

    def call_all(self, return_results=True):
        result_list = [f() for f in self._function_list]
        return_val = (result_list if return_results else None)
        return return_val
