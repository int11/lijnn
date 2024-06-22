import os
import glob
import importlib
import numpy as np
from lijnn import Variable
import lijnn
import copy 

def test_function_backward(*args, **kwargs):
    def check_backpropagation(gpu, input_data, output, grad, f , *args, **kwargs):
        input_data = copy.deepcopy(input_data)
        if isinstance(input_data, tuple) == False:
            input_data = (input_data,)

        if isinstance(grad, tuple) == False:
            grad = (grad,)

        input_data = tuple(lijnn.as_variable(i) for i in input_data)
        output = lijnn.as_variable(output)
        grad = tuple(lijnn.as_variable(i) for i in grad)
 
        for i in input_data:
            if gpu:
                i.to_gpu()
            else:
                i.to_cpu()

        y = f(*input_data, *args, **kwargs)
        

        y.backward(retain_grad=True)

        output_equal = lijnn.allclose(y, output)
        if output_equal == False:
            print(y)
            print(output)
            print(f"{f.__name__} output is not correct")
            

        input_grad = [input_data[i].grad for i in range(len(input_data))]
        a = [lijnn.allclose(input_grad[i], grad[i]) for i in range(len(input_data))]
        if all(a) == False:
            print(input_grad)
            print(grad)
            print(f"{f.__name__} backward is not correct")

        test_pass = output_equal and all(a)
        if test_pass == False:
            print(f"gpu = {gpu}, {f.__name__} test failed")
        return output_equal and all(a)
        
    a = check_backpropagation(False, *args, **kwargs)
    b = check_backpropagation(True, *args, **kwargs)
    if a and b:
        print("Test passed")

def execute_test_function_in_folder(folder_path, function_name):
    # 폴더 내의 모든 .py 파일 찾기
    py_files = glob.glob(os.path.join(folder_path, '*.py'))

    for file_path in py_files:
        # 파일 이름을 모듈 이름으로 변환
        module_name = os.path.basename(file_path)[:-3]  # .py 확장자 제거
        full_module_name = f"{os.path.basename(folder_path)}.{module_name}"

        # 모듈 동적으로 로드
        module = importlib.import_module(full_module_name)

        # 특정 함수가 모듈에 있는지 확인하고 실행
        if hasattr(module, function_name) and callable(getattr(module, function_name)):
            test_function = getattr(module, function_name)
            print(test_function(False))
            print(test_function(True))

if __name__ == '__main__':
    # 실행할 함수명과 폴더 경로 지정
    function_to_execute = 'test'  # 실행할 함수 이름
    folder_to_search = 'backward'  # 실제 폴더 경로로 변경해주세요

    # 함수 실행
    execute_test_function_in_folder(folder_to_search, function_to_execute)