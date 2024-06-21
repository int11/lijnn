import os
import glob
import importlib
import numpy as np
from lijnn import Variable
def test_function_backward(*args, **kwargs):
    
    def check_backpropagation(gpu, input_data, output, backward, f , *args, **kwargs, ):
        if isinstance(input_data, tuple) == False:
            input_data = (input_data,)

        if isinstance(output, tuple) == False:
            output = (output,)

        for i in input_data:
            if isinstance(i, np.array):
                i = Variable(i)

            if gpu:
                i.to_gpu()
            else:
                i.to_cpu()

        y = f(*input_data, *args, **kwargs)
        

        y.backward(retain_grad=True)
        for i in input_data:
            i.grad.to_cpu()

        output_equal = np.allclose(y.data, output.data)
        if output_equal == False:
            print(f"{f.__name__} output is not correct")
            print(output)

        a = [np.allclose(i.grad.data, backward) for i in input_data]
        if backward_equal == False:
            print(f"{f.__name__} backward is not correct")
            print(backward)

    
    check_backpropagation(False, *args, **kwargs)
    check_backpropagation(True, *args, **kwargs)

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