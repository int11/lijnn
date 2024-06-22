import os
import glob
import importlib
import numpy as np
from lijnn import Variable
import lijnn
import copy 

def f_unit_test(*args, **kwargs):
    def check_backpropagation(gpu, input_data, output, grad, f , *args, **kwargs):
        if isinstance(f, lijnn.Layer):
            name = f.__class__
            f.to_gpu() if gpu else f.to_cpu()
        else:
            name = f.__name__
            
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
            print(f"gpu = {gpu}, {name} output is not correct")
            

        input_grad = tuple(input_data[i].grad for i in range(len(input_data)))
        a = [lijnn.allclose(input_grad[i], grad[i]) for i in range(len(input_data))]
        if all(a) == False:
            [print(i) for i in input_grad]
            [print(i) for i in grad]
            print(f"gpu = {gpu}, {name} backward is not correct")

        return output_equal and all(a)
        
    a = check_backpropagation(False, *args, **kwargs)
    b = check_backpropagation(True, *args, **kwargs)
    if a and b:
        print("Test passed")

def f_unit_test_withTorch(input_data_shape, torch_f, lijnn_f, *args, **kwargs):
    import torch
    if isinstance(input_data_shape, tuple) == False:
        input_data_shape = (input_data_shape,)

    input_data = []
    np.random.seed(0)
    for i in input_data_shape:
        if isinstance(i, tuple) == False:
            i = (i, )
        input_data.append(np.random.randint(0, 10, i).astype(np.float32))

    input_data = tuple(input_data)
    input_data_tensor = tuple(torch.tensor(i, requires_grad=True) for i in input_data)

    output = torch_f(*input_data_tensor, *args, **kwargs)

    output.backward(torch.ones_like(output))
    output = output.detach().numpy()
    grad = tuple(i.grad.numpy() for i in input_data_tensor)
    
    f_unit_test(input_data, output, grad, lijnn_f, *args, **kwargs)

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