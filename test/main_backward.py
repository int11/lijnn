import os
import glob
import importlib

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

# 실행할 함수명과 폴더 경로 지정
function_to_execute = 'test'  # 실행할 함수 이름
folder_to_search = 'backward'  # 실제 폴더 경로로 변경해주세요

# 함수 실행
execute_test_function_in_folder(folder_to_search, function_to_execute)