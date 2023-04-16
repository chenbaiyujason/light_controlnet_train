import os
import json

#定义跳过的文件格式
SKIP_FORMATS = {'.png', '.jpg'}

#定义要保存的目录路径
DIRECTORY_PATH = '/mnt/disks/data/'

#遍历目录树，获取目录结构
def get_directory_tree(path):
    # 判断路径是否为文件
    if os.path.isfile(path):
        # 如果是文件，则返回空
        return None
    # 初始化目录结构
    directory_tree = {
        'name': os.path.basename(path),
        'type': 'directory',
        'children': []
    }
    # 遍历子目录和文件
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        # 判断是否需要跳过该文件
        if os.path.isfile(filepath) and any(filepath.endswith(fmt) for fmt in SKIP_FORMATS):
            continue
        # 获取子目录结构
        child = get_directory_tree(filepath)
        # 如果子目录结构不为空，则添加到目录树中
        if child is not None:
            directory_tree['children'].append(child)
    return directory_tree

#将目录结构保存成json文件
def save_directory_tree(directory_tree, file_path):
    with open(file_path, 'w') as f:
        json.dump(directory_tree, f, indent=4)


# 获取目录结构
directory_tree = get_directory_tree(DIRECTORY_PATH)
# 保存目录结构到json文件
save_directory_tree(directory_tree, 'directory_tree.json')