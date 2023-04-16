import os
import json

# 获取当前目录的上级目录
parent_dir = '/mnt/disks/data/'

# 遍历上级目录的文件结构
dir_tree = {}
for root, dirs, files in os.walk(parent_dir):
    if 'jpg' in files or 'png' in files:
        break  # 如果有jpg或png文件，直接退出该目录的遍历
    level = root.replace(parent_dir, '').count(os.sep)
    current_dir = os.path.basename(root)
    dir_tree[current_dir] = {}
    if level == 1:
        for d in dirs:
            dir_tree[current_dir][d] = {}
    else:
        parent_dir = os.path.basename(os.path.dirname(root))
        dir_tree[parent_dir][current_dir] = {}
        for d in dirs:
            dir_tree[parent_dir][current_dir][d] = {}
        for f in files:
            dir_tree[parent_dir][current_dir][f] = ''

# 将目录树保存到json文件中
with open('dir_tree.json', 'w') as f:
    json.dump(dir_tree, f)