import os

def print_tree(root_path, prefix="", is_last=True):
    """递归打印目录树"""
    # 获取路径的显示名
    name = os.path.basename(root_path)
    if not name:  # 根路径本身
        name = root_path

    # 打印当前项
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{name}/" if os.path.isdir(root_path) else f"{prefix}{connector}{name}")

    # 如果是目录，递归处理子项
    if os.path.isdir(root_path):
        try:
            entries = sorted(os.listdir(root_path), key=lambda x: (not os.path.isdir(os.path.join(root_path, x)), x))
        except PermissionError:
            print(f"{prefix}    [权限拒绝]")
            return

        for i, entry in enumerate(entries):
            new_path = os.path.join(root_path, entry)
            is_last_entry = (i == len(entries) - 1)

            # 更新前缀
            if is_last:
                new_prefix = prefix + "    "
            else:
                new_prefix = prefix + "│   "

            # 递归打印
            if os.path.isdir(new_path):
                print_tree(new_path, new_prefix, is_last_entry)
            else:
                connector = "└── " if is_last_entry else "├── "
                print(f"{new_prefix}{connector}{entry}")

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f". ({os.path.abspath(root)})")
    print_tree(os.path.abspath(root), "", True)
