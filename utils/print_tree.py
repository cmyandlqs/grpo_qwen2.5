import os

# 需要忽略的目录和文件列表
IGNORE_PATTERNS = {
    # 隐藏目录/文件（以点开头）
    ".",
    # Git 相关
    ".git", ".gitignore", ".gitattributes", ".gitmodules",
    # IDE 相关
    ".idea", ".vscode", ".vs", "*.swp", "*.swo",
    # Python 相关
    "__pycache__", "*.pyc", "*.pyo", "*.pyd", ".pytest_cache",
    ".mypy_cache", ".tox", ".coverage", ".coverage.*",
    # 虚拟环境
    "venv", "env", ".venv", ".env",
    # 构建产物
    "build", "dist", "*.egg-info",
    # Node.js
    "node_modules",
    # 系统文件
    ".DS_Store", "Thumbs.db",
    # 项目特定
    "*.log", "*.tmp", ".cache",
}

def should_ignore(name):
    """判断是否应该忽略该文件或目录"""
    name_lower = name.lower()

    # 隐藏文件/目录（以点开头）
    if name.startswith("."):
        return True

    # 检查扩展名匹配
    for pattern in IGNORE_PATTERNS:
        if pattern.startswith("*"):
            # 扩展名匹配
            if name_lower.endswith(pattern[1:].lower()):
                return True
        elif name == pattern:
            return True

    return False


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
            # 过滤并排序
            entries = sorted(
                (e for e in os.listdir(root_path) if not should_ignore(e)),
                key=lambda x: (not os.path.isdir(os.path.join(root_path, x)), x)
            )
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
