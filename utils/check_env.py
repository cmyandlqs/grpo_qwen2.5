#!/usr/bin/env python3
"""
GRPO Training Environment Checker
检查 ms-swift GRPO 训练所需的所有依赖和环境配置
增强版：支持检查 uv 和虚拟环境
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Tuple, List, Dict, Optional

# ANSI 颜色代码
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def check_python_version() -> bool:
    """检查 Python 版本（推荐 3.10+）"""
    print_info("检查 Python 版本...")
    version = sys.version_info
    major, minor = version.major, version.minor

    print(f"  当前版本: Python {major}.{minor}.{version.micro}")
    print(f"  Python 路径: {sys.executable}")

    if major == 3 and minor >= 10:
        print_success("Python 版本符合要求 (3.10+)")
        return True
    else:
        print_warning("推荐使用 Python 3.10+，当前版本可能有兼容性问题")
        return False

def check_virtual_env() -> Dict[str, any]:
    """检查虚拟环境状态"""
    venv_info = {
        'in_venv': False,
        'venv_type': None,
        'venv_path': None,
        'base_prefix': None,
        'has_uv': False,
        'uv_version': None
    }

    print_info("检查虚拟环境...")

    # 检查是否在虚拟环境中
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    venv_info['in_venv'] = in_venv

    if in_venv:
        venv_path = sys.prefix
        base_prefix = sys.base_prefix if hasattr(sys, 'base_prefix') else sys.real_prefix
        venv_info['venv_path'] = venv_path
        venv_info['base_prefix'] = base_prefix
        print_success(f"在虚拟环境中: {venv_path}")
        print_info(f"基础 Python: {base_prefix}")

        # 判断虚拟环境类型
        # uv 虚拟环境特征
        uv_marker = Path(venv_path) / 'lib' / 'uv' / '__init__.py'
        # conda 环境特征
        conda_meta = Path(venv_path) / 'conda-meta'
        # 标准 venv 特征
        pyvenv_cfg = Path(venv_path) / 'pyvenv.cfg'

        if uv_marker.exists():
            venv_info['venv_type'] = 'uv'
            print_success("虚拟环境类型: uv")
        elif conda_meta.exists():
            venv_info['venv_type'] = 'conda'
            print_success("虚拟环境类型: conda")
        elif pyvenv_cfg.exists():
            venv_info['venv_type'] = 'venv'
            print_success("虚拟环境类型: standard venv")
        else:
            venv_info['venv_type'] = 'unknown'
            print_info("虚拟环境类型: 未知")
    else:
        print_warning("未检测到虚拟环境")
        print_info("当前使用全局 Python 环境")
        print_info("建议使用虚拟环境隔离项目依赖")

    return venv_info

def check_uv() -> Dict[str, any]:
    """检查 uv 包管理器"""
    print_info("检查 uv 包管理器...")

    uv_info = {
        'installed': False,
        'version': None,
        'path': None,
        'venv_in_project': False
    }

    # 检查 uv 命令
    try:
        result = subprocess.run(
            ['uv', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            uv_info['installed'] = True
            uv_info['version'] = result.stdout.strip()
            print_success(f"uv 已安装: {uv_info['version']}")

            # 获取 uv 路径
            path_result = subprocess.run(
                ['which', 'uv'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if path_result.returncode == 0:
                uv_info['path'] = path_result.stdout.strip()
                print_info(f"uv 路径: {uv_info['path']}")
        else:
            print_warning("uv 未安装")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print_warning("uv 未安装")

    # 检查是否有 .venv 目录（uv 默认创建）
    if Path('.venv').exists():
        uv_info['venv_in_project'] = True
        print_info("检测到 .venv 目录（可能是 uv 创建）")

    return uv_info

def check_uv_project() -> Dict[str, any]:
    """检查 uv 项目配置"""
    print_info("检查 uv 项目配置...")

    project_info = {
        'has_pyproject': False,
        'has_uv_lock': False,
        'has_uv_python': False,
        'dependency_count': 0
    }

    # 检查 pyproject.toml
    if Path('pyproject.toml').exists():
        project_info['has_pyproject'] = True
        print_success("存在 pyproject.toml")

        try:
            with open('pyproject.toml', 'r', encoding='utf-8') as f:
                content = f.read()
                # 检查是否有 uv 配置
                if '[tool.uv]' in content or 'uv =' in content:
                    project_info['has_uv_config'] = True
                    print_success("检测到 uv 配置")
        except Exception as e:
            print_warning(f"读取 pyproject.toml 失败: {e}")

    # 检查 uv.lock
    if Path('uv.lock').exists():
        project_info['has_uv_lock'] = True
        print_success("存在 uv.lock 文件")

    # 检查 .python-version
    if Path('.python-version').exists():
        project_info['has_uv_python'] = True
        try:
            with open('.python-version', 'r') as f:
                version = f.read().strip()
                print_info(f"Python 版本锁定: {version}")
        except Exception:
            pass

    return project_info

def get_package_version(package_name: str, import_name: Optional[str] = None) -> Optional[str]:
    """获取包版本，不打印任何输出"""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)

        # 尝试获取版本
        version = None
        try:
            if hasattr(module, '__version__'):
                version = module.__version__
            elif import_name == 'torch':
                import torch
                version = torch.__version__
            elif import_name == 'transformers':
                import transformers
                version = transformers.__version__
            elif import_name == 'swift':
                import swift
                version = swift.__version__
        except:
            pass

        return version
    except ImportError:
        return None

def check_package(package_name: str, import_name: Optional[str] = None, env_label: str = "") -> bool:
    """检查 Python 包是否已安装"""
    if import_name is None:
        import_name = package_name

    try:
        version = get_package_version(package_name, import_name)

        if version:
            label = f" [{env_label}]" if env_label else ""
            print_success(f"{package_name}{label}: {version}")
        else:
            label = f" [{env_label}]" if env_label else ""
            print_success(f"{package_name}{label}: 已安装")

        return True

    except ImportError:
        label = f" [{env_label}]" if env_label else ""
        print_error(f"{package_name}{label}: 未安装")
        return False

def check_command(command: str) -> Tuple[bool, Optional[str]]:
    """检查系统命令是否可用"""
    try:
        result = subprocess.run(
            ['which', command],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            path = result.stdout.strip()
            return True, path
        return False, None
    except Exception:
        return False, None

def check_cuda() -> Dict[str, any]:
    """检查 CUDA 和 GPU 状态"""
    cuda_info = {
        'available': False,
        'version': None,
        'device_count': 0,
        'device_name': None,
        'vram_total': 0,
        'vram_free': 0
    }

    try:
        import torch
        cuda_info['available'] = torch.cuda.is_available()

        if cuda_info['available']:
            cuda_info['version'] = torch.version.cuda
            cuda_info['device_count'] = torch.cuda.device_count()

            if cuda_info['device_count'] > 0:
                cuda_info['device_name'] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                cuda_info['vram_total'] = props.total_memory / (1024**3)  # GB
                cuda_info['vram_free'] = (
                    torch.cuda.mem_get_info(0)[0] / (1024**3)
                )  # GB

    except Exception as e:
        print_error(f"CUDA 检查失败: {e}")

    return cuda_info

def check_model(model_name: str, cache_dir: str = './models') -> bool:
    """检查模型是否已下载"""
    print_info(f"检查模型: {model_name}")

    try:
        from modelscope import snapshot_download

        # 检查缓存目录
        cache_path = Path(cache_dir)
        model_path = cache_path / model_name.replace('/', '--')

        if model_path.exists():
            size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            size_gb = size / (1024**3)
            print_success(f"模型已下载: {model_path} ({size_gb:.2f} GB)")
            return True
        else:
            print_warning(f"模型未找到，路径: {model_path}")
            return False

    except ImportError:
        print_error("ModelScope 未安装，无法检查模型")
        return False
    except Exception as e:
        print_error(f"模型检查失败: {e}")
        return False

def check_dataset(dataset_name: str, cache_dir: str = './datasets') -> bool:
    """检查数据集是否已下载"""
    print_info(f"检查数据集: {dataset_name}")

    try:
        from modelscope import snapshot_download

        cache_path = Path(cache_dir)
        dataset_path = cache_path / dataset_name.replace('/', '--')

        if dataset_path.exists():
            file_count = len(list(dataset_path.rglob('*')))
            print_success(f"数据集已下载: {dataset_path} ({file_count} 个文件)")
            return True
        else:
            print_warning(f"数据集未找到，路径: {dataset_path}")
            return False

    except ImportError:
        print_error("ModelScope 未安装，无法检查数据集")
        return False
    except Exception as e:
        print_error(f"数据集检查失败: {e}")
        return False

def check_vllm() -> bool:
    """检查 vLLM 是否可用"""
    print_info("检查 vLLM...")

    try:
        import vllm
        print_success(f"vLLM: {vllm.__version__}")

        # 检查 vLLM 命令
        success, path = check_command('python -m vllm.entrypoints.openai.api_server'.split())
        if success:
            print_success("vLLM API 服务器可用")

        return True
    except ImportError:
        print_error("vLLM 未安装")
        return False
    except Exception as e:
        print_error(f"vLLM 检查失败: {e}")
        return False

def check_disk_space(path: str = '.', required_gb: int = 50) -> bool:
    """检查磁盘空间"""
    print_info(f"检查磁盘空间: {path}")

    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)

        if free_gb >= required_gb:
            print_success(f"可用空间: {free_gb:.2f} GB (要求: {required_gb} GB)")
            return True
        else:
            print_warning(f"可用空间不足: {free_gb:.2f} GB (推荐: {required_gb} GB+)")
            return False
    except Exception as e:
        print_error(f"磁盘空间检查失败: {e}")
        return False

def check_global_packages(packages: List[Tuple[str, str]]) -> Dict[str, bool]:
    """检查全局环境的包（使用全局 Python）"""
    print_info("检查全局 Python 环境的包...")

    # 获取全局 Python 路径
    if hasattr(sys, 'base_prefix'):
        global_python = sys.base_prefix
    elif hasattr(sys, 'real_prefix'):
        global_python = sys.real_prefix
    else:
        print_warning("无法确定全局 Python 路径")
        return {}

    # 根据操作系统确定 Python 可执行文件路径
    if os.name == 'nt':  # Windows
        python_exe = os.path.join(global_python, 'python.exe')
    else:  # Linux/Mac
        python_exe = os.path.join(global_python, 'bin', 'python')

    if not os.path.exists(python_exe):
        print_warning(f"找不到全局 Python 可执行文件: {python_exe}")
        return {}

    print_info(f"全局 Python: {python_exe}")

    results = {}

    for display_name, import_name in packages:
        try:
            # 使用全局 Python 运行导入检查
            check_code = (
                f'import importlib; '
                f'module = importlib.import_module("{import_name}"); '
                f'version = getattr(module, "__version__", None) or getattr(module, "version", None); '
                f'print(version or "已安装")'
            )

            result = subprocess.run(
                [python_exe, '-c', check_code],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                version = result.stdout.strip()
                print_success(f"{display_name} [全局]: {version}")
                results[display_name] = True
            else:
                print_error(f"{display_name} [全局]: 未安装")
                results[display_name] = False
        except subprocess.TimeoutExpired:
            print_warning(f"{display_name} [全局]: 检查超时")
            results[display_name] = False
        except Exception as e:
            print_warning(f"{display_name} [全局]: 检查失败 ({e})")
            results[display_name] = False

    return results

def print_uv_setup_guide():
    """打印 uv 安装和使用指南"""
    print_header("uv 安装和使用指南")

    print_info("安装 uv:")
    print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
    print("  # 或在 Windows 上:")
    print("  powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
    print()

    print_info("使用 uv 创建虚拟环境:")
    print("  uv venv                      # 创建 .venv")
    print("  uv venv --python 3.10        # 指定 Python 版本")
    print()

    print_info("使用 uv 安装依赖:")
    print("  uv pip install ms-swift")
    print("  uv pip install transformers accelerate peft")
    print()

    print_info("使用 uv 管理项目（推荐）:")
    print("  uv init                      # 初始化项目")
    print("  uv add ms-swift              # 添加依赖")
    print("  uv sync                      # 同步依赖")
    print()

def main():
    # 解析命令行参数
    check_global = '--check-global' in sys.argv
    only_venv = '--only-venv' in sys.argv

    print_header("GRPO 训练环境检查")

    # 检查 Python 版本
    print_header("1. Python 环境")
    python_ok = check_python_version()

    # 检查虚拟环境
    print_header("2. 虚拟环境")
    venv_info = check_virtual_env()
    venv_ok = venv_info['in_venv']

    # 检查 uv
    print_header("3. uv 包管理器")
    uv_info = check_uv()
    uv_project_ok = False

    if uv_info['installed']:
        print_header("4. uv 项目配置")
        project_info = check_uv_project()
        uv_project_ok = project_info['has_pyproject']

    # 定义核心包列表
    core_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('ms-swift', 'swift'),
        ('modelscope', 'modelscope'),
        ('numpy', 'numpy'),
        ('accelerate', 'accelerate'),
        ('peft', 'peft'),
        ('bitsandbytes', 'bitsandbytes'),
    ]

    # 定义工具包列表
    tool_packages = [
        ('nvitop', 'nvitop'),
        ('swanlab', 'swanlab'),
        ('datasets', 'datasets'),
    ]

    # 检查当前环境的包
    header_num = "5" if not uv_info['installed'] else "6"
    env_label = "虚拟环境" if venv_ok else "全局环境"
    print_header(f"{header_num}. 核心 Python 包 [{env_label}]")

    packages_ok = True
    for display_name, import_name in core_packages:
        label = "venv" if venv_ok else "global"
        if not check_package(display_name, import_name, env_label=label):
            packages_ok = False

    # 如果在虚拟环境且请求检查全局环境
    global_packages_ok = None
    if venv_ok and check_global and not only_venv:
        header_num = "5.5" if not uv_info['installed'] else "6.5"
        print_header(f"{header_num}. 核心 Python 包 [全局环境对比]")
        global_results = check_global_packages(core_packages)
        global_packages_ok = all(global_results.values())

        # 显示对比信息
        if global_packages_ok != packages_ok:
            print_warning("虚拟环境和全局环境的包状态不同")
        else:
            print_success("虚拟环境和全局环境的包状态一致")

    # 检查额外工具
    header_num = "7" if not uv_info['installed'] else "8"
    print_header(f"{header_num}. 监控和工具 [{env_label}]")

    tools_ok = True
    for display_name, import_name in tool_packages:
        label = "venv" if venv_ok else "global"
        if not check_package(display_name, import_name, env_label=label):
            print_warning(f"{display_name}: 未安装（可选但推荐）")
            tools_ok = False

    # 检查 vLLM
    header_num = "8" if not uv_info['installed'] else "9"
    print_header(f"{header_num}. vLLM 加速 [{env_label}]")
    vllm_ok = check_vllm()

    # 检查 CUDA/GPU
    header_num = "9" if not uv_info['installed'] else "10"
    print_header(f"{header_num}. CUDA 和 GPU")
    cuda_info = check_cuda()

    if cuda_info['available']:
        print_success(f"CUDA 可用: {cuda_info['version']}")
        print_success(f"GPU 数量: {cuda_info['device_count']}")
        if cuda_info['device_count'] > 0:
            print_success(f"GPU 名称: {cuda_info['device_name']}")
            print_success(f"显存: {cuda_info['vram_free']:.2f} GB / {cuda_info['vram_total']:.2f} GB")

        # Qwen2.5-0.5B 显存需求检查
        if cuda_info['vram_free'] >= 2:
            print_success("显存足够训练 Qwen2.5-0.5B")
        else:
            print_warning("显存可能不足，建议使用更大的 GPU 或减小 batch size")
    else:
        print_error("CUDA 不可用，请检查 NVIDIA 驱动和 CUDA 安装")

    # 检查磁盘空间
    header_num = "10" if not uv_info['installed'] else "11"
    print_header(f"{header_num}. 磁盘空间")
    disk_ok = check_disk_space('.', required_gb=50)

    # 检查模型
    header_num = "11" if not uv_info['installed'] else "12"
    print_header(f"{header_num}. 模型检查")
    models_to_check = [
        'Qwen/Qwen2.5-0.5B',
    ]

    models_ok = True
    for model in models_to_check:
        if not check_model(model):
            models_ok = False

    # 检查数据集
    header_num = "12" if not uv_info['installed'] else "13"
    print_header(f"{header_num}. 数据集检查")
    datasets_to_check = [
        'AI-ModelScope/gsm8k',
    ]

    datasets_ok = True
    for dataset in datasets_to_check:
        if not check_dataset(dataset):
            datasets_ok = False

    # 总结
    print_header("检查总结")

    results = {
        "Python 版本": python_ok,
        "虚拟环境": venv_ok,
    }

    if uv_info['installed']:
        results["uv 包管理器"] = True
        results["uv 项目配置"] = uv_project_ok

    results.update({
        f"核心包 [{env_label}]": packages_ok,
        "监控工具": tools_ok,
        "vLLM": vllm_ok,
        "CUDA/GPU": cuda_info['available'],
        "磁盘空间": disk_ok,
        "模型文件": models_ok,
        "数据集文件": datasets_ok,
    })

    if global_packages_ok is not None:
        results["核心包 [全局]"] = global_packages_ok

    all_ok = True
    for name, ok in results.items():
        status = f"{Colors.GREEN}✓ 通过{Colors.END}" if ok else f"{Colors.RED}✗ 失败{Colors.END}"
        print(f"  {name:25s} {status}")
        if not ok:
            all_ok = False

    print()

    if all_ok:
        print_success("所有检查通过！环境已准备好进行 GRPO 训练")
        print_info("你可以在服务器上运行以下命令开始训练：")
        print()
        print("python -m torch.distributed.run --nproc_per_node 1 \\")
        print("  $(python -c 'import swift; import os; print(os.path.dirname(swift.__file__) + \"/cli/rlhf.py\")') \\")
        print("  --rlhf_type grpo \\")
        print("  --model Qwen/Qwen2.5-0.5B \\")
        print("  --dataset AI-ModelScope/gsm8k \\")
        print("  --train_type lora \\")
        print("  --lora_rank 64 \\")
        print("  --torch_dtype bfloat16 \\")
        print("  --use_vllm true \\")
        print("  --num_generations 8 \\")
        print("  --reward_funcs accuracy \\")
        print("  --output_dir ./output/grpo_qwen2.5_0.5b")
        return 0
    else:
        print_error("环境检查未完全通过，请先解决上述问题")

        # 提供安装建议
        print()
        print_header("安装建议")

        if not venv_ok:
            print_info("创建虚拟环境:")
            if uv_info['installed']:
                print("  uv venv --python 3.10")
                print("  source .venv/bin/activate  # Linux/Mac")
                print("  # 或 .venv\\Scripts\\activate  # Windows")
            else:
                print("  python -m venv .venv")
                print("  source .venv/bin/activate  # Linux/Mac")
                print("  # 或 .venv\\Scripts\\activate  # Windows")
            print()

        if not uv_info['installed']:
            print_info("安装 uv（推荐）:")
            print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
            print()

        if not packages_ok:
            env_name = "虚拟环境" if venv_ok else "当前环境"
            print_info(f"在 {env_name} 中安装核心依赖:")
            if uv_info['installed']:
                print("  uv pip install ms-swift transformers accelerate peft bitsandbytes")
            else:
                print("  pip install ms-swift transformers accelerate peft bitsandbytes")
            print()

        if not tools_ok:
            print_info("安装监控工具:")
            if uv_info['installed']:
                print("  uv pip install nvitop swanlab datasets")
            else:
                print("  pip install nvitop swanlab datasets")
            print()

        if not vllm_ok:
            print_info("安装 vLLM:")
            if uv_info['installed']:
                print("  uv pip install vllm")
            else:
                print("  pip install vllm")
            print()

        if not models_ok:
            print_info("下载模型:")
            print("  python -c \"from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B', cache_dir='./models')\"")
            print()

        if not datasets_ok:
            print_info("下载数据集:")
            print("  python -c \"from modelscope import snapshot_download; snapshot_download('AI-ModelScope/gsm8k', cache_dir='./datasets')\"")
            print()

        # 如果 uv 未安装，提供详细指南
        if not uv_info['installed']:
            print()
            print_info("需要详细的 uv 使用指南？运行:")
            print("  python check_env.py --uv-guide")

        # 如果在虚拟环境中且全局环境包缺失，提示
        if venv_ok and global_packages_ok is not None and not global_packages_ok:
            print()
            print_info("注意：全局环境的包未完全安装")
            print_info("如需在全局环境使用这些包，请先退出虚拟环境后再安装")

        return 1

if __name__ == '__main__':
    # 检查是否请求 uv 指南
    if '--uv-guide' in sys.argv:
        print_uv_setup_guide()
        sys.exit(0)

    if '-h' in sys.argv or '--help' in sys.argv:
        print_header("check_env.py - GRPO 训练环境检查工具")
        print()
        print("用法: python check_env.py [选项]")
        print()
        print("选项:")
        print("  -h, --help          显示此帮助信息")
        print("  --uv-guide          显示 uv 安装和使用指南")
        print("  --check-global      在虚拟环境中的同时检查全局环境的包")
        print("  --only-venv         仅检查虚拟环境，不检查全局环境")
        print()
        print("示例:")
        print("  python check_env.py              # 基本检查")
        print("  python check_env.py --uv-guide   # 显示 uv 指南")
        print("  python check_env.py --check-global  # 检查虚拟环境和全局环境")
        print()
        sys.exit(0)

    sys.exit(main())
