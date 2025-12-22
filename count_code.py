import os

# ================= 用户配置区域 (USER CONFIGURATION) =================

# 1. 在这里手动填写你需要统计的目录列表
#    可以是相对路径（相对于当前脚本所在位置）或绝对路径
DIRECTORIES_TO_COUNT = [
    "./src",        # 示例：统计 src 目录
    # "./tests",      # 示例：统计 tests 目录
]

# 2. 指定需要统计的文件后缀名
#    如果不限制文件类型，可以将其设置为 None
VALID_EXTENSIONS = [
    '.py', '.js', '.ts', '.html', '.css', 
    '.java', '.c', '.cpp', '.h', '.go', '.rs',
    '.cu', '.cuh'
]

# ===================================================================

def count_lines_in_file(filepath):
    """统计单个文件的行数"""
    count = 0
    try:
        # 使用 errors='ignore' 避免读取二进制文件或特殊编码文件时报错
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 如果你想跳过空行，取消下面两行的注释
                # if line.strip() == "":
                #     continue
                count += 1
    except Exception as e:
        print(f"[跳过] 无法读取文件 {filepath}: {e}")
        return 0
    return count

def main():
    total_lines = 0
    file_count = 0
    
    print(f"正在统计以下目录: {DIRECTORIES_TO_COUNT}")
    print("-" * 50)

    for root_dir in DIRECTORIES_TO_COUNT:
        if not os.path.exists(root_dir):
            print(f"[警告] 目录不存在: {root_dir}")
            continue

        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                # 检查文件后缀
                if VALID_EXTENSIONS:
                    _, ext = os.path.splitext(filename)
                    if ext not in VALID_EXTENSIONS:
                        continue
                
                filepath = os.path.join(dirpath, filename)
                lines = count_lines_in_file(filepath)
                
                if lines > 0:
                    print(f"{lines}\t | {filepath}")
                    total_lines += lines
                    file_count += 1

    print("-" * 50)
    print(f"统计完成！")
    print(f"包含文件数: {file_count}")
    print(f"总代码行数: {total_lines}")
    print("-" * 50)

if __name__ == "__main__":
    main()