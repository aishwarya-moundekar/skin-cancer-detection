import os

def get_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024**3)  # Convert bytes to GB

project_path = r'C:\Users\aishw\PycharmProjects\pythonProject'
size_gb = get_size(project_path)
print(f"Total project size: {size_gb:.2f} GB")
