import os

# Define the folder structure
folder_structure = {
    "my_dnn_acceleration_library": [
        "examples",
        "docs",
        "my_dnn_acceleration_library/core",
        "my_dnn_acceleration_library/extensions/transformers",
        "my_dnn_acceleration_library/extensions/vision_transformers",
        "my_dnn_acceleration_library/utils",
        "my_dnn_acceleration_library/kernels/cuda",
        "my_dnn_acceleration_library/kernels/openblas",
        "my_dnn_acceleration_library/tests",
        "scripts",
        "research/experimental",
        "research/notebooks",
        "research/data",
        "research/reports"
    ]
}

# Create the folder structure
def create_folders(base, structure):
    for root, folders in structure.items():
        for folder in folders:
            path = os.path.join(base, root, folder)
            os.makedirs(path, exist_ok=True)
            print(f"Created folder: {path}")

# Create initial files
def create_initial_files(base):
    files = [
        "README.md",
        "setup.py",
        "requirements.txt",
        "my_dnn_acceleration_library/__init__.py",
        "my_dnn_acceleration_library/core/__init__.py",
        "my_dnn_acceleration_library/extensions/transformers/__init__.py",
        "my_dnn_acceleration_library/extensions/vision_transformers/__init__.py",
        "my_dnn_acceleration_library/utils/__init__.py",
        "my_dnn_acceleration_library/kernels/cuda/__init__.py",
        "my_dnn_acceleration_library/kernels/openblas/__init__.py",
        "my_dnn_acceleration_library/tests/__init__.py",
        "research/README.md"
    ]
    for file in files:
        path = os.path.join(base, file)
        with open(path, 'w') as f:
            f.write(f"# {os.path.basename(file)}\n")
        print(f"Created file: {path}")

# Main execution
if __name__ == "__main__":
    base_path = os.getcwd()
    create_folders(base_path, folder_structure)
    create_initial_files(base_path)
    print("Folder structure created successfully.")
