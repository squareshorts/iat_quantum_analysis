import os
import zipfile
import glob

def build_zenodo_archive():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_zip = os.path.join(base_dir, "zenodo_submission.zip")
    
    # Directories and files to include
    include_dirs = ["data", "outputs", "figures", "tables", "src", "scripts", "env", "tests", "paper"]
    include_files = [
        "README.md", "LICENSE", "DATA_AVAILABILITY.md", 
        "REPRODUCIBILITY.md", "SUBMISSION_AUDIT.md", "SCIENTIFIC_AUDIT.md"
    ]
    
    # Exclusions
    exclude_patterns = ["__pycache__", ".venv", ".git", ".pytest_cache"]
    exclude_extensions = [".pyc", ".ipynb_checkpoints"]
    
    def should_exclude(path):
        for pattern in exclude_patterns:
            if pattern in path:
                return True
        for ext in exclude_extensions:
            if path.endswith(ext):
                return True
        return False

    print(f"Building Zenodo archive: {output_zip} ...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add root files
        for f in include_files:
            file_path = os.path.join(base_dir, f)
            if os.path.exists(file_path):
                zf.write(file_path, f)
                print(f"Added file: {f}")
            else:
                print(f"Warning: Expected file {f} not found.")

        # Add directories
        for d in include_dirs:
            dir_path = os.path.join(base_dir, d)
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    if should_exclude(root):
                        continue
                    for file in files:
                        full_path = os.path.join(root, file)
                        if should_exclude(full_path):
                            continue
                        arcname = os.path.relpath(full_path, base_dir)
                        zf.write(full_path, arcname)
                print(f"Added directory: {d}/")
            else:
                print(f"Warning: Expected directory {d}/ not found.")

    print("✅ Archive successfully created.")
    print(f"Location: {output_zip}")
    print(f"Size: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    build_zenodo_archive()
