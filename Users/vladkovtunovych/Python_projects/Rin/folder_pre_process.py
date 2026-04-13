import os
import shutil



folder_path = '/Volumes/T7/last/after_0.5T/0V_after_strain'  


def _unique_path(dirpath, filename):
    name, ext = os.path.splitext(filename)
    candidate = filename
    i = 1
    while os.path.exists(os.path.join(dirpath, candidate)):
        candidate = f"{name}_{i}{ext}"
        i += 1
    return os.path.join(dirpath, candidate)


def rename_files_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f'Folder not found: {folder_path}')
        return

    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                
                skip_files = ['.DS_Store', 'Thumbs.db', 'desktop.ini', '.directory']
                if filename.startswith('._') or filename.startswith('.') or filename in skip_files:
                    continue

                
                if ' ' in filename:
                    old_path = os.path.join(dirpath, filename)
                    new_filename = filename.replace(' ', '')
                    new_path = os.path.join(dirpath, new_filename)

                    
                    if os.path.exists(new_path):
                        new_path = _unique_path(dirpath, new_filename)
                        new_filename = os.path.basename(new_path)

                    try:
                        os.rename(old_path, new_path)
                        print(f'Renamed: "{old_path}" → "{new_path}"')
                    except Exception as e:
                        print(f'Failed to rename "{old_path}": {e}')
    except PermissionError:
        print(f'Permission denied: {folder_path}')


def sort_files_by_keyword(source_folder, keyword):
    """Sort files containing a keyword into a subfolder."""
    
    destination_folder = os.path.join(source_folder, keyword)
    os.makedirs(destination_folder, exist_ok=True)

    
    for file_name in os.listdir(source_folder):
        
        if keyword in file_name:
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            
            
            if os.path.isfile(source_file):
                shutil.copy2(source_file, destination_file)
                print(f"Copied: {file_name} -> {destination_folder}")


if __name__ == '__main__':
    
    print("Removing spaces")
    rename_files_in_folder(folder_path)
    
    
    print("\nSorting files")
    keywords = ["MFMPhase_Backward", "Height_Backward", "ErrorSignal_Backward"]
    
    for keyword in keywords:
        print(f"\nSorting files with keyword: {keyword}")
        sort_files_by_keyword(folder_path, keyword)
    
    print("\nDone")
