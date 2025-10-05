import os
import shutil

class FileService:
    @staticmethod
    def move_files(files, target_folder):
        """
        Move files to target folder
        """
        for file in files:
            if os.path.exists(file):
                shutil.move(file, os.path.join(target_folder, os.path.basename(file)))
    
    @staticmethod
    def delete_files(files):
        """
        Delete files
        """
        for file in files:
            if os.path.exists(file):
                os.remove(file)
    
    @staticmethod
    def get_folders():
        """
        Get list of available folders
        """
        # Implement folder listing logic here
        pass
