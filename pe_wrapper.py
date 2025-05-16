import os
import sys
from Extract.PE_app import PE_scanner as OriginalPEScanner

class FixedPEScanner(OriginalPEScanner):
    """
    A wrapper around the original PE_scanner class that fixes the path handling issue.
    """
    
    def __init__(self):
        # Initialize the original scanner
        super().__init__()
    
    def PE_mal_classify(self, fpath):
        """
        Wrapper around the original PE_mal_classify that fixes the path handling.
        """
        # Ensure the path is absolute
        abs_path = os.path.abspath(fpath)
        
        # Check if file exists
        if not os.path.exists(abs_path):
            print(f"Error: File not found at {abs_path}")
            return None, None
        
        try:
            # Call the original method but with a modified path
            # We use os.path.basename to get just the filename, then 
            # change to the directory containing the file before scanning
            
            original_dir = os.getcwd()
            file_dir = os.path.dirname(abs_path)
            file_name = os.path.basename(abs_path)
            
            try:
                # Change to the directory containing the file
                os.chdir(file_dir)
                
                # Now call the original method with just the filename
                result = super().PE_mal_classify(file_name)
                return result
            finally:
                # Change back to the original directory
                os.chdir(original_dir)
                
        except Exception as e:
            print(f"Error in PE_mal_classify: {e}")
            return None, None

# For testing
if __name__ == "__main__":
    scanner = FixedPEScanner()
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Scanning: {file_path}")
        result = scanner.PE_mal_classify(file_path)
        print(f"Result: {result}")
