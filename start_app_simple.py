#!/usr/bin/env python3
"""
Simple Launcher - Just starts the server and opens browser
No GUI, just works!
"""
import sys
import time
import webbrowser
import threading
from pathlib import Path
import os

# Get the project directory
PROJECT_DIR = Path(__file__).parent
VENV_PYTHON = PROJECT_DIR / "venv" / "bin" / "python"
APP_FILE = PROJECT_DIR / "app.py"
URL = "http://127.0.0.1:5001"

def main():
    """Main launcher function"""
    print("=" * 60)
    print("SEMANTIC SEARCH ENGINE - STARTING")
    print("=" * 60)
    print()
    
    if not VENV_PYTHON.exists():
        print("ERROR: Virtual environment not found!")
        print(f"Expected: {VENV_PYTHON}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    if not APP_FILE.exists():
        print(f"ERROR: app.py not found at {APP_FILE}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Check if port is already in use
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 5001))
    sock.close()
    
    if result == 0:
        print("WARNING: Port 5001 is already in use!")
        print("Attempting to kill existing process...")
        try:
            import subprocess
            result = subprocess.run(['lsof', '-ti:5001'], capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    subprocess.run(['kill', '-9', pid], check=False)
                print("✓ Killed existing process(es)")
                time.sleep(2)
        except Exception as e:
            print(f"Could not kill existing process: {e}")
            print("Please manually stop the process using port 5001 and try again.")
            input("\nPress Enter to exit...")
            sys.exit(1)
    
    # Change to project directory
    os.chdir(PROJECT_DIR)
    
    # Add venv site-packages to Python path to ensure all packages are available
    venv_lib = PROJECT_DIR / "venv" / "lib"
    if venv_lib.exists():
        # Find the Python version directory (e.g., python3.14)
        python_dirs = [d for d in venv_lib.iterdir() if d.is_dir() and d.name.startswith('python')]
        if python_dirs:
            venv_site_packages = python_dirs[0] / "site-packages"
            if venv_site_packages.exists():
                sys.path.insert(0, str(venv_site_packages))
    
    # Add project to Python path
    sys.path.insert(0, str(PROJECT_DIR))
    
    print("Starting Flask server...")
    print(f"Server will be available at: {URL}")
    print("Waiting for server to be ready, then opening browser...")
    print("Press Ctrl+C to stop the server\n")
    
    # Function to open browser after server is ready
    def wait_and_open_browser():
        """Wait for server to be ready, then open browser"""
        max_wait = 120  # Increased to 2 minutes for large FAISS index loading
        waited = 0
        
        while waited < max_wait:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(('127.0.0.1', 5001))
            sock.close()
            
            if result == 0:
                print(f"\n✓ Server is ready!")
                print(f"Opening browser at {URL}...\n")
                webbrowser.open(URL)
                return
            
            time.sleep(1)
            waited += 1
            if waited % 5 == 0:
                print(f"Waiting... ({waited}s/{max_wait}s)", end="", flush=True)
            else:
                print(".", end="", flush=True)
        
        print(f"\n⚠ WARNING: Server not ready after {max_wait}s, but opening browser anyway...")
        print("The server may still be loading the FAISS index. Please wait and refresh the page.")
        webbrowser.open(URL)
    
    # Start browser opener in background thread
    browser_thread = threading.Thread(target=wait_and_open_browser, daemon=True)
    browser_thread.start()
    
    # Import and run Flask app directly
    try:
        # Import the app
        from app import app
        from config.config import FLASK_HOST, FLASK_PORT
        
        # Run the Flask app (this blocks)
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\nERROR starting server: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
