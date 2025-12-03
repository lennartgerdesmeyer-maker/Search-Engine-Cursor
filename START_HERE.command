#!/bin/bash
# Simple one-click launcher - Double-click this file!

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Open Terminal and start the app
osascript <<EOF
tell application "Terminal"
    activate
    set newTab to do script "cd '$SCRIPT_DIR' && clear && echo '========================================' && echo '  SEMANTIC SEARCH ENGINE - STARTING' && echo '========================================' && echo '' && source venv/bin/activate && python3 start_app_simple.py"
end tell
EOF

