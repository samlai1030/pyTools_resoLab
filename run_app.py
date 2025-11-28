#!/usr/bin/env python
"""
Wrapper script to properly configure Qt environment and run SFR_app_v2.py
"""
import os
import sys
import subprocess

# Set Qt environment variables
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/Users/samlai/miniconda3/envs/Local/lib/python3.12/site-packages/PyQt6/Qt6/plugins'
os.environ['QT_QPA_PLATFORM'] = 'cocoa'

# Import and run the app
sys.path.insert(0, '/Users/samlai/Local_2/agent_test')

from SFR_app_v2 import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

