from setuptools import setup

# Make bundle/app name explicit for build/test scripts
APP_NAME = 'MyApp'

# Set APP to the entry script for the macOS app. Put your main.py next to this file
# or change the path to point to your script.
APP = ['main.py']

DATA_FILES = []

OPTIONS = {
    # argv_emulation is useful for GUI apps started from Finder
    'argv_emulation': True,
    # Add iconfile: 'app.icns' if you add an ICNS file in the same folder
    # 'iconfile': 'app.icns',
    'plist': {
        'CFBundleName': APP_NAME,
        'CFBundleShortVersionString': '0.1.0',
        'CFBundleIdentifier': 'com.yourcompany.myapp',
    },
    # Add any packages listed here if py2app misses them
    'packages': [],
}

setup(
    app=APP,
    name=APP_NAME,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
