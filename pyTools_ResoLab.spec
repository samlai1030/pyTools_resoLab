# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect data files for jaraco.text and cv2
jaraco_datas = collect_data_files('jaraco.text')
cv2_datas = collect_data_files('cv2')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('mainUI.ui', '.')] + jaraco_datas + cv2_datas,
    hiddenimports=[
        'PyQt5.sip',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'scipy.signal',
        'scipy.ndimage',
        'scipy.fftpack',
        'scipy.interpolate',
        'numpy',
        'cv2',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'jaraco.text',
        'jaraco.functools',
        'jaraco.context',
    ] + collect_submodules('cv2'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='pyTools_ResoLab_v2.4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

