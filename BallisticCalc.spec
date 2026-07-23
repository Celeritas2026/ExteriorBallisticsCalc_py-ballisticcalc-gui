# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import copy_metadata

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('ammo_configs.json', '.')] + copy_metadata('py-ballisticcalc') + copy_metadata('py-ballisticcalc-exts'),
    hiddenimports=[
        'matplotlib.backends.backend_tkagg',
        'tkinter',
        'tkinter.ttk',
        'numpy',
        'py_ballisticcalc',
        'py_ballisticcalc_exts',
        'py_ballisticcalc_exts.bind',
        'py_ballisticcalc_exts.traj_data',
        'py_ballisticcalc_exts.base_engine',
        'py_ballisticcalc_exts.euler_engine',
        'py_ballisticcalc_exts.rk4_engine',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BallisticCalc',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
