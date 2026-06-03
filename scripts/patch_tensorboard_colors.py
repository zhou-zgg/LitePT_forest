#!/usr/bin/env python3
"""
Patch TensorBoard dark theme colors for better visibility.
Replaces hard-to-see grey colors with brighter alternatives.
"""

import zipfile
import os
import tempfile
import shutil

TENSORBOARD_ZIP = "/home/caozhou/anaconda3/envs/litept/lib/python3.10/site-packages/tensorboard/webfiles.zip"

# Color replacements for better visibility on dark backgrounds
# Original color -> New color
COLOR_REPLACEMENTS = {
    # Text colors
    b'#616161': b'#b0b0b0',  # axis text color -> brighter
    b'#9e9e9e': b'#c0c0c0',  # axis borders -> brighter
    b'#757575': b'#b0b0b0',  # another grey -> brighter
    # Line/stroke colors
    b'#aaa': b'#dddddd',    # line background -> much brighter
    b'#bbb': b'#cccccc',    # stroke color -> brighter
    b'#848484': b'#aaaaaa', # stroke -> brighter
    b'#b2b2b2': b'#d5d5d5', # borders -> brighter
    b'#bbbbbb': b'#e0e0e0', # borders -> brighter
    b'#d9d9d9': b'#ffffff', # minor lines -> white
}

def patch_colors(content: bytes) -> bytes:
    """Replace grey colors with brighter alternatives."""
    for old, new in COLOR_REPLACEMENTS.items():
        content = content.replace(old, new)
    return content

def backup_original(path: str) -> str:
    """Create a backup of the original file."""
    backup_path = path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
        print(f"Backup created: {backup_path}")
    return backup_path

def patch_tensorboard():
    """Patch the tensorboard webfiles.zip with new colors."""
    print(f"Patching TensorBoard colors in: {TENSORBOARD_ZIP}")

    # Create backup
    backup_original(TENSORBOARD_ZIP)

    # Create temp file
    tmp_path = tempfile.mktemp(suffix='.zip')

    # Read the original zip and write patched version
    with zipfile.ZipFile(TENSORBOARD_ZIP, 'r') as zin:
        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == 'index.js':
                    print(f"Patching: {item.filename}")
                    data = patch_colors(data)
                zout.writestr(item, data)

    # Replace original with patched
    shutil.move(tmp_path, TENSORBOARD_ZIP)
    print(f"Patched TensorBoard webfiles.zip successfully!")

def restore_backup():
    """Restore the original webfiles.zip from backup."""
    backup_path = TENSORBOARD_ZIP + ".backup"
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, TENSORBOARD_ZIP)
        print(f"Restored from backup: {TENSORBOARD_ZIP}")
    else:
        print("No backup found!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        restore_backup()
    else:
        patch_tensorboard()