# -*- mode: python -*-

block_cipher = None

added_files = [
	('path.config', '.'),
	('time.txt', '.'),
	('Resources\\*.bmp', 'Resources'),
	('Resources\\*.png', 'Resources')
	]
a = Analysis(['WBCRev5.py'],
             pathex=['C:\\Users\\david\\Desktop\\Skripsi'],
             binaries=[],
             datas= added_files,
             hiddenimports=['pywt','pywt._extensions._cwt','pywt._estentions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [('v', None, 'OPTION')],
          name='WBCRev5',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True, icon='icon.ico')
