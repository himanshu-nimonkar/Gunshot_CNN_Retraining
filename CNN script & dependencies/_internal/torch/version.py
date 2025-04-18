from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.2.1+cu121'
debug = False
cuda: Optional[str] = '12.1'
git_version = '6c8c5ad5eaf47a62fafbb4a2747198cbffbf1ff0'
hip: Optional[str] = None
