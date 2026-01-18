from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.5.1+cu121'
debug = False
cuda: Optional[str] = '12.1'
git_version = 'a8d6afb511a69687bbb2b7e88a3cf67917e1697e'
hip: Optional[str] = None
