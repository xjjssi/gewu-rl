"""conftest.py —— 测试路径配置：让 tests/ 能 import 教学包与后端模块。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gridworld_edu"))
sys.path.insert(0, str(ROOT / "backend"))
