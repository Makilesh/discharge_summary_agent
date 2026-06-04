import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
namespace = "patient_2__1_a1058ddb1398_auto_gemini_2_5_flash_lite_deepseek_r1_14b_qwen2_5vl_7b"
os.environ["OCR_CACHE_NAMESPACE"] = namespace
os.environ["OCR_CACHE_DIR"] = "output/ocr_cache"

from src.tools import get_extraction_cache_file

cache_file = get_extraction_cache_file(6, "LAB_REPORT_BIOCHEMISTRY")
print(f"Cache file path: {cache_file.resolve()}")
print(f"Cache file exists: {cache_file.exists()}")
