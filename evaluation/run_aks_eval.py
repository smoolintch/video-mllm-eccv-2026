import sys
import os
import logging

# Set up logging to verify import
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path to ensure we can import evaluation
sys.path.append(os.getcwd())

# 1. Patch lmms_eval.models.AVAILABLE_SIMPLE_MODELS
try:
    import lmms_eval.models
    # Map the model name to the full class path string
    # "evaluation.qwen2_5_vl" is the module, "Qwen2_5_VL" is the class
    lmms_eval.models.AVAILABLE_SIMPLE_MODELS["aks_qwen2_5_vl"] = "evaluation.qwen2_5_vl.Qwen2_5_VL"
    logger.info("Registered aks_qwen2_5_vl in lmms_eval.models.AVAILABLE_SIMPLE_MODELS")
except ImportError as e:
    logger.error(f"Failed to patch lmms_eval.models: {e}")

# 2. Explicitly import the custom model module to ensure it can be loaded
# This is useful to catch syntax errors or import errors early
try:
    import evaluation.qwen2_5_vl
    logger.info("Successfully imported evaluation.qwen2_5_vl")
except ImportError as e:
    logger.error(f"Failed to import evaluation.qwen2_5_vl: {e}")

from lmms_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
