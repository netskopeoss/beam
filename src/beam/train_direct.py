"""Direct training entry point for Docker containers.

This bypasses the full BEAM initialization and service setup,
allowing training to run inside containers without needing 
docker-compose.yml access.
"""

import os
# Suppress TensorFlow warnings before importing anything else
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import logging
import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def setup_logging():
    """Setup logging to file using same approach as main BEAM."""
    from beam.constants import LOG_DIR
    
    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)
    
    # Set up file logging to beam.log (same file as main BEAM)
    log_file = LOG_DIR / "beam.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a')]  # Append mode
    )

# Initialize logging
setup_logging()

def main():
    """Direct training entry point for containers."""
    logger = logging.getLogger("beam-training")
    
    parser = argparse.ArgumentParser(description="BEAM Direct Training")
    parser.add_argument(
        "-i",
        "--input_dir",
        help="Directory containing the input files to be processed",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help="The log level to use for logging",
        required=False,
        default="INFO",
    )
    parser.add_argument(
        "--app_name",
        help="Specific app name to train (optional)",
        required=False,
    )
    parser.add_argument(
        "--custom_model_path", 
        help="Custom model path (optional)",
        required=False,
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    logger.info("🔒 BEAM Direct Training Mode")
    logger.info("Running training inside Docker container with TensorFlow support")
    logger.info(f"Input: {args.input_dir}")
    
    try:
        # Import training function
        from beam.run import process_training_data
        
        # Call training directly
        process_training_data(
            input_file_path=args.input_dir,
            app_name=args.app_name,
            custom_model_path=args.custom_model_path,
            logger=logger
        )
        
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()