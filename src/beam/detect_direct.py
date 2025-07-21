"""Direct detection entry point for Docker containers.

This runs BEAM detection inside containers with TensorFlow support
for custom ensemble models.
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
    """Direct detection entry point for containers."""
    logger = logging.getLogger("beam-detection")
    
    parser = argparse.ArgumentParser(description="BEAM Direct Detection")
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
        "--use_custom_models",
        help="Use custom trained models",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    logger.info("🔒 BEAM Direct Detection Mode")
    logger.info("Running detection inside Docker container with TensorFlow support")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Use custom models: {args.use_custom_models}")
    
    try:
        # Import detection functions
        from beam.run import process_input_file
        
        # Process the input file
        input_path = Path(args.input_dir)
        if input_path.is_file():
            # Single file
            process_input_file(
                file_path=str(input_path),
                use_custom_models=args.use_custom_models,
                logger=logger
            )
        elif input_path.is_dir():
            # Directory - find and process all HAR/PCAP files
            import glob
            patterns = ['*.har', '*.pcap', '*.pcapng']
            files = []
            for pattern in patterns:
                files.extend(glob.glob(os.path.join(str(input_path), pattern)))
            
            if not files:
                logger.error(f"No HAR/PCAP files found in {input_path}")
                sys.exit(1)
                
            for file_path in files:
                process_input_file(
                    file_path=file_path,
                    use_custom_models=args.use_custom_models,
                    logger=logger
                )
        else:
            logger.error(f"Input path does not exist: {input_path}")
            sys.exit(1)
        
        logger.info("✅ Detection completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()