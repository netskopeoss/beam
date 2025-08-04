"""Main module for the BEAM package."""

# Copyright 2025 Netskope, Inc.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Authors:
# - Colin Estep
# - Dagmawi Mulugeta

import logging
import logging.config
import os
import sys
from pathlib import Path

def setup_logging():
    """Setup logging to file only and redirect stderr."""
    # Import LOG_DIR here to avoid circular imports
    from beam.constants import LOG_DIR
    
    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)
    
    # Set up file logging only
    log_file = LOG_DIR / "beam.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file)]
    )
    
    # Redirect stderr to log file for TensorFlow warnings
    stderr_log_file = LOG_DIR / "beam_stderr.log"
    sys.stderr = open(stderr_log_file, 'a', buffering=1)
    
    return log_file, stderr_log_file

# Set up logging early, before any imports that might use TensorFlow
log_file, stderr_log_file = setup_logging()

# Now import modules that might trigger TensorFlow
from beam.constants import LOG_CONFIG
from beam.run import MultiHotEncoder, run
from beam.services import setup_environment

def main():
    """Main entry point for BEAM."""
    logger = logging.getLogger("main")
    
    # Set up Docker services before running BEAM
    if not setup_environment():
        print("❌ Failed to set up BEAM environment")
        sys.exit(1)
    
    # Set up environment for native Python execution
    setup_native_environment()
    
    # Run BEAM natively in Python
    try:
        _m = MultiHotEncoder()
        # Main module passes control to the run function which handles command-line arguments
        run(logger=logger)
    except KeyboardInterrupt:
        print("\n⚠️  BEAM execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"BEAM execution failed: {e}")
        print(f"❌ BEAM failed: {e}")
        sys.exit(1)

def setup_native_environment():
    """Set up environment variables for native Python execution."""
    # Set PYTHONPATH to include src directory if not already set
    current_dir = Path(__file__).parent.parent.parent  # Go up to project root
    src_dir = current_dir / "src"
    
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Set default environment variables if not already set
    env_defaults = {
        'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY', ''),
        'LOG_LEVEL': os.environ.get('LOG_LEVEL', 'INFO'),
        'USE_LOCAL_LLM': os.environ.get('USE_LOCAL_LLM', 'false'),
    }
    
    for key, value in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = value

if __name__ == "__main__":
    main()
