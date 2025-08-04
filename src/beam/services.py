"""Service management for BEAM Docker services"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages Docker services required by BEAM"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize service manager
        
        Args:
            project_root: Path to project root (where docker-compose.yml is located)
        """
        if project_root is None:
            # Find project root by looking for docker-compose.yml
            current = Path(__file__).parent
            while current.parent != current:
                if (current / "docker-compose.yml").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                raise RuntimeError("Could not find project root with docker-compose.yml")
        
        self.project_root = project_root
        self.compose_file = project_root / "docker-compose.yml"
        self.compose_cmd = self._get_compose_command()
        
    def _get_compose_command(self) -> str:
        """Get the appropriate docker compose command"""
        try:
            # Check if 'docker compose' (new style) is available
            result = subprocess.run(
                ["docker", "compose", "version"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "docker compose"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Fall back to docker-compose (old style)
        try:
            result = subprocess.run(
                ["docker-compose", "--version"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "docker-compose"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        raise RuntimeError("Neither 'docker compose' nor 'docker-compose' is available")
    
    def _run_compose_command(self, command: list, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a docker compose command"""
        cmd = self.compose_cmd.split() + ["-f", str(self.compose_file)] + command
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                timeout=30
            )
            return result
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"Failed to run command {' '.join(cmd)}: {e}")
            raise
    
    def is_service_running(self, service_name: str) -> bool:
        """Check if a service is running"""
        try:
            result = self._run_compose_command(["ps", service_name])
            # If the service is listed and running, it will be in the output
            return service_name in result.stdout and "Up" in result.stdout
        except Exception:
            return False
    
    def start_service(self, service_name: str, wait_for_ready: bool = True) -> bool:
        """Start a Docker service
        
        Args:
            service_name: Name of the service to start
            wait_for_ready: Whether to wait for the service to be ready
            
        Returns:
            True if service started successfully, False otherwise
        """
        try:
            logger.info(f"Starting {service_name} service...")
            result = self._run_compose_command(["up", "-d", service_name])
            
            if result.returncode != 0:
                logger.error(f"Failed to start {service_name}: {result.stderr}")
                return False
            
            if wait_for_ready:
                return self._wait_for_service(service_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting {service_name}: {e}")
            return False
    
    def _wait_for_service(self, service_name: str, timeout: int = 30) -> bool:
        """Wait for a service to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_service_running(service_name):
                logger.info(f"✓ {service_name} service is ready")
                return True
            time.sleep(1)
        
        logger.warning(f"⚠️  {service_name} service did not become ready within {timeout}s")
        return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a Docker service"""
        try:
            logger.info(f"Stopping {service_name} service...")
            result = self._run_compose_command(["stop", service_name])
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error stopping {service_name}: {e}")
            return False
    
    def ensure_services_running(self, services: list = None) -> bool:
        """Ensure required services are running
        
        Args:
            services: List of service names to ensure are running.
                     If None, uses default services.
                     
        Returns:
            True if all services are running, False otherwise
        """
        if services is None:
            services = ["zeek-processor", "llama-model"]
        
        all_running = True
        
        for service in services:
            if not self.is_service_running(service):
                logger.info(f"🔧 {service} not running, starting...")
                if not self.start_service(service):
                    logger.error(f"❌ Failed to start {service}")
                    all_running = False
                else:
                    logger.info(f"✅ {service} started successfully")
            else:
                logger.debug(f"✓ {service} already running")
        
        return all_running
    
    def check_docker_available(self) -> bool:
        """Check if Docker is available and running"""
        try:
            result = subprocess.run(
                ["docker", "info"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def setup_environment(self) -> bool:
        """Set up the environment for BEAM execution"""
        print("🔧 Setting up BEAM environment...")
        
        # Check Docker availability
        if not self.check_docker_available():
            print("❌ Docker is not running or not installed")
            print("Please start Docker Desktop and try again")
            return False
        
        print("✓ Docker is available")
        
        # Ensure required services are running
        print("🐳 Ensuring required services are running...")
        if not self.ensure_services_running():
            print("❌ Failed to start required services")
            return False
        
        print("✅ Environment setup complete")
        return True


# Global service manager instance
_service_manager = None

def get_service_manager() -> ServiceManager:
    """Get the global service manager instance"""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager


def ensure_services_running(services: list = None) -> bool:
    """Convenience function to ensure services are running"""
    return get_service_manager().ensure_services_running(services)


def setup_environment() -> bool:
    """Convenience function to set up the environment"""
    return get_service_manager().setup_environment()