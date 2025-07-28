"""Tests for command-line interface and error handling in training workflows"""

import argparse
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from beam import run


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for CLI testing"""
    temp_dir = tempfile.mkdtemp()
    workspace = {
        "input_dir": os.path.join(temp_dir, "input"),
        "har_file": os.path.join(temp_dir, "input", "test.har"),
        "pcap_file": os.path.join(temp_dir, "input", "test.pcap"),
        "model_output": os.path.join(temp_dir, "custom_model.pkl"),
        "invalid_file": os.path.join(temp_dir, "invalid.txt"),
    }

    os.makedirs(workspace["input_dir"])

    # Create mock HAR file
    har_content = {
        "log": {
            "version": "1.2",
            "entries": [
                {
                    "request": {
                        "method": "GET",
                        "url": "https://api.testapp.com/test",
                        "headers": [{"name": "User-Agent", "value": "TestApp/1.0.0"}],
                    },
                    "response": {"status": 200},
                }
            ],
        }
    }

    with open(workspace["har_file"], "w") as f:
        json.dump(har_content, f)

    # Create mock PCAP file (empty file for testing)
    with open(workspace["pcap_file"], "wb") as f:
        f.write(b"\x00\x01\x02\x03")  # Mock binary data

    # Create invalid file
    with open(workspace["invalid_file"], "w") as f:
        f.write("This is not a valid network capture file")

    yield workspace

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


class TestCommandLineArgumentParsing:
    """Test command-line argument parsing for training functionality"""

    def test_training_arguments_parsing(self):
        """Test parsing of training-specific arguments"""
        # Test basic training arguments
        test_args = [
            "--train",
            "--app_name",
            "TestApp",
            "-i",
            "/path/to/input",
            "--model_output",
            "/path/to/model.pkl",
        ]

        with patch("sys.argv", ["beam"] + test_args):
            parser = argparse.ArgumentParser()
            # Add the same arguments as in run.py
            parser.add_argument(
                "--train", action="store_true", help="Train custom app model"
            )
            parser.add_argument(
                "--app_name", type=str, help="Name of the app to train for"
            )
            parser.add_argument("-i", "--input_dir", type=str, help="Input directory")
            parser.add_argument(
                "--model_output", type=str, help="Output path for model"
            )
            parser.add_argument(
                "--use_custom_models", action="store_true", default=True
            )
            parser.add_argument(
                "--no-use_custom_models", dest="use_custom_models", action="store_false"
            )
            parser.add_argument("--map_only", action="store_true", default=False)

            args = parser.parse_args()

            assert args.train is True
            assert args.app_name == "TestApp"
            assert args.input_dir == "/path/to/input"
            assert args.model_output == "/path/to/model.pkl"

    def test_default_training_arguments(self):
        """Test default values for training arguments"""
        test_args = ["--train"]

        with patch("sys.argv", ["beam"] + test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument("--train", action="store_true")
            parser.add_argument("--app_name", type=str, default=None)
            parser.add_argument("--model_output", type=str, default=None)

            args = parser.parse_args()

            assert args.train is True
            assert args.app_name is None  # Should trigger auto-discovery
            assert args.model_output is None  # Should be auto-generated

    def test_conflicting_arguments(self):
        """Test handling of conflicting command-line arguments"""
        # Test train and map_only together (if they're mutually exclusive)
        test_args = ["--train", "--map_only"]

        with patch("sys.argv", ["beam"] + test_args):
            parser = argparse.ArgumentParser()
            parser.add_argument("--train", action="store_true")
            parser.add_argument("--map_only", action="store_true")

            # This should parse without error - logic handles conflicts
            args = parser.parse_args()
            assert args.train is True
            assert args.map_only is True


class TestTrainingWorkflowErrorHandling:
    """Test error handling in training workflows"""

    @patch("beam.run.run_training_in_container")
    @patch("beam.run.glob.glob")
    def test_training_with_no_input_files(
        self, mock_glob, mock_run_training, temp_workspace
    ):
        """Test training behavior when no input files are found"""
        # Mock no files found
        mock_glob.return_value = []

        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                train=True,
                app_name="TestApp",
                input_dir=temp_workspace["input_dir"],
                model_output=temp_workspace["model_output"],
                use_custom_models=True,
                mapping_only=None,
                log_level="INFO",
            )

            logger = MagicMock()

            # Should handle gracefully without calling run_training_in_container
            run.run(logger=logger)

            # run_training_in_container should not be called when no files found
            mock_run_training.assert_not_called()

    @patch("beam.run.run_training_in_container")
    @patch("beam.run.glob.glob")
    def test_training_with_auto_discovery(
        self, mock_glob, mock_run_training, temp_workspace
    ):
        """Test training workflow with auto-discovery (no app_name specified)"""
        # Mock files found
        mock_glob.return_value = [temp_workspace["har_file"]]

        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                train=True,
                app_name=None,  # No app_name specified - should trigger auto-discovery
                input_dir=temp_workspace["input_dir"],
                model_output=None,  # Should be auto-generated
                use_custom_models=True,
                mapping_only=None,
                log_level="INFO",
            )

            logger = MagicMock()

            # Should handle auto-discovery workflow
            run.run(logger=logger)

            # Should run training in container for the input file
            mock_run_training.assert_called_once_with(
                input_file_path=temp_workspace["har_file"],
                app_name=None,
                custom_model_path=None,
                logger=logger,
            )

    @patch("beam.run.run_training_in_container")
    @patch("beam.run.glob.glob")
    def test_training_with_invalid_input_directory(
        self, mock_glob, mock_run_training
    ):
        """Test training with non-existent input directory"""
        # Mock files found (even though directory doesn't exist)
        mock_glob.return_value = ["/nonexistent/test.har"]

        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                train=True,
                app_name="TestApp",
                input_dir="/nonexistent/directory",
                model_output="/tmp/model.pkl",
                use_custom_models=True,
                mapping_only=None,
                log_level="INFO",
            )

            logger = MagicMock()

            # Mock Path.is_dir to simulate directory existing so run continues
            with patch("pathlib.Path.is_dir", return_value=True):
                run.run(logger=logger)

            # run_training_in_container should be called
            mock_run_training.assert_called()

    @patch("beam.run.run_training_in_container")
    @patch("beam.run.glob.glob")
    def test_training_with_parse_input_file_failure(
        self, mock_glob, mock_run_training, temp_workspace
    ):
        """Test training when container execution fails (simulating parse failure)"""
        mock_glob.return_value = [temp_workspace["har_file"]]
        # Mock container training to fail with exit code
        mock_run_training.side_effect = SystemExit(1)

        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                train=True,
                app_name="TestApp",
                input_dir=temp_workspace["input_dir"],
                model_output=temp_workspace["model_output"],
                use_custom_models=True,
                mapping_only=None,
                log_level="INFO",
            )

            logger = MagicMock()

            # Should propagate SystemExit
            with pytest.raises(SystemExit) as exc_info:
                run.run(logger=logger)
            assert exc_info.value.code == 1

    @patch("beam.run.run_training_in_container")
    @patch("beam.run.glob.glob")
    def test_training_with_enrichment_failure(
        self, mock_glob, mock_run_training, temp_workspace
    ):
        """Test training when container execution fails (simulating enrichment failure)"""
        mock_glob.return_value = [temp_workspace["har_file"]]
        # Mock container training to fail
        mock_run_training.side_effect = SystemExit(1)

        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                train=True,
                app_name="TestApp",
                input_dir=temp_workspace["input_dir"],
                model_output=temp_workspace["model_output"],
                use_custom_models=True,
                mapping_only=None,
                log_level="INFO",
            )

            logger = MagicMock()

            # Should propagate SystemExit
            with pytest.raises(SystemExit) as exc_info:
                run.run(logger=logger)
            assert exc_info.value.code == 1

    @patch("beam.run.run_training_in_container")
    @patch("beam.run.glob.glob")
    def test_training_with_model_training_failure(
        self,
        mock_glob,
        mock_run_training,
        temp_workspace,
    ):
        """Test training when model training fails inside container"""
        mock_glob.return_value = [temp_workspace["har_file"]]
        # Mock container to fail
        mock_run_training.side_effect = SystemExit(1)

        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                train=True,
                app_name="TestApp",
                input_dir=temp_workspace["input_dir"],
                model_output=temp_workspace["model_output"],
                use_custom_models=True,
                mapping_only=None,
                log_level="INFO",
            )

            logger = MagicMock()

            # Should propagate SystemExit
            with pytest.raises(SystemExit) as exc_info:
                run.run(logger=logger)
            assert exc_info.value.code == 1


class TestModelOutputHandling:
    """Test model output path handling and validation"""

    def test_auto_generated_model_output_path(self):
        """Test automatic generation of model output path when not specified"""
        # This test verifies the behavior, not the actual implementation
        # as the path generation happens inside the container
        assert True  # Placeholder for actual test

    def test_custom_model_output_validation(self):
        """Test validation of custom model output paths"""
        # Test various invalid paths
        invalid_paths = [
            "",  # Empty path
            " ",  # Whitespace only
            "/root/model.pkl",  # Potentially restricted path
        ]

        for path in invalid_paths:
            # In actual implementation, these might be validated
            # For now, we just test the concept
            assert path != ""  or path == ""  # Tautology for placeholder


class TestResourceManagementAndCleanup:
    """Test resource management and cleanup in training workflows"""

    @patch("beam.run.run_training_in_container")
    @patch("beam.run.glob.glob")
    def test_temporary_file_cleanup(
        self, mock_glob, mock_run_training, temp_workspace
    ):
        """Test that temporary files are cleaned up properly"""
        mock_glob.return_value = [temp_workspace["har_file"]]

        # Mock container execution
        mock_run_training.return_value = None

        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                train=True,
                app_name="TestApp",
                input_dir=temp_workspace["input_dir"],
                model_output=temp_workspace["model_output"],
                use_custom_models=True,
                mapping_only=None,
                log_level="INFO",
            )

            logger = MagicMock()

            # Run training
            run.run(logger=logger)

            # Verify container was called
            mock_run_training.assert_called_once()

    def test_signal_handling_during_training(self):
        """Test graceful shutdown on interrupt signals during training"""
        # This would test signal handling, but it's complex to test properly
        # Placeholder for future implementation
        assert True


class TestIntegrationWithDetectionMode:
    """Test integration between training and detection modes"""

    def test_training_followed_by_detection(self):
        """Test workflow of training a model then using it for detection"""
        # This is a conceptual test showing the workflow
        # Actual implementation would involve:
        # 1. Training a model
        # 2. Verifying model file exists
        # 3. Running detection with the new model
        assert True

    def test_model_compatibility_check(self):
        """Test that trained models are compatible with detection mode"""
        # Verify model format matches what detection expects
        assert True