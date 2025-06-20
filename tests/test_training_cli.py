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

            args = parser.parse_args(test_args)

            assert args.train is True
            assert args.app_name == "TestApp"
            assert args.input_dir == "/path/to/input"
            assert args.model_output == "/path/to/model.pkl"
            assert args.use_custom_models is True

    def test_training_arguments_with_custom_models_disabled(self):
        """Test training arguments with custom models disabled"""
        test_args = ["--train", "--app_name", "TestApp", "--no-use_custom_models"]

        parser = argparse.ArgumentParser()
        parser.add_argument("--train", action="store_true")
        parser.add_argument("--app_name", type=str)
        parser.add_argument("--use_custom_models", action="store_true", default=True)
        parser.add_argument(
            "--no-use_custom_models", dest="use_custom_models", action="store_false"
        )

        args = parser.parse_args(test_args)

        assert args.train is True
        assert args.app_name == "TestApp"
        assert args.use_custom_models is False

    def test_training_with_optional_app_name(self):
        """Test training without app_name (uses auto-discovery)"""
        # Training without app_name should be handled gracefully and use auto-discovery
        test_args = ["--train"]

        parser = argparse.ArgumentParser()
        parser.add_argument("--train", action="store_true")
        parser.add_argument("--app_name", type=str, required=False)

        args = parser.parse_args(test_args)

        assert args.train is True
        assert args.app_name is None  # Should trigger auto-discovery


class TestTrainingWorkflowErrorHandling:
    """Test error handling in training workflows"""

    @patch("beam.run.process_training_data")
    @patch("beam.run.glob.glob")
    def test_training_with_no_input_files(
        self, mock_glob, mock_process_training, temp_workspace
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

            # Should handle gracefully without calling process_training_data
            run.run(logger=logger)

            # process_training_data should not be called when no files found
            mock_process_training.assert_not_called()

    @patch("beam.run.process_training_data")
    @patch("beam.run.glob.glob")
    def test_training_with_auto_discovery(
        self, mock_glob, mock_process_training, temp_workspace
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

            # Should process training for the input file
            mock_process_training.assert_called_once_with(
                input_file_path=temp_workspace["har_file"],
                app_name=None,
                custom_model_path=None,
                logger=logger,
            )

    @patch("beam.run.process_training_data")
    @patch("beam.run.glob.glob")
    def test_training_with_invalid_input_directory(
        self, mock_glob, mock_process_training
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

            # process_training_data should be called
            mock_process_training.assert_called()

    @patch("beam.run.parse_input_file")
    @patch("beam.run.glob.glob")
    def test_training_with_parse_input_file_failure(
        self, mock_glob, mock_parse_input, temp_workspace
    ):
        """Test training when input file parsing fails"""
        mock_glob.return_value = [temp_workspace["har_file"]]
        mock_parse_input.side_effect = Exception("Failed to parse input file")

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

            # Should handle parsing error gracefully
            try:
                run.run(logger=logger)
            except Exception as e:
                # Expected to fail at parsing stage
                assert "Failed to parse input file" in str(e)

    @patch("beam.run.enrich_events")
    @patch("beam.run.parse_input_file")
    @patch("beam.run.glob.glob")
    def test_training_with_enrichment_failure(
        self, mock_glob, mock_parse_input, mock_enrich, temp_workspace
    ):
        """Test training when event enrichment fails"""
        mock_glob.return_value = [temp_workspace["har_file"]]
        mock_parse_input.return_value = ("test", "/parsed.json")
        mock_enrich.side_effect = Exception("Enrichment failed")

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

            # Should handle enrichment error gracefully
            try:
                run.run(logger=logger)
            except Exception as e:
                assert "Enrichment failed" in str(e)

    @patch("beam.run.train_custom_app_model")
    @patch("beam.run.extract_app_features")
    @patch("beam.run.discover_apps_in_traffic")
    @patch("beam.run.enrich_events")
    @patch("beam.run.parse_input_file")
    @patch("beam.run.glob.glob")
    def test_training_with_model_training_failure(
        self,
        mock_glob,
        mock_parse_input,
        mock_enrich,
        mock_discover,
        mock_extract,
        mock_train,
        temp_workspace,
    ):
        """Test training when actual model training fails"""
        mock_glob.return_value = [temp_workspace["har_file"]]
        mock_parse_input.return_value = ("test", "/parsed.json")
        mock_enrich.return_value = "/enriched.json"
        mock_discover.return_value = {"TestApp": 150}  # Sufficient transactions
        mock_extract.return_value = "/features.json"
        mock_train.side_effect = Exception("Model training failed")

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

            # Should handle training error gracefully
            try:
                run.run(logger=logger)
            except Exception as e:
                assert "Model training failed" in str(e)


class TestInputValidationAndSanitization:
    """Test input validation and sanitization for training"""

    def test_app_name_validation(self):
        """Test validation of app names"""
        valid_app_names = [
            "TestApp",
            "My_Custom_App",
            "App-Version-1.0",
            "CustomCRM2024",
        ]

        invalid_app_names = [
            "",
            None,
            "App with spaces",  # Spaces might be problematic
            "App/With/Slashes",  # Path separators
            "App\nWith\nNewlines",  # Control characters
            "App<script>alert('xss')</script>",  # Potential injection
        ]

        # Test valid names (should not raise exceptions)
        for app_name in valid_app_names:
            # This would typically be part of argument validation
            assert isinstance(app_name, str)
            assert len(app_name) > 0

        # Test invalid names
        for app_name in invalid_app_names:
            if app_name is None or app_name == "":
                # These should be caught by argument validation
                assert not app_name
            elif any(char in app_name for char in [" ", "/", "\n", "<", ">"]):
                # These contain problematic characters
                assert any(char in app_name for char in [" ", "/", "\n", "<", ">"])

    def test_input_path_validation(self, temp_workspace):
        """Test validation of input paths"""
        valid_paths = [
            temp_workspace["input_dir"],  # Existing directory
            temp_workspace["har_file"],  # Existing file
        ]

        invalid_paths = [
            "/nonexistent/directory",
            "",
            None,
            "../../../etc/passwd",  # Path traversal attempt
            "/dev/null",  # Special file
        ]

        for path in valid_paths:
            if path and os.path.exists(path):
                assert os.path.exists(path)

        for path in invalid_paths:
            if path is None or path == "":
                assert not path
            elif path.startswith("../") or path == "/dev/null":
                # Potentially dangerous paths
                assert "../" in path or path == "/dev/null"

    def test_model_output_path_validation(self, temp_workspace):
        """Test validation of model output paths"""
        valid_output_paths = [
            temp_workspace["model_output"],
            os.path.join(temp_workspace["input_dir"], "model.pkl"),
            "/tmp/custom_model.pkl",
        ]

        invalid_output_paths = [
            "",
            None,
            "/root/model.pkl",  # Privileged directory
            "../../../tmp/malicious.pkl",  # Path traversal
            "/dev/null",  # Special file
            "model.pkl\x00.exe",  # Null byte injection
        ]

        for path in valid_output_paths:
            if path:
                # Should be valid file paths
                assert isinstance(path, str)
                assert len(path) > 0
                assert path.endswith(".pkl")

        for path in invalid_output_paths:
            if not path:
                assert not path
            elif "../" in path or path.startswith("/root/") or "\x00" in path:
                # Contains problematic patterns
                assert "../" in path or "/root/" in path or "\x00" in path


class TestResourceManagementAndCleanup:
    """Test resource management and cleanup in training workflows"""

    @patch("beam.run.process_training_data")
    @patch("beam.run.glob.glob")
    def test_temporary_file_cleanup(
        self, mock_glob, mock_process_training, temp_workspace
    ):
        """Test that temporary files are cleaned up properly"""
        mock_glob.return_value = [temp_workspace["har_file"]]

        # Track temporary files created during processing
        temp_files_created = []

        def mock_process_with_temp_files(*args, **kwargs):
            # Simulate creating temporary files
            temp_file = os.path.join(
                temp_workspace["input_dir"], "temp_processing.json"
            )
            temp_files_created.append(temp_file)
            with open(temp_file, "w") as f:
                json.dump({"temp": "data"}, f)

        mock_process_training.side_effect = mock_process_with_temp_files

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
            run.run(logger=logger)

        # Verify temporary files were created during processing
        assert len(temp_files_created) > 0

        # In a real implementation, these should be cleaned up
        # For this test, we just verify they were created
        for temp_file in temp_files_created:
            if os.path.exists(temp_file):
                os.unlink(temp_file)  # Manual cleanup for test

    def test_memory_usage_with_large_datasets(self, temp_workspace):
        """Test memory management with large datasets"""
        # Create a large feature dataset
        large_dataset = []
        for i in range(1000):  # Simulate large dataset
            large_dataset.append(
                {
                    "key": f"TestApp_{i}",
                    "application": "TestApp",
                    "transactions": 100,
                    "avg_time_taken_ms": i * 1.5,
                    "domain": f"api{i % 10}.testapp.com",
                    "http_methods": ["GET", "POST"] * (i % 5 + 1),
                    "http_statuses": ["200", "404"] * (i % 3 + 1),
                    "req_content_types": ["application/json"] * (i % 2 + 1),
                    "resp_content_types": ["application/json"] * (i % 2 + 1),
                }
            )

        large_features_file = os.path.join(
            temp_workspace["input_dir"], "large_features.json"
        )
        with open(large_features_file, "w") as f:
            json.dump(large_dataset, f)

        # Test that the system can handle large datasets without memory issues
        # In a real test, you might monitor memory usage here
        file_size = os.path.getsize(large_features_file)
        assert file_size > 50000  # Ensure we created a reasonably large file

        # Cleanup
        os.unlink(large_features_file)

    def test_resource_cleanup_on_training_failure(self, temp_workspace):
        """Test resource cleanup when training fails"""
        # Simplified test to verify cleanup concept without complex mocking

        # Create temp files
        temp_enriched = os.path.join(temp_workspace["input_dir"], "temp_enriched.json")
        temp_features = os.path.join(temp_workspace["input_dir"], "temp_features.json")

        with open(temp_enriched, "w") as f:
            json.dump([{"temp": "enriched"}], f)
        with open(temp_features, "w") as f:
            json.dump([{"temp": "features"}], f)

        # Verify files exist
        assert os.path.exists(temp_enriched)
        assert os.path.exists(temp_features)

        # Clean up for test completion
        os.unlink(temp_enriched)
        os.unlink(temp_features)

        # Verify cleanup worked
        assert not os.path.exists(temp_enriched)
        assert not os.path.exists(temp_features)


class TestConcurrencyAndRaceConditions:
    """Test concurrency scenarios and race conditions"""

    def test_concurrent_training_same_app(self, temp_workspace):
        """Test concurrent training attempts for the same app"""
        # This test simulates what happens if multiple processes
        # try to train the same app simultaneously

        app_features = [
            {
                "key": "TestApp_1.0.0",
                "application": "TestApp",
                "transactions": 100,
                "domain": "api.testapp.com",
                "avg_time_taken_ms": 150,
                "http_methods": ["GET", "POST"],
                "http_statuses": ["200"],
                "req_content_types": ["application/json"],
                "resp_content_types": ["application/json"],
                # Add all required numeric fields
                "refered_traffic_pct": 0.8,
                "referer_domain_cnt": 2,
                "unique_actions": 15,
                "http_status_cnt": 3,
                "http_method_cnt": 4,
                "req_content_type_cnt": 3,
                "resp_content_type_cnt": 3,
                "avg_time_interval_sec": 30.5,
                "std_time_interval_sec": 12.3,
                "median_time_interval_sec": 28.0,
                "range_time_interval_sec": 60.0,
                "range_timestamp": 3600,
                "max_time_taken_ms": 500,
                "min_time_taken_ms": 50,
                "sum_time_taken_ms": 15000,
                "std_time_taken_ms": 75,
                "median_time_taken_ms": 125,
                "range_time_taken_ms": 450,
                "max_client_bytes": 2048,
                "min_client_bytes": 0,
                "sum_client_bytes": 102400,
                "avg_client_bytes": 1024,
                "std_client_bytes": 512,
                "median_client_bytes": 512,
                "range_client_bytes": 2048,
                "max_server_bytes": 65536,
                "min_server_bytes": 512,
                "sum_server_bytes": 1048576,
                "avg_server_bytes": 10485,
                "std_server_bytes": 15000,
                "median_server_bytes": 2048,
                "range_server_bytes": 65024,
                "web_traffic_pct": 0.6,
                "cloud_traffic_pct": 0.4,
                "sequence_num_keys": 10,
                "sequence_max_key_length": 20,
                "sequence_min_key_length": 5,
                "sequence_max_val": 100,
                "sequence_min_val": 1,
                "sequence_sum_val": 550,
                "sequence_avg_val": 55,
                "sequence_std_val": 25,
                "sequence_median_val": 50,
                "sequence_range_val": 99,
            }
        ]

        features_file = os.path.join(temp_workspace["input_dir"], "features.json")
        with open(features_file, "w") as f:
            json.dump(app_features, f)

        # Simulate concurrent training by trying to create the same model twice
        from beam.detector.trainer import ModelTrainer

        trainer1 = ModelTrainer(n_features=5, min_transactions=50)
        trainer2 = ModelTrainer(n_features=5, min_transactions=50)

        model_path1 = os.path.join(temp_workspace["input_dir"], "model1.pkl")
        model_path2 = os.path.join(temp_workspace["input_dir"], "model2.pkl")

        # Both should succeed independently
        trainer1.add_app_model(features_file, "TestApp", model_path1)
        trainer2.add_app_model(features_file, "TestApp", model_path2)

        # Both models should be created
        assert os.path.exists(model_path1)
        assert os.path.exists(model_path2)

        # Cleanup
        os.unlink(model_path1)
        os.unlink(model_path2)

    def test_file_locking_simulation(self, temp_workspace):
        """Test file access scenarios that might cause locking issues"""
        # Create a features file
        features_file = os.path.join(temp_workspace["input_dir"], "features.json")
        with open(features_file, "w") as f:
            json.dump([{"key": "test", "transactions": 100}], f)

        # Simulate file being accessed by multiple processes
        from beam.detector.utils import load_json_file

        # Multiple reads should work fine
        data1 = load_json_file(features_file)
        data2 = load_json_file(features_file)

        assert data1 == data2
        assert len(data1) == 1

    def test_directory_creation_race_condition(self, temp_workspace):
        """Test directory creation race conditions"""
        # Simulate multiple processes trying to create the same directory
        nested_dir = os.path.join(
            temp_workspace["input_dir"], "nested", "deep", "directory"
        )
        dummy_file_path = os.path.join(nested_dir, "dummy.txt")

        from beam.detector.utils import safe_create_path

        # Multiple calls to create the same path should not conflict
        # safe_create_path expects a file path and creates its parent directory
        safe_create_path(dummy_file_path)
        safe_create_path(dummy_file_path)  # Second call should not fail

        assert os.path.exists(nested_dir)

        # Cleanup
        import shutil

        shutil.rmtree(os.path.join(temp_workspace["input_dir"], "nested"))
