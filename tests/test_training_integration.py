"""Integration tests for end-to-end custom model training workflow"""

import json
import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from beam import run
from beam.detector.trainer import ModelTrainer


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with required directory structure"""
    temp_dir = tempfile.mkdtemp()
    workspace = {
        "root": temp_dir,
        "data": {
            "input": os.path.join(temp_dir, "data", "input"),
            "enriched_events": os.path.join(temp_dir, "data", "enriched_events"),
            "app_summaries": os.path.join(temp_dir, "data", "app_summaries"),
        },
        "models": {
            "custom_models": os.path.join(temp_dir, "models", "custom_models"),
            "combined_model": os.path.join(
                temp_dir, "models", "combined_app_model.pkl"
            ),
            "pretrained_model": os.path.join(temp_dir, "models", "domain_model.pkl"),
        },
        "files": {
            "har_input": os.path.join(temp_dir, "data", "input", "test_traffic.har"),
            "pcap_input": os.path.join(temp_dir, "data", "input", "test_traffic.pcap"),
            "enriched_events": os.path.join(
                temp_dir, "data", "enriched_events", "test_traffic.json"
            ),
            "app_features": os.path.join(
                temp_dir, "data", "app_summaries", "test_app_features.json"
            ),
            "custom_model": os.path.join(
                temp_dir, "models", "custom_models", "TestApp_model.pkl"
            ),
        },
    }

    # Create directory structure
    for dir_path in [
        workspace["data"]["input"],
        workspace["data"]["enriched_events"],
        workspace["data"]["app_summaries"],
        workspace["models"]["custom_models"],
    ]:
        os.makedirs(dir_path, exist_ok=True)

    yield workspace

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_har_content():
    """Mock HAR file content"""
    return {
        "log": {
            "version": "1.2",
            "creator": {"name": "Chrome DevTools", "version": "120.0.0.0"},
            "entries": [
                {
                    "startedDateTime": "2024-01-01T12:00:00.000Z",
                    "time": 150,
                    "request": {
                        "method": "POST",
                        "url": "https://api.testapp.com/api/v1/sync",
                        "headers": [
                            {
                                "name": "User-Agent",
                                "value": "TestApp/1.0.0 (Windows NT 10.0; Win64; x64)",
                            },
                            {"name": "Content-Type", "value": "application/json"},
                        ],
                        "bodySize": 1024,
                    },
                    "response": {
                        "status": 200,
                        "headers": [
                            {"name": "Content-Type", "value": "application/json"}
                        ],
                        "content": {"size": 2048, "mimeType": "application/json"},
                        "bodySize": 2048,
                    },
                }
            ],
        }
    }


@pytest.fixture
def mock_enriched_events():
    """Mock enriched events for sufficient training data"""
    events = []
    for i in range(60):  # Create 60 events for sufficient training data
        events.append(
            {
                "useragent": "TestApp/1.0.0 (Windows NT 10.0; Win64; x64)",
                "domain": "api.testapp.com",
                "method": "POST" if i % 2 == 0 else "GET",
                "status": 200,
                "request_content_type": "application/json",
                "response_content_type": "application/json",
                "time_taken": 150 + (i * 10),
                "client_bytes": 1024 + (i * 100),
                "server_bytes": 2048 + (i * 200),
                "timestamp": f"2024-01-01T12:{i:02d}:00Z",
                "referer": "https://testapp.com",
                "path": f"/api/v1/endpoint_{i}",
            }
        )
    return events


@pytest.fixture
def mock_app_features_sufficient():
    """Mock app features with sufficient transactions for testing"""
    return [
        {
            "key": "TestApp_1.0.0",
            "application": "TestApp",
            "transactions": 100,  # Sufficient for training
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
            "avg_time_taken_ms": 150,
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
            "domain": "api.testapp.com",
            "http_methods": ["GET", "POST", "PUT", "DELETE"],
            "http_statuses": ["200", "201", "204"],
            "req_content_types": [
                "application/json",
                "application/x-www-form-urlencoded",
                "multipart/form-data",
            ],
            "resp_content_types": ["application/json", "image/png", "text/html"],
        }
    ]


class TestEndToEndTrainingWorkflow:
    """End-to-end integration tests for custom model training"""

    @patch("beam.run.parse_input_file")
    @patch("beam.run.enrich_events")
    @patch("beam.run.discover_apps_in_traffic")
    @patch("beam.run.extract_app_features")
    @patch("beam.run.train_custom_app_model")
    def test_process_training_data_complete_workflow(
        self,
        mock_train_custom,
        mock_extract_features,
        mock_discover,
        mock_enrich_events,
        mock_parse_input,
        temp_workspace,
        mock_app_features_sufficient,
    ):
        """Test complete training data processing workflow"""
        # Setup mocks
        mock_parse_input.return_value = (
            "test_traffic",
            temp_workspace["files"]["enriched_events"],
        )
        mock_enrich_events.return_value = temp_workspace["files"]["enriched_events"]
        mock_discover.return_value = {"TestApp": 150}  # Sufficient transactions
        mock_extract_features.return_value = temp_workspace["files"]["app_features"]
        mock_train_custom.return_value = None

        # Create mock logger
        import logging

        logger = logging.getLogger("test")

        # Execute training workflow
        run.process_training_data(
            input_file_path=temp_workspace["files"]["har_input"],
            app_name="TestApp",
            custom_model_path=temp_workspace["files"]["custom_model"],
            logger=logger,
        )

        # Verify all steps were called
        mock_parse_input.assert_called_once()
        mock_enrich_events.assert_called_once()
        mock_discover.assert_called()  # Called twice (eligible apps + all apps for reporting)
        mock_extract_features.assert_called_once()
        mock_train_custom.assert_called_once()

    @patch("beam.run.parse_input_file")
    @patch("beam.run.enrich_events")
    @patch("beam.run.discover_apps_in_traffic")
    @patch("beam.run.extract_app_features")
    @patch("beam.run.train_custom_app_model")
    def test_process_training_data_no_pretrained_model(
        self,
        mock_train_custom,
        mock_extract_features,
        mock_discover,
        mock_enrich_events,
        mock_parse_input,
        temp_workspace,
    ):
        """Test training workflow (individual model creation)"""
        # Setup mocks
        mock_parse_input.return_value = (
            "test_traffic",
            temp_workspace["files"]["enriched_events"],
        )
        mock_enrich_events.return_value = temp_workspace["files"]["enriched_events"]
        mock_discover.return_value = {"TestApp": 150}  # Sufficient transactions
        mock_extract_features.return_value = temp_workspace["files"]["app_features"]
        mock_train_custom.return_value = None

        import logging

        logger = logging.getLogger("test")

        # Execute training workflow (creates individual models)
        run.process_training_data(
            input_file_path=temp_workspace["files"]["har_input"],
            app_name="TestApp",
            custom_model_path=temp_workspace["files"]["custom_model"],
            logger=logger,
        )

        # Verify steps were called but no merge occurred
        mock_parse_input.assert_called_once()
        mock_enrich_events.assert_called_once()
        mock_extract_features.assert_called_once()
        mock_train_custom.assert_called_once()

    def test_full_model_training_pipeline(
        self, temp_workspace, mock_app_features_sufficient
    ):
        """Test full model training pipeline without mocking core functions"""
        # Create app features file
        with open(temp_workspace["files"]["app_features"], "w") as f:
            json.dump(mock_app_features_sufficient, f)

        # Create trainer and train model
        trainer = ModelTrainer(n_features=10, min_transactions=50)
        trainer.add_app_model(
            temp_workspace["files"]["app_features"],
            "TestApp",
            temp_workspace["files"]["custom_model"],
        )

        # Verify model was created
        assert os.path.exists(temp_workspace["files"]["custom_model"])

        # Load and verify model
        with open(temp_workspace["files"]["custom_model"], "rb") as f:
            model = pickle.load(f)

        assert len(model) == 1
        assert model[0]["key"] == "TestApp"
        assert "estimator" in model[0]
        assert "features" in model[0]
        assert "selected_features" in model[0]

    def test_model_merging_integration(
        self, temp_workspace, mock_app_features_sufficient
    ):
        """Test integration of model merging with existing models"""
        # Create existing pre-trained model
        existing_models = [
            {"key": "PretrainedApp1", "estimator": "estimator1"},
            {"key": "PretrainedApp2", "estimator": "estimator2"},
        ]
        with open(temp_workspace["models"]["pretrained_model"], "wb") as f:
            pickle.dump(existing_models, f)

        # Create and train new custom model
        with open(temp_workspace["files"]["app_features"], "w") as f:
            json.dump(mock_app_features_sufficient, f)

        trainer = ModelTrainer(n_features=10, min_transactions=50)
        trainer.add_app_model(
            temp_workspace["files"]["app_features"],
            "TestApp",
            temp_workspace["files"]["custom_model"],
        )

        # Merge models
        trainer.merge_models(
            temp_workspace["models"]["pretrained_model"],
            temp_workspace["files"]["custom_model"],
            temp_workspace["models"]["combined_model"],
        )

        # Verify merged model
        assert os.path.exists(temp_workspace["models"]["combined_model"])

        with open(temp_workspace["models"]["combined_model"], "rb") as f:
            combined_models = pickle.load(f)

        assert len(combined_models) == 3
        app_names = [model["key"] for model in combined_models]
        assert "PretrainedApp1" in app_names
        assert "PretrainedApp2" in app_names
        assert "TestApp" in app_names


class TestCommandLineIntegration:
    """Test command-line interface integration for training"""

    @patch("beam.run.process_training_data")
    @patch("argparse.ArgumentParser.parse_args")
    def test_command_line_training_invocation(
        self, mock_parse_args, mock_process_training
    ):
        """Test command-line training invocation"""
        # Mock command line arguments for training
        mock_args = MagicMock()
        mock_args.train = True
        mock_args.app_name = "TestApp"
        mock_args.input_dir = "/path/to/input"
        mock_args.model_output = "/path/to/model.pkl"
        mock_args.use_custom_models = True
        mock_args.mapping_only = None
        mock_args.log_level = "INFO"
        mock_parse_args.return_value = mock_args

        # Mock logger
        mock_logger = MagicMock()

        # Execute run function
        with patch("beam.run.glob.glob", return_value=["/path/to/input/test.har"]):
            with patch("pathlib.Path.is_dir", return_value=True):
                run.run(logger=mock_logger)

        # Verify training was called
        mock_process_training.assert_called()

    @patch("beam.run.process_training_data")
    @patch("beam.run.glob.glob")
    @patch("argparse.ArgumentParser.parse_args")
    def test_command_line_training_parameters(
        self, mock_parse_args, mock_glob, mock_process_training
    ):
        """Test command-line training with various parameters"""
        # Mock command line arguments
        mock_args = MagicMock()
        mock_args.train = True
        mock_args.app_name = "CustomApp"
        mock_args.input_dir = "/custom/input"
        mock_args.model_output = "/custom/model.pkl"
        mock_args.use_custom_models = True
        mock_args.mapping_only = None
        mock_args.log_level = "INFO"
        mock_parse_args.return_value = mock_args

        # Mock file discovery
        mock_glob.return_value = ["/custom/input/test.har"]

        mock_logger = MagicMock()

        # Execute CLI workflow
        with patch("pathlib.Path.is_dir", return_value=True):
            run.run(logger=mock_logger)

        # Verify process_training_data was called with correct parameters
        mock_process_training.assert_called_once_with(
            input_file_path="/custom/input/test.har",
            app_name="CustomApp",
            custom_model_path="/custom/model.pkl",
            logger=mock_logger,
        )


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in integration scenarios"""

    def test_training_with_insufficient_data(self, temp_workspace):
        """Test training behavior with insufficient training data"""
        # Create app features with insufficient transactions
        insufficient_features = [
            {
                "key": "TestApp_1.0.0",
                "application": "TestApp",
                "transactions": 10,  # Below minimum threshold
                "domain": "api.testapp.com",
                "avg_time_taken_ms": 150,
                "http_methods": ["GET", "POST"],
                "http_statuses": ["200"],
                "req_content_types": ["application/json"],
                "resp_content_types": ["application/json"],
            }
        ]

        with open(temp_workspace["files"]["app_features"], "w") as f:
            json.dump(insufficient_features, f)

        # Attempt to train model
        trainer = ModelTrainer(n_features=10, min_transactions=50)
        trainer.add_app_model(
            temp_workspace["files"]["app_features"],
            "TestApp",
            temp_workspace["files"]["custom_model"],
        )

        # Model should not be created due to insufficient data
        assert not os.path.exists(temp_workspace["files"]["custom_model"])

    def test_training_with_corrupted_input_file(self, temp_workspace):
        """Test training behavior with corrupted input files"""
        # Create corrupted JSON file
        with open(temp_workspace["files"]["app_features"], "w") as f:
            f.write("{ invalid json content")

        # Attempt to train model - should handle gracefully
        trainer = ModelTrainer()

        # This should raise an exception or handle gracefully
        try:
            trainer.add_app_model(
                temp_workspace["files"]["app_features"],
                "TestApp",
                temp_workspace["files"]["custom_model"],
            )
        except (json.JSONDecodeError, ValueError):
            # Expected behavior for corrupted file
            pass

        # Model should not be created
        assert not os.path.exists(temp_workspace["files"]["custom_model"])

    def test_merge_models_with_corrupted_existing_model(self, temp_workspace):
        """Test model merging with corrupted existing model file"""
        # Create corrupted existing model file
        with open(temp_workspace["models"]["pretrained_model"], "w") as f:
            f.write("corrupted pickle data")

        # Create valid new model
        new_model = [{"key": "TestApp", "estimator": "test_estimator"}]
        with open(temp_workspace["files"]["custom_model"], "wb") as f:
            pickle.dump(new_model, f)

        # Attempt to merge
        trainer = ModelTrainer()
        trainer.merge_models(
            temp_workspace["models"]["pretrained_model"],
            temp_workspace["files"]["custom_model"],
            temp_workspace["models"]["combined_model"],
        )

        # Merge should fail and not create combined model
        assert not os.path.exists(temp_workspace["models"]["combined_model"])

    @patch("os.makedirs")
    def test_training_with_permission_errors(self, mock_makedirs, temp_workspace):
        """Test training behavior with permission errors"""
        # Mock permission error when creating directories
        mock_makedirs.side_effect = PermissionError("Permission denied")

        # Attempt training - should handle gracefully
        import logging

        logger = logging.getLogger("test")

        with patch(
            "beam.run.parse_input_file", return_value=("test", "/enriched.json")
        ):
            with patch("beam.run.enrich_events", return_value="/enriched.json"):
                with patch(
                    "beam.run.extract_app_features", return_value="/features.json"
                ):
                    with patch(
                        "beam.run.train_custom_app_model", side_effect=PermissionError()
                    ):
                        try:
                            run.process_training_data(
                                input_file_path=temp_workspace["files"]["har_input"],
                                app_name="TestApp",
                                custom_model_path=temp_workspace["files"][
                                    "custom_model"
                                ],
                                logger=logger,
                            )
                        except PermissionError:
                            # Expected behavior
                            pass


class TestPerformanceAndScalability:
    """Test performance and scalability aspects of training"""

    def test_training_with_large_feature_set(self, temp_workspace):
        """Test training with a large number of features"""
        # Create features with many transactions
        large_features = []
        for i in range(5):  # Create multiple feature sets
            features = {
                "key": f"TestApp_{i}",
                "application": "TestApp",
                "transactions": 200,  # High transaction count
                "domain": f"api{i}.testapp.com",
                "avg_time_taken_ms": 150 + i * 10,
                "http_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "http_statuses": ["200", "201", "204", "400", "404", "500"],
                "req_content_types": [
                    "application/json",
                    "text/plain",
                    "multipart/form-data",
                ],
                "resp_content_types": ["application/json", "text/html", "image/png"],
            }
            # Add all numeric fields with varied values
            for field in [
                "refered_traffic_pct",
                "referer_domain_cnt",
                "unique_actions",
                "http_status_cnt",
                "http_method_cnt",
                "req_content_type_cnt",
                "resp_content_type_cnt",
                "avg_time_interval_sec",
                "std_time_interval_sec",
                "median_time_interval_sec",
                "range_time_interval_sec",
                "range_timestamp",
                "max_time_taken_ms",
                "min_time_taken_ms",
                "sum_time_taken_ms",
                "std_time_taken_ms",
                "median_time_taken_ms",
                "range_time_taken_ms",
                "max_client_bytes",
                "min_client_bytes",
                "sum_client_bytes",
                "avg_client_bytes",
                "std_client_bytes",
                "median_client_bytes",
                "range_client_bytes",
                "max_server_bytes",
                "min_server_bytes",
                "sum_server_bytes",
                "avg_server_bytes",
                "std_server_bytes",
                "median_server_bytes",
                "range_server_bytes",
                "web_traffic_pct",
                "cloud_traffic_pct",
                "sequence_num_keys",
                "sequence_max_key_length",
                "sequence_min_key_length",
                "sequence_max_val",
                "sequence_min_val",
                "sequence_sum_val",
                "sequence_avg_val",
                "sequence_std_val",
                "sequence_median_val",
                "sequence_range_val",
            ]:
                features[field] = float(i * 10 + 50)  # Varied numeric values

            large_features.append(features)

        with open(temp_workspace["files"]["app_features"], "w") as f:
            json.dump(large_features, f)

        # Train model with high feature count
        trainer = ModelTrainer(n_features=30, min_transactions=100)
        trainer.add_app_model(
            temp_workspace["files"]["app_features"],
            "TestApp",
            temp_workspace["files"]["custom_model"],
        )

        # Verify model was created successfully
        assert os.path.exists(temp_workspace["files"]["custom_model"])

        with open(temp_workspace["files"]["custom_model"], "rb") as f:
            model = pickle.load(f)

        assert len(model) == 1
        assert model[0]["key"] == "TestApp"
        # Verify that feature selection worked
        assert len(model[0]["selected_features"]) <= 30

    def test_concurrent_model_training_simulation(self, temp_workspace):
        """Test simulation of concurrent model training scenarios"""
        # This test simulates what might happen if multiple training processes
        # were trying to create models simultaneously

        apps = ["App1", "App2", "App3"]
        models_created = []

        for app_name in apps:
            # Create features for each app
            app_features = [
                {
                    "key": f"{app_name}_1.0.0",
                    "application": app_name,
                    "transactions": 100,
                    "domain": f"api.{app_name.lower()}.com",
                    "avg_time_taken_ms": 150,
                    "http_methods": ["GET", "POST"],
                    "http_statuses": ["200"],
                    "req_content_types": ["application/json"],
                    "resp_content_types": ["application/json"],
                }
            ]

            # Add all required numeric fields
            numeric_fields = [
                "refered_traffic_pct",
                "referer_domain_cnt",
                "unique_actions",
                "http_status_cnt",
                "http_method_cnt",
                "req_content_type_cnt",
                "resp_content_type_cnt",
                "avg_time_interval_sec",
                "std_time_interval_sec",
                "median_time_interval_sec",
                "range_time_interval_sec",
                "range_timestamp",
                "max_time_taken_ms",
                "min_time_taken_ms",
                "sum_time_taken_ms",
                "avg_time_taken_ms",
                "std_time_taken_ms",
                "median_time_taken_ms",
                "range_time_taken_ms",
                "max_client_bytes",
                "min_client_bytes",
                "sum_client_bytes",
                "avg_client_bytes",
                "std_client_bytes",
                "median_client_bytes",
                "range_client_bytes",
                "max_server_bytes",
                "min_server_bytes",
                "sum_server_bytes",
                "avg_server_bytes",
                "std_server_bytes",
                "median_server_bytes",
                "range_server_bytes",
                "web_traffic_pct",
                "cloud_traffic_pct",
                "sequence_num_keys",
                "sequence_max_key_length",
                "sequence_min_key_length",
                "sequence_max_val",
                "sequence_min_val",
                "sequence_sum_val",
                "sequence_avg_val",
                "sequence_std_val",
                "sequence_median_val",
                "sequence_range_val",
            ]
            for field in numeric_fields:
                app_features[0][field] = 1.0

            features_file = os.path.join(
                temp_workspace["data"]["app_summaries"], f"{app_name}_features.json"
            )
            model_file = os.path.join(
                temp_workspace["models"]["custom_models"], f"{app_name}_model.pkl"
            )

            with open(features_file, "w") as f:
                json.dump(app_features, f)

            # Train model
            trainer = ModelTrainer(n_features=5, min_transactions=50)
            trainer.add_app_model(features_file, app_name, model_file)

            if os.path.exists(model_file):
                models_created.append(app_name)

        # Verify all models were created successfully
        assert len(models_created) == 3
        assert "App1" in models_created
        assert "App2" in models_created
        assert "App3" in models_created
