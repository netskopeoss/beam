"""Tests for auto-discovery functionality in BEAM"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from beam.run import discover_apps_in_traffic, normalize_app_name


class TestNormalizeAppName:
    """Test cases for normalize_app_name function"""

    def test_normalize_simple_name(self):
        """Test normalization of simple app names"""
        assert normalize_app_name("TestApp") == "testapp"
        assert normalize_app_name("MyApp") == "myapp"
        assert normalize_app_name("CustomCRM") == "customcrm"

    def test_normalize_name_with_spaces(self):
        """Test normalization of app names with spaces"""
        assert (
            normalize_app_name("Microsoft Delivery Optimization")
            == "microsoft_delivery_optimization"
        )
        assert normalize_app_name("My Custom App") == "my_custom_app"
        assert normalize_app_name("Test App 2024") == "test_app_2024"

    def test_normalize_name_with_hyphens(self):
        """Test normalization of app names with hyphens"""
        assert normalize_app_name("Test-App") == "test_app"
        assert normalize_app_name("Custom-CRM-System") == "custom_crm_system"
        assert normalize_app_name("App-Version-1.0") == "app_version_1.0"

    def test_normalize_name_with_mixed_case(self):
        """Test normalization of app names with mixed case"""
        assert normalize_app_name("TestApp") == "testapp"
        assert normalize_app_name("Microsoft TEAMS") == "microsoft_teams"
        assert normalize_app_name("Custom CRM V2") == "custom_crm_v2"

    def test_normalize_name_with_numbers(self):
        """Test normalization of app names with numbers"""
        assert normalize_app_name("App 2024") == "app_2024"
        assert normalize_app_name("Version-1.2.3") == "version_1.2.3"
        assert normalize_app_name("Custom App v1.0") == "custom_app_v1.0"

    def test_normalize_name_with_special_characters(self):
        """Test normalization handles special characters"""
        assert normalize_app_name("App & Tool") == "app_&_tool"
        assert normalize_app_name("Custom.App") == "custom.app"
        assert normalize_app_name("Test (Beta)") == "test_(beta)"

    def test_normalize_empty_and_edge_cases(self):
        """Test normalization with edge cases"""
        assert normalize_app_name("") == ""
        assert normalize_app_name(" ") == "_"
        assert (
            normalize_app_name("   Multiple   Spaces   ") == "___multiple___spaces___"
        )
        assert normalize_app_name("---") == "___"


class TestDiscoverAppsInTraffic:
    """Test cases for discover_apps_in_traffic function"""

    @pytest.fixture
    def temp_enriched_events_file(self):
        """Create temporary enriched events file"""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        yield temp_file.name
        os.unlink(temp_file.name)

    def test_discover_apps_basic(self, temp_enriched_events_file):
        """Test basic app discovery functionality"""
        events = [
            {"application": "Microsoft Delivery Optimization", "other": "data1"},
            {"application": "Microsoft Delivery Optimization", "other": "data2"},
            {"application": "Windows Update Agent", "other": "data3"},
            {"application": "Microsoft CryptoAPI", "other": "data4"},
            {"application": "Microsoft CryptoAPI", "other": "data5"},
        ]

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        result = discover_apps_in_traffic(temp_enriched_events_file, min_transactions=1)

        expected = {
            "Microsoft Delivery Optimization": 2,
            "Windows Update Agent": 1,
            "Microsoft CryptoAPI": 2,
        }

        assert result == expected

    def test_discover_apps_with_min_transactions_filter(
        self, temp_enriched_events_file
    ):
        """Test app discovery with minimum transaction filtering"""
        events = [
            {"application": "App A", "other": "data1"},
            {"application": "App A", "other": "data2"},
            {"application": "App A", "other": "data3"},
            {"application": "App B", "other": "data4"},
            {"application": "App B", "other": "data5"},
            {"application": "App C", "other": "data6"},
        ]

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        # With min_transactions=2, only App A and App B should be returned
        result = discover_apps_in_traffic(temp_enriched_events_file, min_transactions=2)

        expected = {"App A": 3, "App B": 2}

        assert result == expected

    def test_discover_apps_with_min_transactions_filter_strict(
        self, temp_enriched_events_file
    ):
        """Test app discovery with strict minimum transaction filtering"""
        events = (
            [
                {"application": "High Volume App", "other": f"data{i}"}
                for i in range(150)
            ]
            + [
                {"application": "Medium Volume App", "other": f"data{i}"}
                for i in range(75)
            ]
            + [
                {"application": "Low Volume App", "other": f"data{i}"}
                for i in range(25)
            ]
        )

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        # With min_transactions=100, only High Volume App should be returned
        result = discover_apps_in_traffic(
            temp_enriched_events_file, min_transactions=100
        )

        expected = {"High Volume App": 150}

        assert result == expected

    def test_discover_apps_empty_events(self, temp_enriched_events_file):
        """Test app discovery with empty events"""
        events = []

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        result = discover_apps_in_traffic(temp_enriched_events_file, min_transactions=1)

        assert result == {}

    def test_discover_apps_missing_application_field(self, temp_enriched_events_file):
        """Test app discovery with missing application field"""
        events = [
            {"useragent": "Test/1.0", "domain": "test.com"},
            {"application": "Valid App", "other": "data"},
            {"other": "data_no_app"},
            {"application": "Valid App", "other": "data2"},
        ]

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        result = discover_apps_in_traffic(temp_enriched_events_file, min_transactions=1)

        # Should only count events with application field
        expected = {"Valid App": 2}

        assert result == expected

    def test_discover_apps_unknown_application(self, temp_enriched_events_file):
        """Test app discovery with 'Unknown' application"""
        events = [
            {"application": "Known App", "other": "data1"},
            {"application": "Unknown", "other": "data2"},
            {"application": "Known App", "other": "data3"},
            {"application": "Unknown", "other": "data4"},
        ]

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        result = discover_apps_in_traffic(temp_enriched_events_file, min_transactions=1)

        # Unknown apps should be filtered out
        expected = {"Known App": 2}

        assert result == expected

    def test_discover_apps_case_sensitivity(self, temp_enriched_events_file):
        """Test that app discovery is case sensitive"""
        events = [
            {"application": "TestApp", "other": "data1"},
            {"application": "testapp", "other": "data2"},
            {"application": "TESTAPP", "other": "data3"},
            {"application": "TestApp", "other": "data4"},
        ]

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        result = discover_apps_in_traffic(temp_enriched_events_file, min_transactions=1)

        # Should treat different cases as different apps
        expected = {"TestApp": 2, "testapp": 1, "TESTAPP": 1}

        assert result == expected

    def test_discover_apps_with_special_characters(self, temp_enriched_events_file):
        """Test app discovery with special characters in app names"""
        events = [
            {"application": "App & Tool", "other": "data1"},
            {"application": "Custom.App", "other": "data2"},
            {"application": "Test (Beta)", "other": "data3"},
            {"application": "App & Tool", "other": "data4"},
        ]

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        result = discover_apps_in_traffic(temp_enriched_events_file, min_transactions=1)

        expected = {"App & Tool": 2, "Custom.App": 1, "Test (Beta)": 1}

        assert result == expected

    @patch("beam.detector.utils.load_json_file")
    def test_discover_apps_file_loading_error(self, mock_load_json):
        """Test error handling when file loading fails"""
        mock_load_json.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            discover_apps_in_traffic("/nonexistent/file.json", min_transactions=1)

    @patch("beam.detector.utils.load_json_file")
    def test_discover_apps_invalid_json_structure(self, mock_load_json):
        """Test error handling with invalid JSON structure"""
        # Return a string instead of a list
        mock_load_json.return_value = "invalid structure"

        with pytest.raises(TypeError):
            discover_apps_in_traffic("dummy_path.json", min_transactions=1)


class TestIntegrationScenarios:
    """Integration tests for auto-discovery functionality"""

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for integration tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        import shutil

        shutil.rmtree(temp_dir)

    def test_full_discovery_and_normalization_workflow(self, temp_directory):
        """Test complete workflow from discovery to normalization"""
        # Create enriched events file
        events = (
            [
                {"application": "Microsoft Delivery Optimization", "other": f"data{i}"}
                for i in range(120)
            ]
            + [
                {"application": "Custom CRM System", "other": f"data{i}"}
                for i in range(80)
            ]
            + [{"application": "Test-App-Beta", "other": f"data{i}"} for i in range(40)]
            + [
                {"application": "Low Volume App", "other": f"data{i}"}
                for i in range(10)
            ]
        )

        enriched_file = os.path.join(temp_directory, "enriched_events.json")
        with open(enriched_file, "w") as f:
            json.dump(events, f)

        # Discover apps with min_transactions=50
        discovered_apps = discover_apps_in_traffic(enriched_file, min_transactions=50)

        # Should return apps with >= 50 transactions
        expected_apps = {
            "Microsoft Delivery Optimization": 120,
            "Custom CRM System": 80,
        }

        assert discovered_apps == expected_apps

        # Test normalization of discovered app names
        normalized_names = {}
        for app_name in discovered_apps.keys():
            normalized_names[app_name] = normalize_app_name(app_name)

        expected_normalized = {
            "Microsoft Delivery Optimization": "microsoft_delivery_optimization",
            "Custom CRM System": "custom_crm_system",
        }

        assert normalized_names == expected_normalized

    def test_realistic_traffic_scenario(self, temp_directory):
        """Test with realistic traffic patterns"""
        # Simulate realistic network traffic with varying app usage
        events = []

        # Heavy usage apps
        for i in range(500):
            events.append(
                {
                    "application": "Microsoft Outlook",
                    "timestamp": f"2024-01-01T{i % 24:02d}:00:00Z",
                }
            )

        for i in range(300):
            events.append(
                {
                    "application": "Google Chrome",
                    "timestamp": f"2024-01-01T{i % 24:02d}:00:00Z",
                }
            )

        # Medium usage apps
        for i in range(150):
            events.append(
                {"application": "Slack", "timestamp": f"2024-01-01T{i % 24:02d}:00:00Z"}
            )

        for i in range(120):
            events.append(
                {
                    "application": "Zoom Desktop Client",
                    "timestamp": f"2024-01-01T{i % 24:02d}:00:00Z",
                }
            )

        # Low usage apps (below threshold)
        for i in range(30):
            events.append(
                {
                    "application": "Adobe Reader",
                    "timestamp": f"2024-01-01T{i % 24:02d}:00:00Z",
                }
            )

        for i in range(15):
            events.append(
                {
                    "application": "Calculator",
                    "timestamp": f"2024-01-01T{i % 24:02d}:00:00Z",
                }
            )

        # Some unknown/unidentified traffic
        for i in range(25):
            events.append(
                {
                    "application": "Unknown",
                    "timestamp": f"2024-01-01T{i % 24:02d}:00:00Z",
                }
            )

        enriched_file = os.path.join(temp_directory, "realistic_traffic.json")
        with open(enriched_file, "w") as f:
            json.dump(events, f)

        # Test with default minimum transactions (100)
        discovered_apps = discover_apps_in_traffic(enriched_file, min_transactions=100)

        expected_apps = {
            "Microsoft Outlook": 500,
            "Google Chrome": 300,
            "Slack": 150,
            "Zoom Desktop Client": 120,
        }

        assert discovered_apps == expected_apps

        # Test model filename generation for discovered apps
        model_filenames = {}
        for app_name in discovered_apps.keys():
            normalized = normalize_app_name(app_name)
            model_filenames[app_name] = f"{normalized}_model.pkl"

        expected_filenames = {
            "Microsoft Outlook": "microsoft_outlook_model.pkl",
            "Google Chrome": "google_chrome_model.pkl",
            "Slack": "slack_model.pkl",
            "Zoom Desktop Client": "zoom_desktop_client_model.pkl",
        }

        assert model_filenames == expected_filenames

    def test_edge_case_app_names(self, temp_directory):
        """Test discovery and normalization with edge case app names"""
        events = []

        # Apps with problematic names
        problematic_names = [
            "App with   Multiple   Spaces",
            "App-With-Many-Hyphens",
            "App.With.Dots",
            "App (with parentheses)",
            "App & Symbols!",
            "CamelCaseAppName",
            "ALL_CAPS_APP",
            "mixed_Case-App.Name",
        ]

        # Generate enough events for each app
        for app_name in problematic_names:
            for i in range(110):  # Above minimum threshold
                events.append({"application": app_name, "other": f"data{i}"})

        enriched_file = os.path.join(temp_directory, "edge_case_apps.json")
        with open(enriched_file, "w") as f:
            json.dump(events, f)

        discovered_apps = discover_apps_in_traffic(enriched_file, min_transactions=100)

        # All apps should be discovered
        assert len(discovered_apps) == len(problematic_names)
        for app_name in problematic_names:
            assert app_name in discovered_apps
            assert discovered_apps[app_name] == 110

        # Test that normalization produces valid filenames
        for app_name in discovered_apps.keys():
            normalized = normalize_app_name(app_name)

            # Normalized names should be lowercase
            assert (
                normalized.islower() or not normalized.isalpha()
            )  # Allow numbers/symbols

            # Should not contain spaces (replaced with underscores)
            assert " " not in normalized

            # Should be a valid filename component (no path separators)
            assert "/" not in normalized
            assert "\\" not in normalized

        # Verify specific normalizations
        assert (
            normalize_app_name("App with   Multiple   Spaces")
            == "app_with___multiple___spaces"
        )
        assert normalize_app_name("App-With-Many-Hyphens") == "app_with_many_hyphens"
        assert normalize_app_name("CamelCaseAppName") == "camelcaseappname"
        assert normalize_app_name("ALL_CAPS_APP") == "all_caps_app"


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for auto-discovery"""

    def test_normalize_app_name_with_none(self):
        """Test normalize_app_name with None input"""
        with pytest.raises(AttributeError):
            normalize_app_name(None)

    def test_normalize_app_name_with_non_string(self):
        """Test normalize_app_name with non-string input"""
        with pytest.raises(AttributeError):
            normalize_app_name(123)

        with pytest.raises(AttributeError):
            normalize_app_name(["list", "input"])

    @pytest.fixture
    def temp_enriched_events_file(self):
        """Create temporary enriched events file"""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        yield temp_file.name
        os.unlink(temp_file.name)

    def test_discover_apps_with_malformed_events(self, temp_enriched_events_file):
        """Test discovery with malformed event objects"""
        events = [
            {"application": "Valid App", "other": "data1"},
            None,  # Malformed event
            {"application": "Valid App", "other": "data2"},
            {},  # Empty event
            {"application": "Valid App", "other": "data3"},
            "invalid_event_type",  # Wrong type
            {"application": "Valid App", "other": "data4"},
        ]

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        # Should handle malformed events gracefully
        result = discover_apps_in_traffic(temp_enriched_events_file, min_transactions=1)

        # Should only count valid events
        expected = {"Valid App": 4}

        assert result == expected

    def test_discover_apps_with_very_large_dataset(self, temp_enriched_events_file):
        """Test discovery performance with large dataset"""
        # Create a large dataset
        events = []
        for i in range(10000):  # 10k events
            app_name = f"App{i % 100}"  # 100 different apps
            events.append({"application": app_name, "event_id": i})

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        # Should handle large datasets efficiently
        result = discover_apps_in_traffic(
            temp_enriched_events_file, min_transactions=50
        )

        # Each app should have 100 events (10000/100)
        assert len(result) == 100
        for app_name, count in result.items():
            assert count == 100
            assert app_name.startswith("App")

    def test_discover_apps_zero_min_transactions(self, temp_enriched_events_file):
        """Test discovery with zero minimum transactions"""
        events = [
            {"application": "App A", "other": "data1"},
            {"application": "App B", "other": "data2"},
        ]

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        result = discover_apps_in_traffic(temp_enriched_events_file, min_transactions=0)

        expected = {"App A": 1, "App B": 1}

        assert result == expected

    def test_discover_apps_negative_min_transactions(self, temp_enriched_events_file):
        """Test discovery with negative minimum transactions"""
        events = [{"application": "App A", "other": "data1"}]

        with open(temp_enriched_events_file, "w") as f:
            json.dump(events, f)

        # Should treat negative as zero
        result = discover_apps_in_traffic(
            temp_enriched_events_file, min_transactions=-5
        )

        expected = {"App A": 1}

        assert result == expected
