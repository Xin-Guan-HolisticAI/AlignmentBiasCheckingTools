import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import json
from abcData import abcData 

class TestLoadFile(unittest.TestCase):
    def setUp(self):
        self.domain = "politics"
        self.category = "fascism"
        self.data_tier = "keywords"
        self.file_path = "dummy_file.json"

    @patch("abcData.abcData.check_format")
    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_load_file_json(self, mock_file, mock_check_format):
        mock_check_format.return_value = None
        instance = abcData.load_file(self.domain, self.category, self.data_tier, self.file_path)
        self.assertIsInstance(instance, abcData)
        self.assertEqual(instance.data, {"key": "value"})
        mock_file.assert_called_once_with(self.file_path, "r")
        mock_check_format.assert_called_once_with(self.data_tier, {"key": "value"})

    @patch("abcData.pd.read_csv")
    @patch("abcData.abcData.check_format")
    def test_load_file_csv(self, mock_check_format, mock_read_csv):
        self.data_tier = "split_sentences"
        mock_df = pd.DataFrame({"column": ["value"]})
        mock_read_csv.return_value = mock_df
        mock_check_format.return_value = None
        instance = abcData.load_file(self.domain, self.category, self.data_tier, self.file_path)
        self.assertIsInstance(instance, abcData)
        self.assertTrue(instance.data.equals(mock_df))
        mock_read_csv.assert_called_once_with(self.file_path)
        mock_check_format.assert_called_once_with(self.data_tier, mock_df)

    @patch("builtins.open", new_callable=mock_open)
    def test_load_file_ioerror(self, mock_file):
        # make sure IO error is correctly processed
        mock_file.side_effect = IOError
        instance = abcData.load_file(self.domain, self.category, self.data_tier, self.file_path)
        self.assertIsNone(instance)

    @patch("builtins.open", new_callable=mock_open, read_data='invalid_json_file')
    def test_load_file_decode_error(self, mock_file):
        # test when invalid json file is passed through as arg
        instance = abcData.load_file(self.domain, self.category, self.data_tier, self.file_path)
        self.assertIsNone(instance)

    @patch("abcData.abcData.check_format")
    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_load_file_assertion_error(self, mock_file, mock_check_format):
        # make sure assertion error is correctly processed
        mock_file.side_effect = AssertionError
        instance = abcData.load_file(self.domain, self.category, self.data_tier, self.file_path)
        self.assertIsNone(instance)

if __name__ == "__main__":
    unittest.main()
