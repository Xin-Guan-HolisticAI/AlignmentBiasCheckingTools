import os
import unittest
from abcData import abcData
import pandas as pd
from abcData import abcData

class TestModelGeneration(unittest.TestCase):

    def test_save(self):
        #Test if data with no file_path works
        no_file_data = abcData(category='rap',
                                 domain='religion',
                                 data_tier='scrapped_sentences')
        
        file_path = abcData.save(no_file_data)
        self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist")

        #Test if data_tier = split_sentence works
        split_sentences = abcData.load_file(category='woman',
                                 domain='gender',
                                 data_tier='scrapped_sentences',
                                 file_path='data/customized/scrapped_sentences/gender_woman_scrapped_sentences.json')
        
        file_path = abcData.save(split_sentences)
        self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist")

        #Test if other data_tier works
        other_data_tier = abcData(category='rap',
                                 domain='music',
                                 data_tier='scrapped_sentences')
        file_path = abcData.save(other_data_tier)
        self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist")

    def test_sub_sample(self):
        #Testing if format that isn't abcData works
        bad_data = abcData.load_file(category='christianity',
                                 domain='religion',
                                 data_tier='scrapped_sentences',
                                 file_path='data/customized/scrapped_sentences/gender_woman_scrapped_sentences.json')
        response = abcData.sub_sample(bad_data)
        self.assertIsNone(response)  # Ensure generated data is not None

        #Testing if regular split_sentences data works
        data = abcData.load_file(category='christianity',
                                 domain='religion',
                                 data_tier='split_sentences',
                                 file_path='data/customized/split_sentences/religion_atheism_split_sentences.csv')
        response = abcData.sub_sample(data)
        self.assertNotIn('keywords_containment', response.columns)

        #Checks to see if old data is the same as the new data when keywords_containment doesn't exist.
        data_two = abcData.load_file(category='christianity',
                                 domain='religion',
                                 data_tier='split_sentences',
                                 file_path='data/customized/split_sentences/politics_nationalism_split_sentences.csv')
        data_two.data.drop(['keywords_containment'], axis=1, inplace=True)
        old_data = data_two.data
        response = abcData.sub_sample(data_two)
        self.assertTrue(response.isin(old_data).all().all())

    def test_model_generation(self):
        # Apply the method

        def generate_response(prompt):
            return f"Response to: {prompt}"
        
        data = abcData.load_file(category='christianity',
                                 domain='religion',
                                 data_tier='split_sentences',
                                 file_path='data/customized/split_sentences/religion_atheism_split_sentences.csv')
        
        result_df = abcData.model_generation(data, generate_response)
        self.assertIsNotNone(result_df)  # Ensure generated data is not None

        # Check that the new column exists and has the expected output
        self.assertIn('generated_output', result_df.columns)
        self.assertEqual(result_df['generated_output'].iloc[0], 'Response to: Atheism , the absence of belief in deities , offers several potential benefits')
        self.assertEqual(result_df['generated_output'].iloc[31], 'Response to: Focus on Human and Natural Issues Human - Centered Approach Atheism often leads to a focus')
        self.assertEqual(result_df['generated_output'].iloc[862], 'Response to: Warm regards ,     explain in long - winding exaggerated positive -')

if __name__ == '__main__':
    unittest.main()