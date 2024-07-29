import unittest
import analytics
import pandas as pd
from abcData import abcData
from assistants import OllamaModel

#example of how to use unitest to test the model generation function
class TestModelGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('initiating the model generation function')

        llama = OllamaModel(model_name='continuation',
                            system_prompt='Continue to finish the following part of the sentence and output nothing else: ')
        cls.generation_function = llama.invoke
        print('Initiation complete')

        print('initiating the benchmark data')
        data = abcData.load_file(category='christianity',
                                 domain='religion',
                                 data_tier='split_sentences',
                                 file_path='data/customized/split_sentences/religion_christianity_split_sentences.csv')
        cls.benchmark = data.sub_sample(4)
        print('Initiation complete')

    def test_default(self):
        print('testing model generation')
        model_generator = analytics.ModelGenerator(self.benchmark)
        generated_data = model_generator.generate(self.generation_function)

        self.assertIsNotNone(generated_data)  # Ensure generated data is not None
        self.assertIn('LLM', generated_data.columns)  # Ensure 'llm' column is present


    def test_generation_name(self):
        print('testing model generation')
        model_generator = analytics.ModelGenerator(self.benchmark)
        generated_data = model_generator.generate(self.generation_function, generation_name='Test')

        self.assertIsNotNone(generated_data)  # Ensure generated data is not None
        self.assertIn('Test', generated_data.columns)  # Ensure 'Test' column is present

    def test_task_prefix(self):
        print('testing model generation')
        model_generator = analytics.ModelGenerator(self.benchmark)
        generated_data = model_generator.generate(self.generation_function, task_prefix='sentence_completion')

        self.assertIsNotNone(generated_data)
        self.assertIn('LLM', generated_data.columns)  # Ensure 'LLM' column is present


if __name__ == '__main__':
    unittest.main()