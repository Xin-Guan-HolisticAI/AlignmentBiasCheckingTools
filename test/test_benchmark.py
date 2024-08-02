import csv
import os
import unittest
from benchmark_building import BenchmarkBuilding

class TestModelGeneration(unittest.TestCase):

    def compare_files(self, file_path_one, file_path_two):

        try:
            with open(file_path_one, 'r') as file1, \
                open(file_path_two, 'r') as file2:

                # Read lines from both files
                lines1 = file1.readlines()
                lines2 = file2.readlines()

                # Check if lengths are different
                self.assertEqual(len(lines1), len(lines2), "Files have different number of lines")

                # Compare lines
                for line1, line2 in zip(lines1, lines2):
                    self.assertEqual(line1.strip(), line2.strip(), "Files have matching lines")

        except IOError as e:
            self.fail(f"An IOError occurred: {e}")

        except AssertionError as e:
            self.fail(f"Assertion failed: {e}")


    def compare_csv_files(self, csv_one, csv_two):
        try:
            with open(csv_one, 'r', newline='', encoding='utf-8') as file1, \
                open(csv_two, 'r', newline='', encoding='utf-8') as file2:

                # Create CSV readers
                reader1 = csv.reader(file1)
                reader2 = csv.reader(file2)

                # Read all rows from both files
                rows1 = list(reader1)
                rows2 = list(reader2)

                # Check if lengths are different
                if len(rows1) != len(rows2):
                    raise AssertionError("Files have different number of rows")

                # Compare rows
                for row1, row2 in zip(rows1, rows2):
                    if row1 != row2:
                        raise AssertionError(f"Rows do not match: {row1} != {row2}")

        except IOError as e:
            raise AssertionError(f"An IOError occurred: {e}")

        except AssertionError as e:
            raise AssertionError(f"Assertion failed: {e}")
    def test_category_pipelines(self):
        '''Tests to make sure that saving_location works for both nonexisting directories and existing directories.'''
        config_1 = {

            #Tests where saving location differs (both default and customized). Also changed keyword_number to 100.
            'keyword_finder':{
                'method': 'hyperlinks_on_wiki',
                'page_name_or_link': "List of Persian violinists",
                'saving_location': f'test/profession_violinist_key_words.json',
            },
            'scrap_area_finder':{
                'saving_location': f'data/customized/scrap_area/profession_violinist_scrap_area.json'
            },
            'scrapper':{
                'saving_location': f'test/profession_violinist_scrapped.json'
            },
            'prompt_maker':{
                'saving_location': f'test/profession_violinist_prompt_area.csv'
            }

        }

        #BenchmarkBuilding.category_pipeline('profession', 'violinist', config_1)
        self.compare_files("test/profession_violinist_key_words.json", "test_results/profession_violonist_key_words_expected.json")
        self.compare_files("data/customized/scrap_area/profession_violinist_scrap_area.json", "test_results/profession_violinist_scrap_area_expected.json")
        self.compare_files("test/profession_violinist_scrapped.json", "test_results/profession_violonist_scrapped_expected.json")
        self.compare_csv_files("test/profession_violinist_prompt_area.csv", "test_results/profession_violinist_prompt_area_expected.csv")


        config_2 = {

            'scrapper':{
                'require': False
            },
            'prompt_maker':{
                'saving_location': f'test/config_2_prompt_area.csv'
            }

        }
        #For sports football, as scrapper is false, no file should be created.
        '''BenchmarkBuilding.category_pipeline('sports', 'football', config_2)
        self.assertFalse(os.path.isfile("data/customized/keywords/sports_football_keywords.json"))
        self.assertFalse(os.path.isfile("data/customized/scrap_area/sports_football_scrap_area.json"))
        self.assertFalse(os.path.isfile("data/customized/scrapped_sentences/sports_football_scrapped_sentences.json"))
        self.assertFalse(os.path.isfile("test/config_2_prompt_area.csv"))
        
        #Even though scrapper is false, since judaism already exists, the prompt area should still be created
        #BenchmarkBuilding.category_pipeline('religion', 'judaism', config_2)
        self.compare_csv_files("test/config_2_prompt_area.csv", "test_results/religion_judaism_prompt_area_expected.csv")

        '''
        #tests what happens when keyword number is great than the length of similar words
        config_3 = {
            'keyword_finder':{
                'keyword_number': 15
            },
            'scrap_area_finder':{
                'method': 'local_files'
            },
            'scrapper':{
                'method': 'local_files'
            },
            'prompt_maker':{
                'saving_location': f'test/atheism_prompt_area.csv'
            }
        }
        BenchmarkBuilding.category_pipeline('religion', 'atheism', config_3)
        self.compare_files("data/customized/keywords/religion_atheism_keywords.json", "test_results/religion_atheism_keywords_expected.json")
        self.compare_files("data/customized/scrap_area/religion_atheism_scrap_area.json", "test_results/religion_atheism_scrap_area_expected.json")
        self.compare_files("data/customized/scrapped_sentences/religion_atheism_scrapped_sentences.json", "test_results/religion_atheism_scrapped_sentences_expected.json")
        self.compare_csv_files("test/atheism_prompt_area.csv", "test_results/atheism_prompt_area_expected.csv")



if __name__ == '__main__':
    unittest.main()