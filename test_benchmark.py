import csv
import importlib
import json
import os
import pkgutil
import shutil
import sys
import unittest
import pandas as pand
from alignmentbiascheckingtools.test.benchmark_building import BenchmarkBuilder
import openai
    
class TestModelGeneration(unittest.TestCase):
    config_1 = {}
    config_2 = {}
    config_3 = {}
    config_4 = {}
    config_5 = {}
    config_6 = {}
    config_7 = {}
    config_8 = {}
    config_9 = {}
    config_10 = {}

    @classmethod
    def setUpClass(cls):
        cls.config_1 = {

            'keyword_finder':{
                'require': False
            },
            'scrap_area_finder':{
                'saving_location': "tests/test_results/profession_technician_scrap_area.json"
            },
            'scrapper':{
                'saving_location': "tests/test_results/profession_technician_scrapped.json"
            },
            'prompt_maker':{
                'saving_location': "tests/test_results/profession_technician_prompts.csv"
            }

        }

        cls.config_2 = {

            'keyword_finder':{
                'saving_location': "tests/test_results/profession_basketball_keywords.json", 
                'max_adjustment': -1
            },
            'scrap_area_finder':{
                'require': False
            },
            'scrapper':{
                'saving_location': "tests/test_results/profession_basketball_scrapped.json"
            },
            'prompt_maker':{
                'saving_location': "tests/test_results/profession_basketball_prompts.csv"
            }

        }

        cls.config_3 = {
            'keyword_finder':{
                'saving_location': "tests/test_results/profession_basketball_keywords.json", 
                'max_adjustment': -1
            },
            'scrap_area_finder':{
                'saving_location': "tests/test_results/profession_basketball_scrap_area.json"
            },
            'scrapper':{
                'require': False
            },
            'prompt_maker':{
                'saving_location': "tests/test_results/profession_basketball_prompts.csv"
            }

        }

        cls.config_4 = {
            'keyword_finder':{
                'saving_location': "tests/test_results/profession_basketball_keywords.json", 
                'max_adjustment': -1
            },
            'scrap_area_finder':{
                'saving_location': "tests/test_results/profession_basketball_scrap_area.json"
            },
            'scrapper':{
                'saving_location': "tests/test_results/profession_basketball_scrapped.json"
            },
            'prompt_maker':{
                'require': False
            }

        }

        cls.config_5 = {
            'keyword_finder':{
                'method': "llm_inquiries",
                'llm_info': {},
                'saving_location': 'tests/test_results/religion_atheism_keywords.json',
            },
            'scrap_area_finder':{
                    'saving_location': "tests/test_results/religion_atheism_scrap_area.json"
            },
            'scrapper':{
                    'saving_location': "tests/test_results/religion_atheism_scrapped.json"
            },
            'prompt_maker':{
                    'saving_location': "tests/test_results/religion_atheism_prompts.csv"
            }  
        }

        cls.config_6 = {
            'keyword_finder':{
                'method': "hyperlinks_on_wiki",
                'hyperlinks_info': {},
                'saving_location': 'tests/test_results/movie_shrek_keywords.json'
            },
            'scrap_area_finder':{
                'require': False
            },
            'scrapper':{
               'require': False
            },
            'prompt_maker':{
                'require': False
            }  
        }

        cls.config_7 = {
            'keyword_finder':{
                'reading_location': 'tests/reading_files/gender_woman_keywords.json',
                'saving': False
            },
           'scrap_area_finder':{
                'saving_location': "tests/test_results/gender_woman_scrap_area.json"
            },
            'scrapper':{
                'saving_location': "tests/test_results/gender_woman_scrapped.json"
            },
            'prompt_maker':{
                'saving_location': "tests/test_results/gender_woman_prompts.csv"
            }   
        }

        cls.config_8 = {
            'keyword_finder':{
                'saving_location': 'tests/test_results/political-ideology_anarchism_keywords.json'
            },
           'scrap_area_finder':{
                'require': False,
                'reading_location': "tests/reading_files/political-ideology_anarchism_scrap_area.json",
                'saving_location': 'tests/test_results/political-ideology_anarchism_scrap_area.json'
            },
            'scrapper':{
                'saving_location': "tests/test_results/political-ideology_anarchism_scrapped.json"
            },
            'prompt_maker':{
                'saving_location': "tests/test_results/political-ideology_anarchism_prompts.csv"
            }   
        }

        cls.config_9 = {
            'keyword_finder':{
                'saving_location': 'tests/test_results/political-ideology_capitalism_keywords.json'
            },
           'scrap_area_finder':{
                'saving_location': 'tests/test_results/political-ideology_capitalism_scrap_area.json'
            },
            'scrapper':{
                'require': False,
                'reading_location': "tests/reading_files/political-ideology_capitalism_scrapped_sentences.json",
                'saving_location': "tests/test_results/political-ideology_capitalism_scrapped.json"
            },
            'prompt_maker':{
                'saving_location': "tests/test_results/political-ideology_capitalism_prompts.csv"
            }   
        }

        cls.config_10 = {
            'keyword_finder':{
                'saving_location': 'tests/test_results/religion_hinduism_keywords.json'
            },
           'scrap_area_finder':{
                'method': 'local_files',
                'local_file': 'tests/local_files',
                'saving_location': 'tests/test_results/religion_hinduism_scrap_area.json'
            },
            'scrapper':{
                'method': 'local_files',
                'saving_location': "tests/test_results/religion_hinduism_scrapped.json"
            },
            'prompt_maker':{
                'saving_location': "tests/test_results/religion_hinduism_prompts.csv"
            }   
        }

    #compares both regular files and csv files
    def compare(self, name, file_one, file_two):
        def compare_files(file_path_one, file_path_two):

            try:
                with open(file_path_one, 'r') as file1, \
                    open(file_path_two, 'r') as file2:

                    # Read lines from both files
                    lines1 = file1.readlines()
                    lines2 = file2.readlines()

                    if len(lines1) != len(lines2):
                        raise AssertionError("Files have different number of rows")

                    # Check if lengths are different
                    # Compare lines
                    for line1, line2 in zip(lines1, lines2):
                        if line1 != line2:
                            raise AssertionError(f"Rows do not match: {line1} != {line2}")

            except IOError as e:
                raise AssertionError(f"An IOError occurred: {e}")

            except AssertionError as e:
                raise AssertionError(f"Assertion failed: {e}")


        def compare_csv_files(csv_one, csv_two):
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
            
        if name == "compare_files":
            compare_files(file_one, file_two)
        elif name == "compare_csv_files":
            compare_csv_files(file_one, file_two)
        else:
            self.assertRaises("Need to enter either compare_files or compare_csv_files")

    #Tests when keyword_require is false. Also tests saving location for scrap_area, scrapped, prompts. Tests for file that doesn't exist, profession/basketball_player and file that does exist, profession/technician.
    def test_one(self):
        with self.assertRaises(ValueError) as cm:
            BenchmarkBuilder.category_pipeline('profession', 'basketball_player', self.config_1)
        
        self.assertEqual(str(cm.exception), "Unable to read keywords from tests/data/customized/keywords/profession_basketball_player_keywords.json. Can't scrap area.")

        BenchmarkBuilder.category_pipeline('profession', 'technician', self.config_1)
        self.compare("compare_files", "tests/test_results/profession_technician_scrap_area.json", "tests/expected_results/profession_technician_scrap_area_expected.json")
        self.compare("compare_files", "tests/test_results/profession_technician_scrapped.json", "tests/expected_results/profession_technician_scrapped_expected.json")
        self.compare("compare_csv_files", "tests/test_results/profession_technician_prompts.csv", "tests/expected_results/profession_technician_prompts_expected.csv")

        print("test one success")
        if os.path.exists("tests/test_results"):
            shutil.rmtree("tests/test_results")

    #Tests when scrap_area_require is false. Also tests saving_location for keyword, scrapped, and prompts. Also tests max_adjustment.  Tests for file that doesn't exist, profession/basketball_player and file that does exist, profession/technician
    def test_two(self):
        with self.assertRaises(ValueError) as cm:
            BenchmarkBuilder.category_pipeline('profession', 'basketball_player', self.config_2)
        self.assertEqual(str(cm.exception), "Unable to scrap areas from tests/data/customized/scrap_area/profession_basketball_player_scrap_area.json. Can't use scrapper.")

        self.config_2['keyword_finder']['saving_location'] = "tests/test_results/profession_technician_keywords.json"
        self.config_2['scrapper']['saving_location'] = "tests/test_results/profession_technician_scrapped.json"
        self.config_2['prompt_maker']['saving_location'] = "tests/test_results/profession_technician_prompts.csv"
        BenchmarkBuilder.category_pipeline('profession', 'technician', self.config_2)
        self.compare("compare_files", "tests/test_results/profession_technician_keywords.json", "tests/expected_results/profession_technician_keywords_expected.json")
        self.compare("compare_files", "tests/test_results/profession_technician_scrapped.json", "tests/expected_results/profession_technician_scrapped_expected.json")
        self.compare("compare_csv_files", "tests/test_results/profession_technician_prompts.csv", "tests/expected_results/profession_technician_prompts_expected.csv")

        print("test two success")
        if os.path.exists("tests/test_results"):
            shutil.rmtree("tests/test_results")

    #Tests when scrapped is false. Also tests saving_location for keyword, scrap_area, and prompts. Also tests max_adjustment. Tests for file that doesn't exist, profession/basketball_player and file that does exist, profession/technician
    def test_three(self):
        with self.assertRaises(ValueError) as cm:
            BenchmarkBuilder.category_pipeline('profession', 'basketball_player', self.config_3)
        
        print(cm.exception)
        self.assertEqual(str(cm.exception), "Unable to scrap from tests/data/customized/scrapped_sentences/profession_basketball_player_scrapped_sentences.json. Can't make prompts.")

        self.config_3['keyword_finder']['saving_location'] = "tests/test_results/profession_technician_keywords.json"
        self.config_3['scrap_area_finder']['saving_location'] = "tests/test_results/profession_technician_scrap_area.json"
        self.config_3['prompt_maker']['saving_location'] = "tests/test_results/profession_technician_prompts.csv"
        BenchmarkBuilder.category_pipeline('profession', 'technician', self.config_3)
        self.compare("compare_files", "tests/test_results/profession_technician_keywords.json", "tests/expected_results/profession_technician_keywords_expected.json")
        self.compare("compare_files", "tests/test_results/profession_technician_scrap_area.json", "tests/expected_results/profession_technician_scrap_area_expected.json")
        self.compare("compare_csv_files", "tests/test_results/profession_technician_prompts.csv", "tests/expected_results/profession_technician_prompts_expected.csv")

        print("test three success")

        if os.path.exists("tests/test_results"):
            shutil.rmtree("tests/test_results")

    #Tests when prompts is false. Also tests saving_location for keyword, scrap_area, and scrapped. Also tests max_adjustment. Tests for file that doesn't exist, profession/basketball_player and file that does exist, profession/technician
    def test_four(self):
        BenchmarkBuilder.category_pipeline('profession', 'basketball_player', self.config_4)

        self.compare("compare_files", "tests/test_results/profession_basketball_keywords.json", "tests/expected_results/profession_basketball_keywords_expected.json")
        self.compare("compare_files", "tests/test_results/profession_basketball_scrap_area.json", "tests/expected_results/profession_basketball_scrap_area_expected.json")
        self.compare("compare_files", "tests/test_results/profession_basketball_scrapped.json", "tests/expected_results/profession_basketball_scrapped_expected.json")
        assert not os.path.isfile('tests/test_results/profession_basketball_prompts.csv'), f"File tests/test_results/profession_basketball_prompts.csv unexpectedly exists"

        print("test four success")

        if os.path.exists("tests/test_results"):
            shutil.rmtree("tests/test_results")

    #Tests when key method is llm_inquiries. 
    def test_five(self):
        openai.api_key = 'PUT YOUR API KEY HERE'

        #I used openai version openai-0.28.0 for this. Feel free to use another model for testing!
        def generate_with_chat(prompt, model="gpt-4"):
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # Access the content from the response
            return response.choices[0].message['content']

        def generation_function(prompt):
            return generate_with_chat(prompt)

        #format = self, n_run=20,n_keywords=20, generation_function=None, model_name=None, embedding_model=None, show_progress=True
        llm_dict = {'generation_function': generation_function}
        self.config_5['keyword_finder']['llm_info'] = llm_dict

        try:
            # Attempt to run the command
            BenchmarkBuilder.category_pipeline('religion', 'atheism', self.config_5)
            command_worked = True
        except Exception as e:
            # If an exception is raised, the command did not work
            command_worked = False
            error_message = str(e)

        # Assert that the command worked
        assert command_worked, f"Command failed with error: {error_message}"
        assert os.path.isfile('tests/test_results/religion_atheism_keywords.json'), f"File tests/test_results/religion_atheism_keywords.csv does not exists"
        assert os.path.isfile('tests/test_results/religion_atheism_scrap_area.json'), f"File tests/test_results/religion_atheism_scrap_area.csv does not exists"
        assert os.path.isfile('tests/test_results/religion_atheism_scrapped.json'), f"File tests/test_results/religion_atheism_scrapped.json does not exists"
        assert os.path.isfile('tests/test_results/religion_atheism_prompts.csv'), f"File tests/test_results/religion_atheism_prompts.csv does not exists"

        print("test five success")

        if os.path.exists("tests/test_results"):
            shutil.rmtree("tests/test_results")

    #Tests when key method is hyperlinks. Tests all three types of hyperlinks.
    def test_six(self):

        #self, format='Paragraph', link=None, page_name=None, name_filter=False,col_info=None, depth=None, source_tag='default', max_keywords = None):
        para_dict = {'format': 'Paragraph', 'link':'https://en.wikipedia.org/wiki/List_of_Shrek_(franchise)_characters', 'name_filter':True}
        self.config_6['keyword_finder']['hyperlinks_info'] = para_dict
        BenchmarkBuilder.category_pipeline('movie', 'shrek', self.config_6)

        table_dict = {'format': 'Table', 'link':'https://en.wikipedia.org/wiki/List_of_female_ambassadors_of_the_United_States', 'col_info':  [{'table_num': 2, 'column_name':['Name']}]}
        self.config_6['keyword_finder']['hyperlinks_info'] = table_dict
        self.config_6['keyword_finder']['saving_location'] = "tests/test_results/profession_female_ambassadors_keywords.json"
        BenchmarkBuilder.category_pipeline('profession', 'female_ambassadors', self.config_6)

        nested_dict = {'format': 'Nested', 'link':'https://en.wikipedia.org/wiki/Category:Grasses_of_Africa', 'depth': 2}
        self.config_6['keyword_finder']['hyperlinks_info'] = nested_dict
        self.config_6['keyword_finder']['saving_location'] = "tests/test_results/plants_grass_keywords.json"
        BenchmarkBuilder.category_pipeline('plants', 'grass', self.config_6)

        self.compare("compare_files", "tests/test_results/movie_shrek_keywords.json", "tests/expected_results/movie_shrek_keywords_expected.json")
        self.compare("compare_files", "tests/test_results/profession_female_ambassadors_keywords.json", "tests/expected_results/profession_female_ambassadors_keywords_expected.json")
        self.compare("compare_files", "tests/test_results/plants_grass_keywords.json", "tests/expected_results/plants_grass_keywords_expected.json")

        print("test six success")

        if os.path.exists("tests/test_results"):
            shutil.rmtree("tests/test_results")


    #Tests when reading location and require=false for keywords.
    def test_seven(self):
        BenchmarkBuilder.category_pipeline('gender', 'woman', self.config_7)

        self.compare("compare_files", "tests/test_results/gender_woman_scrap_area.json", "tests/expected_results/gender_woman_scrap_area_expected.json")
        self.compare("compare_files", "tests/test_results/gender_woman_scrapped.json", "tests/expected_results/gender_woman_scrapped_expected.json")
        self.compare("compare_csv_files", "tests/test_results/gender_woman_prompts.csv", "tests/expected_results/gender_woman_prompts_expected.csv")

        print("test seven success")

        if os.path.exists("tests/test_results"):
            shutil.rmtree("tests/test_results")
    

    #Tests when reading location and require=false for scrap_area.
    def test_eight(self):
        BenchmarkBuilder.category_pipeline('political-ideology', 'anarchism', self.config_8)

        self.compare("compare_files", "tests/test_results/political-ideology_anarchism_keywords.json", "tests/expected_results/political-ideology_anarchism_keywords_expected.json")
        self.compare("compare_files", "tests/test_results/political-ideology_anarchism_scrapped.json", "tests/expected_results/political-ideology_anarchism_scrapped_expected.json")
        self.compare("compare_csv_files", "tests/test_results/political-ideology_anarchism_prompts.csv", "tests/expected_results/political-ideology_anarchism_prompts_expected.csv")

        print("test eight success")

        if os.path.exists("tests/test_results"):
            shutil.rmtree("tests/test_results")

    #Tests when reading location and require=false for scrapped.    
    def test_nine(self):
        BenchmarkBuilder.category_pipeline('political-ideology', 'capitalism', self.config_9)

        self.compare("compare_files", "tests/test_results/political-ideology_capitalism_keywords.json", "tests/expected_results/political-ideology_capitalism_keywords_expected.json")
        self.compare("compare_files", "tests/test_results/political-ideology_capitalism_scrap_area.json", "tests/expected_results/political-ideology_capitalism_scrap_area_expected.json")
        self.compare("compare_csv_files", "tests/test_results/political-ideology_capitalism_prompts.csv", "tests/expected_results/political-ideology_capitalism_prompts_expected.csv")
        
        print("test nine success")

        if os.path.exists("tests/test_results"):
            shutil.rmtree("tests/test_results")

    #Tests local file for scrap_area and scrapped.
    def test_ten(self):
        BenchmarkBuilder.category_pipeline('religion', 'hinduism', self.config_10) 

        def sort_keys(file_path_one, file_path_two):

            with open(file_path_one, 'r', encoding='utf-8') as file_one:
                data_one = json.load(file_one)

            with open(file_path_two, 'r', encoding='utf-8') as file_two:
                data_two = json.load(file_two)

            # Convert JSON data to strings with sorted keys
            json1 = json.dumps(data_one, sort_keys=True)
            json2 = json.dumps(data_two, sort_keys=True)

            self.assertEqual(json1, json2, "The JSON files are not equal.")

        sort_keys("tests/test_results/religion_hinduism_keywords.json", "tests/expected_results/religion_hinduism_keywords_expected.json")
        sort_keys("tests/test_results/religion_hinduism_scrap_area.json", "tests/expected_results/religion_hinduism_scrap_area_expected.json")

        assert os.path.isfile('tests/test_results/religion_hinduism_scrapped.json'), f"File tests/test_results/religion_hinduism_scrapped.json does not exists"   
        assert os.path.isfile('tests/test_results/religion_hinduism_prompts.csv'), f"File tests/test_results/religion_hinduism_prompts.csv does not exists"   
        print("test ten success")

        if os.path.exists("tests/test_results"):
            shutil.rmtree("tests/test_results")
  
    @classmethod
    def tearDownClass(cls):
        if os.path.exists("test"):
            shutil.rmtree("test")

if __name__ == '__main__':
    unittest.main()