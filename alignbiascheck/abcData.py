import json
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations, permutations

tqdm.pandas()


class abcData:

    tier_order = {value: index for index, value in
                  enumerate(['keywords', 'scrap_area', 'scrapped_sentences', 'split_sentences', 'questions'])}

    default_scrap_area_format = [{
        "source_tag": "default",
        "scrap_area_type": "unknown",
        "scrap_area_specification": []
    }]

    default_keyword_metadata = {
        "keyword_type": "sub-concepts",
        "keyword_provider": "manual",
        "targeted_scrap_area": [{
            "source_tag": "default",
            "scrap_area_type": "unknown",
            "scrap_area_specification": []
        }],
        "scrap_mode": "in_page",
        "scrap_shared_area": "Yes"
    }

    def __init__(self, domain, category, data_tier, file_name=None):
        self.domain = domain
        self.category = category
        self.data_tier = data_tier
        assert data_tier in ['keywords', 'scrap_area', 'scrapped_sentences',
                             'split_sentences', 'questions'], "Invalid data tier. Choose from 'keywords', 'scrap_area', 'scrapped_sentences', 'split_sentences', 'questions'."
        # self.tier_order = {value: index for index, value in enumerate(['keywords', 'scrap_area', 'scrapped_sentences', 'split_sentences'])}
        self.file_name = file_name
        self.data = [{
            "category": self.category,
            "domain": self.domain}]

    @staticmethod
    def check_format(data_tier=None, data=None, scrap_area_only=False):

        def check_scrap_area_format(scrap_areas):
            assert isinstance(scrap_areas,
                              list), "scrap_area should be a list of scrap resource dictionary"
            for scrap_area in scrap_areas:
                assert isinstance(scrap_area, dict), "Each item in the scrap_area list should be a dictionary"
                scrap_area_keys = {"source_tag", "scrap_area_type", "scrap_area_specification"}
                assert scrap_area_keys == set(
                    scrap_area.keys()), f"The scrap area dictionary should contain only the keys {scrap_area_keys}"
                assert scrap_area['scrap_area_type'] in ['local_paths', 'wiki_urls',
                                                         'general_links', 'unknown'], "scrap_area_type should be either 'local_paths', 'wiki_urls','general_links', or 'unknown'."
                assert isinstance(scrap_area['scrap_area_specification'],
                                  list), "scrap_area_specification should be a list of URLs or paths"

        if scrap_area_only:
            return check_scrap_area_format

        assert data_tier in ['keywords', 'scrap_area', 'scrapped_sentences',
                             'split_sentences', 'questions'], "Invalid data tier. Choose from 'keywords', 'scrap_area', 'scrapped_sentences', 'split_sentences', 'questions'."
        if data_tier in ['keywords', 'scrap_area', 'scrapped_sentences']:
            assert isinstance(data, list), "Data should be a list of dictionaries."
            for item in data:
                assert isinstance(item, dict), "Each item in the list should be a dictionary."
                assert 'keywords' in item, "Each dictionary should have a 'keywords' key."
                assert 'domain' in item, "Each dictionary should have a 'domain' key."
                assert 'category' in item, "Each dictionary should have a 'category' key."

                if data_tier in ['scrap_area', 'scrapped_sentences']:
                    assert 'category_shared_scrap_area' in item, "Each dictionary should have a 'category_shared_scrap_area' key"
                    scrap_area = item['category_shared_scrap_area']
                    check_scrap_area_format(scrap_area)

                # check keywords format
                keywords = item['keywords']
                assert isinstance(keywords, dict), "keywords should be a dictionary"
                for k, v in keywords.items():
                    assert isinstance(v, dict), f"The value of keyword '{k}' should be a dictionary"
                    required_keys = {"keyword_type", "keyword_provider", "scrap_mode",
                                     "scrap_shared_area"}
                    if data_tier == 'scrapped_sentences':
                        required_keys.add('scrapped_sentences')
                        assert isinstance(v['scrapped_sentences'],
                                          list), "scrapped_sentences should be a list of sentences"


                    # check targeted_scrap_area format
                    if 'targeted_scrap_area' in v.keys():
                        required_keys.add('targeted_scrap_area')
                        assert required_keys == set(
                            v.keys()), f"The keywords dictionary of '{k}' should contain only the keys {required_keys} or with an additional 'targeted_scrap_area' key."
                        required_keys.remove('targeted_scrap_area')
                        check_scrap_area_format(v['targeted_scrap_area'])
                    else:
                        assert required_keys == set(
                            v.keys()), f"The keywords dictionary of '{k}' should contain only the keys {required_keys}."

        elif data_tier == 'split_sentences':
            assert isinstance(data, pd.DataFrame), "Data should be a DataFrame"
            assert 'keyword' in data.columns, "DataFrame must contain 'keyword' column"
            assert 'category' in data.columns, "DataFrame must contain 'category' column"
            assert 'domain' in data.columns, "DataFrame must contain 'domain' column"
            assert 'prompts' in data.columns, "DataFrame must contain 'prompts' column"
            assert 'baseline' in data.columns, "DataFrame must contain 'baseline' column"

        elif data_tier == 'questions':
            print("IN HERE")
            assert isinstance(data, pd.DataFrame), "Data should be a DataFrame"
            assert 'keyword' in data.columns, "DataFrame must contain 'keyword' column"
            assert 'category' in data.columns, "DataFrame must contain 'category' column"
            assert 'domain' in data.columns, "DataFrame must contain 'domain' column"
            assert 'questions' in data.columns, "DataFrame must contain 'questions' column"
            assert 'original_answer' in data.columns, "DataFrame must contain 'original_answer' column"
            

    @classmethod
    def load_file(cls, domain, category, data_tier, file_path):
        instance = cls(domain, category, data_tier, file_path)
        try:
            if data_tier == 'split_sentences' or data_tier == 'questions':
                instance.data = pd.read_csv(file_path)
            else:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                instance.data = data
            cls.check_format(data_tier, instance.data)
        except (IOError, json.JSONDecodeError, AssertionError) as e:
            print(f"Error loading or validating file {file_path}: {e}")
            return None

        return instance

    @classmethod
    def create_data(cls, domain, category, data_tier, data = None):
        instance = cls(domain, category, data_tier)

        if data is None:
            if data_tier == 'keywords':
                instance.data = [{
                    "category": category,
                    "domain": domain,
                    "keywords": {}
                }]
                return instance
            elif data_tier == 'scrap_area':
                instance.data = [{
                    "category": category,
                    "domain": domain,
                    "category_shared_scrap_area": cls.default_scrap_area_format,
                    "keywords": {}
                }]
                return instance
            elif data_tier == 'scrapped_sentences':
                instance.data = [{
                    "category": category,
                    "domain": domain,
                    "category_shared_scrap_area": cls.default_scrap_area_format,
                    "keywords": {}
                }]
                return instance
            elif data_tier == 'split_sentences':
                instance.data = pd.DataFrame(columns=['keyword', 'category', 'domain', 'prompts', 'baseline', 'source_tag'])
                return instance

        try:
            instance.data = data
            cls.check_format(data_tier, instance.data)
            return instance
        except (IOError, json.JSONDecodeError, AssertionError) as e:
            print(f"Error loading or validating data: {e}")
            return None

    def show(self, mode='short', keyword=None, data_tier=None):

        if abcData.tier_order[data_tier] > abcData.tier_order[self.data_tier]:
            print(f"Data in data tier '{data_tier}' is not available in the current data tier {self.data_tier}.")
            return

        if data_tier == 'keywords':
            if mode == 'short':
                for item in self.data:
                    print(f"Category: {item['category']}, Domain: {item['domain']}")
                    keywords = ", ".join(item['keywords'].keys())
                    print(f"  Keywords: {keywords}")
            elif mode == 'metadata':
                for item in self.data:
                    print(f"Category: {item['category']}, Domain: {item['domain']}")
                    for keyword, metadata in item['keywords'].items():
                        print(f"  Keyword: {keyword}, Metadata: {metadata}")
        elif data_tier == 'scrap_area':
            if mode == 'short':
                for item in self.data:
                    print(f"Category: {item['category']}, Domain: {item['domain']}")
                    print(f"  Scrap Area: {item['category_shared_scrap_area']}")
            if mode == 'details':
                for item in self.data:
                    print(f"Category: {item['category']}, Domain: {item['domain']}")
                    print(f"  Scrap Area: {item['category_shared_scrap_area']}")
                    for keyword, metadata in item['keywords'].items():
                        print(f"  Keyword: {keyword}, targeted_scrap_area: {metadata.get('targeted_scrap_area')}")
        elif data_tier == 'scrapped_sentences':
            if keyword == None:
                for item in self.data:
                    print(f"Category: {item['category']}, Domain: {item['domain']}")
                    print(f"  Scrap Area: {item['category_shared_scrap_area']}")
                    for keyword, metadata in item['keywords'].items():
                        print(f"  Keyword: {keyword}, targeted_scrap_area: {metadata.get('scrapped_sentences')}")
        elif data_tier == 'split_sentences':
            print("Split sentences are in a DataFrame")
            print(self.data)
        elif data_tier == 'questions':
            print("Questions are in a DataFrame")
            print(self.data)

    def remove(self, entity, data_tier=None, keyword=None, removal_range='all'):

        def remove_element_from_list(main_list, sublist):
            for element in sublist:
                if element in main_list:
                    main_list.remove(element)
                    break
            return main_list

        if data_tier is None:
            data_tier = self.data_tier
        if not self.data:
            print("No data to modify.")
            return
        if abcData.tier_order[data_tier] > abcData.tier_order[self.data_tier]:
            print(f"Data tier '{data_tier}' is not available in the current data.")
            return

        if data_tier == 'keywords':
            for item in self.data:
                if entity in item['keywords']:
                    del item['keywords'][entity]
                    print(f"Keyword '{entity}' removed from the data.")

        if data_tier == 'scrap_area' and removal_range == 'all':
            if isinstance(entity, dict):
                entity = [entity]

            for item in self.data:
                for sa_dict in item['category_shared_scrap_area']:
                    sa_dict['scrap_area_specification'] = remove_element_from_list(sa_dict['scrap_area_specification'],
                                                                                   entity)
                    print(f"Scrap area '{entity}' removed from the {sa_dict}.")
                for kw_dict in item['keywords']:
                    for sa_dict in kw_dict['targeted_scrap_area']:
                        sa_dict['scrap_area_specification'] = remove_element_from_list(
                            sa_dict['scrap_area_specification'], entity)
                        print(f"Scrap area '{entity}' removed from the {sa_dict}.")

        if data_tier == 'scrap_area' and removal_range == 'targeted':
            # assert that the scrap_area is in the right dictionary format
            if isinstance(entity, dict):
                entity = [entity]
            for index, scrap_area_dict in enumerate(entity):
                # give source_tage if not provided
                if 'source_tag' not in scrap_area_dict.keys():
                    entity[index]['source_tag'] = 'default'
            abcData.check_format(scrap_area_only=True)(entity)

            for item in self.data:
                if keyword:
                    for sa_index, sa_dict in enumerate(item['keywords'][keyword]['targeted_scrap_area']):
                        for sa_index_to_remove, sa_dict_to_remove in enumerate(entity):
                            if sa_dict['source_tag'] == sa_dict_to_remove['source_tag'] and \
                                    sa_dict['scrap_area_type'] == sa_dict_to_remove['scrap_area_type']:
                                remove_element_from_list(sa_dict['scrap_area_specification'],
                                                         sa_dict_to_remove['scrap_area_specification'])
                                print(
                                    f"Scrap area '{entity[sa_index_to_remove]['scrap_area_specification']}' removed from the data.")
                else:
                    for sa_index, sa_dict in enumerate(item['category_shared_scrap_area']):
                        for sa_index_to_remove, sa_dict_to_remove in enumerate(entity):
                            if sa_dict['source_tag'] == sa_dict_to_remove['source_tag'] and \
                                    sa_dict['scrap_area_type'] == sa_dict_to_remove['scrap_area_type']:
                                remove_element_from_list(sa_dict['scrap_area_specification'],
                                                         sa_dict_to_remove['scrap_area_specification'])
                                print(
                                    f"Scrap area '{sa_dict_to_remove['scrap_area_specification']}' removed from the data.")

        if data_tier == 'scrapped_sentences':
            for item in self.data:
                for keyword, metadata in item['keywords'].items():
                    metadata['scrapped_sentences'].remove(entity)
                    print(f"Scrapped sentence '{entity}' removed from the data.")
        if data_tier == 'split_sentences':
            print("Cannot remove from split sentences data, it is a DataFrame.")
        if data_tier == 'questions':
            print("Cannot remove from questions data, it is a DataFrame.")

    def add(self, keyword=None, scrap_area=None, metadata=None, scrap_area_target='common', data_tier=None):

        def merge_scrap_area_specifications(data):
            """
            Merges the scrap_area_specification lists for unique combinations of
            source_tag and scrap_area_type.

            Args:
                data (list): A list of dictionaries containing source_tag, scrap_area_type,
                             and scrap_area_specification.

            Returns:
                list: A list of merged dictionaries with unique source_tag and scrap_area_type.
            """
            merged_data = defaultdict(
                lambda: {"source_tag": "", "scrap_area_type": "", "scrap_area_specification": set()})

            for item in data:
                key = (item["source_tag"], item["scrap_area_type"])
                merged_data[key]["source_tag"] = item["source_tag"]
                merged_data[key]["scrap_area_type"] = item["scrap_area_type"]
                merged_data[key]["scrap_area_specification"].update(item["scrap_area_specification"])

            # Convert the sets back to lists
            result = []
            for value in merged_data.values():
                value["scrap_area_specification"] = list(value["scrap_area_specification"])
                result.append(value)

            return result

        if data_tier is None:
            data_tier = self.data_tier
        if abcData.tier_order[data_tier] > abcData.tier_order[self.data_tier]:
            print(f"Data tier '{data_tier}' is not available in the current data.")
            return self

        if data_tier == 'keywords':
            default_metadata = abcData.default_keyword_metadata.copy()
            if metadata is None:
                metadata = default_metadata
            elif isinstance(metadata, dict):
                # Filter and update metadata based on default values
                filtered_metadata = {key: metadata.get(key, default_value) for key, default_value in
                                     default_metadata.items()}
                metadata = filtered_metadata
                targeted_scrap_area = metadata.get('targeted_scrap_area')
                abcData.check_format(scrap_area_only=True)(targeted_scrap_area)
            else:
                print("Metadata provided is not in the right dictionary format.")
                return self

            for index, item in enumerate(self.data):
                if 'keywords' not in item:
                    self.data[index]['keywords'] = {}
                self.data[index]['keywords'][keyword] = metadata
            return self

        if data_tier == 'scrap_area':
            # assert that the scrap_area is in the right dictionary format
            if isinstance(scrap_area, dict):
                scrap_area = [scrap_area]
            for index, scrap_area_dict in enumerate(scrap_area):
                # give source_tage if not provided
                if 'source_tag' not in scrap_area_dict.keys():
                    scrap_area[index]['source_tag'] = 'default'
            abcData.check_format(scrap_area_only=True)(scrap_area)

            for index, item in enumerate(self.data):
                if 'category_shared_scrap_area' not in item:
                    self.data[index]['category_shared_scrap_area'] = scrap_area
                if scrap_area_target == 'common':
                    self.data[index]['category_shared_scrap_area'].append(scrap_area)
                    self.data[index]['category_shared_scrap_area'] = \
                        merge_scrap_area_specifications(self.data[index]['category_shared_scrap_area'])
                    abcData.check_format(scrap_area_only=True)(self.data[index]['category_shared_scrap_area'])

                elif scrap_area_target == 'targeted':
                    assert keyword is not None, "Keyword must be provided to add targeted scrap area."
                    self.data[index]['keywords'][keyword]['targeted_scrap_area'].append(scrap_area)
                    self.data[index]['keywords'][keyword]['targeted_scrap_area'] = \
                        merge_scrap_area_specifications(self.data[index]['keywords'][keyword]['targeted_scrap_area'])
                    abcData.check_format(scrap_area_only=True)(
                        self.data[index]['keywords'][keyword]['targeted_scrap_area'])

            return self

        if data_tier == 'scrapped_sentences':
            assert keyword is not None, "Keyword must be provided to add scrapped sentences."
            for index, item in enumerate(self.data):
                self.data[index]['keywords'][keyword]['scrapped_sentences'].append(scrap_area)
                return self

        if data_tier == 'split_sentences':
            print("Cannot add to split sentences data, it is a DataFrame.")
            return self

    def save(self, file_path=None, domain_save=False, suffix=None):

        if self.data_tier == 'split_sentences' or self.data_tier == 'questions':
            if file_path is None:
                if domain_save:
                    file_name = f"{self.domain}_{self.data_tier}.csv"
                else:
                    file_name = f"{self.domain}_{self.category}_{self.data_tier}.csv"
                if suffix is not None:
                    file_name = f"{file_name[:-4]}_{suffix}.csv"
                default_path = os.path.join('data', 'customized', self.data_tier)
                os.makedirs(default_path, exist_ok=True)
                file_path = os.path.join(default_path, file_name)
            if isinstance(self.data, pd.DataFrame):
                self.data.to_csv(file_path, index=False)
                print(f"Data saved to {file_path}")
            else:
                print("Data is not in a DataFrame format.")
        else:
            # Generate default file name if not provided
            if file_path is None:
                file_name = f"{self.domain}_{self.category}_{self.data_tier}.json"
                # Ensure the default file path
                default_path = os.path.join('data', 'customized', self.data_tier)
                os.makedirs(default_path, exist_ok=True)
                file_path = os.path.join(default_path, file_name)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            print(f"Data saved to {file_path}")

    @classmethod
    def merge(cls, domain, merge_list, category = 'merged', abc_format = True):
        df = pd.DataFrame()
        for data_item in merge_list:
            assert isinstance(data_item, abcData), "Data to merge should be of type abcData."
            assert data_item.domain == domain, "Data to merge should have the same domain."
            assert data_item.data_tier == 'split_sentences', "Data to merge should be in split_sentences data tier."
            df = pd.concat([df, data_item.data], ignore_index=True)

        merged_data = abcData.create_data(domain, category, 'split_sentences', df)

        if abc_format:
            return merged_data

        return df

    def sub_sample(self, sample=10, seed=42, clean=True, floor = False , abc_format = False):
        if not isinstance(self.data, pd.DataFrame):
            print("Cannot generate sub_sample from non-DataFrame data. You need to perform sentence split.")
            return

        if clean and 'keywords_containment' in self.data.columns:
            df = self.data
            df = df[df['keywords_containment'] == True]
            df = df.copy()  # Make a copy to avoid the SettingWithCopyWarning
            df.drop(['keywords_containment'], axis=1, inplace=True)
            self.data = df

        if floor:
            sample = min(sample, len(self.data))
        else:
           assert sample <= len(self.data), f"Sample size should be less than or equal to the data size {len(self.data)}."
        sample_data = self.data.sample(n=sample, random_state=seed).copy()
        self.data = sample_data

        if abc_format:
            return self

        return sample_data

    def model_generation(self, generation_function, generation_name='generated_output',  abc_format = False):
        if not isinstance(self.data, pd.DataFrame):
            print("Cannot generate model output from non-DataFrame data. You need to perform sentence split.")
            return

        self.data[generation_name] = self.data['prompts'].progress_apply(generation_function)

        if abc_format:
            return self

        return self.data

    # def counterfactualization(self, keywords_mapping = None, mode='all', source_tag='counterfactual', merge = False):
    #     """
    #     This function performs counterfactual insertion by replacing specified keywords in the prompts
    #     with their corresponding replacements based on the provided mapping.
    #
    #     :param keywords_mapping: A dictionary where keys are the keywords to be replaced,
    #                              and values are their replacements.
    #     :param mode: The mode of replacement. Choose from 'one_way' or 'two_way'.
    #     :param source_tag: A tag indicating the source of the modification (default is 'counterfactual').
    #     """
    #     assert mode in ['all', 'one_way', 'two_way'], "Invalid mode. Choose from 'one_way' or 'two_way'."
    #     assert isinstance(self.data, pd.DataFrame), "Data should be a DataFrame."
    #     if keywords_mapping is not None:
    #         assert isinstance(keywords_mapping, list), "Keywords mapping should be a list of tuples."
    #         for keyword_pair in keywords_mapping:
    #             assert keyword_pair[0] in self.data['keyword'].values, f"Keyword '{keyword_pair[0]}' not found in the data."
    #             assert keyword_pair[1] in self.data['keyword'].values, f"Replacement '{keyword_pair[1]}' not found in the data."
    #
    #     # Dictionary to store modified DataFrames
    #     modified_df_dict = {}
    #     kw_cat_mapping = dict(zip(self.data['keyword'], self.data['category']))
    #
    #     if mode == 'all' or keywords_mapping is None:
    #         keyword_list = self.data['keyword'].unique()
    #         keywords_mapping = list(permutations(keyword_list, 2))
    #
    #     if mode == 'two_way':
    #         keywords_inverted = [(kw_pairs[1], kw_pairs[0]) for kw_pairs in keywords_mapping]
    #         keywords_mapping.extend(keywords_inverted)
    #
    #     for keyword_pair in tqdm(keywords_mapping, desc='Replacing keywords'):
    #         keyword = keyword_pair[0]
    #         replacement = keyword_pair[1]
    #         # Filter the data to only include rows with the specified keyword
    #         keyword_data = self.data[self.data['keyword'] == keyword]
    #
    #         if keyword_data.empty:
    #             raise ValueError(f"Keyword '{keyword}' not found in the data.")
    #
    #         # Create a modified DataFrame with the original structure
    #         counterfactual_df = keyword_data.copy()
    #         counterfactual_df['prompts'] = counterfactual_df['prompts'].apply(lambda x: x.lower().replace(keyword, replacement).title())
    #         counterfactual_df['keyword'] = replacement
    #         counterfactual_df['category'] = kw_cat_mapping[replacement]
    #         counterfactual_df['source_tag'] = counterfactual_df['source_tag'] + f'_counterfactual_{keyword}'
    #         if merge:
    #             counterfactual_df = pd.concat([keyword_data, counterfactual_df], ignore_index=True)
    #
    #         # Store the modified DataFrame in the dictionary
    #         modified_df_dict[keyword_pair] = counterfactual_df
    #
    #     # Concatenate all modified DataFrames
    #     all_modified_df = pd.concat(modified_df_dict.values(), ignore_index=True)
    #     self.data = all_modified_df
    #     return all_modified_df



