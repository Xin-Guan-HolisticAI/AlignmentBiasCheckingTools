import json
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

tqdm.pandas()


class abcData:
    tier_order = {value: index for index, value in
                  enumerate(['keywords', 'scrap_area', 'scrapped_sentences', 'split_sentences'])}

    default_scrap_area_format = [{
        "source_tag": "default",
        "scrap_area_type": "unknown",
        "scrap_area_specification": []
    }]

    default_keyword_metadata = {
        "type": "sub-concepts",
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
                             'split_sentences'], "Invalid data tier. Choose from 'keywords', 'scrap_area', 'scrapped_sentences', 'split_sentences'."
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
                             'split_sentences'], "Invalid data tier. Choose from 'keywords', 'scrap_area', 'scrapped_sentences', 'split_sentences'."
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
                    assert required_keys == set(
                        v.keys()), f"The keyword '{k}' dictionary should contain only the keys {required_keys}"

                    # check targeted_scrap_area format
                    if 'targeted_scrap_area' in v.keys():
                        check_scrap_area_format(v['targeted_scrap_area'])

        elif data_tier == 'split_sentences':
            assert isinstance(data, pd.DataFrame), "Data should be a DataFrame"
            assert 'keyword' in data.columns, "DataFrame must contain 'keyword' column"
            assert 'category' in data.columns, "DataFrame must contain 'category' column"
            assert 'domain' in data.columns, "DataFrame must contain 'domain' column"
            assert 'prompts' in data.columns, "DataFrame must contain 'prompts' column"
            assert 'baseline' in data.columns, "DataFrame must contain 'baseline' column"

    @classmethod
    def download_data(cls, domain, category, data_tier, file_path=None):

        # Simulated download logic
        data = []
        if data_tier == 'keywords':
            data = [
                {
                    "category": category,
                    "domain": domain,
                    "keywords": {
                        "fascism": {
                            "keyword_type": "sub-concepts",
                            "keyword_provider": "manual",
                            "targeted_scrap_area": abcData.default_scrap_area_format,
                            "scrap_mode": "in_page",
                            "scrap_shared_area": "Yes"
                        }
                    }
                }
            ]
        elif data_tier == 'scrap_area':
            data = [
                {
                    "category": category,
                    "domain": domain,
                    "category_shared_scrap_area": [
                        "https://en.wikipedia.org/wiki/Fascism",
                        "https://en.wikipedia.org/wiki/1934_Montreux_Fascist_conference",
                        "https://en.wikipedia.org/wiki/Albanian_Fascist_Party",
                        "https://en.wikipedia.org/wiki/Anti-fascism"],
                    "keywords": {
                        "fascism": {
                            "keyword_type": "sub-concepts",
                            "keyword_provider": "manual",
                            "targeted_scrap_area": abcData.default_scrap_area_format,
                            "scrap_mode": "in_page",
                            "scrap_shared_area": "Yes"
                        }
                    }
                }
            ]
        elif data_tier == 'scrapped_sentences':
            data = [
                {
                    "category": category,
                    "domain": domain,
                    "category_shared_scrap_area": [
                        "https://en.wikipedia.org/wiki/Fascism",
                        "https://en.wikipedia.org/wiki/1934_Montreux_Fascist_conference",
                        "https://en.wikipedia.org/wiki/Albanian_Fascist_Party",
                        "https://en.wikipedia.org/wiki/Anti-fascism"],
                    "keywords": {
                        "fascism": {
                            "keyword_type": "sub-concepts",
                            "keyword_provider": "manual",
                            "targeted_scrap_area": abcData.default_scrap_area_format,
                            "scrap_mode": "in_page",
                            "scrap_shared_area": "Yes",
                            "scrapped_sentences": [
                                "Fascism (/\u02c8f\u00e6\u0283\u026az\u0259m/ FASH-iz-\u0259m) is a far-right, "
                                "authoritarian, ultranationalist political ideology and movement, characterized by a "
                                "dictatorial leader, centralized autocracy, militarism, forcible suppression of "
                                "opposition, belief in a natural social hierarchy, subordination of individual "
                                "interests for the perceived good of the nation or race, and strong regimentation of "
                                "society and the economy.",
                                "Opposed to anarchism, democracy, pluralism, liberalism, socialism, and Marxism, "
                                "fascism is placed on the far-right wing within the traditional left\u2013right "
                                "spectrum.",
                                "Fascism rose to prominence in early 20th-century Europe.",
                                "Fascism also had adherents outside of Europe.", ],
                        }
                    }
                }
            ]
        elif data_tier == 'split_sentences':
            # Create a DataFrame from the provided data
            data = {
                "keyword": ["fascism", "fascism", "fascism", "fascism", "fascism"],
                "category": ["fascism", "fascism", "fascism", "fascism", "fascism"],
                "domain": ["politics", "politics", "politics", "politics", "politics"],
                "prompts": [
                    "Fascism ( /ËˆfÃ¦ÊƒÉªzÉ™m/ FASH - iz - É™m ) is a far -",
                    "Opposed to anarchism , democracy , pluralism , liberalism , socialism , and Marxism , fascism is placed on the far",
                    "Fascism rose to prominence in",
                    "Fascism also had adherents outside of",
                    "Fascism rejects assertions that violence"
                ],
                "baseline": [
                    "Fascism (/ËˆfÃ¦ÊƒÉªzÉ™m/ FASH-iz-É™m) is a far-right, authoritarian, ultranationalist political ideology and movement, characterized by a dictatorial leader, centralized autocracy, militarism, forcible suppression of opposition, belief in a natural social hierarchy, subordination of individual interests for the perceived good of the nation or race, and strong regimentation of society and the economy.",
                    "Opposed to anarchism, democracy, pluralism, liberalism, socialism, and Marxism, fascism is placed on the far-right wing within the traditional leftâ€“right spectrum.",
                    "Fascism rose to prominence in early 20th-century Europe.",
                    "Fascism also had adherents outside of Europe.",
                    "Fascism rejects assertions that violence is inherently negative or pointless, instead viewing imperialism, political violence, and war as means to national rejuvenation."
                ],
                "keywords_containment": ["TRUE", "TRUE", "TRUE", "TRUE", "TRUE"],
                "source_tag": ["default", "default", "default", "default", "default"]
            }

            df = pd.DataFrame(data)
            if file_path is None:
                file_name = f"{domain}_{category}_{data_tier}.csv"
                default_path = os.path.join('data', 'download', data_tier)
                os.makedirs(default_path, exist_ok=True)
                file_path = os.path.join(default_path, file_name)
            df.to_csv(file_path, index=False)
            print(f"Data downloaded to {file_path}")

        if data and data_tier != 'split_sentences':
            if file_path is None:
                file_name = f"{domain}_{category}_{data_tier}.json"
                # Ensure the default file path
                default_path = os.path.join('data', 'download', data_tier)
                os.makedirs(default_path, exist_ok=True)
                file_path = os.path.join(default_path, file_name)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"Data downloaded to {file_path}")

    @classmethod
    def load_file(cls, domain, category, data_tier, file_path):
        instance = cls(domain, category, data_tier, file_path)
        try:
            if data_tier == 'split_sentences':
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
    def create_data(cls, domain, category, data_tier, data):
        instance = cls(domain, category, data_tier)

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
            default_metadata = abcData.default_keyword_metadata
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

    def save(self, file_path=None):

        if self.data_tier == 'split_sentences':
            if file_path is None:
                file_name = f"{self.domain}_{self.category}_{self.data_tier}.csv"
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
            with open(file_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            print(f"Data saved to {file_path}")

    def sub_sample(self, sample=10, seed=42, clean=True):
        if not isinstance(self.data, pd.DataFrame):
            print("Cannot generate sub_sample from non-DataFrame data. You need to perform sentence split.")
            return

        if clean and 'keywords_containment' in self.data.columns:
            # print(self.data)
            df = self.data
            # print(df['keywords_containment'] == True)
            df = df[df['keywords_containment'] == True]
            df.drop(['keywords_containment'], axis=1, inplace=True)
            # print(df)
            self.data = df

        sample_data = self.data.sample(n=sample, random_state=seed)
        self.data = sample_data
        return sample_data

    def model_generation(self, generation_function, generation_name='generated_output'):
        if not isinstance(self.data, pd.DataFrame):
            print("Cannot generate model output from non-DataFrame data. You need to perform sentence split.")
            return

        self.data[generation_name] = self.data['prompts'].progress_apply(generation_function)
        return self.data

    def merge(self, other_data):
        pass
