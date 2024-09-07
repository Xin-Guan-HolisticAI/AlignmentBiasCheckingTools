import abcData

from scrape import check_generation_function
from scrape import KeywordFinder, ScrapAreaFinder, Scrapper
from assembler import PromptMaker

class BenchmarkBuilder:
    default_category_configuration = {}
    default_domain_configuration = {}
    default_branching_configuration = {}

    def reset(cls):
        cls.default_branching_configuration = {
            'branching_pairs': 'all',
            'direction': 'both',
            'source_restriction': None,
            'replacement_descriptor_require': True,
            'descriptor_threshold': 'Auto',
            'descriptor_embedding_model': 'paraphrase-Mpnet-base-v2',
            'descriptor_distance': 'cosine',
            'replacement_description': {},
            'replacement_description_saving': True,
            'replacement_description_saving_location': f'data/customized/split_sentences/replacement_description.json',
            'counterfactual_baseline': True,
            'generation_function': None,
        }
        cls.default_category_configuration = {
            'keyword_finder': {
                'require': True,
                'reading_location': 'default',
                'method': 'embedding_on_wiki',  # 'embedding_on_wiki' or 'llm_inquiries' or 'hyperlinks_on_wiki'
                'keyword_number': 7,  # keyword_number works for both embedding_on_wiki and hyperlinks_on_wiki
                # If hyperlinks_info is method chosen, can give more info... format='Paragraph', link=None, page_name=None, name_filter=False, col_info=None, depth=None, source_tag='default', max_keywords = None). col_info format is [{'table_num': value, 'column_name':List}]
                'llm_info': {},
                # If llm_inequiries is method chosen, can give more info... self, n_run=20,n_keywords=20, generation_function=None, model_name=None, embedding_model=None, show_progress=True
                'max_adjustment': 150,
                # max_adjustment for embedding_on_wiki. If max_adjustment is equal to -1, then max_adjustment is not taken into account.
                'embedding_model': 'paraphrase-Mpnet-base-v2',
                'saving': True,
                'saving_location': 'default',
                'manual_keywords': None,
            },
            'scrap_area_finder': {
                'require': True,
                'reading_location': 'default',
                'method': 'wiki',  # 'wiki' or 'local_files',
                'local_file': None,
                'scrap_number': 5,
                'saving': True,
                'saving_location': 'default',
                'scrap_backlinks': 0,
            },
            'scrapper': {
                'require': True,
                'reading_location': 'default',
                'saving': True,
                'method': 'wiki',  # This is related to the scrap_area_finder method,
                'saving_location': 'default'},
            'prompt_maker': {
                'require': True,
                'method': 'split_sentences',  # can also have "questions" as a method
                # prompt_maker_generation_function and prompt_maker_keyword_list are needed for questions
                'generation_function': None,
                # prompt_maker_keyword_list must contain at least one keyword. The first keyword must be the keyword
                # of the original scrapped data.
                'keyword_list': None,
                # User will enter False if they don't want their questions answer checked.
                'answer_check': False,
                'saving_location': 'default',
                'max_benchmark_length': 500,
            },
        }
        cls.default_domain_configuration = {
            'categories': [],
            'branching': False,  # If branching is False, then branching_config is not taken into account
            'branching_config': cls.default_branching_configuration,
            'shared_config':cls.default_category_configuration,
            'category_specified_config':{},
            'saving': True,  # If saving is False, then saving_location is not taken into account
            'saving_location': 'default',
        }

    def __init__(self):
        pass

    @staticmethod
    def update_configuration(default_configuration, updated_configuration):
        """
        Update the default configuration dictionary with the values from the updated configuration
        only if the keys already exist in the default configuration.

        Args:
        - default_category_configuration (dict): The default configuration dictionary.
        - updated_configuration (dict): The updated configuration dictionary with new values.

        Returns:
        - dict: The updated configuration dictionary.
        """

        for key, value in updated_configuration.items():
            if key in default_configuration.copy():
                # print(f"Updating {key} recursively")
                if isinstance(default_configuration[key], dict) and isinstance(value, dict):
                    # Recursively update nested dictionaries
                    default_configuration[key] = BenchmarkBuilder.update_configuration(default_configuration[key].copy(),
                                                                                       value)
                else:
                    # print(f"Skipping key: {key} due to type mismatch")
                    # Update the value for the key
                    default_configuration[key] = value
        return default_configuration

    @staticmethod
    def merge_category_specified_configuration(domain_configuration):
        specified_config = domain_configuration['category_specified_config'].copy()
        domain_configuration = BenchmarkBuilder.update_configuration(BenchmarkBuilder.default_domain_configuration.copy(),
                                              domain_configuration)
        domain_configuration['shared_config'] = BenchmarkBuilder.update_configuration(BenchmarkBuilder.default_category_configuration.copy(), domain_configuration['shared_config'].copy())

        base_category_config = {}
        for cat in domain_configuration['categories']:
            base_category_config[cat] = domain_configuration['shared_config'].copy()

        # print('start ====================== \n\n')
        merge_category_config = BenchmarkBuilder.update_configuration(base_category_config.copy(), specified_config.copy())
        print(merge_category_config)
        return merge_category_config

    @classmethod
    def category_pipeline(cls, domain, demographic_label, configuration=None):
        cls.reset(cls)
        if configuration is None:
            configuration = cls.default_category_configuration.copy()
        else:
            configuration = cls.update_configuration(cls.default_category_configuration.copy(), configuration)

        # Unpacking keyword_finder section
        keyword_finder_config = configuration['keyword_finder']
        keyword_finder_require, keyword_finder_reading_location, keyword_finder_method, \
        keyword_finder_keyword_number, keyword_finder_hyperlinks_info, keyword_finder_llm_info, \
        keyword_finder_max_adjustment, keyword_finder_embedding_model, keyword_finder_saving, \
        keyword_finder_saving_location, keyword_finder_manual_keywords = (
            keyword_finder_config[key] for key in [
            'require', 'reading_location', 'method', 'keyword_number', 'hyperlinks_info',
            'llm_info', 'max_adjustment', 'embedding_model', 'saving', 'saving_location',
            'manual_keywords'
        ]
        )

        # Unpacking scrap_area_finder section
        scrap_area_finder_config = configuration['scrap_area_finder']
        scrap_area_finder_require, scrap_area_finder_reading_location, scrap_area_finder_method, \
        scrap_area_local_file, scrap_area_finder_saving, scrap_area_finder_saving_location, \
        scrap_area_finder_scrap_area_number, scrap_area_finder_scrap_backlinks = (
            scrap_area_finder_config[key] for key in [
            'require', 'reading_location', 'method', 'local_file', 'saving',
            'saving_location', 'scrap_number', 'scrap_backlinks'
        ]
        )

        # Unpacking scrapper section
        scrapper_config = configuration['scrapper']
        scrapper_require, scrapper_reading_location, scrapper_saving, \
        scrapper_method, scrapper_saving_location = (
            scrapper_config[key] for key in [
            'require', 'reading_location', 'saving', 'method', 'saving_location'
        ]
        )

        # Unpacking prompt_maker section
        prompt_maker_config = configuration['prompt_maker']
        prompt_maker_require, prompt_maker_method, prompt_maker_generation_function, \
        prompt_maker_keyword_list, prompt_maker_answer_check, prompt_maker_saving_location, \
        prompt_maker_max_sample_number = (
            prompt_maker_config[key] for key in [
            'require', 'method', 'generation_function', 'keyword_list', 'answer_check',
            'saving_location', 'max_benchmark_length'
        ]
        )

        # check the validity of the configuration
        assert keyword_finder_method in ['embedding_on_wiki', 'llm_inquiries',
                                         'hyperlinks_on_wiki'], "Invalid keyword finder method. Choose either 'embedding_on_wiki', 'llm_inquiries', or 'hyperlinks_on_wiki'."
        if keyword_finder_method == 'llm_inquiries':
            assert 'generation_function' in keyword_finder_llm_info and keyword_finder_llm_info[
                'generation_function'] is not None, "generation function must be provided if llm_inquiries is chosen as the method"
            check_generation_function(keyword_finder_llm_info['generation_function'])
        assert scrap_area_finder_method in ['wiki',
                                            'local_files'], "Invalid scrap area finder method. Choose either 'wiki' or 'local_files'"
        assert scrapper_method in ['wiki',
                                   'local_files'], "Invalid scrapper method. Choose either 'wiki' or 'local_files'"
        assert scrap_area_finder_method == scrapper_method, "scrap_area_finder and scrapper methods must be the same"
        assert prompt_maker_method in ['split_sentences', 'questions'], "Invalid prompt maker method. Choose 'split_sentences' or 'questions'"

        '''
        # make sure only the required loading is done
        if not scrapper_require:
            scrap_area_finder_require = False

        if not scrap_area_finder_require:
            keyword_finder_require = False
        '''

        if keyword_finder_require:
            if keyword_finder_method == 'embedding_on_wiki':
                kw = KeywordFinder(domain=domain, category=demographic_label).find_keywords_by_embedding_on_wiki(
                    n_keywords=keyword_finder_keyword_number, embedding_model=keyword_finder_embedding_model,
                    max_adjustment=keyword_finder_max_adjustment).add(
                    keyword=demographic_label)
            elif keyword_finder_method == 'llm_inquiries':
                # format = self, n_run=20,n_keywords=20, generation_function=None, model_name=None, embedding_model=None, show_progress=True
                default_values = {'n_run': 20, 'n_keywords': 20, 'generation_function': None, 'model_name': None,
                                  'embedding_model': None, 'show_progress': True}

                # Create a new dictionary with only non-default fields
                for key in keyword_finder_llm_info:
                    if key in default_values:
                        default_values[key] = keyword_finder_llm_info[key]

                kw = KeywordFinder(domain=domain, category=demographic_label).find_keywords_by_llm_inquiries(
                    **default_values).add(
                    keyword=demographic_label)
            elif keyword_finder_method == 'hyperlinks_on_wiki':
                # format='Paragraph', link=None, page_name=None, name_filter=False, col_info=None, depth=None, source_tag='default', max_keywords = None):
                if ('link' in keyword_finder_hyperlinks_info and keyword_finder_hyperlinks_info['link'] != None) or (
                        'page_name' in keyword_finder_hyperlinks_info and keyword_finder_hyperlinks_info[
                    'page_name'] != None):
                    default_values = {'format': 'Paragraph', 'link': None, 'page_name': None, 'name_filter': False,
                                      'col_info': None, 'depth': None, 'source_tag': 'default', 'max_keywords': None}
                else:
                    raise AssertionError("For hyperlinks of wiki, must provide either the page name or link")

                default_values = {'format': 'Paragraph', 'link': None, 'page_name': None, 'name_filter': False,
                                  'col_info': None, 'depth': None, 'source_tag': 'default', 'max_keywords': None}

                # Create a new dictionary with only non-default fields
                for key in keyword_finder_hyperlinks_info:
                    if key in default_values:
                        default_values[key] = keyword_finder_hyperlinks_info[key]

                kw = KeywordFinder(domain=domain, category=demographic_label).find_name_keywords_by_hyperlinks_on_wiki(
                    **default_values).add(keyword=demographic_label)

                # if manual keywords are provided, add them to the keyword finder
                if isinstance(keyword_finder_manual_keywords, list):
                    for keyword in keyword_finder_manual_keywords:
                        kw = kw.add(keyword)

            if keyword_finder_saving:
                if keyword_finder_saving_location == 'default':
                    kw.save()
                else:
                    kw.save(file_path=keyword_finder_saving_location)

        elif (not keyword_finder_require) and isinstance(keyword_finder_manual_keywords, list):
            kw = abcData.create_data(domain=domain, category=demographic_label, data_tier='keywords')
            for keyword in keyword_finder_manual_keywords:
                kw = kw.add(keyword)

        elif scrap_area_finder_require and (keyword_finder_manual_keywords is None):
            filePath = ""
            if keyword_finder_reading_location == 'default':
                filePath = f'data/customized/keywords/{domain}_{demographic_label}_keywords.json'
                kw = abcData.load_file(domain=domain, category=demographic_label,
                                       file_path=filePath,
                                       data_tier='keywords')
            else:
                filePath = keyword_finder_reading_location
                kw = abcData.load_file(domain=domain, category=demographic_label,
                                       file_path=filePath, data_tier='keywords')

            if kw != None:
                print(f'Keywords loaded from {filePath}')
            else:
                raise ValueError(f"Unable to read keywords from {filePath}. Can't scrap area.")

        if scrap_area_finder_require:
            if scrap_area_finder_method == 'wiki':
                sa = ScrapAreaFinder(kw, source_tag='wiki').find_scrap_urls_on_wiki(
                    top_n=scrap_area_finder_scrap_area_number, scrap_backlinks=scrap_area_finder_scrap_backlinks)
            elif scrap_area_finder_method == 'local_files':
                if scrap_area_local_file == None:
                    raise ValueError(f"Unable to read keywords from {scrap_area_local_file}. Can't scrap area.")
                sa = ScrapAreaFinder(kw, source_tag='local').find_scrap_paths_local(scrap_area_local_file)
            print('Scrap areas located.')

            if scrap_area_finder_saving:
                if scrap_area_finder_saving_location == 'default':
                    sa.save()
                else:
                    sa.save(file_path=scrap_area_finder_saving_location)


        elif scrapper_require:
            filePath = ""
            if scrap_area_finder_reading_location == 'default':
                filePath = f'data/customized/scrap_area/{domain}_{demographic_label}_scrap_area.json'
                sa = abcData.load_file(domain=domain, category=demographic_label,
                                       file_path=filePath,
                                       data_tier='scrap_area')
            else:
                filePath = scrap_area_finder_reading_location
                sa = abcData.load_file(domain=domain, category=demographic_label,
                                       file_path=scrap_area_finder_reading_location, data_tier='scrap_area')

            if sa != None:
                print(f'Scrap areas loaded from {filePath}')
            else:
                raise ValueError(f"Unable to scrap areas from {filePath}. Can't use scrapper.")

        if scrapper_require:
            if scrapper_method == 'wiki':
                sc = Scrapper(sa).scrap_in_page_for_wiki_with_buffer_files()
            elif scrapper_method == 'local_files':
                sc = Scrapper(sa).scrap_local_with_buffer_files()
            print('Scrapped sentences completed.')

            if scrapper_saving:
                if scrapper_saving_location == 'default':
                    sc.save()
                else:
                    sc.save(file_path=scrapper_saving_location)
        elif prompt_maker_require:
            filePath = ""
            if scrapper_reading_location == 'default':
                filePath = f'data/customized/scrapped_sentences/{domain}_{demographic_label}_scrapped_sentences.json'
                sc = abcData.load_file(domain=domain, category=demographic_label,
                                       file_path=filePath,
                                       data_tier='scrapped_sentences')
                print(
                    f'Scrapped sentences loaded from data/customized/scrapped_sentences/{domain}_{demographic_label}_scrapped_sentences.json')
            else:
                filePath = scrapper_reading_location
                sc = abcData.load_file(domain=domain, category=demographic_label, file_path=scrapper_reading_location,
                                       data_tier='scrapped_sentences')
                print(f'Scrapped sentences loaded from {scrapper_reading_location}')

            if sc != None:
                print(f'Scrapped loaded from {filePath}')
            else:
                raise ValueError(f"Unable to scrap from {filePath}. Can't make prompts.")

        pm_result = None
        if prompt_maker_method == 'split_sentences' and prompt_maker_require:
            pm = PromptMaker(sc)
            pm_result = pm.split_sentences()
        elif prompt_maker_method == 'questions' and prompt_maker_require:
            pm = PromptMaker(sc)
            pm_result = pm.make_questions(generation_function = prompt_maker_generation_function, keyword_reference=prompt_maker_keyword_list, answer_check = prompt_maker_answer_check, max_questions = prompt_maker_max_sample_number)
        if pm_result is None:
            raise ValueError(f"Unable to make prompts out of no scrapped sentences")
        pm_result = pm_result.sub_sample(prompt_maker_max_sample_number, floor=True, abc_format=True) ### There is likely a bug
        if prompt_maker_saving_location == 'default':
            pm_result.save()
        else:
            pm_result.save(file_path=prompt_maker_saving_location)

        print(f'Benchmark building for {demographic_label} completed.')
        print('\n=====================================================\n')

        return pm_result
    @classmethod
    def domain_pipeline(cls, domain, configuration=None):
        cls.reset(cls)
        category_list = configuration['categories']
        category_specified_configuration = cls.merge_category_specified_configuration(configuration.copy())
        configuration = cls.update_configuration(cls.default_domain_configuration.copy(), configuration)
        domain_benchmark = abcData.create_data(domain=domain, category='all', data_tier='split_sentences')
        for category in category_list:
            cat_result = cls.category_pipeline(domain, category, category_specified_configuration[category])
            print(f'Benchmark building for {category} completed.')
            domain_benchmark = abcData.merge(domain, [domain_benchmark, cat_result])

            if configuration['saving']:
                if configuration['saving_location'] == 'default':
                    domain_benchmark.save()
                else:
                    domain_benchmark.save(file_path=configuration['saving_location'])

        if configuration['branching']:
            empty_ss = abcData.create_data(category='merged', domain='AI-company', data_tier='scrapped_sentences')
            pmr = PromptMaker(empty_ss)
            pmr.output_df = domain_benchmark.data
            domain_benchmark = pmr.branching(branching_config=configuration['branching_config'])
            if configuration['saving']:
                if configuration['saving_location'] == 'default':
                    domain_benchmark.save(suffix='branching')
                else:
                    domain_benchmark.save(file_path=configuration['saving_location'])

        return domain_benchmark


if __name__ == '__main__':
    pass