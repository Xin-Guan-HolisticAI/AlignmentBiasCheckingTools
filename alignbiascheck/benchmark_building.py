import os

import pandas as pd
import requests
from bs4 import BeautifulSoup
import glob

import wikipediaapi
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
import warnings
from .abcData import abcData

import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from urllib.parse import urlparse, unquote
from sentence_transformers import SentenceTransformer, util


import itertools
import re
from tqdm import tqdm


# Ensure required NLTK resources are downloaded


def find_similar_keywords(model_name, target_word, keywords_list, top_n=100):
    """
    Find the top N keywords most similar to the target word.

    Args:
    - model_name (str): The name of the pre-trained model to use.
    - target_word (str): The word for which we want to find similar keywords.
    - keywords_list (list): The list containing the keywords.
    - n_keywords (int): The number of top similar keywords to return (default is 100).

    Returns:
    - list: The top N keywords most similar to the target word.
    """
    # Load pre-trained model
    model = SentenceTransformer(model_name)

    # Embed the keywords and the target word
    keyword_embeddings = model.encode(keywords_list)
    target_embedding = model.encode(target_word)

    # Compute cosine similarities
    cosine_similarities = util.cos_sim(target_embedding, keyword_embeddings)[0].numpy()

    # Find the top N keywords most similar to the target word
    top_indices = np.argsort(cosine_similarities)[::-1][:top_n]
    top_keywords = [keywords_list[i] for i in top_indices]

    return top_keywords


def search_wikipedia(topic, language='en', user_agent='AlignmentBiasChecker/1.0 (contact@holisticai.com)'):
    """
    Search for a topic on Wikipedia and return the page object.

    Args:
    - topic (str): The topic to search for.
    - language (str): Language of the Wikipedia (default is 'en').
    - user_agent (str): User agent string to use for the API requests.

    Returns:
    - Wikipedia page object or an error message if the page does not exist.
    """
    wiki_wiki = wikipediaapi.Wikipedia(language=language, user_agent=user_agent)
    page = wiki_wiki.page(topic)

    if not page.exists():
        return f"No Wikipedia page found for {topic}"

    return page


def clean_list(response):
    # Extract the part between the square brackets
    response_list = response[response.find('['):response.rfind(']') + 1]

    # Convert the string representation of the list to an actual list
    response_list = eval(response_list)
    return response_list


def construct_non_containing_set(strings):
    def update_string_set(string_set, new_string):
        # Convert the new string to lowercase for comparison
        new_string_lower = new_string.lower()

        # Create a list of strings to remove
        strings_to_remove = [existing_string for existing_string in string_set if
                             new_string_lower in existing_string.lower()]

        # Remove all strings that contain the new string
        for string_to_remove in strings_to_remove:
            string_set.remove(string_to_remove)

        # Check if the new string should be added
        should_add = True
        for existing_string in string_set:
            if existing_string.lower() in new_string_lower:
                should_add = False
                break

        if should_add:
            string_set.add(new_string)

    result_set = set()
    for string in strings:
        update_string_set(result_set, string)
    return result_set


def check_generation_function(generation_function, test_mode=None):
    assert callable(generation_function), "The generation function must be a function."

    try:
        test_response = generation_function('test')
        assert isinstance(test_response, str), "The generation function must return a string as output."
    except TypeError as e:
        raise AssertionError("The generation function must take only one string as a positional argument.")

    if test_mode == 'list':
        try:
            test_response = generation_function('Give me a list of birds in Python format.')
            clean_list(test_response)
        except Exception as e:
            warnings.warn("The generation function seems not capable enough to respond in Python list format.")


class KeywordFinder:
    def __init__(self, category, domain):
        self.category = category
        self.domain = domain
        self.model = ''
        self.keywords = []
        self.finder_mode = None
        self.kw_targeted_scrap_area_dict = None

    def keywords_to_abcData(self):
        keywords = self.keywords

        if self.finder_mode == "llm" or self.finder_mode == "embedding":
            keywords_dictionary = {}
            if self.finder_mode == "llm":
                keyword_provider = f'{self.finder_mode}_{self.model}'
            else:
                keyword_provider = f'{self.finder_mode}'

            for keyword in keywords:
                keywords_dictionary[keyword] = {
                    "keyword_type": "sub-concepts",
                    "keyword_provider": keyword_provider,
                    "scrap_mode": "in_page",
                    "scrap_shared_area": "Yes"
                }

                if isinstance(self.kw_targeted_scrap_area_dict, dict):
                    targeted_scrap_area = self.kw_targeted_scrap_area_dict[keyword]
                    abcData.check_format(scrap_area_only=True)(targeted_scrap_area)
                    keywords_dictionary[keyword]["targeted_scrap_area"] = targeted_scrap_area

            keyword_entry = {
                "category": self.category,
                "domain": self.domain,
                "keywords": keywords_dictionary
            }

            keywords_abcData = abcData.create_data(category=self.category, domain=self.domain, data_tier='keywords',
                                                   data=[keyword_entry])

            return keywords_abcData

    def find_keywords_by_llm_inquiries(self, n_run=20,
                                       n_keywords=20,
                                       generation_function=None,
                                       model_name=None,
                                       embedding_model=None,
                                       show_progress=True):

        def get_llm_template(template_type, **kwargs):
            category = kwargs.get('category')
            domain = kwargs.get('domain')

            if category and domain:
                template_people = f"""
                Given {category} under the topic {domain}, \
                provide a list of famous names that are representative to the {category}.
                Output your answer in a python list format only and nothing else.
                """

                template_people_short = f"""
                Given {category} under the topic {domain}, \
                provide a list of famous names that are representative to {category}.
                These names can be short family names for which they are identified within the context.
                Output your answer in a python list format only and nothing else.
                """

                template_characteristics = f"""
                Given the {category} under the topic {domain}, \
                provide a list of characteristics that are necessary (!!!) to {category}.
                Output your answer in a python list format only and nothing else.
                """

                template_subcategories = f"""
                Given the {category} under the topic {domain}, \
                provide a list of sub-categories of {category}.
                Output your answer in a python list format only and nothing else.
                """

                template_syn = f"""
                Given the {category} under the topic {domain}, \
                provide a list of synonyms of {category}.
                Output your answer in a python list format only and nothing else.
                """

                template_root = f"""
                Given the {category} under the topic {domain}, \
                provide a list of words that share the same grammatical roots with {category}.
                Output your answer in a python list format only and nothing else.
                """

                if template_type == 'people':
                    return template_people
                elif template_type == 'people_short':
                    return template_people_short
                elif template_type == 'characteristics':
                    return template_characteristics
                elif template_type == 'subcategories':
                    return template_subcategories
                elif template_type == 'synonym':
                    return template_syn
                elif template_type == 'root':
                    return template_root

            print('Template type not found')
            return None

        category = self.category
        domain = self.domain
        if model_name:
            self.model = model_name
        else:
            warnings.warn("Model name not provided. Using the default model name 'user_LLM'")
            self.model = 'user_LLM'
        final_set = {category}
        check_generation_function(generation_function, test_mode='list')

        for _ in tqdm(range(n_run), desc='finding keywords by LLM', unit='run'):
            try:
                if _ == 0 or _ == 1:
                    response = clean_list(
                        generation_function(get_llm_template('root', category=category, domain=domain)))
                    final_set.update(response)
                if _ % 5 == 0:
                    # response = clean_list(agent.invoke(get_template('people_short', category=category, domain=domain)))
                    response = clean_list(
                        generation_function(get_llm_template('subcategories', category=category, domain=domain)))
                elif _ % 5 == 1:
                    # response = clean_list(agent.invoke(get_template('people', category=category, domain=domain)))
                    response = clean_list(
                        generation_function(get_llm_template('characteristics', category=category, domain=domain)))
                elif _ % 5 == 2:
                    response = clean_list(
                        generation_function(get_llm_template('synonym', category=category, domain=domain)))
                elif _ % 5 == 3:
                    response = clean_list(
                        generation_function(get_llm_template('people', category=category, domain=domain)))
                elif _ % 5 == 4:
                    response = clean_list(
                        generation_function(get_llm_template('people_short', category=category, domain=domain)))
                if show_progress:
                    print(f"Response: {response}")
                # Extend the final_set with the response_list
                final_set.update(response)

            except Exception as e:
                print(f"Invocation failed at iteration {_}: {e}")

        self.keywords = list(construct_non_containing_set(list(final_set)))
        if show_progress:
            print('final_set')
            print(final_set)
            print('summary')
            print(self.keywords)
        if len(self.keywords) > n_keywords:
            if embedding_model:
                self.keywords = find_similar_keywords(embedding_model, category, self.keywords, top_n=n_keywords)
            else:
                self.keywords = find_similar_keywords('paraphrase-MiniLM-L6-v2', self.category, self.keywords,
                                                      top_n=n_keywords)
        self.finder_mode = "llm"
        return self.keywords_to_abcData()

    def find_keywords_by_embedding_on_wiki(self, keyword=None,
                                           n_keywords=40, embedding_model='paraphrase-Mpnet-base-v2',
                                           language='en', max_adjustment = 150,
                                           user_agent='AlignBiasCheck/1.0 (contact@holisticai.com)'):
        if not keyword:
            keyword = self.category

        # Search Wikipedia for the keyword
        print('Initiating the embedding model...')
        model = SentenceTransformer(embedding_model)
        try:
            page_content = search_wikipedia(keyword, language, user_agent).text
        except AttributeError as e:
            raise f"Page not Found: {e}"

        if isinstance(page_content, str) and page_content.startswith("No Wikipedia page found"):
            return page_content

        # Tokenize the Wikipedia page content
        tokens = re.findall(r'\b\w+\b', page_content.lower())

        # Get unique tokens
        unique_tokens = list(set(tokens))

        # Get embeddings for unique tokens
        embeddings = {}
        token_embeddings = model.encode(unique_tokens, show_progress_bar=True)
        for token, embedding in zip(unique_tokens, token_embeddings):
            embeddings[token] = embedding
        # # Find most similar words to the keyword
        # if keyword not in embeddings:
        #     raise AssertionError(f"Keyword '{keyword}' not found in the embeddings")

        keyword_embedding = model.encode([keyword.lower()], show_progress_bar=True)[0]
        similarities = {}
        for token, embedding in tqdm(embeddings.items(), desc="Calculating similarities"):
            similarities[token] = util.pytorch_cos_sim(keyword_embedding, embedding).item()

        ADDITIONAL_ITEMS = 20

        # Ensure n_keywords is non-negative and within valid range. Adjusts n_keywords accordingly
        if max_adjustment > 0 and n_keywords + max_adjustment > len(similarities) / 2:
            n_keywords = max(int(len(similarities) / 2) - max_adjustment - 1, 0)

        # Sort tokens by similarity score and select top candidates
        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

        # Ensure not to exceed the available number of items
        num_items_to_select = min(len(sorted_similarities), n_keywords * 2 + ADDITIONAL_ITEMS)
        similar_words_first_pass = sorted_similarities[:num_items_to_select]

        # Convert to dictionary
        words_dict = dict(similar_words_first_pass)

        # Construct non-containing set
        non_containing_set = construct_non_containing_set(words_dict.keys())

        # Filter based on non-containing set
        similar_words_dict = {k: v for k, v in words_dict.items() if k in non_containing_set}

        # Select top keywords
        self.keywords = sorted(similar_words_dict, key=lambda k: similar_words_dict[k], reverse=True)[
                        :min(n_keywords, len(similar_words_dict))]

        # Set mode and return processed data
        self.finder_mode = "embedding"
        return self.keywords_to_abcData()

    def find_name_keywords_by_hyperlinks_on_wiki(self, format='Paragraph', link=None, page_name=None, name_filter=False,
                                                 col_info=None, depth=None, source_tag='default', max_keywords=None):

        def complete_link_page_name_pair(link, page_name, wiki_html):
            def extract_title_from_url(url):
                parsed_url = urlparse(url)
                path = parsed_url.path
                if path.startswith('/wiki/'):
                    title = path[len('/wiki/'):]
                    title = unquote(title.replace('_', ' '))
                    return title
                else:
                    return None

            if page_name == None:
                page_name = extract_title_from_url(link)
                return link, page_name
            if link == None:
                page = wiki_html.page(page_name)
                if page.exists():
                    return page.fullurl, page_name
                else:
                    raise AssertionError('Page link not found. Please provide a valid link or page name.')
            if link == None and page_name == None:
                raise AssertionError("You must enter either the page_name or the link")
            return link, page_name

        def bullet(link, page_name, wiki_html, max_keywords=None):

            p_html = wiki_html.page(page_name)
            # all_urls stores title as key and link as value
            all_urls = {}

            links = p_html.links

            # keywords counter, if max_keywords is given, will only output the top max_keywords from the bullet list.
            counter = 0
            for key in tqdm(links.keys(), desc=f'Finding Keywords by Hyperlinks on Wiki Page {page_name}'):
                k_entity = links[key]
                if k_entity != None and k_entity.title != None and "Category" not in k_entity.title and "List" not in k_entity.title and "Template" not in k_entity.title and "Citation" not in k_entity.title and 'Portal' not in k_entity.title:
                    try:
                        all_urls[k_entity.title] = (k_entity.fullurl)
                        counter += 1

                        if max_keywords != None and counter >= max_keywords:
                            break
                    except KeyError:
                        continue

            return all_urls

        def table(link, col_info, max_keywords=100):
            matched_links = {}
            try:
                page = requests.get(link)
                page.raise_for_status()  # Check for request errors

                soup = BeautifulSoup(page.content, 'html.parser')
                # Find the div with id="mw-content-text"
                mw_content_text_div = soup.find('div', id='mw-content-text')

                # If max_keywords is given, will print out keywords from current table until max_keywords is exceeded.
                # Note: This method will finish keywords in the current table, meaning max_keywords may be exceeded.
                counter = 0
                if mw_content_text_div:
                    # Find all <table> tags within mw-content-text div
                    table_tags = mw_content_text_div.find_all('table')

                    # keeps track of what table user requested for scraping
                    table_num = 1
                    for tab in table_tags:
                        index = 0

                        for each in tqdm(col_info, desc='checking through columns info'):
                            if table_num in each.values():
                                tr_tags = tab.find_all('tr')

                                # Columns are the first row of columns in order to see which column needs scraping according to user's table_info
                                first_row = tr_tags[0].find_all('th')

                                for i in range(len(first_row)):

                                    for col_name in each['column_name']:
                                        if col_name in first_row[i]:
                                            index = i

                                    for tr in tr_tags:
                                        # Find the <td> tag within each <tr> tag according to index
                                        tds = tr.find_all('td')
                                        if tds != None and len(tds) > 0:
                                            term = tds[index + 1]
                                            if term:
                                                # Find the <a> tag within the first <td> tag
                                                a_tag = term.find('a')
                                                if a_tag:
                                                    href_value = a_tag.get('href')
                                                    # Check if href attribute starts with '/wiki/' and does not contain "Category"
                                                    if href_value and href_value.startswith(
                                                            '/wiki/') and "Category" not in href_value:
                                                        matched_links[href_value[6:]] = (
                                                                'https://en.wikipedia.org/' + href_value)
                                                        counter += 1

                        if max_keywords != None and counter >= max_keywords:
                            break

                        table_num += 1
                else:
                    print("Div with id='mw-content-text' not found.")
            except requests.RequestException as e:
                print(f"Request failed: {e}")
            except Exception as e:
                print(f"Failed to find keywords in the wiki page: {e}")

            return matched_links

        def nested(link, page_name, depth, wiki_html, max_keywords=None):

            def default_depth(categorymembers, level=1):

                max_depth = level
                for c in categorymembers.values():
                    if c.ns == wikipediaapi.Namespace.CATEGORY:
                        current_depth = default_depth(c.categorymembers, level + 1)
                        if current_depth > max_depth:
                            max_depth = current_depth
                return max_depth

            def print_categorymembers(categorymembers, all_urls, max_level=1, level=0, max_keywords=None):
                # Initialize a counter to track the number of links added
                if 'count' not in print_categorymembers.__dict__:
                    print_categorymembers.count = 0

                # If max_keywords is specified and the count has reached this limit, return
                if max_keywords is not None and print_categorymembers.count >= max_keywords:
                    return

                for c in categorymembers.values():
                    if c is not None and c.title is not None and "Category" not in c.title and "List" not in c.title and "Template" not in c.title and "Citation" not in c.title and 'Portal' not in c.title:
                        # Try-catch block to prevent KeyError in case fullurl is not present
                        try:
                            if max_keywords is not None and print_categorymembers.count >= max_keywords:
                                return

                            all_urls[c.title] = c.fullurl
                            print_categorymembers.count += 1

                        except KeyError:
                            continue

                    # As long as Category is still the name of the site and level is lower than max_level, recursively call the method again
                    if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
                        print_categorymembers(c.categorymembers, all_urls, max_level=max_level, level=level + 1,
                                              max_keywords=max_keywords)

            p_html = wiki_html.page(page_name)
            all_urls = {}

            if depth == None:
                depth = default_depth(p_html.categorymembers)
                print(f"Default depth is {depth}.")
            # Calls recursive method to iterate through links according to depth
            print_categorymembers(p_html.categorymembers, all_urls, depth, 0, max_keywords)

            return all_urls

        def named_entity_recognition(all_links, max_keywords=None):

            # Uses spacy for named_entity recognition
            nlp = spacy.load("en_core_web_sm")
            pd.set_option("display.max_rows", 200)
            new_links = {}

            count = 0
            for key, url in all_links.items():

                if max_keywords != None and count >= max_keywords:
                    break

                doc = nlp(key)

                # Goes through each entity in the key to search for PERSON label
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        new_links[key] = url
                        count += 1
                        break

            return new_links

        def turn_results_to_abcData(results):
            keywords_dictionary = {}
            for keyword, link in results.items():
                targeted_scrap_area = {
                    "source_tag": source_tag,
                    "scrap_area_type": "wiki_urls",
                    "scrap_area_specification": [link]
                }
                keywords_dictionary[keyword] = {
                    "keyword_type": "name_entity",
                    "keyword_provider": "wiki_hyperlinks",
                    "scrap_mode": "in_page",
                    "scrap_shared_area": "Yes",
                    "targeted_scrap_area": [targeted_scrap_area]
                }

            keyword_entry = {
                "category": self.category,
                "domain": self.domain,
                "keywords": keywords_dictionary
            }

            keywords_abcData = abcData.create_data(category=self.category, domain=self.domain, data_tier='keywords',
                                                   data=[keyword_entry])

            return keywords_abcData

        # Use wikipedia api for parsing through bulleted lists
        wiki_html = wikipediaapi.Wikipedia(
            user_agent="AlignmentBiasCheckingTools/1.0 (contact@holisticai.com)",
            language='en',
            extract_format=wikipediaapi.ExtractFormat.HTML
        )

        link, page_name = complete_link_page_name_pair(link, page_name, wiki_html)

        result_map = {}
        assert format in ['Paragraph', 'Bullet', 'Table',
                          'Nested'], "Invalid format type. It must be either Paragraph, Bullet, Table, or Nested"
        if format == 'Bullet' or format == 'Paragraph':
            result_map = bullet(link, page_name, wiki_html, max_keywords)
        elif format == 'Table':
            if col_info == 'None':
                print("Missing table information")
                return
            result_map = table(link, col_info, max_keywords)
        elif format == 'Nested':
            result_map = nested(link, page_name, depth, wiki_html, max_keywords)

        # Checks to see if user wants NER. This will work for all formats however is most helpful for Paragraph
        if name_filter:
            result_map = named_entity_recognition(result_map, max_keywords)

        return turn_results_to_abcData(result_map)


class ScrapAreaFinder:
    def __init__(self, keyword_abcdata, source_tag='default'):
        assert isinstance(keyword_abcdata, abcData), "You need an abcData as an input."
        keyword_data_tier = keyword_abcdata.data_tier
        assert abcData.tier_order[keyword_data_tier] >= abcData.tier_order['keywords'], "You need an abcData with " \
                                                                                        "data_tier higher than " \
                                                                                        "keywords. "
        self.category = keyword_abcdata.category
        self.domain = keyword_abcdata.domain
        self.data = keyword_abcdata.data
        self.scrap_area = []
        self.source_tage = source_tag
        self.scrap_area_type = 'unknown'

    def scrap_area_to_abcData(self):

        formatted_scrap_area = [{
            "source_tag": self.source_tage,
            "scrap_area_type": self.scrap_area_type,
            "scrap_area_specification": self.scrap_area
        }]

        self.data[0]["category_shared_scrap_area"] = formatted_scrap_area
        scrap_area = abcData.create_data(category=self.category, domain=self.domain, data_tier='scrap_area',
                                         data=self.data)

        return scrap_area

    def find_scrap_urls_on_wiki(self, top_n=5, bootstrap_url=None, language='en',
                                user_agent='AlignmentBiasChecker/1.0 (contact@holisticai.com)'):
        """
        Main function to search Wikipedia for a topic and find related pages.
        """

        def get_related_pages(topic, page, max_depth=1, current_depth=0, visited=None, top_n=50):
            """
            Recursively get related pages up to a specified depth.

            Args:
            - topic (str): The main topic to start the search from.
            - page (Wikipedia page object): The Wikipedia page object of the main topic.
            - max_depth (int): Maximum depth to recurse.
            - current_depth (int): Current depth of the recursion.
            - visited (set): Set of visited pages to avoid loops.

            Returns:
            - list: A list of tuples containing the title and URL of related pages.
            """
            links = page.links
            related_pages = []

            if visited is None:
                visited = set()
                # related_pages.extend([(page.title, page.fullurl)])
                related_pages.extend([page.fullurl])

            visited.add(page.title)

            title_list = [title for title, link_page in links.items()]
            if len(title_list) > top_n:
                title_list = find_similar_keywords('paraphrase-MiniLM-L6-v2', topic, title_list, top_n)

            for title, link_page in tqdm(links.items()):
                if link_page.title not in visited and link_page.title in title_list:
                    try:
                        related_pages.extend([link_page.fullurl])
                    except Exception as e:
                        print(f"Error: {e}")
                if current_depth + 1 < max_depth:
                    related_pages.extend(get_related_pages(topic, link_page, max_depth, current_depth + 1, visited))

            return related_pages

        topic = self.category

        print(f"Searching Wikipedia for topic: {topic}")
        main_page = search_wikipedia(topic, language=language, user_agent=user_agent)
        if top_n == 0:
            main_page_link = search_wikipedia(topic, language=language, user_agent=user_agent).fullurl
            self.scrap_area = [main_page_link]
            self.scrap_area_type = 'wiki_urls'
            return self.scrap_area_to_abcData()

        if isinstance(main_page, str):
            return self.scrap_area_to_abcData()
        else:
            print(f"Found Wikipedia page: {main_page.title}")
            related_pages = get_related_pages(topic, main_page, max_depth=1, top_n=top_n)
            self.scrap_area = related_pages
            self.scrap_area_type = 'wiki_urls'
            return self.scrap_area_to_abcData()

    def find_scrap_paths_local(self, directory_path):
        # Use glob to find all text files in the directory and its subdirectories
        text_files = glob.glob(os.path.join(directory_path, '**/*.txt'), recursive=True)
        file_paths = [file_path.replace('\\', '/') for file_path in text_files]
        self.scrap_area = file_paths
        self.scrap_area_type = 'local_paths'
        return self.scrap_area_to_abcData()


class Scrapper:
    def __init__(self, scrap_area_abcdata):
        assert isinstance(scrap_area_abcdata, abcData), "You need an abcData with data_tier higher than scrap_area."
        keyword_data_tier = scrap_area_abcdata.data_tier
        assert abcData.tier_order[keyword_data_tier] >= abcData.tier_order['scrap_area'], "You need an abcData with " \
                                                                                          "data_tier higher than " \
                                                                                          "scrap_area. "
        self.category = scrap_area_abcdata.category
        self.domain = scrap_area_abcdata.domain
        self.data = scrap_area_abcdata.data
        self.scrap_areas = scrap_area_abcdata.data[0]["category_shared_scrap_area"]
        self.keywords = self.data[0]["keywords"].keys()
        self.extraction_expression = r'(?<=\.)\s+(?=[A-Z])|(?<=\?”)\s+|(?<=\.”)\s+'  # Regex pattern to split sentences
        self.source_tag = 'default'

    def scrapped_sentence_to_abcData(self):
        scrapped_sentences = abcData.create_data(category=self.category, domain=self.domain,
                                                 data_tier='scrapped_sentences',
                                                 data=self.data)
        return scrapped_sentences

    def scrap_in_page_for_wiki_with_buffer_files(self):

        url_links = []
        source_tags_list = []
        for sa_dict in self.scrap_areas:
            if sa_dict["scrap_area_type"] == "wiki_urls":
                url_links.extend(sa_dict["scrap_area_specification"])
                source_tags_list.extend([sa_dict["source_tag"]] * len(sa_dict["scrap_area_specification"]))

        temp_dir = 'temp_results'
        os.makedirs(temp_dir, exist_ok=True)
        for url, source_tag in tqdm(zip(url_links, source_tags_list), desc='Scraping through URL', unit='url',
                                    total=min(len(url_links), len(source_tags_list))):
            url_results = []
            source_tag_buffer = []
            for keyword in tqdm(self.keywords, desc='Scraping in page', unit='keyword'):

                # Fetch the HTML content of the URL
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find all text elements in the HTML
                text_elements = soup.find_all(['p', 'caption', 'figcaption'])

                # Compile regex pattern to match keywords
                keyword_regex = re.compile(r'\b(' + '|'.join([keyword]) + r')\b', re.IGNORECASE)

                # Iterate through each text element
                for element in text_elements:
                    # Remove references like '[42]' and '[page needed]'
                    clean_text = re.sub(r'\[\d+\]|\[.*?\]', '', element.get_text())

                    # Split text into sentences
                    sentences = re.split(self.extraction_expression, clean_text)

                    # Check each sentence for the keyword
                    for sentence in sentences:
                        if len(sentence.split()) >= 6 and keyword_regex.search(sentence):
                            url_results.append(sentence.strip())
                            source_tag_buffer.append(source_tag)

                # Store results in a temporary file
                temp_file = os.path.join(temp_dir, f'{url.replace("https://", "").replace("/", "_")}_{keyword}.txt')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    for sentence in url_results:
                        f.write(f'{sentence}\n')

                temp_file_source_tag = os.path.join(temp_dir,
                                                    f'{url.replace("https://", "").replace("/", "_")}_{keyword}_source_tag.txt')
                with open(temp_file_source_tag, 'w', encoding='utf-8') as f:
                    for source_tag in source_tag_buffer:
                        f.write(f'{source_tag}\n')

        # Read from temporary files and aggregate results
        for keyword in self.keywords:
            aggregated_results = []
            aggregated_source_tags = []
            for url in url_links:
                temp_file = os.path.join(temp_dir, f'{url.replace("https://", "").replace("/", "_")}_{keyword}.txt')
                temp_file_source_tag = os.path.join(temp_dir,
                                                    f'{url.replace("https://", "").replace("/", "_")}_{keyword}_source_tag.txt')
                if os.path.exists(temp_file) and os.path.exists(temp_file_source_tag):
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        sentences = f.readlines()
                        aggregated_results.extend([sentence.strip() for sentence in sentences])
                    with open(temp_file_source_tag, 'r', encoding='utf-8') as f:
                        source_tags = f.readlines()
                        aggregated_source_tags.extend([source_tag.strip() for source_tag in source_tags])
                    os.remove(temp_file)
                    os.remove(temp_file_source_tag)
            aggregated_results_with_source_tag = list(zip(aggregated_results, aggregated_source_tags))

            # Store aggregated results in the data structure
            if "scrapped_sentences" in self.data[0]["keywords"][keyword].keys():
                self.data[0]["keywords"][keyword]["scrapped_sentences"].extend(aggregated_results_with_source_tag)
            else:
                self.data[0]["keywords"][keyword]["scrapped_sentences"] = aggregated_results_with_source_tag
        return self.scrapped_sentence_to_abcData()

    def scrap_local_with_buffer_files(self):

        temp_dir = 'temp_results'
        os.makedirs(temp_dir, exist_ok=True)

        file_paths = []
        source_tags_list = []
        for sa_dict in self.scrap_areas:
            if sa_dict["scrap_area_type"] == "local_paths":
                file_paths.extend(sa_dict["scrap_area_specification"])
                source_tags_list.extend([sa_dict["source_tag"]] * len(sa_dict["scrap_area_specification"]))

        for file_path, source_tag in tqdm(zip(file_paths, source_tags_list), desc='Scraping through loacal files',
                                          unit='file', total=min(len(file_paths), len(source_tags_list))):
            path_results = []
            source_tag_buffer = []

            for keyword in tqdm(self.keywords, desc='Scraping in page', unit='keyword'):

                # Read the file content
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                # Clean the text by removing citations and other patterns within square brackets
                text = text.replace('.\n', '. ').replace('\n', ' ')
                clean_text = re.sub(r'\[\d+\]|\[.*?\]', '', text)

                # Define a regex pattern for the keyword
                keyword_regex = re.compile(re.escape(keyword), re.IGNORECASE)

                # Split the cleaned text into sentences
                sentences = re.split(r'(?<=\.)\s+(?=[A-Z])|(?<=\?”)\s+|(?<=\.”)\s+', clean_text)

                # Extract desired sentences
                for sentence in sentences:
                    if len(sentence.split()) >= 6 and keyword_regex.search(sentence):
                        path_results.append(sentence.strip())
                        source_tag_buffer.append(source_tag)

                # Store results in a temporary file
                temp_file = os.path.join(temp_dir,
                                         f'{file_path.replace(".", "_").replace("/", "_")}_{keyword}.txt')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    for sentence in path_results:
                        f.write(f'{sentence}\n')

                temp_file_source_tag = os.path.join(temp_dir,
                                                    f'{file_path.replace(".", "_").replace("/", "_")}_{keyword}_source_tag.txt')
                with open(temp_file_source_tag, 'w', encoding='utf-8') as f:
                    for source_tag in source_tag_buffer:
                        f.write(f'{source_tag}\n')

        for keyword in self.keywords:
            aggregated_results = []
            aggregated_source_tags = []
            for file_path in file_paths:
                temp_file = os.path.join(temp_dir,
                                         f'{file_path.replace(".", "_").replace("/", "_")}_{keyword}.txt')
                temp_file_source_tag = os.path.join(temp_dir,
                                                    f'{file_path.replace(".", "_").replace("/", "_")}_{keyword}_source_tag.txt')
                if os.path.exists(temp_file) and os.path.exists(temp_file_source_tag):
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        sentences = f.readlines()
                        aggregated_results.extend([sentence.strip() for sentence in sentences])
                    with open(temp_file_source_tag, 'r', encoding='utf-8') as f:
                        source_tags = f.readlines()
                        aggregated_source_tags.extend([source_tag.strip() for source_tag in source_tags])
                    os.remove(temp_file)
                    os.remove(temp_file_source_tag)
            aggregated_results_with_source_tag = list(zip(aggregated_results, aggregated_source_tags))

            # Store aggregated results in the data structure
            if "scrapped_sentences" in self.data[0]["keywords"][keyword].keys():
                self.data[0]["keywords"][keyword]["scrapped_sentences"].extend(
                    aggregated_results_with_source_tag)
            else:
                self.data[0]["keywords"][keyword]["scrapped_sentences"] = aggregated_results_with_source_tag
        return self.scrapped_sentence_to_abcData()


class PromptMaker:
    def __init__(self, scrapped_sentence_abcdata):
        assert isinstance(scrapped_sentence_abcdata, abcData), "You need an abcData of scrapped_sentences data_tier."
        keyword_data_tier = scrapped_sentence_abcdata.data_tier
        assert abcData.tier_order[keyword_data_tier] == abcData.tier_order[
            'scrapped_sentences'], "You need an abcData of scrapped_sentences data_tier."
        self.category = scrapped_sentence_abcdata.category
        self.domain = scrapped_sentence_abcdata.domain
        self.data = scrapped_sentence_abcdata.data
        self.output_df = pd.DataFrame()

        download('punkt')
        download('stopwords')
        # Load the English model for spaCy
        self.nlp = spacy.load("en_core_web_sm")

    def output_df_to_abcData(self):
        return abcData.create_data(category=self.category, domain=self.domain, data_tier='split_sentences',
                                   data=self.output_df)

    def split_sentences(self, kw_check=False, keyword=None):

        def split_individual_sentence(sentence, kw_check=False, keyword=None):

            def is_within_brackets(token, doc):
                """
                Check if a token is within parentheses.
                """
                open_paren = False
                for t in doc[:token.i]:
                    if t.text == '(':
                        open_paren = True
                for t in doc[token.i:]:
                    if t.text == ')':
                        return open_paren
                return False

            # Process the sentence with spaCy
            doc = self.nlp(sentence)

            # Initialize verb_index to -1 (meaning no verb found yet)
            verb_index = -1

            # Flag to indicate if a verb was found after 6 words
            found_after_six_words = False

            # Loop to find the first verb after the first six words
            for i, token in enumerate(doc):
                if i > 5 and (token.pos_ == "VERB" or token.dep_ == "ROOT") and not is_within_brackets(token,
                                                                                                       doc) and not token.text.istitle():
                    verb_index = token.i
                    found_after_six_words = True
                    break

            # If no verb is found after the first six words, search for the first verb in the sentence
            if not found_after_six_words:
                for token in doc:
                    if (token.pos_ == "VERB" or token.dep_ == "ROOT") and not is_within_brackets(token,
                                                                                                 doc) and not token.text.istitle():
                        verb_index = token.i
                        break

            # If no verb is found, return the original sentence
            if verb_index == -1:
                return sentence, "", False

            # Calculate the split index (3 words after the verb)
            split_index = verb_index + 4  # Including the verb itself and three words after it

            # Ensure the split index is within bounds
            if split_index >= len(doc):
                split_index = len(doc)

            # Convert doc to list of tokens
            tokens = [token.text for token in doc]

            # Split the sentence
            part1 = " ".join(tokens[:split_index])
            part2 = " ".join(tokens[split_index:])
            success = True

            if kw_check and keyword:
                if keyword.lower() not in part1.lower():
                    success = False

            return part1, part2, success

        # Initialize the list to store the split sentences
        results = []
        for category_item in self.data:
            category = category_item.get("category")
            domain = category_item.get("domain")
            for keyword, keyword_data in tqdm(category_item['keywords'].items()):
                for sentence_with_tag in tqdm(keyword_data['scrapped_sentences']):
                    part1, part2, success = split_individual_sentence(sentence_with_tag[0], True, keyword=keyword)

                    if part2:
                        result = {
                            "keyword": keyword,
                            "category": category,
                            "domain": domain,
                            "prompts": part1,
                            "baseline": sentence_with_tag[0],
                            "keywords_containment": success,
                            "source_tag": sentence_with_tag[1],
                        }
                        results.append(result)

                # Create a DataFrame
                self.output_df = pd.DataFrame(results)

        return self.output_df_to_abcData()

    def make_questions(self, generation_function, keyword_reference=None, answer_check=True, max_questions=None):

        def get_question(sentence, generation_function, keyword, keyword_list=None, bad_questions=None, Example=True):
            prompt_qa_making = f" Write a question about '{keyword}' such that the following sentence '''{sentence}''' can act as an accurate answer (!!!). \n" \
                               f" For each generated question make sure that the '{keyword}' is included. This is very important to do so, the question must have atleast one mention of '{keyword}'. " \
                               f"So don't include any names, specific details, or situations belonging specifically to the keyword." \

            if (keyword_list is not None) and (len(keyword_list) > 1):
                prompt_qa_making = ''.join([prompt_qa_making,
                                            f" Make these questions generic enough that all the keywords in '{keyword_list}' can also be subsituted as the current keyword as the question would still make sense. "])

            if (bad_questions is not None) and (len(bad_questions) > 0):
                prompt_qa_making = ''.join([prompt_qa_making,
                                            f" Make sure that this question is not similar to any of the questions in '{bad_questions.values()}'. If it is, recreate a new question. "])

            if Example:
                addition = f"\n For example, don't ask a very specific question like What announcement was made by Sam Altman and Peng Xiao regarding Open AI technology?! because if I replaced" \
                           f" Open AI with Google Deepmind this question wouldn't make sense. Instead ask a question like what did the co-founders say regarding Open AI technology? " \
                           f" Or for a question like What did OpenAI clarify or deny regarding its relationship with Elon and Microsoft? Instead say How did OpenAI respond regarding relationships with other companies?" \
                           f" If you can not generate a question, output 'NO'. Otherwise, output the question directly without anything else. "
                prompt_qa_making = ''.join([prompt_qa_making, addition])

            format_instruct = f" Don't add any notes or extra comments in paranthesis. No extra lines either. Just the response"
            prompt_qa_making = ''.join([prompt_qa_making, format_instruct])

            gf_question = generation_function(prompt_qa_making)

            return gf_question

        def check_question(question, generation_function, keyword, keyword_list=None, bad_questions=None,
                           answer_check=True):
            keyword_dict = {}
            if bad_questions == None:
                bad_questions = []
            if keyword_list == None:
                keyword_list = []
            for each in keyword_list:
                if keyword != each:
                    prompt_new_question = f"In the following {question}, replace every instance of {keyword} with {each}. There should be no {keyword} in the question" \
                                          f"Don't modify anything else in the question. The rest of the question must be the same " \
                                          f"Make sure to only replace {keyword} with {each} and to keep the rest of the question the same. " \
                                          f"Don't add any other of your notes, comments, or questions. " \
                                          f"No text before or after the question, the response must be only be the question."
                    prompt_new_question = generation_function(prompt_new_question)
                else:
                    prompt_new_question = question

                keyword_dict[each] = prompt_new_question

                # only if user puts answer_check. This is an added feature to make sure that the generated
                # question actually has a valid question and that the answer makes sense.
                if answer_check:
                    prompt_new_answer = f"Answer the following question: {prompt_new_question}. " \
                                        f"Find the answer of the question in a sentence from an actual online" \
                                        f"source, don't just make up an answer. Don't answer back in a question. " \
                                        f"Also add the source where you found the answer to the end."
                    prompt_new_answer = generation_function(prompt_new_answer)

                    prompt_check = f"Check if {prompt_new_answer} answers {prompt_new_question} correctly " \
                                   f"and if it makes sense. Be able to check if an answer properly answers the" \
                                   f"question given by checking to see if the answer makes sense given the question. " \
                                   f"Answer in simple Yes or No, I don't want any explanation or extra lines or extra words, the answer must be only one word, either a Yes or No."
                    prompt_check = generation_function(prompt_check)

                    # If prompt_check is No which means the answer doesn't make sense in context of the question, add
                    # the bad_question to the list of bad_questions and returns so the LLM won't regenerate the same bad question.
                    if prompt_check == 'No':
                        bad_questions.append(prompt_new_question)
                        return False, {'bad questions': bad_questions}

            return True, keyword_dict

        results = []
        for category_item in self.data:
            category = category_item.get("category")
            domain = category_item.get("domain")
            key_dict = {}

            for keyword, keyword_data in tqdm(category_item['keywords'].items(), desc="Going through keywords"):
                for sentence_with_tag in tqdm(keyword_data['scrapped_sentences'],
                                              desc="Going through scrapped sentences"):
                    if max_questions != None and len(results) >= max_questions:
                        break
                    question = get_question(sentence=sentence_with_tag[0], generation_function=generation_function,
                                            keyword=category, keyword_list=keyword_reference)
                    if keyword_reference is not None:
                        if len(keyword_reference) > 1 and answer_check:
                            # only check_question to find questions for other keywords if key_word list is greater than 1
                            check, key_dict = check_question(question=question, generation_function=generation_function,
                                                             keyword=category, keyword_list=keyword_reference,
                                                             answer_check=answer_check)
                        else:
                            check = True
                    else:
                        # otherwise, if only one keyword then no need to check question.
                        check = True

                    # Tries two more times to see if valid question can be made from scrapped sentence.
                    bad_count = 0
                    while check == False and bad_count < 2:
                        print("Trying question generation again")
                        question = get_question(sentence=sentence_with_tag[0], generation_function=generation_function,
                                                keyword=category, keyword_list=keyword_reference,
                                                bad_questions=key_dict)
                        check, key_dict = check_question(question=question, generation_function=generation_function,
                                                         keyword=category, keyword_list=keyword_reference,
                                                         bad_questions=key_dict, answer_check=answer_check)
                        bad_count += 1

                    if check:
                        result = {
                            "keyword": category,
                            "category": category,
                            "domain": domain,
                            "prompts": question,
                            "baseline": sentence_with_tag[0],
                            "source_tag": sentence_with_tag[1],
                        }
                        results.append(result)

                self.output_df = pd.DataFrame(results)

            return self.output_df_to_abcData()

    def merge(self, prompt_df):
        self.output_df = pd.concat([self.output_df, prompt_df])
        return self.output_df_to_abcData()

    def branching(self, branching_config = None):

        df = self.output_df
        default_branching_config = {
            'branching_pairs': 'all',
            'direction': 'both',
            'source_restriction': None,
            'replacement_descriptor_require': True,
            'descriptor_threshold': 'Auto',
            'descriptor_embedding_model': 'paraphrase-Mpnet-base-v2',
            'descriptor_distance': 'cosine',
            'replacement_description': {},
            'replacement_description_saving': True,
            'replacement_description_saving_location': f'data/customized/split_sentences/{self.domain}_replacement_description.json',
            'counterfactual_baseline': True,
            'generation_function': None,
        }
        if branching_config is None:
            branching_config = {}
        branching_config = BenchmarkBuilder.update_configuration(default_branching_config, branching_config)


        def replacement_descriptor(df, original_category, replace_category, replacing: list[str] or str, gf=None,
                                   embedding_model='paraphrase-Mpnet-base-v2',
                                   descriptor_threold='Auto', descriptor_distance='cosine'):

            if isinstance(replacing, str):
                replacing = [replacing]

            def find_similar_words(word_bank, target_word, model_name=embedding_model, threshold=0.2):
                model = SentenceTransformer(model_name)
                tokens = word_bank.lower().split()  # Tokenize the word_bank

                # Get embeddings for each token and the target word
                target_vector = model.encode(target_word, convert_to_tensor=True)
                similar_tokens = []

                def cosine_similarity(vec1, vec2):
                    return util.cos_sim(vec1, vec2).item()

                for token in tokens:
                    token_vector = model.encode(token, convert_to_tensor=True)
                    similarity = cosine_similarity(target_vector, token_vector)
                    if similarity >= threshold:  # Apply the threshold
                        similar_tokens.append((token, similarity))

                similar_tokens.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity
                return similar_tokens

            def clean_sentences_and_join(sentence_list):
                return ' '.join(sentence_list) \
                    .replace('?', '').replace('.', '') \
                    .replace(',', '').replace('!', '') \
                    .replace(':', '').replace(';', '') \
                    .replace('(', '').replace(')', '') \
                    .replace('[', '').replace(']', '') \
                    .replace('{', '').replace('}', '') \
                    .replace('"', '').replace("'", '') \
                    .replace('`', '').replace('~', '') \
                    .replace('@', '').replace('#', '') \
                    .replace('$', '').replace('%', '') \
                    .replace('^', '').replace('&', '') \
                    .replace('*', '')

            def check_if_threshold_can_go_higher(similar_tokens, threshold, target_word, gf=gf):

                vocabs = [word for word, similarity in similar_tokens if similarity < threshold][:15]

                prompts = f"Do you find any words in the below list that can be associated with the word '''{target_word}'''? \n" \
                          f"Words: {vocabs}\n " \
                          f"For example: 'actress' or 'ovum' are associated with 'females' while 'Sam Altman' is associated with the 'Open AI'. \n" \
                          f"Output 'Yes' or 'No' directly."

                response = gf(prompts)

                if response.startswith('Yes'):
                    return False
                else:
                    return True

            def iterative_guessing(sorted_list_threshold, check_go_higher, max_iterations = 50):
                remaining = sorted_list_threshold[:]
                lower_bound = None
                iteration_count = 0

                while len(remaining) > 2 and iteration_count < max_iterations:
                    guess = remaining[len(remaining) // 2]
                    iteration_count += 1

                    if check_go_higher(guess):
                        lower_bound = guess
                        remaining = [i for i in remaining if i >= guess]
                    else:
                        remaining = [i for i in remaining if i <= guess]
                        # print(f"Remaining: {remaining}")

                # After narrowing down, the remaining list will contain the correct number
                correct_guess = remaining[-1]

                return correct_guess

            def replacer_prompts(target_word, branch, words):
                return f" The following words are about'''{target_word}''' and your job is to find the analogous words about '''{branch} \n" \
                       f" Words: {words}\n " \
                       f" For example: 'women' for female is associated with 'men' for male " \
                       ' Give a dictionary of the following python json format only: [{"word1": "analogy1", "word2": "analogy2" ....}]. ' \
                       f" You only need to provide words that you can find analogy."

            def dict_extraction(response):
                # If the JSON is embedded in a larger string, extract it
                pattern = r'\[.*?\]'
                match = re.search(pattern, response, re.DOTALL)

                if match:
                    json_string = match.group(0).replace("'", '"')
                    try:
                        # Parse the JSON
                        json_data = json.loads(json_string)
                        return (json_data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                else:
                    print("No JSON found in the string.")


            df_category = df[df['category'] == original_category]
            word_bank = ''
            for replace in replacing:
                word_bank += ' '.join(list(set(clean_sentences_and_join(df_category[replace].tolist()).split(' '))))
            if descriptor_threold == 'Auto':
                print('Obtaining the similar words...')
                similar_tokens = find_similar_words(word_bank, original_category)
                print('Obtaining the threshold...')
                thresholds_list = [similarity for word, similarity in similar_tokens]
                checker = lambda x: check_if_threshold_can_go_higher(similar_tokens, x, original_category)
                threshold = iterative_guessing(thresholds_list, checker)
                words = [word for word, similarity in similar_tokens if similarity >= threshold]

            else:
                similar_tokens = find_similar_words(word_bank, original_category, threshold=float(descriptor_threold))
                words = [word for word, similarity in similar_tokens]

            print('Obtaining the replacement...')
            k = 0
            while k < 5:
                k += 1
                try:
                    result = dict_extraction(gf(replacer_prompts(original_category, replace_category, words)))
                    assert isinstance(result, list) and all(isinstance(item, dict) for item in result)
                    combined_dict = {k: v for d in result for k, v in d.items()}
                    return combined_dict
                except AssertionError:
                    print('Try again...')
                    continue
            print(f'Failed to obtain the replacement for {original_category} and {replace_category}...')
            return {}

        def add_and_clean_replacement_pairs(replacement_dict):
                for outer_key, inner_dict in replacement_dict.items():
                    for sub_key, replacements in inner_dict.items():
                        # Collect pairs to remove
                        pairs_to_remove = []

                        for a, b in replacements.items():
                            # Check if a contains outer_key or outer_key is contained by a
                            if outer_key.lower() in a.lower() or a.lower() in outer_key.lower():
                                pairs_to_remove.append(a)
                            # Check if b contains sub_key or sub_key is contained by b
                            elif sub_key.lower() in b.lower() or b.lower() in sub_key.lower():
                                pairs_to_remove.append(a)

                        # Remove conflicting pairs
                        for a in pairs_to_remove:
                            del replacements[a]

                        # After removing conflicts, add the new pair if no conflict
                        if outer_key not in replacements and sub_key not in replacements.values():
                            if outer_key not in replacements and sub_key not in replacements:
                                replacements[outer_key] = sub_key

                return replacement_dict

        def replace_terms(sentence, replacement_dict):

            replacement_dict = {k.lower(): v for k, v in replacement_dict.items()}
            reverse_replacement_dict = {v.lower(): k for k, v in replacement_dict.items()}
            replacement_dict.update(reverse_replacement_dict)

            sentence = sentence.lower()

            # Create a regular expression pattern that matches any of the phrases
            pattern = re.compile("|".join(re.escape(phrase) for phrase in replacement_dict.keys()))

            # Function to replace matched phrases using the replacement dictionary
            def replace_match(match):
                return replacement_dict[match.group(0)]

            # Replace all matched phrases with their corresponding replacements
            modified_sentence = pattern.sub(replace_match, sentence)

            return modified_sentence

        def replace_gender_terms_arc(sentence, replacement_dictionary):
            # Step 1: Define the replacement dictionary

            # Step 2: Extend the dictionary to include reverse replacements
            reverse_replacement_dictionary = {v: k for k, v in replacement_dictionary.items()}
            full_replacement_dictionary = {**replacement_dictionary, **reverse_replacement_dictionary}
            # Lower the keys
            full_lower_replacement_dictionary = {k.lower(): v for k, v in full_replacement_dictionary.items()}
            print(full_lower_replacement_dictionary)


            # Step 3: Tokenize the sentence
            tokens = re.findall(r'\b\w+\b', sentence)

            # Step 4: Replace the words according to the dictionary
            replaced_tokens = [full_lower_replacement_dictionary.get(token.lower(), token) for token in tokens]

            # Step 5: Reassemble the sentence
            replaced_sentence = ' '.join(replaced_tokens)

            return replaced_sentence



        replacement_description = branching_config['replacement_description']
        gef = branching_config['generation_function']

        if branching_config['branching_pairs'] == 'all':
            branching_pairs = list(itertools.combinations(df['category'].unique().tolist(), 2))
            # Include the reverse of each pair
            branching_pairs = branching_pairs + [(b, a) for a, b in branching_pairs]
            # Optionally, you can remove duplicates if needed
            branching_pairs = list(set(branching_pairs))
        else:
            branching_pairs = [(key, sub_key) for key, sub_dict in replacement_description.items() for sub_key in
                               sub_dict.keys()]
            # Include the reverse of each pair
            if branching_config['direction'] == 'both':
                branching_pairs = branching_pairs + [(b, a) for a, b in branching_pairs]
                # Optionally, you can remove duplicates if needed
                branching_pairs = list(set(branching_pairs))

        if branching_config['source_restriction'] is not None:
            df = df[df['source_tag'] == branching_config['source_restriction']]

        df_result = df.copy()
        for category_pair in tqdm(branching_pairs, desc='Branching pairs'):
            if branching_config['replacement_descriptor_require']:
                assert gef is not None, "Generation function is required for replacement descriptor generation."
                if branching_config['counterfactual_baseline']:
                    rd = replacement_descriptor(df, category_pair[0], category_pair[1], ['baseline', 'prompts'], gf=gef)
                else:
                    rd = replacement_descriptor(df, category_pair[0], category_pair[1], ["prompts"], gf=gef)
                # Ensure category_pair[0] exists in replacement_description
                if category_pair[0] not in replacement_description:
                    replacement_description[category_pair[0]] = {}

                # Ensure category_pair[1] exists within the nested dictionary
                if category_pair[1] not in replacement_description[category_pair[0]]:
                    replacement_description[category_pair[0]][category_pair[1]] = {}

                # Update the existing dictionary with the contents of rd
                replacement_description[category_pair[0]][category_pair[1]].update(rd)
                replacement_description = add_and_clean_replacement_pairs(replacement_description)
                if branching_config['replacement_description_saving']:
                    with open(branching_config['replacement_description_saving_location'], 'w', encoding='utf-8') as f:
                        json.dump(replacement_description, f)
            else:
                replacement_description = add_and_clean_replacement_pairs(replacement_description)

            rd = replacement_description[category_pair[0]][category_pair[1]]
            print(rd)
            print('Replacing...')
            df_new = df[df['category'] == category_pair[0]].copy()
            df_new['prompts'] = df_new['prompts'].apply(lambda x: replace_terms(x, rd).title())
            if branching_config['counterfactual_baseline']:
                df_new['baseline'] = df_new['baseline'].apply(lambda x: replace_terms(x, rd).title())
            df_new['source_tag'] = df_new.apply(lambda row: f'br_{row["source_tag"]}_cat_{row["category"]}', axis=1)
            df_new['category'] = df_new['category'].apply(lambda x: replace_terms(x, rd))
            df_new['keyword'] = df_new['keyword'].apply(lambda x: replace_terms(x, rd))
            df_result = pd.concat([df_result, df_new])

            self.output_df = df_result

        return self.output_df_to_abcData()




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
                'hyperlinks_info': [],
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

        keyword_finder_require = configuration['keyword_finder']['require']
        keyword_finder_reading_location = configuration['keyword_finder']['reading_location']
        keyword_finder_method = configuration['keyword_finder']['method']
        keyword_finder_keyword_number = configuration['keyword_finder']['keyword_number']
        keyword_finder_hyperlinks_info = configuration['keyword_finder']['hyperlinks_info']
        keyword_finder_llm_info = configuration['keyword_finder']['llm_info']
        keyword_finder_max_adjustment = configuration['keyword_finder']['max_adjustment']
        keyword_finder_embedding_model = configuration['keyword_finder']['embedding_model']
        keyword_finder_saving = configuration['keyword_finder']['saving']
        keyword_finder_saving_location = configuration['keyword_finder']['saving_location']
        keyword_finder_manual_keywords = configuration['keyword_finder']['manual_keywords']

        scrap_area_finder_require = configuration['scrap_area_finder']['require']
        scrap_area_finder_reading_location = configuration['scrap_area_finder']['reading_location']
        scrap_area_finder_method = configuration['scrap_area_finder']['method']
        scrap_area_local_file = configuration['scrap_area_finder']['local_file']
        scrap_area_finder_saving = configuration['scrap_area_finder']['saving']
        scrap_area_finder_saving_location = configuration['scrap_area_finder']['saving_location']
        scrap_area_finder_scrap_area_number = configuration['scrap_area_finder']['scrap_number']

        scrapper_require = configuration['scrapper']['require']
        scrapper_reading_location = configuration['scrapper']['reading_location']
        scrapper_saving = configuration['scrapper']['saving']
        scrapper_method = configuration['scrapper']['method']
        scrapper_saving_location = configuration['scrapper']['saving_location']

        prompt_maker_require = configuration['prompt_maker']['require']
        prompt_maker_method = configuration['prompt_maker']['method']
        prompt_maker_generation_function = configuration['prompt_maker']['generation_function']
        prompt_maker_keyword_list = configuration['prompt_maker']['keyword_list']
        prompt_maker_answer_check = configuration['prompt_maker']['answer_check']
        prompt_maker_saving_location = configuration['prompt_maker']['saving_location']
        prompt_maker_max_sample_number = configuration['prompt_maker']['max_benchmark_length']

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
                filePath = f'tests/data/customized/keywords/{domain}_{demographic_label}_keywords.json'
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
                    top_n=scrap_area_finder_scrap_area_number)
            elif scrap_area_finder_method == 'local_files':
                if scrap_area_local_file == None:
                    raise ValueError(f"Unable to read keywords from {scrap_area_local_file}. Can't scrap area.")
                sa = ScrapAreaFinder(kw, source_tag='local').find_scrap_paths_local(scrap_area_local_file)


            if scrap_area_finder_saving:
                if scrap_area_finder_saving_location == 'default':
                    sa.save()
                else:
                    sa.save(file_path=scrap_area_finder_saving_location)


        elif scrapper_require:
            filePath = ""
            if scrap_area_finder_reading_location == 'default':
                filePath = f'tests/data/customized/scrap_area/{domain}_{demographic_label}_scrap_area.json'
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
                filePath = f'tests/data/customized/scrapped_sentences/{domain}_{demographic_label}_scrapped_sentences.json'
                sc = abcData.load_file(domain=domain, category=demographic_label,
                                       file_path=filePath,
                                       data_tier='scrapped_sentences')
                print(
                    f'Scrapped sentences loaded from tests/data/customized/scrapped_sentences/{domain}_{demographic_label}_scrapped_sentences.json')
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