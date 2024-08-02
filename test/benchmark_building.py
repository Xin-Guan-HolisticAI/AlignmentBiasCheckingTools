import os

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import glob

from assistants import OllamaModel
import wikipediaapi
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
import warnings
from abcData import abcData

import spacy
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download

from urllib.parse import urlparse, unquote



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
                # print(title)
                related_pages.extend([link_page.fullurl])
            except Exception as e:
                print(f"Error: {e}")
        if current_depth + 1 < max_depth:
            related_pages.extend(get_related_pages(topic, link_page, max_depth, current_depth + 1, visited))

    return related_pages


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
        # print(f"Current set: {result_set}")
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
            response_list = clean_list(test_response)
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
                        generation_function(self.get_llm_template('root', category=category, domain=domain)))
                    final_set.update(response)
                if _ % 5 == 0:
                    # response = clean_list(agent.invoke(get_template('people_short', category=category, domain=domain)))
                    response = clean_list(
                        generation_function(self.get_llm_template('subcategories', category=category, domain=domain)))
                elif _ % 5 == 1:
                    # response = clean_list(agent.invoke(get_template('people', category=category, domain=domain)))
                    response = clean_list(
                        generation_function(self.get_llm_template('characteristics', category=category, domain=domain)))
                elif _ % 5 == 2:
                    response = clean_list(
                        generation_function(self.get_llm_template('synonym', category=category, domain=domain)))
                elif _ % 5 == 3:
                    response = clean_list(
                        generation_function(self.get_llm_template('people', category=category, domain=domain)))
                elif _ % 5 == 4:
                    response = clean_list(
                        generation_function(self.get_llm_template('people_short', category=category, domain=domain)))
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
                                           language='en',
                                           user_agent='YourAppName/1.0 (yourname@example.com)'):
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
            print("No wiki")
            return page_content

        # Tokenize the Wikipedia page content
        tokens = re.findall(r'\b\w+\b', page_content.lower())
        # print("Tokens:", tokens)

        # Get unique tokens
        unique_tokens = list(set(tokens))
        print("Unique Tokens:", unique_tokens)

        # Get embeddings for unique tokens
        embeddings = {}
        token_embeddings = model.encode(unique_tokens, show_progress_bar=True)
        for token, embedding in zip(unique_tokens, token_embeddings):
            embeddings[token] = embedding
        print("Embeddings:", {k: v[:5] for k, v in embeddings.items()})  # Printing first 5 values for brevity

        # # Find most similar words to the keyword
        # if keyword not in embeddings:
        #     raise AssertionError(f"Keyword '{keyword}' not found in the embeddings")

        keyword_embedding = model.encode([keyword.lower()], show_progress_bar=True)[0]
        similarities = {}
        for token, embedding in tqdm(embeddings.items(), desc="Calculating similarities"):
            similarities[token] = util.pytorch_cos_sim(keyword_embedding, embedding).item()
        
        MAX_ADJUSTMENT = 150
        ADDITIONAL_ITEMS = 20

        # Ensure n_keywords is non-negative and within valid range. Adjusts n_keywords accordingly
        if n_keywords + MAX_ADJUSTMENT > len(similarities) / 2:
            n_keywords = max(int(len(similarities) / 2) - MAX_ADJUSTMENT - 1, 0)

        print("Adjusted n_keywords:")
        print(n_keywords)

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

        print("Filtered Similar Words Dictionary:")
        print(similar_words_dict)

        # Select top keywords
        self.keywords = sorted(similar_words_dict, key=lambda k: similar_words_dict[k], reverse=True)[:min(n_keywords, len(similar_words_dict))]

        print("Top Keywords:")
        print(self.keywords)

        # Set mode and return processed data
        self.finder_mode = "embedding"
        return self.keywords_to_abcData()
    
    def find_name_keywords_by_hyperlinks_on_wiki(self, format='Paragraph', link=None, page_name=None, name_filter=False,
                                                 col_info=None, depth=1, source_tag='default'):

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

        def bullet(page_name, wiki_html):

            p_html = wiki_html.page(page_name)
            # all_urls stores title as key and link as value
            all_urls = {}

            links = p_html.links

            for key in tqdm(links.keys(), desc=f'Finding Keywords by Hyperlinks on Wiki Page {page_name}'):
                # print(key)
                k_entity = links[key]
                if k_entity != None and k_entity.title != None and "Category" not in k_entity.title and "List" not in k_entity.title and "Template" not in k_entity.title and "Citation" not in k_entity.title and 'Portal' not in k_entity.title:
                    try:
                        all_urls[k_entity.title] = (k_entity.fullurl)
                    except KeyError:
                        continue

            return all_urls

        def table(link, col_info):
            matched_links = {}
            try:
                page = requests.get(link)
                page.raise_for_status()  # Check for request errors

                soup = BeautifulSoup(page.content, 'html.parser')
                # Find the div with id="mw-content-text"
                mw_content_text_div = soup.find('div', id='mw-content-text')

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

                                    print(each['column_name'])
                                    print(first_row)

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
                        table_num += 1
                else:
                    print("Div with id='mw-content-text' not found.")
            except requests.RequestException as e:
                print(f"Request failed: {e}")
            except Exception as e:
                print(f"Failed to find keywords in the wiki page: {e}")

            return matched_links

        def nested(page_name, depth):

            def default_depth(categorymembers, level=1):

                max_depth = level
                for c in tqdm(categorymembers.values(), desc = "Calculating depth", unit = "category", total= len(categorymembers)):
                    if c.ns == wikipediaapi.Namespace.CATEGORY:
                        current_depth = default_depth(c.categorymembers, level + 1)
                        if current_depth > max_depth:
                            max_depth = current_depth
                return max_depth

            def print_categorymembers(categorymembers, all_urls, max_level=1, level=0):
                for c in tqdm(categorymembers.values(), desc = "Processing categories", unit = "category", total=len(categorymembers)):
                    if c != None and c.title != None and "Category" not in c.title and "List" not in c.title and "Template" not in c.title and "Citation" not in c.title and 'Portal' not in c.title:
                        # Try catch block to prevent KeyError in case fullurl is not present
                        try:
                            all_urls[c.title] = c.fullurl
                        except KeyError:
                            continue

                    # As long as Category is still the name of the site and level is lower than max_level, recursively calls method again.
                    if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
                        print_categorymembers(c.categorymembers, all_urls, max_level=max_level, level=level + 1, )

            p_html = wiki_html.page(page_name)
            all_urls = {}

            if depth == None:
                depth = default_depth(p_html.categorymembers)
                print(f"Default depth is {depth}.")
            # Calls recursive method to iterate through links according to depth
            print_categorymembers(p_html.categorymembers, all_urls, depth)

            return all_urls

        def named_entity_recognition(all_links):

            # Uses spacy for named_entity recognition
            nlp = spacy.load("en_core_web_sm")
            pd.set_option("display.max_rows", 200)
            new_links = {}

            for key, url in tqdm(all_links.items(), desc="Processing Links", unit="link", total=len(all_links)):
                doc = nlp(key)

                # Goes through each entity in the key to search for PERSON label
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        new_links[key] = url
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
            result_map = bullet(page_name, wiki_html)
        elif format == 'Table':
            if col_info == 'None':
                print("Missing table information")
                return
            result_map = table(link, col_info)
        elif format == 'Nested':
            result_map = nested(page_name, depth)

        # Checks to see if user wants NER. This will work for all formats however is most helpful for Paragraph
        if name_filter:
            result_map = named_entity_recognition(result_map)

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
                        # print(title)
                        related_pages.extend([link_page.fullurl])
                    except Exception as e:
                        print(f"Error: {e}")
                if current_depth + 1 < max_depth:
                    related_pages.extend(get_related_pages(topic, link_page, max_depth, current_depth + 1, visited))

            return related_pages

        topic = self.category

        print(f"Searching Wikipedia for topic: {topic}")
        main_page = search_wikipedia(topic, language=language, user_agent=user_agent)

        if isinstance(main_page, str):
            return self.scrap_area_to_abcData()
        else:
            print(f"Found Wikipedia page: {main_page.title}")
            related_pages = get_related_pages(topic, main_page, max_depth=1, top_n=top_n)
            print(f"Related pages saved to related_pages.json")
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
                # print(sentences)

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

        # Read from temporary files and aggregate results
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


class BenchmarkBuilding:
    default_configuration = {
        'keyword_finder': {
            'require': True,
            'reading_location': 'default',
            'method': 'embedding_on_wiki',  # 'embedding_on_wiki' or 'llm_inquiries' or 'hyperlinks_on_wiki'
            'page_name_or_link': None, # if hyperlinks_on_wiki is chosen, page_name or link must be provided
            'keyword_number': 7,
            'embedding_model': 'paraphrase-Mpnet-base-v2',
            'saving': True,
            'saving_location': 'default',
            'generation_function': None,  # if llm is chosen as the method, this function should be provided
        },
        'scrap_area_finder': {
            'require': True,
            'reading_location': 'default',
            'local_file_location': 'default',
            'method': 'wiki',  # 'wiki' or 'local_files'
            'scrap_number': 5,
            'saving': True,
            'saving_location': 'default',
        },
        'scrapper': {
            'require': True,
            'reading_location': 'default',
            'saving': True,
            'method': 'wiki',  # This is related to the scrap_area_finder method
            'saving_location': 'default'},
        'prompt_maker': {
            'method': 'split_sentences',
            'saving_location': 'default',
        },
    }

    def __init__(self):
        pass

    @staticmethod
    def update_configuration(default_configuration, updated_configuration):
        """
        Update the default configuration dictionary with the values from the updated configuration
        only if the keys already exist in the default configuration.

        Args:
        - default_configuration (dict): The default configuration dictionary.
        - updated_configuration (dict): The updated configuration dictionary with new values.

        Returns:
        - dict: The updated configuration dictionary.
        """
        print(updated_configuration)
        print("\n")
        print(default_configuration)
        print("\n")
        for key, value in updated_configuration.items():
            if key in default_configuration:
                if isinstance(default_configuration[key], dict) and isinstance(value, dict):
                    print("recursive")
                    # Recursively update nested dictionaries
                    default_configuration[key] = BenchmarkBuilding.update_configuration(default_configuration[key],
                                                                                        value)
                else:
                    # Update the value for the key
                    default_configuration[key] = value
        
        print(default_configuration)
        return default_configuration

    @classmethod
    def category_pipeline(cls, domain, demographic_label, configuration=None):
        if configuration is None:
            configuration = cls.default_configuration
        else:
            configuration = cls.update_configuration(cls.default_configuration, configuration)

        keyword_finder_require = configuration['keyword_finder']['require']
        keyword_finder_reading_location = configuration['keyword_finder']['reading_location']
        keyword_finder_method = configuration['keyword_finder']['method']
        keyword_finder_page_name_or_link = configuration['keyword_finder']['page_name_or_link']
        keyword_finder_keyword_number = configuration['keyword_finder']['keyword_number']
        keyword_finder_embedding_model = configuration['keyword_finder']['embedding_model']
        keyword_finder_saving = configuration['keyword_finder']['saving']
        keyword_finder_saving_location = configuration['keyword_finder']['saving_location']
        keyword_finder_generation_function = configuration['keyword_finder']['generation_function']

        scrap_area_finder_require = configuration['scrap_area_finder']['require']
        scrap_area_local_files_location = configuration['scrap_area_finder']['local_file_location']
        scrap_area_finder_reading_location = configuration['scrap_area_finder']['reading_location']
        scrap_area_finder_method = configuration['scrap_area_finder']['method']
        scrap_area_finder_saving = configuration['scrap_area_finder']['saving']
        scrap_area_finder_saving_location = configuration['scrap_area_finder']['saving_location']
        scrap_area_finder_scrap_area_number = configuration['scrap_area_finder']['scrap_number']

        scrapper_require = configuration['scrapper']['require']
        scrapper_reading_location = configuration['scrapper']['reading_location']
        scrapper_saving = configuration['scrapper']['saving']
        scrapper_method = configuration['scrapper']['method']
        scrapper_saving_location = configuration['scrapper']['saving_location']

        prompt_maker_method = configuration['prompt_maker']['method']
        prompt_maker_saving_location = configuration['prompt_maker']['saving_location']

        # check the validity of the configuration
        assert keyword_finder_method in ['embedding_on_wiki', 'llm_inquiries',
                                         'hyperlinks_on_wiki'], "Invalid keyword finder method. Choose either 'embedding_on_wiki', 'llm_inquiries', or 'hyperlinks_on_wiki'."
        if keyword_finder_method == 'llm_inquiries':
            assert keyword_finder_generation_function is not None, "generation function must be provided if llm_inquiries is chosen as the method"
            check_generation_function(keyword_finder_generation_function)
        assert scrap_area_finder_method in ['wiki',
                                            'local_files'], "Invalid scrap area finder method. Choose either 'wiki' or 'local_files'"
        assert scrapper_method in ['wiki',
                                   'local_files'], "Invalid scrapper method. Choose either 'wiki' or 'local_files'"
        assert scrap_area_finder_method == scrapper_method, "scrap_area_finder and scrapper methods must be the same"
        assert prompt_maker_method in ['split_sentences'], "Invalid prompt maker method. Choose 'split_sentences'"

        # make sure only the required loading is done
        if not scrapper_require:
            scrap_area_finder_require = False

        if not scrap_area_finder_require:
            keyword_finder_require = False

        if keyword_finder_require:
            if keyword_finder_method == 'embedding_on_wiki':
                kw = KeywordFinder(domain=domain, category=demographic_label).find_keywords_by_embedding_on_wiki(
                    n_keywords=keyword_finder_keyword_number, embedding_model=keyword_finder_embedding_model).add(
                    keyword=demographic_label)
            elif keyword_finder_method == 'llm_inquiries':
                kw = KeywordFinder(domain=domain, category=demographic_label).find_keywords_by_llm_inquiries(
                    generation_function=keyword_finder_generation_function).add(
                    keyword=demographic_label)
            elif keyword_finder_method == 'hyperlinks_on_wiki':
                kw = KeywordFinder(domain=domain,
                                   category=demographic_label).find_name_keywords_by_hyperlinks_on_wiki(page_name=keyword_finder_page_name_or_link).add(
                    keyword=demographic_label)
            print('Keywords found.')

            if keyword_finder_saving:
                if keyword_finder_saving_location == 'default':
                    kw.save()
                else:
                    kw.save(file_path=keyword_finder_saving_location)

        elif scrap_area_finder_require:
            if keyword_finder_reading_location == 'default':
                kw = abcData.load_file(domain=domain, category=demographic_label,
                                       file_path=f'data/customized/keywords/{domain}_{demographic_label}_keywords.json',
                                       data_tier='keywords')
                print(f'Keywords loaded from data/customized/keywords/{domain}_{demographic_label}_keywords.json')
            else:
                print(keyword_finder_reading_location)
                kw = abcData.load_file(domain=domain, category=demographic_label,
                                       file_path=keyword_finder_reading_location, data_tier='keywords')
                print(f'Keywords loaded from {keyword_finder_reading_location}')

        if scrap_area_finder_require:
            if scrap_area_finder_method == 'wiki':
                sa = ScrapAreaFinder(kw, source_tag='wiki').find_scrap_urls_on_wiki(top_n=scrap_area_finder_scrap_area_number)
            elif scrap_area_finder_method == 'local_files':
                if scrap_area_finder_reading_location == 'default':
                    sa = ScrapAreaFinder(kw, source_tag='local').find_scrap_paths_local(
                        f'data/customized/local_files/{demographic_label}')
                else:
                    sa = ScrapAreaFinder(kw, source_tag='local').find_scrap_paths_local(scrap_area_local_files_location)
            print('Scrapped areas found.')

            if scrap_area_finder_saving:
                if scrap_area_finder_saving_location == 'default':
                    sa.save()
                else:
                    sa.save(file_path=scrap_area_finder_saving_location)

        elif scrapper_require:
            if scrap_area_finder_reading_location == 'default':
                sa = abcData.load_file(domain=domain, category=demographic_label,
                                       file_path=f'data/customized/scrap_area/{domain}_{demographic_label}_scrap_area.json',
                                       data_tier='scrap_area')
                print(f'Scrap areas loaded from data/customized/scrap_area/{domain}_{demographic_label}_scrap_area.json')
            else:
                sa = abcData.load_file(domain=domain, category=demographic_label,
                                       file_path=scrap_area_finder_reading_location, data_tier='scrap_area')
                print(f'Scrap areas loaded from {scrap_area_finder_reading_location}')

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

        else:
            if scrapper_reading_location == 'default':
                sc = abcData.load_file(domain=domain, category=demographic_label,
                                       file_path=f'data/customized/scrapped_sentences/{domain}_{demographic_label}_scrapped_sentences.json',
                                       data_tier='scrapped_sentences')
                if (sc!=None): print(f'Scrapped sentences loaded from data/customized/scrapped_sentences/{domain}_{demographic_label}_scrapped_sentences.json')
            else:
                sc = abcData.load_file(domain=domain, category=demographic_label, file_path=scrapper_reading_location,
                                       data_tier='scrapped_sentences')
                if (sc!=None): print(f'Scrapped sentences loaded from {scrapper_reading_location}')

        if prompt_maker_method == 'split_sentences' and sc !=None:
            print(prompt_maker_saving_location)
            pm = PromptMaker(sc).split_sentences()
            if prompt_maker_saving_location == 'default':
                pm.save()
            else:
                pm.save(file_path=prompt_maker_saving_location)
            print(f'Benchmark building for {demographic_label} completed.')


