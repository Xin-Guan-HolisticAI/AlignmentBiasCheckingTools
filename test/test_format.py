import unittest
import pandas as pd 
from abcData import abcData

class TestCheckFormat(unittest.TestCase):
    def setUp(self):
        self.check_format = abcData.check_format
        
    def test_valid_keywords_with_tsa(self):    
        # includes optional targeted_scrap_area
        data = [
            {
                "category": "fascism",
                "domain": "politics",
                "keywords": {
                    "fascism": {
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
                }
            }
        ]

        try:
            self.check_format('keywords', data)
        except AssertionError:
            self.fail("check_format rasied AssertionError for 'keywords' data")

    def test_valid_keywords_without_tsa(self):
        # does not include optional targeted_scrap_area
        data = [
            {
                "category": "man",
                "domain": "gender",
                "keywords": {
                    "fascism": {
                        "keyword_type": "sub-concepts",
                        "keyword_provider": "manual",  
                        "scrap_mode": "not_in_page",
                        "scrap_shared_area": "Yes"
                    }
                }
            }
        ]

        try:
            self.check_format('keywords', data)
        except AssertionError:
            self.fail("check_format rasied AssertionError without targeted_scrap_area included in 'keywords' data")

    def test_invalid_keywords_without_scrap_mode(self):
        # leaves out scrap_mode 
        invalid_data = [
            {
                "category": "atheism",
                "domain": "religion",
                "keywords": {
                    "fascism": {
                        "keyword_type": "sub-concepts",
                        "keyword_provider": "manual", 
                        "targeted_scrap_area": [{  
                            "source_tag": "default",
                            "scrap_area_type": "unknown",
                            "scrap_area_specification": []
                        }], 
                        "scrap_shared_area": "Yes"
                    }
                }
            }
        ]

        with self.assertRaises(AssertionError):
            self.check_format('keywords', invalid_data)

    def test_invalid_keywords_with_string(self):
        # ensures that keywords should be a dictionary 
        invalid_data = [
            {
                "category": "nationalism",
                "domain": "politics",
                "keywords": "this should be a dictionary and not a string"
            }
        ]

        with self.assertRaises(AssertionError):
            self.check_format('keywords', invalid_data)

    def test_valid_scrap_area(self):
        data = [
            {
                "category_shared_scrap_area": [
                {
                    "source_tag": "wiki",
                    "scrap_area_type": "wiki_urls",
                    "scrap_area_specification": [
                    "https://en.wikipedia.org/wiki/Woman",
                    "https://en.wikipedia.org/wiki/Female",
                    "https://en.wikipedia.org/wiki/Lady",
                    "https://en.wikipedia.org/wiki/Wife",
                    "https://en.wikipedia.org/wiki/Woman",
                    "https://en.wikipedia.org/wiki/Woman"
                    ]
                }
            ]
            }
        ]

        try:
            self.check_format('scrap_area', data, scrap_area_only=True)
        except AssertionError:
            self.fail("check_format rasied AssertionError for 'scrap_area' data")

    def test_false_scrap_area(self):
        data = [
            {
                "category_shared_scrap_area": [
                {
                    "source_tag": "default",
                    "scrap_area_type": "invalid_type",
                    "scrap_area_specification": [
                    "https://en.wikipedia.org/wiki/Woman",
                    "https://en.wikipedia.org/wiki/Female",
                    "https://en.wikipedia.org/wiki/Lady",
                    "https://en.wikipedia.org/wiki/Wife",
                    "https://en.wikipedia.org/wiki/Woman",
                    "https://en.wikipedia.org/wiki/Woman"
                    ]
                }
            ]
            }
        ]

        with self.assertRaises(AssertionError):
            # set scrap_area_only to false
            self.check_format('scrap_area', data, scrap_area_only=False)

    def test_valid_scrapped_sentences(self):
        data = [
            {
                "category": "democracy",
                "domain": "politics",
                "keywords": {
                    "fascism": {
                        "keyword_type": "sub-concepts",
                        "keyword_provider": "manual",
                        "scrap_mode": "in_page",
                        "scrap_shared_area": "Yes",
                        "scrapped_sentences": ["Social democracy, democratic socialism, and the dictatorship of the proletariat are some examples."]
                    }
                },
                "category_shared_scrap_area": [
                    {
                        "source_tag": "default",
                        "scrap_area_type": "wiki_urls",
                        "scrap_area_specification": [
                        "https://en.wikipedia.org/wiki/Democracy",
                        "https://en.wikipedia.org/wiki/Democracy_in_America",
                        "https://en.wikipedia.org/wiki/Emergent_democracy",
                        "https://en.wikipedia.org/wiki/Empowered_democracy",
                        "https://en.wikipedia.org/wiki/Popular_democracy",
                        "https://en.wikipedia.org/wiki/Liberal_democracy"
                        ]
                    }
                ]
            }
        ]
        try:
            self.check_format('scrapped_sentences', data)
        except AssertionError:
            self.fail("check_format raised AssertionError for 'scrapped_sentences' data")

    def test_invalid_scrapped_sentences(self):
        # scrapped_sentences is a string instead of a list
        invalid_data = [
            {
                "category": "fascism",
                "domain": "politics",
                "keywords": {
                    "fascism": {
                        "keyword_type": "sub-concepts",
                        "keyword_provider": "manual",
                        "scrap_mode": "in_page",
                        "scrap_shared_area": "Yes",
                        "scrapped_sentences": "This should be a list"
                    }
                },
                "category_shared_scrap_area": [
                    {
                        "source_tag": "default",
                        "scrap_area_type": "wiki_urls",
                        "scrap_area_specification": [
                        "https://en.wikipedia.org/wiki/Democracy",
                        "https://en.wikipedia.org/wiki/Democracy_in_America",
                        "https://en.wikipedia.org/wiki/Emergent_democracy",
                        "https://en.wikipedia.org/wiki/Empowered_democracy",
                        "https://en.wikipedia.org/wiki/Popular_democracy",
                        "https://en.wikipedia.org/wiki/Liberal_democracy"
                        ]
                    }
                ]
            }
        ]
        with self.assertRaises(AssertionError):
            self.check_format('scrapped_sentences', invalid_data)

    def test_valid_split_sentences(self):
        data = pd.DataFrame({
            "keyword": ["fascism"],
            "category": ["political_ideologies"],
            "domain": ["politics"],
            "prompts": ["prompt1"],
            "baseline": ["baseline1"]
        })
        try:
            self.check_format('split_sentences', data)
        except AssertionError:
            self.fail("check_format raised AssertionError for 'split_sentences' data")

    def test_invalid_split_sentences(self):
        invalid_data = pd.DataFrame({
            "keyword": ["fascism"],
            "category": ["political_ideologies"]
            # Missing required columns: 'domain', 'prompts', 'baseline'
        })
        with self.assertRaises(AssertionError):
            self.check_format('split_sentences', invalid_data)

    def test_scrap_area_only(self):
        scrap_area_data = [
            {
                "source_tag": "default",
                "scrap_area_type": "wiki_urls",
                "scrap_area_specification": ["http://example.com"]
            }
        ]
        check_scrap_area_format = self.check_format(scrap_area_only=True)
        try:
            check_scrap_area_format(scrap_area_data)
        except AssertionError:
            self.fail("check_scrap_area_format raised AssertionError for valid scrap_area data")

    def test_invalid_scrap_area_only(self):
        invalid_scrap_area_data = [
            {
                "source_tag": "default",
                "scrap_area_type": "invalid_type",
                "scrap_area_specification": ["http://example.com"]
            }
        ]
        check_scrap_area_format = self.check_format(scrap_area_only=True)
        with self.assertRaises(AssertionError):
            check_scrap_area_format(invalid_scrap_area_data)

    

if __name__ == '__main__':
    unittest.main()
