import unittest
import pandas as pd 
from abcData import check_format 

class TestCheckFormat(unittest.TestCase):
    def test_valid_keywords(self):
        data = [
            {
                "category": "fascism",
                "domain": "politics",
                "keywords": {
                    "fascism": {
                        "keyword_type": "sub-concepts",
                        "keyword_provider": "embedding",
                        "scrap_mode": "in_page",
                        "scrap_shared_area": "Yes"
                    }
                }
            }
        ]
        try:
            check_format(data_tier='keywords', data=data)
        except AssertionError:
            self.fail("check_format raised AssertionError unexpectedly for 'keywords' data_tier")

if __name__ == '__main__':
    unittest.main()
    
