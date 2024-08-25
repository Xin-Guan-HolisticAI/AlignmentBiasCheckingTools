from alignbiascheck.analytics import AlignmentBiasChecker
from assistants import GPTAgent, OllamaModel  # Assuming you have this import

def test_pipeline():
    # Set up test configuration
    configuration = {
        'generation': {
            'require': False,  # Set to False if you already have generated data
            'reading_location': 'tests/data/customized/benchmarks/political-ideology_benchmark_LLM_generation_counterfactual.csv',  # Provide the path to your test data
        },
        'feature_extraction': {
            'reading_location': 'data/customized/benchmarks/political-ideology_benchmark_LLM_cluster_counterfactual.csv',
            'feature': 'cluster',
            'require': False,
            'theme_insight': True,
            'num_clusters': 5,
            'generation_function': OllamaModel(model_name='ollama').invoke,  # Specify the correct model name
        },
        'alignment': {
            'method': 'kl_divergence',
            'visualization': True,
        },
        'bias': {
            'require': False,  # Set to True if you want to test bias checking as well
        }
    }

    # Run the pipeline
    domain = 'political-ideology'  # Replace with the domain of your test data
    AlignmentBiasChecker.domain_pipeline(domain, None, configuration)

if __name__ == "__main__":
    test_pipeline()