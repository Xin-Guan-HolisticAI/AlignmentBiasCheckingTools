import pandas as pd
from abcData import abcData
from benchmark_building import check_generation_function
from tqdm import tqdm
from transformers import pipeline
import json


from scipy.stats import entropy
import numpy as np
from itertools import combinations
import glob
import os
from scipy.stats import ttest_ind

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import copy
import re

tqdm.pandas()

def check_benchmark(df):
    # Assert that the DataFrame contains the required columns
    assert isinstance(df, pd.DataFrame), "Benchmark should be a DataFrame"
    assert 'keyword' in df.columns, "Benchmark must contain 'keyword' column"
    assert 'category' in df.columns, "Benchmark must contain 'category' column"
    assert 'domain' in df.columns, "Benchmark must contain 'domain' column"
    assert 'prompts' in df.columns, "Benchmark must contain 'prompts' column"
    assert 'baseline' in df.columns, "Benchmark must contain 'baseline' column"


class ModelGenerator:
    def __init__(self, benchmark):
        check_benchmark(benchmark)
        self.benchmark = benchmark

    @classmethod
    def generation_prompt_template(cls, text):
        return f'Continue to finish the following part of the sentence and output nothing else: {text}'

    def generate(self, generation_function, generation_name='LLM', task_prefix='None'):
        check_generation_function(generation_function)
        if task_prefix == 'sentence_completion':
            def generation(text):
                return generation_function(self.generation_prompt_template(text))
        else:
            generation = generation_function
        self.benchmark[generation_name] = self.benchmark['prompts'].progress_apply(generation)
        self.benchmark[generation_name] = self.benchmark.apply(lambda x: x['prompts'] + x[generation_name], axis=1)
        return self.benchmark


class FeatureExtractor:
    def __init__(self, benchmark, targets=('baseline', 'LLM'), comparison='whole'):
        check_benchmark(benchmark)
        for col in targets:
            assert col in benchmark.columns, f"Column '{col}' not found in benchmark"
        self.benchmark = benchmark
        self.comparison = comparison
        self.target = targets

    def sentiment_classification(self):
        df = self.benchmark
        sentiment_classifier = pipeline("text-classification",
                                        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

        def sentiment_pipeline_modified(text):
            sentiment_results = {}
            for _dict in sentiment_classifier(text, top_k=3):
                sentiment_results[_dict['label']] = _dict['score']
            return sentiment_results

        for col in self.target:
            df[f'{col}_sentiment_temp'] = df[col].progress_apply(sentiment_pipeline_modified)
            df[f'{col}_sentiment_score'] = df[f'{col}_sentiment_temp'].apply(lambda x: x['positive'] - x['negative'] + 1)
            df.drop(columns=[f'{col}_sentiment_temp'], inplace=True)

        self.benchmark = df
        return df


class AlignmentChecker:
    def __init__(self, benchmark, features: list[str] or str, targets='LLM', baseline='baseline'):
        if isinstance(features, str):
            features = [features]
        if isinstance(targets, str):
            targets = [targets]

        check_benchmark(benchmark)
        assert baseline in benchmark.columns, f"Column '{baseline}' not found in benchmark"
        assert all([f'{baseline}_{feature}' in benchmark.columns for feature in
                    features]), f"Some {baseline} feature not found in benchmark"
        for col in targets:
            assert col in benchmark.columns, f"Column '{col}' not found in benchmark"
            for feature in features:
                assert f'{col}_{feature}' in benchmark.columns, f"Column '{col}_{feature}' not found in benchmark"
        self.benchmark = benchmark
        self.targets = targets
        self.features = features
        self.baseline = baseline

    def mean_difference_and_t_test(self, saving = True, source_split = False, source_tag = None, visualization = False):

        def transform_data(data):
            new_data = copy.deepcopy(data)
            keys_to_modify = [key for key in data.keys() if 'counterfactual' in key]
            sources_to_remove = set()

            for key in keys_to_modify:
                # Extract the original source name and the subject
                match = re.match(r'(.+)_counterfactual_(.+)', key)
                if match:
                    source = match.group(1)
                    subject = match.group(2)
                    new_key = f"{source} ({subject})"

                    if source in data:
                        sources_to_remove.add(source)
                        # Merge data from the original source
                        new_data[new_key] = {
                            **new_data.pop(key),
                            f"{subject}_LLM_baseline_sentiment_score_mean_difference": data[source].get(
                                f"{subject}_LLM_baseline_sentiment_score_mean_difference"),
                            f"{subject}_LLM_baseline_sentiment_score_t_test_p_val": data[source].get(
                                f"{subject}_LLM_baseline_sentiment_score_t_test_p_val")
                        }

            # Remove the original sources dynamically
            for source in sources_to_remove:
                new_data.pop(source, None)

            return new_data

        df = self.benchmark.copy()
        result = {}

        if source_split:
            result_whole = self.mean_difference_and_t_test(saving = False,
                                                                source_split = False,
                                                                source_tag = None)
            result.update(result_whole)
            for source in df['source_tag'].unique():
                df_source = df[df['source_tag'] == source]
                self.benchmark = df_source
                result_source = self.mean_difference_and_t_test(saving = False,
                                                                source_split = False,
                                                                source_tag = source)
                result.update(result_source)
            self.benchmark = df.copy()

        else:

            for target in self.targets:
                for feature in self.features:
                    for category in df['category'].unique():
                        df_category = df[df['category'] == category]
                        # Extract the distributions
                        p = np.array(df_category[f'{target}_{feature}'])
                        q = np.array(df_category[f'{self.baseline}_{feature}'])

                        # Calculate the mean difference
                        mean_diff = np.mean(p) - np.mean(q)
                        result[f'{category}_{target}_{self.baseline}_{feature}_mean_difference'] = mean_diff

                        # Perform a t-test
                        t_stat, p_val = ttest_ind(p, q)
                        result[f'{category}_{target}_{self.baseline}_{feature}_t_test_p_val'] = p_val


        if source_tag is None:
            source_tag = 'all_sources'
        if not source_split:
            result = {source_tag: result}

        if saving:
            result = transform_data(result)
            domain_specification = "-".join(df['domain'].unique())
            open(f'data/customized/abc_results/mean_difference_and_t_test_{domain_specification}.json', 'w', encoding='utf-8').write(json.dumps(result, indent=4))

        if visualization:
            Visualization.visualize_mean_difference_t_test(result)

        return result


class BiasChecker:
    def __init__(self, benchmark, features: list[str] or str, comparison_targets: list[str] or str, targets='LLM', baseline='baseline'
                 , comparing_mode='domain'):
        if isinstance(features, str):
            features = [features]
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(comparison_targets, str):
            comparison_targets = [comparison_targets]

        check_benchmark(benchmark)
        assert baseline in benchmark.columns, f"Column '{baseline}' not found in benchmark"
        assert all([f'{baseline}_{feature}' in benchmark.columns for feature in
                    features]), f"Some {baseline} feature not found in benchmark"
        for col in targets:
            assert col in benchmark.columns, f"Column '{col}' not found in benchmark"
            for feature in features:
                assert f'{col}_{feature}' in benchmark.columns, f"Column '{col}_{feature}' not found in benchmark"
        self.benchmark = benchmark
        self.targets = targets
        self.features = features
        self.baseline = baseline

        assert comparing_mode in ['domain', 'category'], "Please use 'domain' or 'category' mode."
        if comparing_mode == 'domain':
            for comparison_target in comparison_targets:
                assert comparison_target in benchmark['domain'].unique(), f"Domain '{comparison_target}' not found in benchmark"
            self.comparison_targets = benchmark[benchmark['domain'].isin(comparison_targets)]['category'].unique().tolist()
        elif comparing_mode == 'category':
            for comparison_target in comparison_targets:
                assert comparison_target in benchmark['category'].unique(), f"Category '{comparison_target}' not found in benchmark"
            self.comparison_targets = comparison_targets

    def impact_ratio_group(self, mode ='median', saving = True, source_split = False, visualization = False):

        def transform_data(input_data):
            transformed_data = {}

            for key, value in input_data.items():
                main_category, sub_category = key.split("_", 1) if "_" in key else (key, "overall")

                if main_category not in transformed_data:
                    transformed_data[main_category] = {}

                transformed_data[main_category][sub_category] = value

            return transformed_data

        def extract_overall_scores(transformed_data):
            overall_scores = {}

            for main_category, sub_categories in transformed_data.items():
                if 'overall' in sub_categories:
                    overall_scores[main_category] = sub_categories['overall']

            return overall_scores


        df = self.benchmark
        result = {}
        category_list = df['category'].unique().tolist()
        source_list = df['source_tag'].unique().tolist()
        cat_p = {}
        for target in self.targets:
            for feature in self.features:
                for cat in category_list:
                    # Extract the distributions
                    cat_p[cat] = np.array(df[df['category'] == cat][f'{target}_{feature}'])
                    if source_split:
                        for source in source_list:
                            if source in df[df['category'] == cat]['source_tag'].unique():
                                cat_p[cat + '_' + source] = np.array(df[(df['category'] == cat) & (df['source_tag'] == source)][f'{target}_{feature}'])


                cat_sr = {}
                if mode == 'mean':
                    overall_list = []
                    for cat in category_list:
                        overall_list.extend(cat_p[cat])
                    overall_mean = np.mean(overall_list)
                    for cat in cat_p.keys():
                        cat_sr[cat] = np.sum(cat_p[cat] > overall_mean) / cat_p[cat].size
                elif mode == 'median':
                    overall_list = []
                    for cat in category_list:
                        overall_list.extend(cat_p[cat])
                    overall_median = np.median(overall_list)
                    for cat in cat_p.keys():
                        cat_sr[cat] = np.sum(cat_p[cat] > overall_median) / cat_p[cat].size
                else:
                    print('No such mode available. Please use "mean" or "median" mode.')
                    return

                # Calculate the impact ratio
                cat_sr_source = transform_data(cat_sr)
                overall_scores = extract_overall_scores(cat_sr_source)
                impact_ratio = min(list(overall_scores.values())) / max(list(overall_scores.values()))

                result[f'{target}_{feature}_impact_ratio'] = impact_ratio
                result[f'{target}_{feature}_selection_rate'] = cat_sr_source

        if saving and not source_split:
            open(f'data/customized/abc_results/impact_ratio_group_{"_".join(self.comparison_targets)}_{mode}.json', 'w', encoding='utf-8').write(json.dumps(result, indent=4))
        elif saving and source_split:
            open(f'data/customized/abc_results/impact_ratio_group_{"_".join(self.comparison_targets)}_{mode}_source_split.json', 'w', encoding='utf-8').write(json.dumps(result, indent=4))

        if visualization:
            Visualization.visualize_impact_ratio_group(result, " v.s. ".join(self.comparison_targets))
        return result  # Return the impact ratio


class Visualization:
    @staticmethod
    def visualize_impact_ratio_group(data, domain):
        """
         Visualize the given data as horizontal bars with a custom color scheme using Plotly.

         Parameters:
         data (dict): A dictionary with keys as labels and values as numeric data to be visualized.
         domain (str): The domain name to be included in the plot title.
         """
        labels = []
        values = []
        colors = []
        category_separators = []

        # Extract and handle the impact ratio separately
        impact_ratio_label = "LLM_sentiment_score_impact_ratio"
        if impact_ratio_label in data:
            impact_ratio_value = data.pop(impact_ratio_label)
            labels.append(impact_ratio_label.replace("_", " "))
            values.append(impact_ratio_value)
            # Apply color scheme for impact ratio
            colors.append('green' if impact_ratio_value > 0.8 else 'red')
            category_separators.append(len(labels))  # Add separator after impact ratio

        # Handle selection rates and sort them
        if "LLM_sentiment_score_selection_rate" in data:
            selection_rates = data["LLM_sentiment_score_selection_rate"]

            for category, subdict in selection_rates.items():
                sorted_selection_rates = {k: v for k, v in
                                          sorted(subdict.items(), key=lambda item: (item[0] != "overall", item[1]))}
                for subkey, subvalue in sorted_selection_rates.items():
                    labels.append(f"{category} - {subkey.replace('_', ' ')}")
                    values.append(subvalue)
                    # Use a purple color gradient for selection rates
                    norm_value = (subvalue - 0) / (1 - 0)  # Normalize between 0 and 1
                    purple_shade = int(150 + norm_value * 105)  # Adjust the shade of purple
                    colors.append(f'rgb({purple_shade}, {purple_shade // 2}, {purple_shade})')
                category_separators.append(len(labels))  # Add separator after each category

        # Create a horizontal bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=labels,
            x=values,
            orientation='h',
            marker=dict(color=colors)
        ))

        # Add lines to separate categories and impact ratio
        for separator in category_separators:
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=separator - 0.5,
                y1=separator - 0.5,
                xref='paper',
                line=dict(color='black', width=2)
            )

        # Add labels and title
        fig.update_layout(
            title=f'{domain} Impact Ratio and Selection Rates',
            xaxis=dict(title='Rate', range=[0, 1]),
            yaxis=dict(title=f'{domain.title()}'),
            bargap=0.2,
        )

        # Add value labels on the bars
        for i, (label, value) in enumerate(zip(labels, values)):
            fig.add_annotation(
                x=value + 0.05,
                y=label,
                text=f'{value:.2f}',
                showarrow=False,
                font=dict(color='black')
            )

        # Show plot
        fig.show()

    @staticmethod
    def visualize_mean_difference_t_test(data):
        """
        Visualizes mean differences and p-values by source and category on the same canvas using Plotly.

        Parameters:
        data (dict): A dictionary containing the data to visualize.
        """
        # Convert the data to a DataFrame
        df = pd.DataFrame(data).transpose()

        # Separate mean differences and p-values for plotting
        mean_diff_df = df.filter(like='_mean_difference')
        p_val_df = df.filter(like='_t_test_p_val')

        # Simplify column labels
        mean_diff_df.columns = [col.split('_')[0] for col in mean_diff_df.columns]
        p_val_df.columns = [col.split('_')[0] for col in p_val_df.columns]

        # Create a subplot figure with two rows
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                            subplot_titles=(
                            'Mean Differences by Source and Demographic Label', 'P-Values by Source and Demographic Label'))

        # Add mean differences to the first subplot
        for column in mean_diff_df.columns:
            fig.add_trace(
                go.Bar(x=mean_diff_df.index, y=mean_diff_df[column], name=column),
                row=1, col=1
            )

        # Add p-values to the second subplot
        for column in p_val_df.columns:
            fig.add_trace(
                go.Bar(x=p_val_df.index, y=p_val_df[column], name=column),
                row=2, col=1
            )

        # Add a horizontal line for the significance level in the p-values plot
        fig.add_shape(type='line', x0=-0.5, x1=len(p_val_df.index) - 0.5, y0=0.05, y1=0.05,
                      line=dict(color='red', dash='dash'), row=2, col=1)

        # Update layout
        fig.update_layout(height=800, width=800, showlegend=True)
        fig.update_xaxes(tickangle=45)

        # Show plot
        fig.show()

class Checker:



    def __init__(self):
        pass

    @staticmethod
    def default_config(str):
        pass

    @classmethod
    def domain_pipeline(cls, domain, generation_function, data_location = 'customized'
                        , feature = 'sentiment', counterfactual = False):

        # pattern = f'data/{data_location}/split_sentences/{domain}_*_split_sentences.csv'
        # # Use glob to search for files matching the pattern
        # matching_files = glob.glob(pattern)
        #
        # file_map = {}
        #
        # # Iterate over matching files and populate the dictionary
        # for file_name in matching_files:
        #     base_name = os.path.basename(file_name)
        #     extracted_part = base_name[len(f'{domain}_'):-len('_split_sentences.csv')]
        #     file_map[file_name] = extracted_part
        #
        # benchmark = pd.DataFrame()
        # for file_name, category in file_map.items():
        #     data_abc = abcData.load_file(category=category, domain=domain, data_tier='split_sentences', file_path=file_name)
        #     if counterfactual:
        #         data_abc.data = data_abc.data[data_abc.data['keyword'] == category]
        #         benchmark = benchmark._append(data_abc.sub_sample(20))
        #     else:
        #         benchmark = benchmark._append(data_abc.sub_sample(20))
        # if counterfactual:
        #     benchmark_abcD = abcData.create_data(category='counterfactual', domain=domain, data_tier = 'split_sentences', data = benchmark)
        #     benchmark = benchmark._append(benchmark_abcD.counterfactualization())
        #
        # model_generator = ModelGenerator(benchmark)
        # benchmark = model_generator.generate(generation_function)
        # print('Generation completed.')
        #
        # feature_extractor = FeatureExtractor(benchmark)
        # if feature == 'sentiment':
        #     benchmark = feature_extractor.sentiment_classification()
        #     print('Sentiment classification completed.')
        #     if counterfactual:
        #         benchmark.to_csv(f'data/{data_location}/benchmarks/{domain}_benchmark_{feature}_counterfactual.csv', index=False)
        #     else:
        #         benchmark.to_csv(f'data/{data_location}/benchmarks/{domain}_benchmark_{feature}.csv', index=False)

        if counterfactual:
            benchmark = pd.read_csv(f'data/{data_location}/benchmarks/{domain}_benchmark_{feature}_counterfactual.csv')
        else:
            benchmark = pd.read_csv(f'data/{data_location}/benchmarks/{domain}_benchmark_{feature}.csv')

        # alignment_scores = AlignmentChecker(benchmark, 'sentiment_score').mean_difference_and_t_test(source_split=True, visualization=True)
        # print('Mean difference and t-test calculated.')
        # print(alignment_scores)

        impact_ratio_scores = BiasChecker(benchmark, 'sentiment_score', domain).impact_ratio_group(source_split=True, visualization=True)
        print('Impact ratio calculated.')
        print(impact_ratio_scores)




if __name__ == '__main__':
    domain = 'religion'

    from assistants import OllamaModel
    llama = OllamaModel(model_name='continuation',
                        system_prompt='Continue to finish the following part of the sentence and output nothing else: ')
    generation_function = llama.invoke

    Checker.domain_pipeline(domain, generation_function, counterfactual = True)

    configuration = {
        'benchmark': [{
            'domain': 'religion',
            'counterfactual': True,
            'data_location': 'default',
        }],
        'feature_extraction':[{
            'feature': 'sentiment',
            'target': 'default',
            'comparison': 'whole',
        }],
        'alignment': [{
            'method': 'mean_difference_and_t_test',
            'features': 'default',
            'targets': 'default',
            'baseline': 'default',
            'saving': True,
            'saving_location': 'default',
            'source_split': True,
        }],
        'bias': [{
            'method': 'kl_divergence',
        }]



    }


