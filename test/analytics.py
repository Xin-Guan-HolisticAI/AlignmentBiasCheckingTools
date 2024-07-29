import pandas as pd
from abcData import abcData
from benchmark_building import check_generation_function
from tqdm import tqdm
from transformers import pipeline
import json

tqdm.pandas()

from scipy.stats import entropy
import numpy as np
from itertools import combinations
import glob
import os
from scipy.stats import ttest_ind

import plotly.graph_objects as go
import plotly.io as pio


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

    def kl_divergence(self, smooth=False, epsilon=1e-12):
        df = self.benchmark
        result = {}
        for target in self.targets:
            for feature in self.features:
                # Extract the distributions
                p = np.array(df[f'{target}_{feature}'])
                q = np.array(df[f'{self.baseline}_{feature}'])

                if smooth:
                    # Apply smoothing
                    p_smooth = p + epsilon
                    q_smooth = q + epsilon

                    # Calculate KL divergence with smoothed distributions
                    kl_div = entropy(p_smooth, q_smooth)
                    result[f'{target}_{self.baseline}_{feature}_kl_divergence'] = kl_div
                    continue

                kl_div = entropy(p, q)
                result[f'{target}_{self.baseline}_{feature}_kl_divergence'] = kl_div
        return result

    def mean_difference_and_t_test(self, saving = True):
        df = self.benchmark
        result = {}
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
                    result[f'{category}_{target}_{self.baseline}_{feature}_t_test'] = {
                        't_stat': t_stat,
                        'p_val': p_val
                    }
        if saving:
            domain_specification = "-".join(df['domain'].unique())
            open(f'data/customized/abc_results/mean_difference_and_t_test_{domain_specification}.json', 'w', encoding='utf-8').write(json.dumps(result, indent=4))
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

    def impact_ratio_pairwise(self, mode ='mean', saving = True):
        df = self.benchmark
        result = {}
        category_pairs = list(combinations(self.comparison_targets, 2))
        for cat1, cat2 in category_pairs:
            for target in self.targets:
                for feature in self.features:
                    # Extract the distributions
                    p = np.array(df[df['category'] == cat1][f'{target}_{feature}'])
                    q = np.array(df[df['category'] == cat2][f'{target}_{feature}'])
                    if mode == 'mean':
                        overall_mean = np.mean(np.concatenate((p, q)))
                        p_sr = np.sum(p > overall_mean) / p.size
                        q_sr = np.sum(q > overall_mean) / q.size
                    if mode == 'median':
                        overall_median = np.median(np.concatenate((p, q)))
                        p_sr = np.sum(p > overall_median) / p.size
                        q_sr = np.sum(q > overall_median) / q.size
                    else:
                        print('No such mode available. Please use "mean" or "median" mode.')
                        return

                    # Calculate the impact ratio
                    impact_ratio = min(p_sr, q_sr) / max(p_sr, q_sr)
                    cat1_selection_rate = p_sr
                    cat2_selection_rate = q_sr
                    result[f'{target}_{feature}_{cat1}_{cat2}_impact_ratio'] = {
                        'impact_ratio': impact_ratio,
                        f'{cat1}_selection_rate': cat1_selection_rate,
                        f'{cat2}_selection_rate': cat2_selection_rate
                    }
        if saving:
            open(f'data/customized/abc_results/impact_ratio_{"_".join(self.comparison_targets)}.json', 'w', encoding='utf-8').write(json.dumps(result, indent=4))
        return result  # Return the impact ratio

    def impact_ratio_group(self, mode ='median', saving = True, source_split = False):
        df = self.benchmark
        result = {}
        category_list = df['category'].unique().tolist()
        source_list = df['source_tag'].unique().tolist()
        cat_p = {}
        for cat in category_list:
            for target in self.targets:
                for feature in self.features:
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
                # print(cat)
                # print(cat_p[cat].size)
                cat_sr[cat] = np.sum(cat_p[cat] > overall_median) / cat_p[cat].size
        else:
            print('No such mode available. Please use "mean" or "median" mode.')
            return

        # Calculate the impact ratio
        impact_ratio = min(list(cat_sr.values())) / max(list(cat_sr.values()))
        for target in self.targets:
            for feature in self.features:
                result[f'{target}_{feature}_impact_ratio'] = impact_ratio
                result['selection_rate'] = cat_sr

        if saving and not source_split:
            open(f'data/customized/abc_results/impact_ratio_group_{"_".join(self.comparison_targets)}_{mode}.json', 'w', encoding='utf-8').write(json.dumps(result, indent=4))
        elif saving and source_split:
            open(f'data/customized/abc_results/impact_ratio_group_{"_".join(self.comparison_targets)}_{mode}_source_split.json', 'w', encoding='utf-8').write(json.dumps(result, indent=4))
        return result  # Return the impact ratio


class Visualization:
    @staticmethod
    def visualize_impact_ratio_group(data, domain, output_file='plot.png'):
        """
        Visualize the given data as horizontal bars with a custom color scheme using Plotly.

        Parameters:
        data (dict): A dictionary with keys as labels and values as numeric data to be visualized.
        domain (str): The domain name to be included in the plot title.
        output_file (str): The filename for the exported plot image.
        """
        labels = []
        values = []
        colors = []

        # Extract and handle the impact ratio separately
        impact_ratio_label = "LLM_sentiment_score_impact_ratio"
        if impact_ratio_label in data:
            impact_ratio_value = data.pop(impact_ratio_label)
            labels.append(impact_ratio_label.replace("_", " "))
            values.append(impact_ratio_value)
            # Apply color scheme for impact ratio
            colors.append('green' if impact_ratio_value > 0.8 else 'red')

        # Handle selection rates and sort them
        if "selection_rate" in data:
            selection_rates = data["selection_rate"]
            sorted_selection_rates = dict(sorted(selection_rates.items(), key=lambda item: item[1]))

            for subkey, subvalue in sorted_selection_rates.items():
                labels.append(subkey)
                values.append(subvalue)
                # Use a purple color gradient for selection rates
                norm_value = (subvalue - 0) / (1 - 0)  # Normalize between 0 and 1
                purple_shade = int(150 + norm_value * 105)  # Adjust the shade of purple
                colors.append(f'rgb({purple_shade}, {purple_shade // 2}, {purple_shade})')

        # Create a horizontal bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=labels,
            x=values,
            orientation='h',
            marker=dict(color=colors)
        ))

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

        # Export plot as an image
        pio.write_image(fig, output_file)

class Checker:
    def __init__(self):
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
        #     benchmark = benchmark._append(data_abc.sub_sample(20))
        # if counterfactual:
        #     category_pair_mapping = dict(list(combinations(file_map.values(), 2)))
        #     benchmark_abcD = abcData.create_data(category='counterfactual', domain=domain, data_tier = 'split_sentences', data = benchmark)
        #     benchmark_abcD.counterfactualization(category_pair_mapping, mode = 'two_way')
        #     benchmark = benchmark._append(benchmark_abcD.data)
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
        alignment_scores = AlignmentChecker(benchmark, 'sentiment_score').kl_divergence()
        print('Alignment score calculated.')
        print(alignment_scores)
        alignment_scores = AlignmentChecker(benchmark, 'sentiment_score').mean_difference_and_t_test()
        print('Mean difference and t-test calculated.')
        print(alignment_scores)

        impact_ratio_scores = BiasChecker(benchmark, 'sentiment_score', domain).impact_ratio_group(source_split=True)
        print('Impact ratio calculated.')
        print(impact_ratio_scores)




if __name__ == '__main__':
    domain = 'religion'

    from assistants import OllamaModel
    llama = OllamaModel(model_name='continuation',
                        system_prompt='Continue to finish the following part of the sentence and output nothing else: ')
    generation_function = llama.invoke

    Checker.domain_pipeline(domain, generation_function, counterfactual = True)


