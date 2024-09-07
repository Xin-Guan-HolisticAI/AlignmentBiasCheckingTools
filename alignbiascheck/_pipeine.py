from .analytics import check_benchmark, ModelGenerator, FeatureExtractor, Analyzer
import pandas as pd



def _update_configuration(scheme_dict, default_dict, updated_dict):
    """
    Update the scheme dictionary with values from the updated dictionary, or from the default
    dictionary if the updated value is not available.

    Args:
    - scheme_dict (dict): The scheme dictionary with keys and None values.
    - default_dict (dict): The dictionary containing default values.
    - updated_dict (dict): The dictionary containing updated values.

    Returns:
    - dict: The configuration dictionary with updated values.
    """

    for key, value in scheme_dict.items():
        if value is None:
            if key in updated_dict:
                if isinstance(updated_dict[key], dict) and isinstance(scheme_dict[key], dict):
                    # Recursively update nested dictionaries
                    scheme_dict[key] = _update_configuration(
                        scheme_dict[key],
                        default_dict.get(key, {}),
                        updated_dict[key]
                    )
                else:
                    # Use the value from updated_dict if available
                    scheme_dict[key] = updated_dict[key]
            else:
                # Use the value from default_dict if available
                scheme_dict[key] = default_dict.get(key, None)
        elif isinstance(value, dict):
            # If the value itself is a dictionary, recursively update it
            scheme_dict[key] = _update_configuration(
                scheme_dict[key],
                default_dict.get(key, {}),
                updated_dict.get(key, {})
            )

    return scheme_dict


class Pipeline:
    _analytics_config_scheme = {}
    _analytics_default_config = {}

    @classmethod
    def _set_config(cls):
        cls._analytics_config_scheme = {
            "benchmark": None,
            "generation": {
                "require": None,
                "generate_dict": None,
                "generation_saving_location": None,
                "generation_list": None,
            },
            "extraction": {
                "feature_extractors": None,
                'extractor_configs': None,
                "calibration": None,
                "extraction_saving_location": None,
            },
            "analysis": {
                "specifications": None,
                "analyzers": None,
                "analyzer_configs": None,
                'statistics_saving_location': None,
                "disparity_saving_location": None,
            }
        }
        cls._analytics_default_config = {
            "generation": {
                "require": True,
                "generate_dict": {},
                "generation_saving_location": 'data/customized/' + '_' + 'sbg_benchmark.csv',
                "generation_list": [],
                "baseline": 'baseline',
            },
            "extraction": {
                "feature_extractors": [
                    'personality_classification',
                    'toxicity_classification',
                    'sentiment_classification',
                    'stereotype_classification',
                    'regard_classification'
                ],
                'extractor_configs': {},
                "calibration": True,
                "extraction_saving_location": 'data/customized/' + '_' + 'sbge_benchmark.csv',
            },
            "analysis": {
                "specifications": ['category', 'source_tag'],
                "analyzers": ['mean', 'selection_rate', 'precision'],
                "analyzer_configs": {
                    'selection_rate': {'standard_by': 'mean'},
                    'precision': {'tolerance': 0.1}
                },
                'statistics_saving_location': 'data/customized/' + '_' + 'sbgea_statistics.csv',
                "disparity_saving_location": 'data/customized/' + '_' + 'sbgea_disparity.csv',
            }
        }

    @classmethod
    def config_helper(cls):
        pass

    @classmethod
    def benchmark_building(cls, config, domain='unspecified'):
        pass

    @classmethod
    def analytics(cls, config, domain='unspecified'):
        cls._set_config()
        v = _update_configuration(
            cls._analytics_config_scheme.copy(),
            cls._analytics_default_config.copy(),
            config.copy())


        if v['generation']['require']:
            gen = ModelGenerator(v['benchmark'])

            for name, gf in v['generation']['generate_dict'].items():
                gen.generate(gf, generation_name=name)
            sbg_benchmark = gen.benchmark.copy()
            sbg_benchmark.to_csv(v['generation']['generation_saving_location'], index=False)

            generation_list = list(v['generation']['generate_dict'].keys())
            glb = ['baseline'] + generation_list.copy()
        else:
            sbg_benchmark = v['benchmark']
            generation_list = v['generation']['generation_list']
            glb = ['baseline'] + generation_list

        fe = FeatureExtractor(sbg_benchmark, generations=glb, calibration=v['extraction']['calibration'])

        sbge_benchmark = pd.DataFrame()
        for x in v['extraction']['feature_extractors']:
            try:
                method_to_call = getattr(fe, x)
                sbge_benchmark = method_to_call(**v['extraction']['extractor_configs'].get(x, {}))
            except AttributeError as e:
                print(f"Method {x} does not exist: {e}")
            except Exception as e:
                print(f"Error calling method {x}: {e}")
        sbge_benchmark.to_csv(v['extraction']['extraction_saving_location'], index=False)
        raw_features = fe.classification_features + fe.cluster_features
        calibrated_features = fe.calibrated_features

        anas = []
        anas.append(Analyzer(sbge_benchmark.copy(), features=raw_features, generations=glb))
        if v['extraction']['calibration']:
            anas.append(
                Analyzer(sbge_benchmark.copy(), features=calibrated_features, generations=generation_list))

        for k, ana in enumerate(anas):
            ana.specifications = v['analysis']['specifications']
            for x in v['analysis']['analyzers']:
                try:
                    method_to_call = getattr(ana, x)
                    sbgea_benchmark = method_to_call(test=False, **v['analysis']['analyzer_configs'].get(x, {}))
                    if k == 0:
                        sbgea_benchmark.to_csv(v['analysis']['statistics_saving_location'].replace('.csv', f'_{x}.csv'), index=False)
                    elif k == 1:
                        disparity_calibrated_saving_location = v['analysis']['disparity_saving_location'].replace(
                            '.csv',
                            f'_calibrated_{x}.csv')
                        sbgea_benchmark.to_csv(disparity_calibrated_saving_location, index=False)
                except AttributeError as e:
                    print(f"Method {x} does not exist: {e}")
                except Exception as e:
                    print(f"Error calling method {x}: {e}")
            df = ana.statistics_disparity()
            if k == 0:
                df.to_csv(v['analysis']['disparity_saving_location'], index=False)
            elif k == 1:
                disparity_calibrated_saving_location = v['analysis']['disparity_saving_location'].replace('.csv',
                                                                                                          '_calibrated.csv')
                df.to_csv(disparity_calibrated_saving_location, index=False)

    @classmethod
    def pipeline(cls, config, domain='unspecified'):
        pass
