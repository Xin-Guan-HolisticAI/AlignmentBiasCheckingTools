from ..alignbiascheck import analytics as an


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
    _analytics_config_scheme = {
        "benchmark": None,
        "generation": {
            "generate_dict": None,
            "generation_saving_location": None,
        },
        "extraction": {
            "feature_extractors": None,
            "calibration": None,
            "extraction_saving_location": None,
        },
        "analysis": {
            "specifications": None,
            "analyzers": None,
            "analyzers_configs": None,
            "disparity_saving_location": None,
        }
    }

    _analytics_default_config = {
        "generation": {
            "generate_dict": {},
            "generation_saving_location": 'data/customized/' + '_' + 'sbg_benchmark.csv',
        },
        "extraction": {
            "feature_extractors": [
                'personality_classification',
                'toxicity_classification',
                'sentiment_classification',
                'stereotype_classification',
                'regard_classification'
            ],
            "calibration": True,
            "extraction_saving_location": 'data/customized/' + '_' + 'sbge_benchmark.csv',
        },
        "analysis": {
            "specifications": ['category', 'source_tag'],
            "analyzers": ['mean', 'selection_rate', 'precision'],
            "analyzers_configs": {
                'selection_rate': {'standard_by': 'mean'},
                'precision': {'tolerance': 0.1}
            },
            "disparity_saving_location": 'data/customized/' + '_' + 'sbgea_disparity.csv',
        }
    }

    @classmethod
    def analytics_pipeline(cls, config):
        v = _update_configuration(
            cls._analytics_config_scheme.copy(),
            cls._analytics_default_config.copy(),
            config.copy())

        gen = an.ModelGenerator(v['benchmark'])

        for name, gf in v['generation']['generate_dict'].items():
            gen.generate(gf, generation_name=name)
        sbg_benchmark = gen.benchmark.copy()
        sbg_benchmark.to_csv(v['generation']['generation_saving_location'], index=False)

        generation_list = list(v['generation']['generate_dict'].keys())
        glb = ['baseline'] + generation_list.copy()

        fe = ann.FeatureExtractor(sbg_benchmark, generations=glb, calibration=v['extraction']['calibration'])

        sbge_benchmark = pd.DataFrame()
        for x in v['extraction']['feature_extractors']:
            try:
                method_to_call = getattr(fe, x)
                sbge_benchmark = method_to_call()
            except AttributeError as e:
                print(f"Method {x} does not exist: {e}")
            except Exception as e:
                print(f"Error calling method {x}: {e}")
        sbge_benchmark.to_csv(v['extraction']['extraction_saving_location'], index=False)
        classification_features = fe.classification_features
        calibrated_features = fe.calibrated_features

        anas = []
        anas.append(ann.Analyzer(sbge_benchmark.copy(), features=classification_features, generations=glb))
        if v['extraction']['calibration']:
            anas.append(ann.Analyzer(sbge_benchmark.copy(), features=calibrated_features, generations=generation_list))

        for k, ana in enumerate(anas):
            ana.specifications = v['analysis']['specifications']
            for x in v['analysis']['analyzers']:
                try:
                    method_to_call = getattr(ana, x)
                    method_to_call(test=False, **v['analysis']['analyzers_configs'].get(x, {}))
                except AttributeError as e:
                    print(f"Method {x} does not exist: {e}")
                except Exception as e:
                    print(f"Error calling method {x}: {e}")
            df = ana.statistics_disparity()
            if k == 1:
                df.to_csv(v['analysis']['disparity_saving_location'], index=False)
            elif k == 2:
                disparity_calibrated_saving_location = v['analysis']['disparity_saving_location'].replace('.csv',
                                                                                                          '_calibrated.csv')
                df.to_csv(disparity_calibrated_saving_location, index=False)