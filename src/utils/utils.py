from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer


def label_encode_column(column):
    le = LabelEncoder()
    return le.fit_transform(column)


def encode_region_column(x):
    return label_encode_column(x["region"]).reshape(-1, 1)


def get_feature_names(preprocessor, input_features):
    output_feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name != "remainder":
            if isinstance(transformer, Pipeline):
                # Get the actual transformer from the pipeline
                transformer = transformer.named_steps[transformer.steps[-1][0]]
            if isinstance(transformer, OneHotEncoder):
                # Get feature names for OneHotEncoder
                transformed_feature_names = transformer.get_feature_names_out(features)
                output_feature_names.extend(transformed_feature_names)
            elif isinstance(transformer, StandardScaler):
                # StandardScaler does not change feature names
                output_feature_names.extend(features)
            elif isinstance(transformer, FunctionTransformer):
                # FunctionTransformer should ideally not change feature names but we provide a generic name
                output_feature_names.extend(features)
    return output_feature_names
