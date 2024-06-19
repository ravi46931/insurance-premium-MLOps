from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    data_file_path: str


@dataclass
class DataTransformationArtifact:
    train_transform_file_path: str
    test_transform_file_path: str
    preprocessor_file_path: str


@dataclass
class ModelTrainerArtifact:
    top_models_file_name_path: str
    model1_file_path: str
    model2_file_path: str
    model3_file_path: str
    model4_file_path: str


@dataclass
class ModelEvaluationArtifact:
    best_model_file_path: str
