import os
from pathlib import Path

list_of_files = [
    ".github/workflows/.gitkeep",
    ".github/workflows/ci.yaml",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/constants/__init__.py",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/utils/__init__.py",
    "src/utils/utils.py",
    "src/logger/__init__.py",
    "src/exception/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "docs/docs/index.md",
    "docs/docs/getting-started.md",
    "docs/mkdocs.yml",
    "docs/README.md",
    "data/.gitkeep",
    ".env",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "app.py",
    "main.py",
    "experiment/experiments.ipynb",
    "README.md",
    "implement.md",
    ".gitignore"
]

for file in list_of_files:
    file = Path(file)
    filedir, filename = os.path.split(file)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(file)) or (os.path.getsize(file)==0):
        with open (file, 'wb') as f:
            pass
 