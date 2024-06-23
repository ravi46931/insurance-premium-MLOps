# name: MkDocs

# on:
#   push:
#     branches:
#       - main  

# permissions:
#   contents: write

# jobs:
#   deploy:
#     runs-on: ubuntu-latest

#     steps:
#       - name: Checkout code
#         uses: actions/checkout@v3

#       - name: Set up Python
#         uses: actions/setup-python@v4
#         with:
#           python-version: 3.x  
#       - uses: actions/cache@v2
#         with:
#           key: ${{ github.ref }}
#           path: .cache
#       - run: |
#           python -m pip install --upgrade pip

#       - run: pip install mkdocs
#       - run: |
#           cd docs
#           mkdocs build -f mkdocs.yml
#           mkdocs gh-deploy --force
      