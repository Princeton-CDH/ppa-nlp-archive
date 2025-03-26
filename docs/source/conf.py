# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "corppa"
copyright = "2024, Princeton CDH RSE Team"
author = "Princeton CDH RSE Team"
release = "0.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    #    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
]

source_suffix = [".rst", ".md"]
templates_path = ["_templates"]
exclude_patterns = []
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Ignore annotation-related modules
autodoc_mock_imports = [
    "prodigy",
    "pyarrow",
    "rapidfuzz",
    "spacy",
    "unidecode",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
