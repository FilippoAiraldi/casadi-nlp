# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

import csnlp

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = csnlp.__name__
copyright = "2024, Filippo Airaldi"
author = "Filippo Airaldi"
version = release = csnlp.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_autodoc_typehints",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# enables intersphinx to pick references to external libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
highlight_language = "python3"

# config for the sphinx gallery of examples
sphinx_gallery_conf = {
    "doc_module": "csnlp",
    "backreferences_dir": os.path.join("generated/generated"),
    "reference_url": {"csnlp": None},
    "filename_pattern": "",
    "default_thumb_file": "_static/csnlp.logo.examples.png",
}

# for references
bibtex_bibfiles = ["references.bib"]

# other options
add_function_parentheses = False
autosummary_generate = True
autosummary_imported_members = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_logo = "_static/csnlp.logo.png"
html_theme = "alabaster"
html_theme_options = {
    "github_user": "FilippoAiraldi",
    "github_repo": "casadi-nlp",
    "github_button": "true",
    "link": "#aa560c",
}
