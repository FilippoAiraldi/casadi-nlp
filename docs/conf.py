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

# decide which version of the docs is being built
version_match = os.environ.get("READTHEDOCS_VERSION")
if (
    not version_match or version_match.isdigit() or version_match == "latest"
) and "rc" in release:
    version_match = "latest"
else:
    version_match = "stable"

# options for the html output and theme
html_static_path = ["_static"]
html_logo = "_static/csnlp.logo.png"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/FilippoAiraldi/casadi-nlp",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/csnlp/",
            "icon": "fa-solid fa-box",
        },
    ],
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "logo": {"text": "csnlp"},
    "use_edit_page_button": True,
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "switcher": {
        "json_url": "https://casadi-nlp.readthedocs.io/en/latest/_static/switcher.json",
        "version_match": version_match,
    },
}
html_context = {
    "github_user": "FilippoAiraldi",
    "github_repo": "casadi-nlp",
    "github_version": "main",
    "doc_path": "docs",
}
