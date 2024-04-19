import sphinx_rtd_theme
numfig = True

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'TRex'
copyright = '2020, Tristan Walter'
author = 'Tristan Walter'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_dark_mode",
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton'
]

default_dark_mode = True
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 4

html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_dark_mode"
master_doc = 'contents'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [ 
    "custom.css"
]
html_css_files = [
    "custom.css"
]

html_theme_options = {
    'canonical_url': '',
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
#    'style_nav_header_background': '',
    # Toc options
    'collapse_navigation': True,
#    'sticky_navigation': False,
#    'navigation_depth': 3,
    'includehidden': True,
#    'titles_only': True
}

from docutils import nodes, utils
from docutils.parsers.rst.roles import set_classes

def green_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    set_classes(options)
    text = utils.unescape(text)
    node = nodes.inline(rawtext, text, classes=['green-text'])
    return [node], []

def setup(app):
    app.add_css_file('custom.css')  # may also be an URL
    app.add_role('green', green_role)