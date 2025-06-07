import sphinx_rtd_theme
import os
import sys

# Enable automatic figure numbering
numfig = True
numfig_secnum_depth = 3  # Adjusts depth for including section numbers

# Customize numbering format (optional)
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
    'section': 'Section %s'
}

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
copyright = '2024, Tristan Walter'
author = 'Tristan Walter'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_dark_mode",
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'hoverxref.extension'  # Add the hoverxref extension
]

default_dark_mode = True
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3

hoverxref_auto_ref = True

html_sidebars = { '**': ['localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

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
    "css",
    "images"
]
html_css_files = [
    "custom.css"
]
html_js_files = [
]

html_theme_options = {
    'canonical_url': '',
    'logo_only': True,
    #'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
#    'style_nav_header_background': '',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
#    'titles_only': True
}

# Configuration for hoverxref
hoverxref_api_host = 'https://trex.run/docs/docs2'

hoverxref_roles = [
    'py:func',
    'ref',
    'param'              # Enable hover previews for the param role
]  
hoverxref_role_types = {
    'class': 'tooltip',  # Use tooltip for class role,
    'py:func': 'tooltip',  # Use tooltip for py:func role,
    'param': 'tooltip',  # Use tooltip for param role
    'ref': 'tooltip',  # Use tooltip for ref role
}

from docutils import nodes, utils
from docutils.parsers.rst.roles import set_classes
from sphinx.addnodes import pending_xref

def green_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    set_classes(options)
    text = utils.unescape(text)
    node = nodes.inline(rawtext, text, classes=['green-text'])
    return [node], []

def param_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Create a cross-reference to a function defined in parameters_trex.rst.
    Example: :param:`adaptive_threshold_scale`
    """
    set_classes(options)
    # Create a pending_xref node that references a Python function
    node = pending_xref(
        rawsource=rawtext,
        refdomain='py',  # Use the Python domain
        reftype='func',  # Use the function reference type
        reftarget=text,  # The target is the function name
        modname=None,
        classname=None,
        refexplicit=False,
        refwarn=True,
        classes=['param-reference'],
        **options
    )
    node += nodes.literal(text, text)
    return [node], []

def setup(app):
    app.add_css_file('custom.css')  # may also be an URL
    app.add_role('green', green_role)
    app.add_role('param', param_role)  # So you can do :param:`my_parameter`