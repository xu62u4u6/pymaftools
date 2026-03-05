"""Sphinx configuration for pymaftools documentation."""

project = "pymaftools"
copyright = "2026, xu62u4u6"
author = "xu62u4u6"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Napoleon settings (NumPy style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Type hints settings
typehints_defaults = "comma"
always_use_bars_union = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
