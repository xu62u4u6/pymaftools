"""Sphinx configuration for pymaftools documentation."""

from docutils.parsers.rst import roles

project = "pymaftools"
copyright = "2026, xu62u4u6"
author = "xu62u4u6"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_llms_txt",
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
autodoc_mock_imports = ["anndata"]

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

# llms.txt / llms-full.txt (https://llmstxt.org) for coding agents, generated
# from these same sources so the agent-facing docs never drift from the site.
html_baseurl = "https://dionic.xyz/pymaftools/"
llms_txt_title = "pymaftools"
llms_txt_summary = (
    "Python library for cancer-genomics analysis: read MAF/VCF event tables, "
    "build feature-by-sample matrices with aligned sample and feature metadata "
    "(PivotTable), compute mutation statistics, and draw oncoplots, lollipop "
    "and interaction plots through layered .plot accessors."
)
# Link rendered pages, not raw .rst sources: the API pages are autodoc
# directives whose docstrings only exist in the HTML.
llms_txt_uri_template = "{base_url}{docname}.html"
# Internal QA record, not API guidance for agents.
llms_txt_exclude = ["agent_test_report"]

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


def setup(app):
    """Register the Matplotlib type role used by upstream docstrings."""
    roles.register_local_role("mpltype", roles.code_role)
