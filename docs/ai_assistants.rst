AI assistants
=============

``pymaftools`` distributes reusable AI assets from the package itself so that
contributors and users install the same reviewed definitions. Personal
``~/.codex`` or ``~/.claude`` directories are deployment targets, not the
source of truth.

Bundled assets
--------------

``bioinformatics-researcher``
   Plans and runs reproducible example studies. It records provenance, produces
   rerunnable analyses and artifacts, separates observations from interpretation,
   and hands results to independent validation.

``bioinformatics-validator``
   Independently checks study design, data provenance, package API use,
   statistics, reported values, interpretation, and limitations. It reports
   ``PASS``, ``PASS WITH CONDITIONS``, or ``FAIL``.

``reproducibility-reviewer``
   Independently checks whether documented inputs, environments, commands,
   seeds, tables, and figures can be rerun and compared. It does not validate
   biological interpretation.

``pymaftools`` skill
   Provides the package architecture, current API workflows, contribution rules,
   and research-validation guidance needed by an AI coding or research agent.

Install
-------

Preview the available assets:

.. code-block:: console

   pymaftools-ai list

Install all agents and the skill for Codex:

.. code-block:: console

   pymaftools-ai install --target codex --dry-run
   pymaftools-ai install --target codex

Use ``--target claude`` for Claude. The default scope is the current user's
``~/.codex`` or ``~/.claude`` directory. To keep the assets inside a checked-out
project, use:

.. code-block:: console

   pymaftools-ai install --target codex --scope project

The installer does not overwrite different existing files unless ``--force`` is
provided. It does not modify unrelated tool configuration.

Source layout
-------------

Canonical assets live under ``pymaftools/ai_assets``. Codex agents use the
bundled TOML directly; Claude agent Markdown is rendered from the same TOML.
The Skill is copied without rewriting so both targets receive identical
instructions and references.
