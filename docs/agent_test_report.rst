AI Agent Test Report
====================

:Date: 2026-08-28
:Branch: ``dev``
:Overall result: **Partial pass (assets and installation pass; some runtime
   checks remain blocked by the environment)**

Scope
-----

This report tests the AI collaboration assets bundled with pymaftools and the
recommended workflow:

``scientific-user-tester`` → ``bioinformatics-researcher`` →
``bioinformatics-validator`` → ``biostatistics-reviewer`` →
``reproducibility-reviewer`` → ``scientific-manuscript-writer``.

This is a suggested order of responsibilities; the package does not ship an
orchestrator that chains the roles automatically. Testing covers TOML format,
the installer, package build, documentation, and actual runtime behaviour. A
runtime test counts as passed only when the named role was really loaded and
its role instructions were observed; a generic child agent with a similar name
does not count.

Bundled agents
--------------

``scientific-user-tester``
   Reads only the public documentation and, from a clean environment, runs
   installation, data loading, and table and figure output, recording the
   blockers and documentation gaps a user would hit.

``bioinformatics-researcher``
   Designs and runs exploratory, reproducible bioinformatics studies, then hands
   the results to independent review.

``bioinformatics-validator``
   Independently checks study design, API usage, statistical methods, results,
   biological interpretation, and limitations.

``biostatistics-reviewer``
   Attacks denominators, independence, confounding, selection, multiple
   testing, confidence intervals, sparse cells, and leakage.

``reproducibility-reviewer``
   Independently checks whether others can rerun the workflow from the
   documentation, versions, checksums, and commands, and obtain substantively
   identical results and figures.

``scientific-manuscript-writer``
   Assembles a manuscript from the claim ledger, executable artifacts, and
   independent reviews, preserving the VERIFIED, PARTIAL, PROPOSED, BLOCKED, and
   NOT MEASURED evidence boundaries.

Automated tests
---------------

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Test
     - Result
     - Evidence
   * - Agent TOML format
     - Pass
     - All six files parse with ``tomllib`` and contain ``name``,
       ``description``, and ``developer_instructions``.
   * - Installer unit tests
     - Pass
     - All 6 tests in ``tests/test_ai_install.py`` pass, covering Codex file
       installation for all six agents, Claude Markdown conversion, and dry-run.
   * - Ruff static checks
     - Pass
     - ``pymaftools`` and ``tests/test_ai_install.py`` pass Ruff.
   * - Full project test suite
     - Pass
     - 304 passed, 7 skipped, 10 deselected, 48 warnings; coverage 68.52%.
   * - Wheel / sdist contents
     - Pass
     - The wheel and sdist built with network access pass ``twine check``; the
       wheel contains all six agent definition files.
   * - Codex project install
     - Pass
     - The installer writes the six TOML files into an isolated project's
       ``.codex/agents``.
   * - Claude project install
     - Pass
     - The installer writes six Markdown files, rendered from the same TOML,
       into an isolated project's ``.claude/agents``.

Claude runtime tests
--------------------

Previously saved evidence from Claude Code 2.1.170 shows that the first three
agents load through the real ``--agent <name>`` flag and answer in role:

.. list-table::
   :header-rows: 1
   :widths: 32 22 46

   * - Agent
     - Runtime result
     - Role evidence
   * - ``bioinformatics-researcher``
     - Pass
     - Refused to endorse its own conclusions and required hand-off to two
       independent reviewer roles.
   * - ``bioinformatics-validator``
     - Pass
     - Used only the ``PASS``, ``PASS WITH CONDITIONS``, and ``FAIL`` review
       states.
   * - ``reproducibility-reviewer``
     - Pass
     - Refused to review biological interpretation and dealt only with
       reproducibility.

The ``scientific-user-tester``, ``biostatistics-reviewer``, and
``scientific-manuscript-writer`` agents added in this round have passed format,
installation, and content-contract tests, but no new external-model runtime
evidence has been collected. Claude Code 2.1.233 in the test shell reported that
organization subscription access was disabled (an Anthropic API key or
administrator enablement is required), so runtime behaviour that was not
executed is not reported as passing.

Claude role-behaviour tests (existing evidence)
-----------------------------------------------

No project files were sent to Claude; the tests used synthetic cases that
contain no private repository content.

Researcher
~~~~~~~~~~

Result: **Pass after fix**. For six synthetic MAF samples the researcher
produced an exploratory study specification, data-source fields, filtering
rules, machine-readable outputs, study limitations, and a hand-off checklist,
and labelled the data as synthetic and not for clinical decisions. The first
run used cosine distance with Ward linkage, which are incompatible; the role
definition now includes a method-compatibility guard and the regression test
uses ``Euclidean distance + Ward linkage``.

Bioinformatics validator
~~~~~~~~~~~~~~~~~~~~~~~~

Result: **Pass**. The validator rejected a causal survival conclusion that the
study design did not support, flagged the missing cohort, reference genome,
statistical test, effect size, confidence interval, and multiple-testing
correction, declined to pretend to recompute without data, and returned
``FAIL``.

Reproducibility reviewer
~~~~~~~~~~~~~~~~~~~~~~~~

Result: **Pass**. Given a deliberately incomplete workflow, the reviewer
flagged missing input sources and checksums, dependency versions, random seed,
complete commands, reference genome, expected outputs, and comparison
tolerances, and returned ``FAIL``.

Codex runtime tests
-------------------

Environment: Codex CLI 0.149.1.

Result: **Fail: the current runtime cannot select a custom role**. Tests placed
the agent TOML in ``.codex/agents`` and tried an isolated Git repository,
multi-agent mode, stable ``multi_agent_v2``, and ``[agents.<role>]``
registration. Codex could create a generic child thread, but the child
session's ``agent_role`` metadata stayed empty and the package's
``developer_instructions`` were not loaded. The current ``spawn_agent``
interface has no ``role`` or ``agent_type`` parameter, so a generic child cannot
be claimed as a passing custom-agent test.

Other confirmed limitations
---------------------------

* The three new Claude roles have only format, installation, and
  content-contract evidence. Once external-model access is restored, their
  runtime behaviour must be tested with synthetic cases that contain no private
  data.
* The Claude safety layer does not allow sending pymaftools repository content
  to an external model, so the existing behaviour tests use synthetic cases.
* The existing ``compare_cohorts()`` returns odds ratio, p-value, and FDR, but
  no confidence interval.
* Codex custom-role selection is a runtime compatibility gap, not a format
  error in the package's agent definitions.
* The Sphinx ``-W --keep-going`` build with network access succeeded in this
  round; an offline build still fails on DNS resolution of the intersphinx
  inventories, which is an environment limitation rather than a documentation
  syntax error.

Criteria for a full pass
------------------------

The agent workflow counts as fully passing only when:

#. Claude runtime behaviour tests are completed for the three new agents.
#. Codex offers a custom-role selection parameter, and the child session
   records the correct role and the package's ``developer_instructions``.
#. The researcher, validator, biostatistics reviewer, and reproducibility
   reviewer keep handing off in order in later regression tests, with traceable
   evidence for every conclusion.

Official format reference
-------------------------

The current OpenAI Codex documentation specifies that project agents live in
``.codex/agents`` and must include ``name``, ``description``, and
``developer_instructions``:
`OpenAI Codex subagents documentation
<https://learn.chatgpt.com/docs/agent-configuration/subagents>`_.
