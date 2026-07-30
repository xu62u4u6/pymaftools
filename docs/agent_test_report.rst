AI Agent test report
====================

:Date: 2026-07-30
:Branch: ``dev``
:Overall result: **Partial pass**

Scope
-----

This report tests the packaged AI workflow:

``bioinformatics-researcher`` → ``bioinformatics-validator`` →
``reproducibility-reviewer``.

It distinguishes file-format checks, actual Agent runtime loading, and
role-behavior tests. A generic subagent with a matching task name does not count
as a custom-role runtime pass.

Packaged agents
---------------

``bioinformatics-researcher``
   Designs and runs exploratory, reproducible bioinformatics studies and hands
   the results to independent review.

``bioinformatics-validator``
   Independently checks scientific design, API use, statistics, results,
   interpretation, and limitations.

``reproducibility-reviewer``
   Independently checks whether another person can rerun the workflow and
   reproduce material outputs.

Automated tests
---------------

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Test
     - Result
     - Evidence
   * - Agent TOML schema
     - PASS
     - All three files contain ``name``, ``description``, and
       ``developer_instructions``
   * - Installer unit tests
     - PASS
     - ``6 passed`` in ``tests/test_ai_install.py``
   * - Ruff static checks
     - PASS
     - Installer and tests passed
   * - Full project test suite
     - PASS
     - 264 passed, 7 skipped, and 10 deselected
   * - Wheel package contents
     - PASS
     - Offline wheel build contained all three Agent definitions
   * - Codex project installation
     - PASS
     - All three TOML files were installed into ``.codex/agents``
   * - Claude project installation
     - PASS
     - All three Markdown files were installed into ``.claude/agents``

Claude runtime tests
--------------------

Environment: Claude Code 2.1.170.

Each generated Agent was loaded by its real ``--agent <name>`` option. The
command returned success and each Agent answered from role-specific
instructions:

.. list-table::
   :header-rows: 1
   :widths: 32 22 46

   * - Agent
     - Runtime result
     - Role evidence
   * - ``bioinformatics-researcher``
     - PASS
     - Refused to certify its own conclusions and required both independent
       reviewers
   * - ``bioinformatics-validator``
     - PASS
     - Returned only ``PASS``, ``PASS WITH CONDITIONS``, and ``FAIL`` as its
       allowed statuses
   * - ``reproducibility-reviewer``
     - PASS
     - Refused to validate biological interpretation

Claude role-behavior tests
--------------------------

No project files were sent to Claude. The tests used synthetic prompts without
private repository content.

Researcher
~~~~~~~~~~

Result: **PASS after remediation**

The researcher produced an exploratory six-sample MAF study specification,
provenance fields, filtering rules, machine-readable outputs, limitations, and
handoff checklists. It clearly marked the example as synthetic and non-clinical.

The initial run found one material method problem: it combined
cosine distance with Ward linkage. Ward linkage assumes Euclidean geometry, so
that clustering plan was invalid. The researcher definition now explicitly
requires checking method/input compatibility and prohibits Ward linkage with
cosine distance. The focused Claude runtime regression returned
``Euclidean distance + Ward linkage`` and correctly explained why the pair is
mathematically compatible.

Validator
~~~~~~~~~

Result: **PASS**

Given an intentionally invalid six-sample report, the validator:

* rejected an unsupported causal survival claim;
* flagged the missing cohort, reference genome, test, effect size, confidence
  interval, and multiple-testing correction;
* refused to recompute absent data; and
* returned ``FAIL``.

Reproducibility reviewer
~~~~~~~~~~~~~~~~~~~~~~~~

Result: **PASS**

Given an intentionally incomplete ``analysis.py data.maf`` workflow, the
reviewer flagged missing input provenance and checksum, dependency versions,
random seed, exact command, genome build, expected outputs, and comparison
tolerances. It correctly returned ``FAIL``.

Codex runtime tests
-------------------

Environment: Codex CLI 0.145.0.

Result: **FAIL — custom role selection is unavailable in this runtime**

The following paths were tested:

#. Standalone Agent TOML files under ``.codex/agents``.
#. A real isolated Git repository.
#. Default multi-agent mode.
#. Stable ``multi_agent_v2``.
#. Explicit ``[agents.<role>]`` registrations in ``.codex/config.toml``.

Codex could create a generic child thread, but the child session metadata showed
``agent_role: null`` and did not contain the packaged
``developer_instructions``. The available ``spawn_agent`` interface had no
``role`` or ``agent_type`` parameter. Explicit registration did not add one.

Enabling the under-development ``use_agent_identity`` feature was also tested.
It attempted remote identity registration but still did not expose the custom
role, so it is not a valid workaround.

This is a runtime compatibility gap: the packaged TOML matches the current
documented custom-Agent schema, but this installed Codex CLI cannot select that
role through its exposed spawn interface. A generic child with a similar name
must not be reported as a successful custom-Agent run.

Other confirmed limitations
---------------------------

* Claude successfully loaded all three roles, but a real PyMAFTools study was
  not sent to Claude because the safety layer rejected disclosing repository
  contents to the external model. Synthetic behavior tests were used instead.
* The existing PyMAFTools ``compare_cohorts()`` API reports odds ratio, p-value,
  and FDR but not a confidence interval.
* The full Sphinx ``-W`` build still contains an existing unknown ``mpltype``
  role in ``api/plot`` and offline intersphinx warnings. The Agent report itself
  parses successfully.

Acceptance criteria
-------------------

The packaged workflow is fully passing only when:

#. Codex exposes a custom-role selector and the child session records the
   expected non-null role plus the packaged developer instructions.
#. The researcher continues to select statistically compatible
   distance/linkage pairs in runtime regression tests.
#. A policy-approved end-to-end study is handed from researcher to both
   independent reviewers.

Official format source
----------------------

The current OpenAI Codex manual documents project agents under
``.codex/agents`` with required ``name``, ``description``, and
``developer_instructions`` fields:
`OpenAI Codex subagents documentation
<https://learn.chatgpt.com/docs/agent-configuration/subagents>`_.
