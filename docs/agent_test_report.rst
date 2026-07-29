AI Agent test report
====================

:Date: 2026-07-29
:Branch: ``dev``
:Commit under test: ``fd53fcf``
:Overall result: **Partial pass — end-to-end Codex spawn failed**

Purpose
-------

This report verifies that the bundled ``bioinformatics-researcher`` agent:

* is packaged in the expected format;
* can be installed for Codex and Claude;
* gives a useful, evidence-based research response; and
* can be discovered and spawned by Codex at runtime.

Test results
------------

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Test
     - Result
     - Evidence
   * - Unit tests for the installer
     - PASS
     - ``6 passed`` in ``tests/test_ai_install.py``
   * - Ruff static checks
     - PASS
     - ``pymaftools/ai_install.py`` and its tests passed
   * - Codex project installation
     - PASS
     - Wrote and parsed
       ``.codex/agents/bioinformatics-researcher.toml``
   * - Claude project installation
     - PASS
     - Wrote and checked
       ``.claude/agents/bioinformatics-researcher.md``
   * - Agent behavior smoke test
     - PASS WITH LIMITATION
     - A subagent was manually instructed to load the bundled TOML; it produced
       a reproducible MAF study without modifying the repository
   * - Independent result reproduction
     - PASS
     - The main agent reran the analysis and reproduced every reported number
   * - Codex named-agent runtime spawn
     - FAIL
     - Two attempts failed with
       ``collab spawn failed: no thread with id``
   * - Claude named-agent runtime spawn
     - NOT TESTED
     - No Claude CLI runtime test was run

Behavior smoke test
-------------------

The researcher used the bundled ``multisample`` MAF and proposed an exploratory
study of nonsynonymous mutation burden and six-class SNV substitutions. The
following observations were independently reproduced:

* 6 samples;
* 272 nonsynonymous variants;
* 39–50 variants per sample;
* 262 classifiable SNPs;
* ``C>A`` was the largest substitution class: 102/262 (38.9%);
* ``USH2A``, ``RYR2``, and ``SPTA1`` were the top genes, each present in
  3 samples.

The referenced public APIs exist:

* ``load_example_maf(name="multisample", **kwargs)``
* ``MAF.filter_maf(mutation_types)``
* ``mutation_burden_by_class(maf)``
* ``top_mutated_genes(maf, top=20)``
* ``summarize_titv(maf)``
* ``compare_cohorts(...)``

The Agent correctly described this as exploratory evidence and did not turn it
into a clinical claim.

Confirmed problems
------------------

1. **Codex end-to-end integration is not passing.**

   The installed agent file matches the official custom-agent schema, but an
   isolated ``codex exec`` session could not create any subagent thread. Both
   attempts ended with ``no thread with id``. This means the test has not yet
   proven that a user can install the asset and successfully spawn it by name.
   The failure may be in the CLI/session router rather than this agent file, but
   that distinction is not yet verified.

2. **The behavioral test was not a discovery test.**

   The successful research task explicitly told a subagent to read the TOML.
   It proves that the instructions are usable, but not that Codex automatically
   discovered the installed custom role.

3. **The handoff roles named by the researcher do not exist in the package.**

   The agent requires handoff to an independent bioinformatics validator and a
   reproducibility reviewer, but neither agent is currently bundled. The
   workflow therefore stops at a checklist instead of performing independent
   validation.

4. **Claude has only been format-tested.**

   The Markdown conversion is valid, but no Claude runtime was used to discover
   or execute the generated agent.

5. **Formal cohort comparison is missing uncertainty intervals.**

   ``compare_cohorts()`` reports an odds ratio, p-value, and FDR, but no
   confidence interval. The bundled example MAF also lacks a defensible clinical
   cohort variable, so it cannot support a meaningful two-cohort biological
   conclusion by itself.

Official format check
---------------------

The current OpenAI Codex manual says project agents belong in
``.codex/agents/`` and require ``name``, ``description``, and
``developer_instructions``. The generated TOML satisfies those requirements.

Source:
`OpenAI Codex subagents documentation
<https://learn.chatgpt.com/docs/agent-configuration/subagents>`_.

Next acceptance test
--------------------

Run the same isolated named-agent test after resolving the Codex
``no thread with id`` failure. Acceptance requires a runtime-created thread
whose selected role is ``bioinformatics-researcher`` and whose response follows
the bundled developer instructions without those instructions being copied into
the test prompt.
