Related projects
================

pymaftools covers mutation and multi-omics *matrices*: reading MAF/VCF files,
building feature x sample tables, testing them, and plotting them. Interpreting
a resulting gene list (pathway enrichment, literature, knowledge graphs) is a
separate job, handled by a sibling project.

pymaftools
   MAF/VCF ingest, mutation and CNV matrices, frequency and enrichment tests
   between sample groups, oncoplots, machine-readable results. No gene set
   enrichment, literature search or LLM interpretation.

`EnrichRAG <https://github.com/xu62u4u6/EnrichRAG>`_
   GO/KEGG enrichment (via gseapy/Enrichr), a local knowledge graph (STRING,
   KEGG, Reactome, PubTator), PubMed search and an LLM-written report. No MAF
   handling or cohort matrices.

The two projects do not import each other. They exchange plain data: a gene
list, one symbol per line.

From a mutation table to an enrichment report
---------------------------------------------

.. code-block:: python

   import pymaftools as pm

   maf = pm.load_example_maf()
   table = maf.filter_maf(pm.MAF.nonsynonymous_types).to_gene_table()

   frequency = table.calculate_feature_frequency()
   genes = frequency[frequency > 0].sort_values(ascending=False).index[:50]
   with open("genes.txt", "w") as handle:
       handle.write("\n".join(genes) + "\n")

.. code-block:: bash

   enrichrag analyze --file genes.txt --disease "lung cancer" --output result.json

.. warning::

   On the bundled example, the most frequently mutated genes include
   ``USH2A``, ``RYR2`` and ``TTN``: long genes collect passenger mutations.
   Enrichment of a raw "most mutated" list against an all-genes background
   is biased toward long genes and can report pathways that reflect gene
   length, not biology. Prefer a list from a group comparison
   (:meth:`~pymaftools.core.PivotTable.PivotTable.mutation_enrichment_test`)
   or a driver-gene list, and restrict the background to genes the assay
   covers. Passing that background to EnrichRAG is planned, not yet available.
