Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install pymaftools

Quick Start
-----------

Read a MAF file and create an OncoPlot. A small multi-sample MAF is bundled with
the package, so this runs as-is after ``pip install``:

.. code-block:: python

   from pymaftools import load_example_maf

   # Bundled example (use MAF.read_maf("your.maf") for your own data)
   maf = load_example_maf("multisample")

   # Build + prepare the mutation table
   table = (
       maf.to_gene_table()
       .add_freq()
       .sort_features(by="freq")
       .sort_samples_by_mutations()
   )
   plot_table = table.subset(features=table.index[:20])

   # Compose the oncoplot from tracks, then render once
   op = (
       plot_table.plot.oncoplot(figsize=(12, 8))
       .main()                        # mutation matrix
       .add_bar("mutations_count", side="top")
       .add_freq(side="right")
       .render()
   )
   op.save("oncoplot.png", dpi=300)

``plot_table`` limits only the visualization to the 20 most frequently mutated
genes; ``table`` still contains the full gene-level result.

Saving Results
--------------

Use HDF5 for both individual tables and multi-omics cohorts. It preserves data
and metadata without SQLite's practical limit of roughly 2,000 columns.

.. code-block:: python

   from pymaftools import Cohort, PivotTable

   table.to_h5("mutation_table.h5")
   table = PivotTable.read_h5("mutation_table.h5")

   cohort.to_hdf5("cohort.h5")
   cohort = Cohort.read_hdf5("cohort.h5")

.. note::

   ``to_sqlite`` and ``read_sqlite`` are deprecated as of 0.5.0. They remain
   available for existing files during the deprecation period, but should not
   be used for new high-dimensional datasets.

Reading MAF Files
-----------------

``MAF.read_maf`` reads a tab-separated MAF file. Leading comment lines
(``#``-prefixed, e.g. the GDC ``#version 2.4`` header) are detected and
skipped automatically, so files with zero, one, or many comment lines all
work.

**Required columns.** These must be present:

- ``Hugo_Symbol``
- ``Start_Position``
- ``End_Position``
- ``Reference_Allele``
- ``Tumor_Seq_Allele1``
- ``Tumor_Seq_Allele2``
- ``Variant_Classification``

The first six build the per-mutation index; ``Variant_Classification`` is the
value used by ``to_gene_table``. (``Variant_Type`` and ``Protein_position``
are only needed for base-change / lollipop analyses.)

**Sample identity.** By default each row's sample comes from the
``Tumor_Sample_Barcode`` column, so a standard multi-sample MAF keeps its
samples distinct:

.. code-block:: python

   # Multi-sample MAF: samples taken from Tumor_Sample_Barcode
   maf = MAF.read_maf("cohort.maf")

   # Per-sample file: assign one sample_ID to every row (overrides the column)
   maf_a = MAF.read_maf("sample_A.maf", sample_ID="sample_A")
   maf_b = MAF.read_maf("sample_B.maf", sample_ID="sample_B")
   maf = MAF.merge_mafs([maf_a, maf_b])

.. note::

   If ``sample_ID`` is not given and ``Tumor_Sample_Barcode`` is absent,
   ``read_maf`` raises ``ValueError`` rather than silently mislabelling
   samples. Use ``sample_col`` to point at a differently named column.

Computing TMB
-------------

Scientific TMB requires an explicit eligible sample universe, an event-filter
policy, and sample-specific callable territory. A universal 40 Mb denominator
is not assumed:

.. code-block:: python

   import pandas as pd
   from pymaftools import MAF, SampleManifest

   manifest = SampleManifest(
       pd.DataFrame(
           {
               "patient_id": ["patient-1", "patient-2"],
               "eligible": [True, True],
               "callable_mb": [38.7, 41.2],
           },
           index=["sample-1", "sample-2"],
       )
   )
   manifest.assert_independent("patient")
   audit = maf.calculate_tmb_audit(
       manifest,
       variant_classifications=MAF.nonsynonymous_types,
   )
   audit.summary[["mutation_count", "callable_mb", "TMB"]]
   audit.events[["included", "exclusion_reason"]]

``callable_mb`` must come from the callable territory for each analyzed sample
under the same genome build and assay policy. Pass ``pass_col``,
``somatic_col``, and ``genome_build_col`` when those event fields are available.

.. warning::

   ``PivotTable.calculate_tmb(default_capture_size=40)`` remains temporarily for
   backward compatibility. It is a legacy normalized count without an event
   audit and should not be reported as scientific TMB.

Subsetting Data
---------------

``PivotTable.subset()`` lets you filter by features (rows) and samples (columns),
with metadata automatically kept in sync.

.. note::

   These snippets are illustrative — the bundled ``multisample`` MAF has no
   ``subtype`` column and its genes are not ``TP53``/``KRAS``/``EGFR``. Use
   names/columns from your own data, or load the bundled HDF5 fixture
   (``load_example_table()``), which
   has ``subtype``. See :doc:`pivottable` for details.

.. code-block:: python

   # By feature names (use names present in your table)
   subset = table.subset(features=["TP53", "KRAS", "EGFR"])

   # By boolean mask — select samples of a specific subtype
   luad = table.subset(samples=table.sample_metadata["subtype"] == "LUAD")

   # Combine both — specific genes in specific samples
   result = table.subset(
       features=table.feature_metadata["freq"] > 0.1,
       samples=table.sample_metadata["subtype"] == "LUSC",
   )

   # Compute group-wise mutation frequencies straight from a sample_metadata
   # column (adds LUAD_freq, LUSC_freq, ... plus the overall freq):
   table = table.add_freq(group_col="subtype")

   # Equivalent explicit form (still supported) — pass per-group subsets yourself:
   table = table.add_freq(
       groups={
           "LUAD": table.subset(samples=table.sample_metadata.subtype == "LUAD"),
           "LUSC": table.subset(samples=table.sample_metadata.subtype == "LUSC"),
       }
   )

Multi-omics Integration
-----------------------

.. code-block:: python

   from pymaftools import PivotTable, Cohort

   # Build a cohort from multiple omics layers
   # add_table(table, table_name) — the PivotTable comes first
   cohort = Cohort("my_cohort")
   cohort.add_table(mutation_table, "mutation")
   cohort.add_table(expression_table, "expression")
   cohort.add_table(cnv_table, "cnv")

   # Subset to shared samples
   cohort = cohort.subset(samples=shared_samples)

For full API details, see the :doc:`API Reference <api/core>`.
