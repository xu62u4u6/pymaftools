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
       maf.to_pivot_table()
       .add_freq()
       .calculate_TMB(default_capture_size=40)
       .sort_features(by="freq")
       .sort_samples_by_mutations()
   )

   # Compose the oncoplot from tracks, then render once
   op = (
       table.plot.oncoplot(figsize=(12, 8))
       .main()                        # mutation matrix
       .add_bar("TMB", side="top")
       .add_freq(side="right")
       .render()
   )
   op.save("oncoplot.png", dpi=300)

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
value used by ``to_pivot_table``. (``Variant_Type`` and ``Protein_position``
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

.. code-block:: python

   table = maf.to_pivot_table()          # provides `mutations_count`, not TMB
   table = table.calculate_TMB(default_capture_size=40)  # TMB = count / size (Mb)
   table.sample_metadata["TMB"]

.. note::

   ``to_pivot_table`` does not compute TMB. ``calculate_TMB`` returns a **new**
   table rather than modifying in place, so capture the return value
   (``table = table.calculate_TMB(...)``) or the TMB column will not appear.

Subsetting Data
---------------

``PivotTable.subset()`` lets you filter by features (rows) and samples (columns),
with metadata automatically kept in sync.

.. code-block:: python

   # By feature names
   subset = table.subset(features=["TP53", "KRAS", "EGFR"])

   # By boolean mask — select samples of a specific subtype
   luad = table.subset(samples=table.sample_metadata["subtype"] == "LUAD")

   # Combine both — specific genes in specific samples
   result = table.subset(
       features=table.feature_metadata["freq"] > 0.1,
       samples=table.sample_metadata["subtype"] == "LUSC",
   )

   # Use with add_freq to compute group-wise mutation frequencies
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
   cohort = Cohort()
   cohort.add_table("mutation", mutation_table)
   cohort.add_table("expression", expression_table)
   cohort.add_table("cnv", cnv_table)

   # Subset to shared samples
   cohort = cohort.subset(samples=shared_samples)

For full API details, see the :doc:`API Reference <api/core>`.
