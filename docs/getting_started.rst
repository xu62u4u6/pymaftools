Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install pymaftools

Quick Start
-----------

Read a MAF file and create an OncoPlot:

.. code-block:: python

   from pymaftools import MAF, OncoPlot

   # Read MAF file
   maf = MAF.read_maf("data/tcga_paad.maf")

   # Create mutation table
   table = maf.to_pivot_table()

   # Plot top 20 mutated genes
   (
       OncoPlot(table)
       .set_config(figsize=(12, 8))
       .oncoplot()
       .add_barplot()
       .add_legend()
   )

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
