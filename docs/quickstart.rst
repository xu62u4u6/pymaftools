Five-Minute Quickstart
======================

From your own MAF file to a checked oncoplot. The whole flow is in
``examples/quickstart.py``; run it on the bundled example or your file:

.. code-block:: bash

   python examples/quickstart.py                       # bundled 6-sample example
   python examples/quickstart.py cohort.maf.gz         # your file
   python examples/quickstart.py cohort.maf.gz --samples samples.txt

1. Read and check
-----------------

.. code-block:: python

   from pymaftools import MAF

   maf = MAF.read_maf("cohort.maf.gz")
   print(maf.ingest_report)

``read_maf`` accepts GDC, cBioPortal, vcf2maf, Funcotator/Oncotator and
maftools-style files (``#`` comment headers, ``.gz``, tab or comma). The
report says what it detected and changed, for example::

   MAF ingest report: cohort.maf.gz
     source guess : gdc (tab-delimited, 3 comment line(s) skipped)
     events       : 9,412 rows, 512 samples
     genome build : GRCh38
     no changes made, no problems found

Problems that would silently change results stop the read with an explanation
(``MAFValidationError``):

.. list-table::
   :header-rows: 1

   * - Problem
     - Why it matters
     - What to do
   * - blank ``Hugo_Symbol``
     - the event cannot be placed in a gene row
     - fill (e.g. ``Unknown``) or drop those rows
   * - duplicate events
     - each copy becomes a false ``Multi_Hit``
     - ``read_maf(..., deduplicate=True)``
   * - mixed ``NCBI_Build`` or ``chr1``/``1``
     - two coordinate systems in one table
     - lift over / harmonize first
   * - alternate allele in ``Tumor_Seq_Allele1``
     - Ti/Tv and spectra read ``Tumor_Seq_Allele2``
     - fix the allele columns

``validate="warn"`` loads such a file anyway with one warning;
``validate="off"`` only records problems in the report. Changes that are safe
are made and noted instead: ``End_position`` is renamed to ``End_Position``, a
missing ``Tumor_Seq_Allele1`` is copied from ``Reference_Allele``, and sample
barcodes, chromosomes and gene symbols are kept as text (``"001"`` stays
``"001"``).

2. Build the gene table over every sequenced sample
---------------------------------------------------

.. code-block:: python

   samples = open("samples.txt").read().split()   # all sequenced samples
   events = maf.filter_maf(MAF.nonsynonymous_types)
   table = events.to_gene_table(samples=samples).add_freq()

A MAF only lists samples that have events. Samples with no protein-altering
mutation still belong in every frequency's denominator, so pass the full list
(``samples=`` or a :class:`~pymaftools.core.SampleManifest.SampleManifest`).
Without one, at least use ``maf["sample_ID"].unique()`` from the *unfiltered*
MAF, as the quickstart script does. When clinical metadata is attached with
``add_sample_metadata``, IDs that do not match raise (none match) or warn
(some match): mixing TCGA aliquot, sample and patient barcodes is the usual
cause.

3. Plot
-------

.. code-block:: python

   top = table.sort_features(by="freq").sort_samples_by_mutations()
   top = top.subset(features=top.index[:20])
   (
       top.plot.oncoplot(figsize=(12, 7))
       .main()
       .add_bar("mutations_count", side="top")
       .add_freq(side="right")
       .render()
       .save("oncoplot.png", dpi=150)
   )

Next: :doc:`pivottable` for filtering and statistics, :doc:`oncoplot` for
tracks and grouping.
