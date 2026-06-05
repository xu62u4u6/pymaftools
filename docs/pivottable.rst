PivotTable
==========

``PivotTable`` is the core data structure in pymaftools: a feature × sample
matrix with two aligned metadata frames that travel with it through every
operation.

.. code-block:: text

   PivotTable
   ├── (the matrix)        features (rows) × samples (columns)
   ├── feature_metadata    one row per feature  (freq, pathway, exon_size, ...)
   └── sample_metadata     one row per sample   (subtype, TMB, sex, ...)

Subsetting, sorting and frequency calculations all keep the two metadata frames
in sync with the matrix, so you never have to re-align them by hand.

From a MAF
----------

A mutation ``PivotTable`` is produced from a :class:`~pymaftools.core.MAF.MAF`.
There are two granularities:

.. code-block:: python

   from pymaftools import load_example_maf

   maf = load_example_maf("multisample")

   table = maf.to_gene_table()       # gene × sample   (one row per gene)
   muts  = maf.to_mutation_table()   # mutation × sample (one row per variant)

``to_gene_table`` is what you want for an oncoplot. The returned object is a
``SmallVariationTable`` (a ``PivotTable`` subclass), so every method below
applies.

.. note::

   **Wild-type cells are the boolean ``False``; mutated cells hold the
   ``Variant_Classification`` string.** The matrix is therefore an ``object``
   dtype mixing ``False`` and strings like ``"Missense_Mutation"``. A
   ``PivotTable`` *is* a ``pandas.DataFrame`` (the matrix itself), so to get a
   plain mutated/not-mutated boolean mask, compare it directly against
   ``False``::

       mutated = table != False        # boolean feature × sample mask

   ``to_pivot_table`` is a backward-compatible alias for ``to_gene_table``;
   prefer ``to_gene_table`` in new code — the name states the granularity.

A typical prep chain turns the raw matrix into something ready to plot:

.. code-block:: python

   table = (
       maf.to_gene_table()
       .add_freq()                                  # feature_metadata["freq"]
       .calculate_tmb(default_capture_size=40)      # sample_metadata["TMB"]
       .sort_features(by="freq", ascending=False)   # high-freq genes on top
       .sort_samples_by_mutations()                 # waterfall the columns
   )

.. note::

   ``calculate_tmb`` returns a **new** table rather than mutating in place, so
   capture the return value (``table = table.calculate_tmb(...)``) or the TMB
   column will not appear. The same is true of ``add_freq`` / ``sort_*`` — they
   all return a new table, so keep them in one chain as above.

Inspecting the structure
-------------------------

.. code-block:: python

   table.shape                       # (n_features, n_samples)
   table.feature_metadata.head()     # per-gene annotations (freq, ...)
   table.sample_metadata.head()      # per-sample annotations (TMB, subtype, ...)
   table.sample_metadata["mutations_count"]   # added by to_gene_table

Subsetting
----------

``PivotTable.subset()`` filters by features (rows) and/or samples (columns),
with both metadata frames kept in sync automatically.

.. note::

   The examples in this and the next two sections are **illustrative**: the
   bundled ``multisample`` MAF carries only ``mutations_count`` in
   ``sample_metadata`` (no ``subtype``/``sex``/``age``), and its genes are not
   ``TP53``/``KRAS``/``EGFR`` (run ``table.feature_metadata.index[:10]`` to see
   the real ones). Substitute names/columns that exist in *your* data. For a
   runnable cohort that already has ``subtype`` (plus ``LUAD_freq``/``LUSC_freq``),
   load the bundled HDF5 fixture instead::

       from pymaftools import read_h5
       table = read_h5("pymaftools/data/example_tcga_lung_mutation_grouped.h5")

.. code-block:: python

   # By feature names (use names present in your table)
   subset = table.subset(features=["TP53", "KRAS", "EGFR"])

   # By boolean mask — samples of a specific subtype
   luad = table.subset(samples=table.sample_metadata["subtype"] == "LUAD")

   # Combine both — recurrent genes within one subtype
   result = table.subset(
       features=table.feature_metadata["freq"] > 0.1,
       samples=table.sample_metadata["subtype"] == "LUSC",
   )

.. tip::

   Masks are ordinary pandas boolean Series indexed by feature/sample name, so
   any pandas expression works (``.isin([...])``, ``&``, ``|``, ``~``). Because
   ``subset`` re-slices both metadata frames, the result is a self-consistent
   ``PivotTable`` you can hand straight to the next step.

Mutation frequency
------------------

``add_freq`` writes a ``freq`` column into ``feature_metadata`` (fraction of
samples mutated). Pass ``group_col=`` to additionally get one column per group:

.. code-block:: python

   # Overall freq only
   table = table.add_freq()

   # Per-subtype columns straight from a sample_metadata column:
   # adds LUAD_freq, LUSC_freq, ... alongside the overall freq
   table = table.add_freq(group_col="subtype")

   # Equivalent explicit form — pass the per-group subsets yourself
   table = table.add_freq(
       groups={
           "LUAD": table.subset(samples=table.sample_metadata.subtype == "LUAD"),
           "LUSC": table.subset(samples=table.sample_metadata.subtype == "LUSC"),
       }
   )

These per-group columns are what the grouped oncoplot draws as per-section
frequency strips (see :doc:`oncoplot`).

Sorting
-------

.. code-block:: python

   # Rows: by a feature_metadata column (single or multiple keys)
   table = table.sort_features(by="freq", ascending=False)
   table = table.sort_features(by=["size_group", "log2OR"], ascending=[True, False])

   # Columns: waterfall by mutation pattern (classic oncoplot look)
   table = table.sort_samples_by_mutations(top=10)

   # Columns: keep groups contiguous, waterfall within each group
   table = table.sort_samples_by_group(
       group_col="subtype", group_order=["LUAD", "LUSC"], top=10
   )

.. note::

   The two axes sort differently *on purpose*, which is why there is no generic
   ``sort_samples(by=...)`` to mirror ``sort_features(by=...)``. A **feature** is
   sorted by a single scalar per gene (a ``feature_metadata`` column such as
   ``freq``). A **sample**, in an oncoplot, is sorted by its mutation *pattern
   across genes* — the waterfall staircase — which is not a single column value,
   so it gets the dedicated ``sort_samples_by_mutations`` /
   ``sort_samples_by_group`` methods instead.

.. note::

   Grouping in an oncoplot only draws clean sections if the rows/columns are
   already **contiguous** by the grouping key. Always ``sort_features`` /
   ``sort_samples_by_group`` *before* ``group_features`` / ``group_samples``.

Multi-omics integration
------------------------

Stack several omics layers that share samples into a ``Cohort``:

.. code-block:: python

   from pymaftools import Cohort

   cohort = Cohort("my_cohort")
   cohort.add_table(mutation_table, "mutation")   # PivotTable first, name second
   cohort.add_table(expression_table, "expression")
   cohort.add_table(cnv_table, "cnv")

   cohort = cohort.subset(samples=shared_samples)  # align to shared samples

For full method signatures, see the :doc:`API Reference <api/core>`.
