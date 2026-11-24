Machine-readable results
========================

Agents and pipelines cannot look at a figure or read a notebook, so every
standard result can be serialised to strict JSON together with *where it came
from*. None of this changes a computed number: statistics functions still
return plain :class:`pandas.DataFrame` objects, and the extra record rides in
``df.attrs``.

Serialise any result
--------------------

.. code-block:: python

   import pymaftools as pm

   maf = pm.load_example_maf()
   table = maf.filter_maf(pm.MAF.nonsynonymous_types).to_gene_table()
   stats = table.somatic_interactions(top=10)

   pm.to_json(stats, "interactions.json")   # {"provenance", "n_rows", "columns", "records"}
   pm.get_provenance(stats)["parameters"]   # {"top": 10, "alpha": 0.05}
   maf.ingest_report.to_json("ingest.json") # includes the input file's sha256

``NaN`` becomes ``null``; ``+inf``/``-inf`` become the strings ``"inf"`` /
``"-inf"`` (infinite odds ratios are kept, never dropped). The output never
contains the non-standard ``NaN``/``Infinity`` literals.

Serialisable results: ``MAF.ingest_report`` (``IngestReport.to_dict`` /
``to_json``), ``TMBAudit.to_dict(include_events=False)`` / ``to_json``, and the
DataFrames from ``somatic_interactions()`` and
``PivotTable.mutation_enrichment_test()``.

Provenance
----------

A provenance record holds ``analysis``, ``parameters``, ``inputs`` (path, size
and SHA-256 of the source MAF, carried from ``read_maf`` through
``to_gene_table`` and later table operations), ``sample_universe`` (count,
sorted-ID hash, IDs, and whether it came from a ``SampleManifest``),
``pymaftools_version`` and ``created_at`` (UTC).

``pymaftools.session_info()`` returns the package, Python, platform and
dependency versions for a methods section or a run log.

Figure summaries
----------------

``plot.save(path, summary=True)`` also writes ``path`` with a ``.json``
extension describing what was drawn: per-axes artist counts, an ``empty`` flag
and ``warnings``. ``OncoPlot`` adds the matrix shape, fill fraction, top
features, auto-hidden sample labels and the estimated cell size in pixels, so a
blank or unreadable heatmap is detectable without viewing the PNG. The sidecar
is opt-in so existing scripts produce exactly the same files.

``plot.summary()`` returns the same dict, and
``pymaftools.figure_summary(fig)`` works on any matplotlib figure (for example
the one returned by ``table.plot.somatic_interactions()``).

JSON schema version
-------------------

Every JSON record pymaftools writes -- provenance records, ``IngestReport``,
``TMBAudit``, figure-summary sidecars and CLI output -- carries
``"schema_version"`` (currently ``"1.0"``, see
``pymaftools.provenance.SCHEMA_VERSION``). The minor part grows when fields are
only added; the major part changes when a field is renamed, removed or changes
meaning, so a consumer can refuse output it does not understand. CLI records
also start with ``"pymaftools_version"``.

Command line
------------

.. code-block:: console

   pymaftools inspect cohort.maf --json
   pymaftools oncoplot cohort.maf -o oncoplot.png --top 20     # also writes oncoplot.json
   pymaftools interactions cohort.maf --top 25 --json -o stats.csv
   pymaftools session-info

Declare the full sample universe with ``--samples clinical_samples.txt`` (one ID
per line, or a table such as cBioPortal's ``data_clinical_sample.txt`` whose
``SAMPLE_ID`` column is detected; ``--sample-col`` picks another column).
Without it the universe is only the samples that have events in the MAF, so
sequenced samples without mutations are missing from frequencies and test
denominators; every command warns about this.

``python -m pymaftools`` is equivalent. With ``--json`` stdout is a single JSON
document (library messages go to stderr) and Python warnings are returned in
``python_warnings``. Exit codes: ``0`` success, ``1`` error (a JSON ``error``
object with ``--json``), ``2`` input rejected by ``--validate strict``.
``pymaftools --version`` prints the installed version.

``inspect`` defaults to ``--validate warn`` so it can report problems;
``oncoplot`` and ``interactions`` default to ``strict``, and plot or test only
nonsynonymous classes unless ``--all-classes`` is given.
