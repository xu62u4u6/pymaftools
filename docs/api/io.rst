I/O Modules
===========

TCGA Builders
-------------

The TCGA builders in ``pymaftools.io.tcga`` are the canonical TCGA API. They
resolve downloaded GDC files to cases, select the requested sample type, and
return the appropriate table class with aligned metadata. By default matrices
use ``case_id`` columns for backwards compatibility. Pass
``sample_key="sample_id"`` when joining modalities at exact specimen level;
the selected GDC specimen barcode is then used for matrix columns and sample
metadata indices while case and file provenance remain available in metadata.
The standalone
functions in ``pymaftools.io.tcga_readers`` remain available as lower-level
compatibility utilities.

.. automodule:: pymaftools.io.tcga
   :members:
   :show-inheritance:

VCF Parsing
-----------

.. automodule:: pymaftools.io.vcf.parsers
   :members:
   :exclude-members: VCFRecord

.. automodule:: pymaftools.io.vcf.record
   :members:
   :exclude-members: chrom, pos, ref, alt, filter, caller, tumor_dp, tumor_ad, tumor_af, normal_dp, normal_ad
