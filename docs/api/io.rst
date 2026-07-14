I/O Modules
===========

TCGA Builders
-------------

The case-level builders in ``pymaftools.io.tcga`` are the canonical TCGA API.
They resolve downloaded GDC files to cases, select the requested sample type,
and return the appropriate table class with aligned metadata. The standalone
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
