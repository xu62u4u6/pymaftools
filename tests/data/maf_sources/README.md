# Producer-style MAF fixtures

Small inputs used by `tests/core/test_maf_ingest.py` to check that `MAF.read_maf`
handles MAFs from different producers. Rebuild with
`python scripts/build_maf_source_fixtures.py`.

| File | Origin | What it exercises |
| --- | --- | --- |
| `cbioportal_acc_tcga_pan_can_atlas_2018.maf` | 114-row slice (all TP53/CTNNB1 rows + 80 random) of cBioPortal datahub `acc_tcga_pan_can_atlas_2018/data_mutations.txt` (TCGA open-access), 24 columns kept | no comment lines, 15-character sample barcodes, bare `Protein_position` (`74`), `Splice_Region` |
| `gdc_style_grch38.maf.gz` | the cBioPortal slice rewritten in GDC layout | `#version gdc-1.0.0` comment block, gzip, `chr` prefix, `NCBI_Build` GRCh38 |
| `funcotator_style.maf` | the cBioPortal slice rewritten in Funcotator layout | `#`/`##` header block, `Protein_Change` instead of HGVSp/Protein_position, Oncotator classifications |
| `maftools_tcga_laml_head.maf.gz` | first 80 rows of maftools `inst/extdata/tcga_laml.maf.gz` (MIT) | legacy TCGA columns, `End_position` casing, `Protein_Change` |
| `maftools_brca_head.maf` | first 60 rows of maftools `inst/extdata/brca.maf.gz` (MIT) | 9-column minimal MAF without `Tumor_Seq_Allele1` |

GDC- and Funcotator-style files are derived, not downloaded: the GDC API and
Funcotator test data were not reachable when these were built. The coordinates
in the GDC-style file are the original GRCh37 values relabelled, so do not use
it for anything coordinate-specific.
