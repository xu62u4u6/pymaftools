# Bundled Data Sources

The project license covers pymaftools source code. Bundled third-party and
derived datasets remain subject to their upstream data-use, attribution, and
redistribution terms. Review those terms before redistributing a wheel or
source archive.

| File | Source and transformation | SHA-256 |
| --- | --- | --- |
| `cytoBand.txt` | UCSC Genome Browser hg38 cytoband table, stored as the four-column interval/name/stain format consumed by the TCGA CNV readers. | `ce1b6033a5243e7c5022660b952d2ec33243e307e909afcaeec1894641a5208f` |
| `ensembl_gene_map.tsv` | Ensembl BioMart human gene coordinate export. The schema records Ensembl gene ID, HGNC symbol, chromosome, start, end, and biotype. | `b318ee074d723faa6efe498f69e8842875239819f5cf3d008144575e06736faf` |
| `ensembl_gene_sizes.tsv` | Small Ensembl canonical-transcript size cache produced by `scripts/download_gene_sizes.py` or its REST fallback. | `2da9ae5eec7881cb458e4c6b7a6b82832f6d6981e7cb83a8aecf6b02c0d9997e` |
| `protein_domains.csv` | Tabular conversion of the `protein_domains.RDs` data distributed by the Bioconductor `maftools` project. | `03742f68e3fcda9f719bb6963ea5a84e656d586d52c34600f460a6d9e1e96335` |
| `example_multisample.maf` | Derived subset of public, open-access TCGA-LUAD/LUSC masked somatic mutation data from the NCI Genomic Data Commons. | `60945b00d5313e685ea2c7a0f3b86ce20121824956715ddd6067a6b3f1ba25f1` |
| `example_single_sample.maf` | Derived single-aliquot subset of the same GDC mutation source. | `850ff09e13ab05fbb9afcc76aa22866e69b636d88e4e150fd36a85f6b59c0165` |
| `example_tcga_lung_mutation_grouped.h5` | Derived TCGA lung mutation table used by documentation and plot tests. Built from GDC mutation data and package metadata operations. | `3518d7c63b2306238ca325234975bc34648a703f87453499e62803ac00ae7fa5` |

## Upstream References

- NCI Genomic Data Commons: <https://portal.gdc.cancer.gov/>
- GDC data-use policies: <https://gdc.cancer.gov/access-data/data-access-processes-and-tools>
- Ensembl BioMart: <https://www.ensembl.org/biomart/martview>
- UCSC hg38 database downloads: <https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/>
- Bioconductor maftools: <https://bioconductor.org/packages/maftools>

## Reproducibility Limitation

The existing example MAF and HDF5 files predate a checked-in GDC manifest, so
their exact file UUID/accession set and transformation command cannot currently
be reconstructed from the repository alone. `scripts/download_demo_samples.py`
and `scripts/build_cohort.py` document the current reproducible workflow, but
they must not be presented as the exact lineage of these older fixtures. A
future fixture refresh should commit the GDC manifest, case list, generation
command, upstream release/build identifiers, and updated checksums together.
