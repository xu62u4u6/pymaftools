"""TCGA-specific data builders and GDC client."""

from .base import TCGATableBuilder
from .client import GDCClient, parse_tcga_barcode
from .clinical import TCGAClinicalBuilder
from .cnv_gene import TCGACNVGeneBuilder
from .cnv_segment import TCGACNVSegmentBuilder
from .expression import TCGAExpressionBuilder
from .mapping import load_file_mapping, resolve_files
from .methylation import TCGAMethylationBuilder
from .mutation import TCGAMutationBuilder

__all__ = [
    "GDCClient",
    "parse_tcga_barcode",
    "TCGATableBuilder",
    "TCGAExpressionBuilder",
    "TCGAMutationBuilder",
    "TCGACNVSegmentBuilder",
    "TCGACNVGeneBuilder",
    "TCGAMethylationBuilder",
    "TCGAClinicalBuilder",
    "load_file_mapping",
    "resolve_files",
]
