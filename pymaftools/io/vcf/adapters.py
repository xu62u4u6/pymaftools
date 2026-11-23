from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SampleMetrics:
    """Harmonized per-sample depth and allele-fraction metrics."""

    dp: int
    ref_count: int
    alt_count: int
    af: float


def safe_af(alt_count: int, total_depth: int) -> float:
    """Derive alternate allele fraction from counts with zero-depth validation.

    Parameters
    ----------
    alt_count : int
        Alternate allele read depth.
    total_depth : int
        Total read depth (or ref + alt depth for allele-count callers).

    Returns
    -------
    float
        Alternate allele fraction in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If ``total_depth`` is zero while ``alt_count`` is positive, or if
        ``alt_count`` exceeds ``total_depth``.
    """
    if total_depth == 0:
        if alt_count == 0:
            return 0.0
        raise ValueError("Unable to derive AF when DP is zero and AD is positive.")
    if alt_count > total_depth:
        return 1.0
    return alt_count / total_depth


def parse_format_mapping(format_str: str, value_str: str) -> dict[str, str]:
    """Zip a VCF ``FORMAT`` specification with a sample's colon-delimited values."""
    keys = format_str.split(":")
    values = value_str.split(":")
    if len(keys) != len(values):
        raise ValueError(
            f"Failed to parse FORMAT '{format_str}' with value '{value_str}': "
            "field counts differ."
        )
    return dict(zip(keys, values))


def parse_info_mapping(info_str: str) -> dict[str, str]:
    """Parse a semicolon-delimited VCF ``INFO`` field into key/value pairs."""
    if not info_str or info_str == ".":
        return {}
    parsed: dict[str, str] = {}
    for token in info_str.split(";"):
        if not token:
            continue
        if "=" in token:
            key, val = token.split("=", 1)
            parsed[key] = val
        else:
            parsed[token] = "true"
    return parsed


def _parse_int_field(raw: str, field_name: str) -> int:
    if raw in {".", ""}:
        raise ValueError(f"Missing integer value for '{field_name}'.")
    return int(raw)


def _parse_float_field(raw: str, field_name: str) -> float:
    if raw in {".", ""}:
        raise ValueError(f"Missing float value for '{field_name}'.")
    return float(raw)


class BaseCallerAdapter(ABC):
    """Base interface for caller-specific VCF parsing adapters.

    Subclasses encapsulate caller-specific header signatures, sample-column
    resolution, ``FORMAT``/``INFO`` metric extraction, and supported variant
    categories while emitting one harmonized :class:`VCFRecord` schema.
    """

    name: str
    """Canonical lowercase caller name (e.g. ``"mutect2"``)."""

    aliases: tuple[str, ...] = ()
    """Alternative lowercase names accepted by :func:`get_caller_adapter`."""

    supported_variant_types: frozenset[str] = frozenset({"SNV", "INDEL"})
    """Variant categories this caller can emit (subset of ``{"SNV", "INDEL"}``)."""

    default_accepted_filters: frozenset[str] = frozenset({"PASS", "."})
    """Default ``FILTER`` values retained when ``accepted_filters`` is omitted."""

    site_only: bool = False
    """``True`` when the caller writes 8-column site-only VCFs (no sample columns)."""

    def detect(self, header_lines: Sequence[str]) -> bool:
        """Return ``True`` if *header_lines* uniquely identify this caller."""
        return False

    def accept_filter(
        self,
        filter_value: str,
        accepted_filters: frozenset[str] | None = None,
    ) -> bool:
        """Return ``True`` when a record's ``FILTER`` value should be kept."""
        allowed = (
            self.default_accepted_filters
            if accepted_filters is None
            else accepted_filters
        )
        return filter_value in allowed

    def resolve_samples(
        self,
        meta_tumor: str | None,
        meta_normal: str | None,
        sample_columns: Sequence[str],
        *,
        tumor_sample: str | None = None,
        normal_sample: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Resolve ``(tumor_column, normal_column)`` from VCF header metadata."""
        cols = list(sample_columns)
        if self.site_only and not cols:
            if tumor_sample is not None or normal_sample is not None:
                raise ValueError(
                    f"Caller '{self.name}' produced a site-only VCF with no sample "
                    "columns, so tumor_sample/normal_sample cannot be selected."
                )
            return None, None

        if tumor_sample is not None:
            if tumor_sample not in cols:
                raise ValueError(
                    f"Requested tumor_sample '{tumor_sample}' not found in VCF "
                    f"sample columns: {cols}."
                )
            resolved_normal = normal_sample
            if resolved_normal is not None and resolved_normal not in cols:
                raise ValueError(
                    f"Requested normal_sample '{resolved_normal}' not found in VCF "
                    f"sample columns: {cols}."
                )
            if resolved_normal is None and len(cols) == 2:
                other = [c for c in cols if c != tumor_sample][0]
                if (
                    other == meta_normal
                    or other.upper() == "NORMAL"
                    or other.endswith("N")
                ):
                    resolved_normal = other
            return tumor_sample, resolved_normal

        resolved_tumor: str | None = None
        resolved_normal: str | None = normal_sample
        if resolved_normal is not None and resolved_normal not in cols:
            raise ValueError(
                f"Requested normal_sample '{resolved_normal}' not found in VCF "
                f"sample columns: {cols}."
            )

        # Priority 1: explicit header metadata (e.g. Mutect2 ##tumor_sample=...)
        if meta_tumor is not None:
            for col in cols:
                if col == meta_tumor:
                    resolved_tumor = col
                elif resolved_normal is None and col == meta_normal:
                    resolved_normal = col

        # Priority 2: literal TUMOR / NORMAL column names
        if resolved_tumor is None:
            for col in cols:
                if col.upper() == "TUMOR":
                    resolved_tumor = col
                elif resolved_normal is None and col.upper() == "NORMAL":
                    resolved_normal = col

        # Priority 3: suffix T / N (Pindel / custom barcodes)
        if resolved_tumor is None:
            for col in cols:
                if col.endswith("T"):
                    resolved_tumor = col
                elif resolved_normal is None and col.endswith("N"):
                    resolved_normal = col

        # Priority 4: single-sample tumor-only VCF (e.g. DeepSomatic, tumor-only Mutect2)
        if resolved_tumor is None and len(cols) == 1:
            only_col = cols[0]
            if only_col.upper() != "NORMAL":
                resolved_tumor = only_col

        if resolved_tumor is None:
            raise ValueError(
                f"Unable to identify tumor sample column from: {cols}. "
                "Expected ##tumor_sample= header, a 'TUMOR' column, a sample ID "
                "ending in 'T', or pass tumor_sample=... explicitly to VCF.read_vcf()."
            )

        return resolved_tumor, resolved_normal

    @abstractmethod
    def parse_record_metrics(
        self,
        *,
        ref: str,
        alt: str,
        format_str: str | None,
        sample_value: str | None,
        info_str: str,
        alt_index: int = 0,
    ) -> SampleMetrics:
        """Extract harmonized ``(dp, ref_count, alt_count, af)`` for one sample."""


def _parse_standard_ad_format(
    format_str: str,
    value_str: str,
    *,
    alt_index: int = 0,
    af_keys: tuple[str, ...] = ("AF", "VAF"),
) -> SampleMetrics:
    """Parse callers encoding ``AD`` as ``Number=R`` (``ref,alt1,...``)."""
    parsed = parse_format_mapping(format_str, value_str)
    try:
        if "AD" not in parsed:
            raise KeyError("AD")
        ad_values = [_parse_int_field(x, "AD") for x in parsed["AD"].split(",")]
        ref_count = ad_values[0]
        alt_count = ad_values[alt_index + 1]
        if "DP" in parsed and parsed["DP"] not in {".", ""}:
            dp = _parse_int_field(parsed["DP"], "DP")
        else:
            dp = sum(ad_values)

        af: float | None = None
        for af_key in af_keys:
            if af_key in parsed and parsed[af_key] not in {".", ""}:
                af_parts = parsed[af_key].split(",")
                af = _parse_float_field(af_parts[alt_index], af_key)
                break
        if af is None:
            af = safe_af(alt_count, dp)
        return SampleMetrics(dp=dp, ref_count=ref_count, alt_count=alt_count, af=af)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Failed to parse FORMAT '{format_str}' with value '{value_str}'."
        ) from exc


class Mutect2Adapter(BaseCallerAdapter):
    """Adapter for GATK4 Mutect2 somatic VCFs (paired or tumor-only)."""

    name = "mutect2"
    aliases = ("mutect", "gatk", "gatk_mutect2")
    supported_variant_types = frozenset({"SNV", "INDEL"})

    def detect(self, header_lines: Sequence[str]) -> bool:
        joined = "\n".join(header_lines).lower()
        return "mutect" in joined

    def parse_record_metrics(
        self,
        *,
        ref: str,
        alt: str,
        format_str: str | None,
        sample_value: str | None,
        info_str: str,
        alt_index: int = 0,
    ) -> SampleMetrics:
        if not format_str or sample_value is None:
            raise ValueError("Mutect2 records require FORMAT and sample values.")
        return _parse_standard_ad_format(
            format_str, sample_value, alt_index=alt_index, af_keys=("AF",)
        )


class MuseAdapter(BaseCallerAdapter):
    """Adapter for MuSE somatic SNV VCFs (SNV-only caller)."""

    name = "muse"
    aliases = ()
    supported_variant_types = frozenset({"SNV"})

    def detect(self, header_lines: Sequence[str]) -> bool:
        joined = "\n".join(header_lines).lower()
        return "muse" in joined or "##filter=<id=tier1," in joined

    def parse_record_metrics(
        self,
        *,
        ref: str,
        alt: str,
        format_str: str | None,
        sample_value: str | None,
        info_str: str,
        alt_index: int = 0,
    ) -> SampleMetrics:
        if not format_str or sample_value is None:
            raise ValueError("MuSE records require FORMAT and sample values.")
        return _parse_standard_ad_format(
            format_str, sample_value, alt_index=alt_index, af_keys=("AF",)
        )


class VarScan2Adapter(BaseCallerAdapter):
    """Adapter for VarScan2 somatic SNP and INDEL VCFs."""

    name = "varscan2"
    aliases = ("varscan", "varscan_somatic")
    supported_variant_types = frozenset({"SNV", "INDEL"})

    def detect(self, header_lines: Sequence[str]) -> bool:
        joined = "\n".join(header_lines).lower()
        return "##source=varscan" in joined or (
            "##format=<id=rd," in joined and "##format=<id=freq," in joined
        )

    def parse_record_metrics(
        self,
        *,
        ref: str,
        alt: str,
        format_str: str | None,
        sample_value: str | None,
        info_str: str,
        alt_index: int = 0,
    ) -> SampleMetrics:
        if not format_str or sample_value is None:
            raise ValueError("VarScan2 records require FORMAT and sample values.")
        parsed = parse_format_mapping(format_str, sample_value)
        try:
            if "RD" in parsed and "AD" in parsed and "," not in parsed["AD"]:
                if alt_index != 0:
                    raise ValueError(
                        "VarScan2 FORMAT does not encode per-ALT depths; split "
                        "multiallelic sites with 'bcftools norm -m -any' first."
                    )
                ref_count = _parse_int_field(parsed["RD"], "RD")
                alt_count = _parse_int_field(parsed["AD"], "AD")
                dp = (
                    _parse_int_field(parsed["DP"], "DP")
                    if "DP" in parsed and parsed["DP"] not in {".", ""}
                    else ref_count + alt_count
                )
                if "FREQ" in parsed and parsed["FREQ"] not in {".", ""}:
                    af = float(parsed["FREQ"].rstrip("%")) / 100.0
                else:
                    af = safe_af(alt_count, dp)
                return SampleMetrics(
                    dp=dp, ref_count=ref_count, alt_count=alt_count, af=af
                )
            # Fallback if a harmonized/post-processed VarScan VCF uses Number=R AD
            return _parse_standard_ad_format(
                format_str, sample_value, alt_index=alt_index
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse VarScan2 FORMAT '{format_str}' with value "
                f"'{sample_value}'."
            ) from exc


class PindelAdapter(BaseCallerAdapter):
    """Adapter for Pindel somatic VCFs (``GT:AD``)."""

    name = "pindel"
    aliases = ()
    supported_variant_types = frozenset({"SNV", "INDEL"})

    def detect(self, header_lines: Sequence[str]) -> bool:
        joined = "\n".join(header_lines).lower()
        return "pindel" in joined

    def parse_record_metrics(
        self,
        *,
        ref: str,
        alt: str,
        format_str: str | None,
        sample_value: str | None,
        info_str: str,
        alt_index: int = 0,
    ) -> SampleMetrics:
        if not format_str or sample_value is None:
            raise ValueError("Pindel records require FORMAT and sample values.")
        return _parse_standard_ad_format(
            format_str, sample_value, alt_index=alt_index, af_keys=("AF",)
        )


class Strelka2Adapter(BaseCallerAdapter):
    """Adapter for Illumina Strelka2 somatic SNV (``AU/CU/GU/TU``) and INDEL (``TAR/TIR``) VCFs."""

    name = "strelka2"
    aliases = ("strelka", "strelka2_somatic")
    supported_variant_types = frozenset({"SNV", "INDEL"})

    def detect(self, header_lines: Sequence[str]) -> bool:
        joined = "\n".join(header_lines).lower()
        return (
            "##source=strelka" in joined
            or ("##format=<id=au," in joined and "##format=<id=tu," in joined)
            or ("##format=<id=tar," in joined and "##format=<id=tir," in joined)
        )

    def parse_record_metrics(
        self,
        *,
        ref: str,
        alt: str,
        format_str: str | None,
        sample_value: str | None,
        info_str: str,
        alt_index: int = 0,
    ) -> SampleMetrics:
        if not format_str or sample_value is None:
            raise ValueError("Strelka2 records require FORMAT and sample values.")
        parsed = parse_format_mapping(format_str, sample_value)
        try:
            ref_upper = ref.upper()
            alt_upper = alt.upper()
            ref_key = f"{ref_upper}U"
            alt_key = f"{alt_upper}U"

            # Strelka2 somatic SNVs use tier1 counts in AU/CU/GU/TU
            if (
                len(ref_upper) == 1
                and len(alt_upper) == 1
                and ref_key in parsed
                and alt_key in parsed
            ):
                ref_count = _parse_int_field(parsed[ref_key].split(",")[0], ref_key)
                alt_count = _parse_int_field(parsed[alt_key].split(",")[0], alt_key)
                allele_sum = ref_count + alt_count
                dp = (
                    _parse_int_field(parsed["DP"], "DP")
                    if "DP" in parsed and parsed["DP"] not in {".", ""}
                    else allele_sum
                )
                af = safe_af(alt_count, allele_sum)
                return SampleMetrics(
                    dp=dp, ref_count=ref_count, alt_count=alt_count, af=af
                )

            # Strelka2 somatic INDELs use tier1 counts in TAR/TIR
            if "TAR" in parsed and "TIR" in parsed:
                if alt_index != 0:
                    raise ValueError(
                        "Strelka2 INDEL TAR/TIR FORMAT does not encode per-ALT "
                        "depths; split multiallelic sites with 'bcftools norm -m -any'."
                    )
                ref_count = _parse_int_field(parsed["TAR"].split(",")[0], "TAR")
                alt_count = _parse_int_field(parsed["TIR"].split(",")[0], "TIR")
                allele_sum = ref_count + alt_count
                dp = (
                    _parse_int_field(parsed["DP"], "DP")
                    if "DP" in parsed and parsed["DP"] not in {".", ""}
                    else allele_sum
                )
                af = safe_af(alt_count, allele_sum)
                return SampleMetrics(
                    dp=dp, ref_count=ref_count, alt_count=alt_count, af=af
                )

            raise ValueError(
                f"Expected Strelka2 AU/CU/GU/TU or TAR/TIR fields in FORMAT '{format_str}'."
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse Strelka2 FORMAT '{format_str}' with value "
                f"'{sample_value}'."
            ) from exc


class DeepSomaticAdapter(BaseCallerAdapter):
    """Adapter for Google DeepSomatic somatic VCFs (``GT:GQ:DP:AD:VAF:PL``)."""

    name = "deepsomatic"
    aliases = ("deep_somatic", "deepvariant_somatic")
    supported_variant_types = frozenset({"SNV", "INDEL"})

    def detect(self, header_lines: Sequence[str]) -> bool:
        joined = "\n".join(header_lines).lower()
        return "deepsomatic" in joined

    def parse_record_metrics(
        self,
        *,
        ref: str,
        alt: str,
        format_str: str | None,
        sample_value: str | None,
        info_str: str,
        alt_index: int = 0,
    ) -> SampleMetrics:
        if not format_str or sample_value is None:
            raise ValueError("DeepSomatic records require FORMAT and sample values.")
        return _parse_standard_ad_format(
            format_str, sample_value, alt_index=alt_index, af_keys=("VAF", "AF")
        )


class DragenAdapter(BaseCallerAdapter):
    """Adapter for Illumina DRAGEN Somatic VCFs."""

    name = "dragen"
    aliases = ("dragen_somatic", "illumina_dragen")
    supported_variant_types = frozenset({"SNV", "INDEL"})

    def detect(self, header_lines: Sequence[str]) -> bool:
        joined = "\n".join(header_lines).lower()
        return "dragen" in joined

    def parse_record_metrics(
        self,
        *,
        ref: str,
        alt: str,
        format_str: str | None,
        sample_value: str | None,
        info_str: str,
        alt_index: int = 0,
    ) -> SampleMetrics:
        if not format_str or sample_value is None:
            raise ValueError("DRAGEN records require FORMAT and sample values.")
        return _parse_standard_ad_format(
            format_str, sample_value, alt_index=alt_index, af_keys=("AF", "VAF")
        )


class LoFreqAdapter(BaseCallerAdapter):
    """Adapter for LoFreq somatic 8-column site-only VCFs (``INFO`` DP/AF/DP4)."""

    name = "lofreq"
    aliases = ("lofreq_somatic", "lofreq2")
    supported_variant_types = frozenset({"SNV", "INDEL"})
    site_only = True

    def detect(self, header_lines: Sequence[str]) -> bool:
        joined = "\n".join(header_lines).lower()
        return "lofreq" in joined

    def parse_record_metrics(
        self,
        *,
        ref: str,
        alt: str,
        format_str: str | None,
        sample_value: str | None,
        info_str: str,
        alt_index: int = 0,
    ) -> SampleMetrics:
        if format_str and sample_value is not None:
            return _parse_standard_ad_format(
                format_str, sample_value, alt_index=alt_index, af_keys=("AF", "VAF")
            )
        info = parse_info_mapping(info_str)
        try:
            if "DP4" in info:
                dp4 = [_parse_int_field(x, "DP4") for x in info["DP4"].split(",")]
                if len(dp4) != 4:
                    raise ValueError(
                        f"Expected 4 integers in DP4, got '{info['DP4']}'."
                    )
                ref_count = dp4[0] + dp4[1]
                alt_count = dp4[2] + dp4[3]
                dp = (
                    _parse_int_field(info["DP"], "DP")
                    if "DP" in info
                    else ref_count + alt_count
                )
                af = (
                    _parse_float_field(info["AF"].split(",")[alt_index], "AF")
                    if "AF" in info
                    else safe_af(alt_count, dp)
                )
                return SampleMetrics(
                    dp=dp, ref_count=ref_count, alt_count=alt_count, af=af
                )

            dp = _parse_int_field(info["DP"], "DP")
            af = _parse_float_field(info["AF"].split(",")[alt_index], "AF")
            alt_count = int(round(dp * af))
            ref_count = max(dp - alt_count, 0)
            return SampleMetrics(dp=dp, ref_count=ref_count, alt_count=alt_count, af=af)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse LoFreq INFO '{info_str}'. Expected DP4 or DP+AF."
            ) from exc


class GenericFormatAdapter(BaseCallerAdapter):
    """Fallback adapter for custom caller names using standard ``FORMAT`` patterns."""

    name = "generic"
    aliases = ()
    supported_variant_types = frozenset({"SNV", "INDEL"})

    def __init__(self, caller_name: str = "generic") -> None:
        self.name = caller_name

    def parse_record_metrics(
        self,
        *,
        ref: str,
        alt: str,
        format_str: str | None,
        sample_value: str | None,
        info_str: str,
        alt_index: int = 0,
    ) -> SampleMetrics:
        if not format_str or sample_value is None:
            raise ValueError(
                f"Caller '{self.name}' requires FORMAT and sample columns."
            )
        parsed = parse_format_mapping(format_str, sample_value)
        if "RD" in parsed and "AD" in parsed and "," not in parsed["AD"]:
            return VarScan2Adapter().parse_record_metrics(
                ref=ref,
                alt=alt,
                format_str=format_str,
                sample_value=sample_value,
                info_str=info_str,
                alt_index=alt_index,
            )
        return _parse_standard_ad_format(
            format_str, sample_value, alt_index=alt_index, af_keys=("AF", "VAF")
        )


_CALLER_REGISTRY: dict[str, BaseCallerAdapter] = {}
_ALIAS_MAP: dict[str, str] = {}


def register_caller_adapter(
    adapter: BaseCallerAdapter,
    *,
    overwrite: bool = False,
) -> None:
    """Register a :class:`BaseCallerAdapter` for lookup by name or alias."""
    canonical = adapter.name.strip().lower()
    if not canonical:
        raise ValueError("Caller adapter name must be a non-empty string.")
    if canonical in _CALLER_REGISTRY and not overwrite:
        raise ValueError(
            f"Caller adapter '{canonical}' is already registered. "
            "Pass overwrite=True to replace it."
        )
    _CALLER_REGISTRY[canonical] = adapter
    _ALIAS_MAP[canonical] = canonical
    for alias in adapter.aliases:
        key = alias.strip().lower()
        if key in _ALIAS_MAP and _ALIAS_MAP[key] != canonical and not overwrite:
            raise ValueError(
                f"Caller alias '{key}' is already registered to '{_ALIAS_MAP[key]}'."
            )
        _ALIAS_MAP[key] = canonical


def list_caller_adapters() -> list[str]:
    """Return sorted canonical names of all registered caller adapters."""
    return sorted(_CALLER_REGISTRY)


def get_caller_adapter(
    caller: str | BaseCallerAdapter,
    *,
    allow_generic: bool = False,
) -> BaseCallerAdapter:
    """Resolve a caller string or adapter instance into a :class:`BaseCallerAdapter`.

    Parameters
    ----------
    caller : str or BaseCallerAdapter
        Registered caller name (or alias, case-insensitive) or an adapter
        instance.
    allow_generic : bool, default False
        When ``True``, unknown caller names fall back to
        :class:`GenericFormatAdapter` preserving the caller label.

    Returns
    -------
    BaseCallerAdapter
        The resolved adapter.
    """
    if isinstance(caller, BaseCallerAdapter):
        return caller
    key = str(caller).strip().lower()
    canonical = _ALIAS_MAP.get(key)
    if canonical is not None:
        return _CALLER_REGISTRY[canonical]
    if allow_generic:
        return GenericFormatAdapter(caller_name=str(caller).strip())
    available = ", ".join(list_caller_adapters())
    raise ValueError(
        f"Unknown VCF caller '{caller}'. Available callers: [{available}], "
        "or pass caller='auto' for header detection, or register a custom "
        "BaseCallerAdapter with register_caller_adapter()."
    )


def detect_caller_adapter(header_lines: Sequence[str]) -> BaseCallerAdapter:
    """Auto-detect the appropriate caller adapter from VCF ``##`` header lines."""
    matches = [
        adapter for adapter in _CALLER_REGISTRY.values() if adapter.detect(header_lines)
    ]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(list_caller_adapters())
    if len(matches) > 1:
        names = ", ".join(adapter.name for adapter in matches)
        raise ValueError(
            f"Ambiguous VCF header matched multiple callers: [{names}]. "
            f"Pass caller=... explicitly (choices: [{available}])."
        )
    raise ValueError(
        "Unable to auto-detect VCF caller from header lines. "
        f"Pass caller=... explicitly (choices: [{available}])."
    )


for _builtin_adapter in (
    Mutect2Adapter(),
    MuseAdapter(),
    VarScan2Adapter(),
    PindelAdapter(),
    Strelka2Adapter(),
    DeepSomaticAdapter(),
    DragenAdapter(),
    LoFreqAdapter(),
):
    register_caller_adapter(_builtin_adapter)
