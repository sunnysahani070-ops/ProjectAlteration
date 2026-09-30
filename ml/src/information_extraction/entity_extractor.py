"""Rule-based extraction of drilling entities from report text."""

from __future__ import annotations

import re


class EntityExtractor:
    """Extract deterministic fields such as well ID, depth, and severity."""

    FORMATIONS = (
        "barail", "tipam", "kopili", "tura", "limestone", "sandstone",
        "shale", "dolomite", "granite", "chalk", "anhydrite",
        "conglomerate", "siltstone",
    )

    def extract_well_id(self, text: str) -> str | None:
        lines = (text or "").splitlines()
        accepted_labels = {"well", "well id", "well name", "well identifier"}
        identifier_pattern = re.compile(r"\b([A-Z]{1,6}-\d{2,5})\b", flags=re.IGNORECASE)

        for index, line in enumerate(lines):
            label, separator, value = line.strip().partition(":")
            if label.strip().lower() not in accepted_labels:
                continue
            candidates = [value] if separator else []
            if not candidates and line.strip().lower() in accepted_labels and index + 1 < len(lines):
                candidates.append(lines[index + 1])
            for candidate in candidates:
                match = identifier_pattern.search(candidate)
                if match:
                    return match.group(1).upper()

        for match in identifier_pattern.finditer(text or ""):
            if match.group(1).upper() != "SHA-256":
                return match.group(1).upper()
        return None

    def extract_depth_md(self, text: str) -> int | float | None:
        match = re.search(
            r"\b(?:at\s+|depth\s*(?:of\s*)?)?((?:\d{1,3}(?:,\d{3})+|\d{3,5})(?:\.\d+)?)\s*(?:m|meters?)\b",
            text or "",
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        value = float(match.group(1).replace(",", ""))
        return int(value) if value.is_integer() else value

    def extract_severity(self, text: str) -> str | None:
        normalized = (text or "").lower()
        explicit_match = re.search(
            r"\bseverity\s*[:=]\s*(low|medium|high|critical)\b",
            normalized,
        )
        if explicit_match:
            return explicit_match.group(1).upper()
        if re.search(r"\bcritical\b", normalized):
            return "CRITICAL"
        if re.search(r"\b(high|severe)\b", normalized):
            return "HIGH"
        if re.search(r"\b(medium|moderate)\b", normalized):
            return "MEDIUM"
        if re.search(r"\b(low|minor|slight)\b", normalized):
            return "LOW"
        return None

    def extract_formation(self, text: str) -> str | None:
        normalized = (text or "").lower()
        matches = [
            (match.start(), formation)
            for formation in self.FORMATIONS
            if (match := re.search(rf"\b{formation}\b", normalized))
        ]
        return min(matches)[1].title() if matches else None

    def extract_npt_hours(self, text: str) -> float | None:
        match = re.search(
            r"\bNPT\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:hrs?|hours?)\b",
            text or "",
            flags=re.IGNORECASE,
        )
        return float(match.group(1)) if match else None

    def extract_cause(self, text: str) -> str | None:
        cause_patterns = (
            r"\b(?:cause|caused by|due to|because of|resulting from|attributed to|triggered by)\s*[:=]?\s*([^;\n.]+)",
            r"\b(?:after|following)\s+(losses?\s+(?:reduced|lowered|stopped)(?:\s+[^;,\n.]+)?|static\s+pipe\s+(?:during|across)|loss\s+of\s+returns)",
            r"\b(?:on entering|upon entering|while entering)\s+([^;\n.]+)",
            r"\bin\s+((?:fractured|high[- ]permeability|weak|unconsolidated|depleted|overpressured|permeable)\s+[^;\n.]+)",
            r"\b(?:due to|from)\s+((?:losses?|kick|influx|packoff|differential sticking|formation pressure)[^;\n.]*)",
        )
        for pattern in cause_patterns:
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip(" ,")
        return None

    def extract_mitigation(self, text: str) -> str | None:
        patterns = (
            r"\b(?:mitigation|response|treatment|corrective action|action taken)\s*[:=]\s*(.+?)(?=;\s*(?:cured|resolved|restored|freed|returns? restored|losses? stopped)\b|\n\s*(?:lesson|cause|outcome|result|effectiveness)\s*:|\n\s*\d+\.\s+[A-Z]|$)",
            r"\b(?:mitigated by|treated with|treatment was|responded by|resolved by)\s+(.+?)(?=\n\s*(?:lesson|cause|outcome|result)\s*:|\n\s*\d+\.\s+[A-Z]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)
            if match:
                value = re.sub(r"\s+", " ", match.group(1)).strip().rstrip(" ;,")
                if value:
                    return value
        return None

    def extract_outcome(self, text: str) -> str | None:
        normalized_text = text or ""
        labeled_match = re.search(
            r"\b(?:outcome|result)\s*:\s*([^\n]+?)(?<!\d)(?:\.(?=\s|$)|$)",
            normalized_text,
            flags=re.IGNORECASE,
        )
        if labeled_match:
            return labeled_match.group(1).strip().rstrip(".")

        action_match = re.search(
            r"\b((?:cured|resolved|recovered|restored|regained|freed|circulated out|circulated\s+(?:the\s+)?(?:influx|kick)\s+out|killed|flow stopped|losses? stopped|returns? restored|influx removed)[^;\n]*)",
            normalized_text,
            flags=re.IGNORECASE,
        )
        return action_match.group(1).strip().rstrip(".") if action_match else None
