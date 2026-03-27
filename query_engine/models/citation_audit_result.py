
from dataclasses import dataclass

@dataclass
class CitationAuditResult:
    hallucinated:    list[str]
    verified:        list[str]
    missing:         list[str]
    misattributed:   list[tuple]
    orphaned_claims: list[str]
    audit_passed:    bool