"""Manually curated lookup: law abbreviation (as it appears inline in DE/FR/IT statutory
text, e.g. "art. 97 CO") -> SR/RS systematic number.

This is deliberately small and hand-curated, not derived automatically from ingested
`law_short_name` values (those are full act titles, e.g. "Codice civile svizzero del 10
dicembre 1907", not the abbreviation actually used in cross-references) — see
`processing.reference_extractor`, the only consumer. Seeded from the same codes covered
by `config.PREFIX_LABELS`; extend both together when adding a new code to the ingestion
scope. Ambiguous abbreviations shared across languages for the same law (e.g. "CC" for
both "Code civil" and "Codice civile") are not actually ambiguous here since they resolve
to the same systematic number.
"""

from __future__ import annotations

# abbreviation -> SR/RS systematic number. Longer/more specific keys are checked first
# by `reference_extractor` where a shorter key could otherwise shadow one (e.g. "CPC"
# before "CP", "CPP" before "CP").
LAW_ABBREVIATIONS: dict[str, str] = {
    # Codice civile / ZGB / Code civil (SR 210)
    "ZGB": "210",
    "CC": "210",
    # Codice delle obbligazioni / OR / Code des obligations (SR 220)
    "OR": "220",
    "CO": "220",
    # Codice penale / StGB / Code pénal (SR 311.0)
    "StGB": "311.0",
    "CP": "311.0",
    # Procedura penale / StPO / Code de procédure pénale (SR 312.0)
    "StPO": "312.0",
    "CPP": "312.0",
    # Procedura civile / ZPO / Code de procédure civile (SR 272)
    "ZPO": "272",
    "CPC": "272",
    # Costituzione federale / BV / Constitution (SR 101)
    "BV": "101",
    "Cst.": "101",
    "Cst": "101",
    "Cost.": "101",
    "Cost": "101",
    # Legge federale sull'imposta federale diretta / LIFD / DBG (SR 642.11) — the single
    # most commonly cross-referenced tax act, more specific than the bare "642" prefix.
    "LIFD": "642.11",
    "DBG": "642.11",
    # Legge sull'IVA / LTVA / MWSTG (SR 641.20) — not in the default ingestion scope
    # (config.PREFIX_LABELS only covers 640/642), kept here so a reference to it is at
    # least captured with a resolvable target if 641 is added to scope later.
    "LTVA": "641.20",
    "MWSTG": "641.20",
}
