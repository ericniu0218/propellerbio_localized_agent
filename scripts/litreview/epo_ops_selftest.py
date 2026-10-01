#!/usr/bin/env python3
"""Run once, by hand, against the real EPO OPS account — not part of `pytest`.

Everything in `bedrock.litreview.sources.epo_ops` is unit-tested offline
against a mock transport, which proves the code does what we told it the API
does. It cannot prove the API actually behaves that way; only a live account
can. One piece specifically could not be checked against the primary document
at all — the claims XML shape (see the module's own docstring) — so this
script exists to close that gap before the client relies on it in a report.

Usage:

    export EPO_OPS_KEY=...
    export EPO_OPS_SECRET=...
    python scripts/litreview/epo_ops_selftest.py

Prints each step's result and stops at the first thing that does not match
what the code assumes. Nothing here touches the pipeline or the database —
it is a smoke test of one source, in isolation.
"""

from __future__ import annotations

import os
import sys

from bedrock.litreview.sources.epo_ops import EpoOps, PatentSearchFailed, cql_query
from bedrock.litreview.sources.query import Query


def main() -> int:
    key, secret = os.environ.get("EPO_OPS_KEY", ""), os.environ.get("EPO_OPS_SECRET", "")
    if not key or not secret:
        print("Set EPO_OPS_KEY and EPO_OPS_SECRET first (from developers.epo.org > My Apps).")
        return 1

    ops = EpoOps(key, secret)

    print("1. Requesting an OAuth token...")
    header = ops.tokens.header()
    print(f"   ok: {header['Authorization'][:20]}...")

    print("\n2. Searching for a known, findable topic (biblio+abstract, one page)...")
    query = Query(label="test", groups=(("interleukin-23",),))
    cql = cql_query(query)
    print(f"   CQL: {cql}")
    try:
        page, total = ops.search_page(cql, 1, 10)
    except PatentSearchFailed as exc:
        print(f"   FAILED: {exc}")
        return 1
    print(f"   ok: {total} total hits, {len(page)} on this page")
    if not page:
        print("   No hits for a term this common — something is wrong with query building or auth scope.")
        return 1

    for i, record in enumerate(page[:3], 1):
        print(f"   [{i}] {record.id}  family={record.family_id}  {record.title[:70]!r}")
        print(f"       assignee={record.assignee!r}  pub_date={record.publication_date}  "
              f"priority_date={record.priority_date}  abstract={'yes' if record.abstract else 'NO'}")

    print("\n3. Fetching claim 1 for the first result "
          "(THE UNCONFIRMED PART — read this output carefully)...")
    target = page[0]
    found = ops.claim_1(target)
    claim = found[0] if found else None
    if claim is None:
        print("   claim_1() returned None. Either this patent genuinely has no claims text")
        print("   available (some jurisdictions don't), or parse_claim_1()'s assumed <claim>/")
        print("   <claim-text> shape is wrong. Check the warning above, then fetch this URL by")
        print("   hand and look at the real element names:")
        print(f"     https://ops.epo.org/3.2/rest-services/published-data/publication/docdb/"
              f"{target.jurisdiction}.{target.id[len(target.jurisdiction or ''):]}/claims")
    else:
        print(f"   ok ({len(claim)} chars{', cut at the cap' if found[1] else ''}): {claim[:200]!r}")
        if len(claim) > 4000:
            print("   TOO LONG for one claim: the splitter has not found where claim 2 starts.")
        print("   Read this against the real patent (search its number on Espacenet) and confirm")
        print("   it really is claim 1, not a fragment or the wrong claim.")

    print("\n4. Throttle status after these calls:")
    status = ops.throttle.last_status
    if status:
        print(f"   system state: {status.system_state}")
        for service, (colour, limit) in status.services.items():
            print(f"   {service}: {colour} ({limit})")
    else:
        print("   No X-Throttling-Control header was seen — note this, the code assumes one "
              "is always present on a successful response.")

    print("\nDone. If step 3 needed hand-checking, update parse_claim_1() and this script's")
    print("comment before anyone relies on claim_1 in a real report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
