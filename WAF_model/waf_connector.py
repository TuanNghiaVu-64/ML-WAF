"""
waf_connector.py
================
Real WAF interface targeting DVWA's SQLi endpoint.

TARGET
------
    GET /vulnerabilities/sqli/?id=<payload>&Submit=Submit
    Host: localhost:9003
    Cookie: PHPSESSID=...; security=low

LABEL LOGIC
-----------
    HTTP 403  →  "B"  (Blocked  — WAF caught the payload)
    HTTP 200  →  "P"  (Passed   — payload reached the application)
    anything else → logged as error, treated as "B" (safe default)

WHAT THIS MODULE PROVIDES
--------------------------
    WafConnector          class — holds session config, sends requests
        .check(attack)    → "P" | "B"
        .check_batch(lst) → list of "P" | "B"

    MockWafConnector      drop-in replacement for offline testing
        same interface, uses a naive keyword blacklist

Both connectors share the same interface so the rest of the pipeline
(slice extractor, ML classifier, EA loop) never needs to change when
you swap between real and mock.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class DvwaConfig:
    """
    All connection parameters for the DVWA target in one place.
    Change these to match your local setup.

    Attributes
    ----------
    host        : base URL including port, no trailing slash
    path        : vulnerable endpoint path
    param       : GET parameter to inject the payload into
    phpsessid   : session cookie value (copy from browser DevTools)
    security    : DVWA security level cookie (keep as "low")
    timeout     : HTTP request timeout in seconds
    delay       : seconds to wait between requests (be gentle on localhost)
    """
    host:      str = "http://localhost:9003"
    path:      str = "/vulnerabilities/sqli/"
    param:     str = "id"
    phpsessid: str = "7782690643ce9018e8f5e0b45efe2dcb"  # ← paste yours here
    security:  str = "low"
    timeout:   float = 5.0
    delay:     float = 0.05   # 50 ms between requests


# ── Real WAF connector ────────────────────────────────────────────────────────

class WafConnector:
    """
    Sends SQLi payloads to the real DVWA endpoint and reads the WAF response.

    How a single check works
    ------------------------
    1. URL-encode the payload  →  safe to embed in a GET parameter
    2. Build the full URL:
           http://localhost:9003/vulnerabilities/sqli/?id=<encoded>&Submit=Submit
    3. Attach the session cookie header so DVWA recognises the session
    4. Send the GET request
    5. Read the HTTP status code:
           403  →  WAF blocked it  →  label "B"
           200  →  reached the app →  label "P"
           other→  treat as "B"   →  safe default

    Parameters
    ----------
    config : DvwaConfig   connection settings (see class above)
    """

    def __init__(self, config: DvwaConfig | None = None):
        self.cfg = config or DvwaConfig()

    # ── internal ──────────────────────────────────────────────────────────

    def _build_url(self, payload: str) -> str:
        """
        Construct the full request URL with the payload injected.
        """
        # Pass 1: decode existing %-sequences back to raw bytes.
        #   MUST use encoding='latin-1' (iso-8859-1) not the default utf-8.
        #   Reason: grammar tokens like %a0, %0b, %0d are single-byte values
        #   in the 0x80-0xFF range. They are valid latin-1 but NOT valid utf-8.
        #   With utf-8 (default), unquote(%a0) → replacement char U+FFFD which
        #   then re-encodes as %EF%BF%BD — completely wrong.
        #   With latin-1, unquote(%a0) → chr(0xa0) = non-breaking space ✓
        decoded = urllib.parse.unquote(payload, encoding='latin-1')

        # Pass 2: re-encode cleanly. Also latin-1 so chr(0xa0) → %A0, not %EF%BF%BD
        encoded = urllib.parse.quote(decoded, safe="", encoding='latin-1')
        return (
            f"{self.cfg.host}{self.cfg.path}"
            f"?{self.cfg.param}={encoded}&Submit=Submit"
        )

    def _build_request(self, url: str) -> urllib.request.Request:
        """
        Build the HTTP request object with required headers.

        The Cookie header carries:
          PHPSESSID  — identifies the logged-in session (required, else redirect)
          security   — DVWA security level (must be "low" for the SQLi lab)

        The remaining headers match the browser request you provided
        so the WAF sees a realistic-looking request.
        """
        cookie = (
            f"PHPSESSID={self.cfg.phpsessid}; "
            f"security={self.cfg.security}"
        )
        headers = {
            "Host":                    f"localhost:{self.cfg.host.split(':')[-1]}",
            "Cookie":                  cookie,
            "User-Agent":              (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.0.0 Safari/537.36"
            ),
            "Accept":                  (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language":         "en-US,en;q=0.9",
            "Accept-Encoding":         "gzip, deflate",   # no br — easier to read
            "Referer":                 f"{self.cfg.host}/security.php",
            "Connection":              "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        return urllib.request.Request(url, headers=headers, method="GET")

    # ── public API ────────────────────────────────────────────────────────

    def check(self, payload: str) -> str:
        """
        Send one payload to the WAF and return its label.

        Parameters
        ----------
        payload : str   raw SQLi string, e.g. "' OR 1=1 #"

        Returns
        -------
        "P"  — payload bypassed the WAF (HTTP 200)
        "B"  — payload was blocked       (HTTP 403)
        "B"  — on any network/timeout error (safe default)
        """
        url = self._build_url(payload)
        req = self._build_request(url)

        try:
            # urllib raises HTTPError for 4xx/5xx, so we catch it separately
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                status = resp.status          # 200 → passed through
                label  = "P" if status == 200 else "B"

        except urllib.error.HTTPError as e:
            # 403 Forbidden lands here
            label = "B" if e.code == 403 else "B"

        except urllib.error.URLError as e:
            # Connection refused, timeout, DNS failure, etc.
            print(f"[WARN] Network error for payload {repr(payload)}: {e.reason}")
            label = "B"

        except Exception as e:
            print(f"[WARN] Unexpected error for payload {repr(payload)}: {e}")
            label = "B"

        # Polite delay between requests
        if self.cfg.delay > 0:
            time.sleep(self.cfg.delay)

        return label

    def check_batch(self, payloads: list[str]) -> list[str]:
        """
        Send a list of payloads and return a label for each.

        Parameters
        ----------
        payloads : list[str]   raw attack strings

        Returns
        -------
        list[str]   same length as payloads, each entry "P" or "B"
        """
        labels = []
        for i, payload in enumerate(payloads):
            label = self.check(payload)
            labels.append(label)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(payloads)}] checked — "
                      f"{labels.count('P')} bypass, "
                      f"{labels.count('B')} blocked so far")
        return labels

    def label_corpus(self, attacks: list[dict]) -> list[dict]:
        """
        Convenience wrapper: take a list of attack dicts (with "attack"
        and "derivation" keys) and add a "label" key to each by querying
        the WAF.

        Parameters
        ----------
        attacks : list of dicts with at least {"attack": str, "derivation": list}

        Returns
        -------
        Same list with "label" added to each dict — ready for
        SliceRegistry.encode_corpus()
        """
        print(f"Checking {len(attacks)} payloads against WAF at {self.cfg.host}…")
        for i, entry in enumerate(attacks):
            entry["label"] = self.check(entry["attack"])
            if (i + 1) % 20 == 0:
                bypass  = sum(1 for e in attacks[:i+1] if e.get("label") == "P")
                blocked = sum(1 for e in attacks[:i+1] if e.get("label") == "B")
                print(f"  [{i+1}/{len(attacks)}] bypass={bypass} blocked={blocked}")
        return attacks


# ── Mock WAF connector (offline testing) ─────────────────────────────────────

class MockWafConnector:
    """
    Drop-in replacement for WafConnector when no real WAF is available.

    Uses a simple keyword blacklist — intentionally incomplete so that
    obfuscated payloads (using %20, /**/, &#39; etc.) slip through,
    producing realistic P/B mix for testing the ML pipeline.

    Same public interface as WafConnector:
        .check(payload)        → "P" | "B"
        .check_batch(payloads) → list["P" | "B"]
        .label_corpus(attacks) → attacks with "label" added
    """

    # Patterns the mock WAF "knows" about — naive, easily bypassed
    _BLACKLIST = [
        "union",
        "select",
        "sleep(",
        "' or ",
        "\" or ",
        "1=1",
        "--",
    ]

    def check(self, payload: str) -> str:
        low = payload.lower()
        for pattern in self._BLACKLIST:
            if pattern in low:
                return "B"
        return "P"

    def check_batch(self, payloads: list[str]) -> list[str]:
        return [self.check(p) for p in payloads]

    def label_corpus(self, attacks: list[dict]) -> list[dict]:
        for entry in attacks:
            entry["label"] = self.check(entry["attack"])
        return attacks


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from random_sampler import generate_unique_attacks

    # ── choose connector based on command-line arg ────────────────────────
    # Usage:
    #   python waf_connector.py         → uses real DVWA
    #   python waf_connector.py mock    → uses mock WAF (no network needed)

    USE_MOCK = len(sys.argv) > 1 and sys.argv[1] == "mock"

    if USE_MOCK:
        print("Using MockWafConnector (offline mode)\n")
        waf = MockWafConnector()
    else:
        print("Using WafConnector → DVWA at localhost:9003\n")
        print("Make sure DVWA is running and PHPSESSID is current.\n")
        waf = WafConnector()   # uses default DvwaConfig

    # ── generate 20 attacks ───────────────────────────────────────────────
    print("Generating 20 attacks…")
    raw_attacks = generate_unique_attacks(20)

    # Build corpus dicts (no label yet)
    corpus = [
        {"attack": atk, "derivation": drv}
        for atk, drv in raw_attacks
    ]

    # ── label via WAF ─────────────────────────────────────────────────────
    corpus = waf.label_corpus(corpus)

    # ── print results ─────────────────────────────────────────────────────
    print("\n── Results ─────────────────────────────────────────────────")
    print(f"  {'#':<4}  {'Label'}  {'Payload'}")
    print("  " + "-" * 70)
    for i, entry in enumerate(corpus):
        short = entry["attack"][:55]
        print(f"  {i:<4}  {entry['label']}      {short}")

    bypass  = sum(1 for e in corpus if e["label"] == "P")
    blocked = sum(1 for e in corpus if e["label"] == "B")
    print(f"\n  bypass (P): {bypass}   blocked (B): {blocked}")

    # ── show URL that would be sent for attack #1 ─────────────────────────
    if not USE_MOCK:
        connector = WafConnector()
        example_url = connector._build_url(corpus[0]["attack"])
        print(f"\n── Example URL for attack #1 ───────────────────────────────")
        print(f"  {example_url}")