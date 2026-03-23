"""
test_waf.py
===========
Interactive WAF tester — enter a payload, see exactly what gets sent
and what label the WAF returns. Type 'quit' to exit.

Run:
    python test_waf.py
"""

import sys
import os
import urllib.parse

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODULE_DIR)

from waf_connector import WafConnector, DvwaConfig

# ── config ────────────────────────────────────────────────────────────────────
cfg = DvwaConfig(
    host      = "http://localhost:9003",
    phpsessid = "f59690908df4860b89bddd3eaba6922c",  # ← update if session expired
    security  = "low",
    delay     = 0.0,
)
waf = WafConnector(cfg)

# ── helpers ───────────────────────────────────────────────────────────────────
def print_result(payload: str) -> None:
    url           = waf._build_url(payload)
    encoded_param = url.split("?id=")[1].split("&")[0]
    server_sees   = urllib.parse.unquote(encoded_param, encoding="latin-1")
    label         = waf.check(payload)

    label_display = "✓ BYPASS (P)" if label == "P" else "✗ BLOCKED (B)"

    print()
    print(f"  Result       : {label_display}")
    print(f"  Original     : {repr(payload)}")
    print(f"  Encoded param: {encoded_param}")
    print(f"  Server sees  : {repr(server_sees)}")
    print(f"  Full URL     : {url}")
    print()

# ── main loop ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Interactive WAF Tester")
print(f"  Target : {cfg.host}")
print("  Type a payload and press Enter to test.")
print("  Type 'quit' to exit.")
print("=" * 65)

while True:
    try:
        payload = input("\nPayload > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not payload:
        continue

    if payload.lower() in ("quit", "exit", "q"):
        print("Exiting.")
        break

    print_result(payload)