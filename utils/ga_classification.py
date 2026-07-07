"""Manufacturer/model-based classification of conventional, light, fixed-wing
General Aviation aircraft, for building a homogeneous light-GA motion-prior
training set out of the full aircraft database.

Classification is regex-based on `manufacturername` / `model` text rather
than the `icaoaircrafttype` column, because that column is inconsistently
populated (often blank) and sometimes mistagged in the source data.

Three passes, applied in order by classify():
  1. BRAND_RULES         -- per-manufacturer keep/exclude patterns
  2. strict exclusions   -- military variants, mistag safety net, large twins
  3. icao24 format check -- drop malformed (non-6-hex) transponder codes
"""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

REQUIRED_COLUMNS = ["manufacturername", "model", "icaoaircrafttype", "icao24"]

# A well-formed ICAO 24-bit address: exactly six hex characters.
ICAO24_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{6}$")

# Overrides any brand-level keep match, regardless of family.
GLOBAL_EXCLUDE = re.compile(
    r"HELICOPTER|ROTORCRAFT|GYROCOPTER|GYROPLANE|\bGLIDER\b|SAILPLANE|MOTORGLIDER|BALLOON|AIRSHIP"
)


def num(code) -> str:
    """Regex for a numeric model code with digit-only boundaries.

    Plain `\\b` boundaries fail on codes like "58TC" or "PA-28R-200", where a
    letter is glued directly to the digits. Requiring "not preceded/followed
    by another digit" instead lets those letter suffixes through untouched.
    """
    return rf"(?<!\d){code}(?!\d)"


def any_num(codes: List[int]) -> str:
    """OR-combine several `num()` patterns into one regex alternation."""
    return "|".join(num(c) for c in codes)


class BrandRule:
    """One manufacturer's keep/exclude patterns.

    `brand` matches the manufacturer name; `keep` matches that brand's
    conventional piston-GA lineup; `exclude` (optional) strips turbine /
    military / jet variants that would otherwise match a shared model code
    (e.g. Beech "65" is both the piston Queen Air and, as "65-A90", an
    early King Air).
    """

    def __init__(self, name: str, brand: str, keep: str, exclude: Optional[str] = None):
        self.name = name
        self.brand = re.compile(brand)
        self.keep = re.compile(keep)
        self.exclude = re.compile(exclude) if exclude else None


BRAND_RULES = [
    # Reims was the licensed French Cessna builder. Keep = piston singles
    # (120-210) and piston twins (310-421); jets/turboprops/ag lines excluded.
    BrandRule(
        "CESSNA",
        brand=r"\bCESSNA\b|\bREIMS\b",
        keep=any_num([
            120, 140, 145, 150, 152, 162, 165, 170, 172, 175, 177, 180, 182, 185, 190, 195,
            205, 206, 207, 210, 240, 310, 320, 335, 336, 337, 340, 400, 401, 402, 404, 411, 414, 421,
        ]),
        exclude=(
            r"CITATION|CARAVAN|CONQUEST|CORSAIR|BOBCAT|BIRD ?DOG|MUSTANG|SOVEREIGN|ENCORE|EXCEL|\bXLS\b|"
            r"\bT-?37\b|\bT-?41\b|\bO-?1A?\b|\bO-?2A?\b|\bL-19\b|AGWAGON|AGTRUCK|AG ?HUSKY|AGPICKUP|" +
            any_num([188, 208, 406, 425, 441, 500, 501, 510, 525, 526, 530, 550, 551, 560, 650, 680, 700, 750])
        ),
    ),
    # PA-11 through PA-60 piston family, plus Cub-era J/L designations and
    # common nicknames. Excluded: crop dusters (PA-25/36), the turboprop-only
    # PA-42, and any Cheyenne/JetPROP/Meridian turboprop variant.
    BrandRule(
        "PIPER",
        brand=r"\bPIPER\b",
        keep=(
            r"PA[\.\-\s]?(?:" + "|".join(str(c) for c in
                [11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 28, 29, 30, 31, 32, 34, 38, 39, 44, 46, 60]) + r")(?!\d)"
            r"|\bJ-?[2345][A-Z]{0,3}(?:-\d+)?|\bE-2\b|\bL-?4[A-Z]?\b|\bL-?18[A-Z]?\b|\bL-?21[A-Z]?\b|"
            r"AEROSTAR|\bCUB\b|CHEROKEE|COMANCHE|ARCHER|WARRIOR|TOMAHAWK|SARATOGA|SENECA|MALIBU|MIRAGE|"
            r"MATRIX|NAVAJO|CHIEFTAIN|AZTEC|APACHE|\bPACER\b|\bCOLT\b|VAGABOND|CLIPPER|SEMINOLE|DAKOTA|\bARROW\b"
        ),
        exclude=(
            r"CHEYENNE|JETPROP|MERIDIAN|\bM500\b|\bM600\b|500TP|600TP|PAWNEE|"
            r"PA[\.\-\s]?25\b|PA[\.\-\s]?36\b|PA[\.\-\s]?42\b|PA-?31T"
        ),
    ),
    # Bonanza/Baron/Musketeer/Duke/Queen Air piston family (model numbers
    # 19-95). King Air/Hawker/1900/T-34/Model 18 share numbers with the piston
    # lineup, so they need an explicit override rather than just omission.
    BrandRule(
        "BEECH",
        brand=r"\bBEECH\b|\bBEECHCRAFT\b|\bRAYTHEON\b",
        keep=(
            any_num([19, 23, 24, 33, 35, 36, 55, 56, 58, 60, 65, 70, 76, 77, 80, 95]) +
            r"|BONANZA|DEBONAIR|\bBARON\b|MUSKETEER|SUNDOWNER|\bSIERRA\b|DUCHESS|SKIPPER|\bDUKE\b|"
            r"QUEEN ?AIR|TRAVEL ?AIR|TWIN BONANZA"
        ),
        exclude=(
            r"\bA90\b|65-?90|U-?21|\bUTE\b|RC-?12|JC-?12|QU-?22|RU-?21|GU-?21|L-?23[A-Z]?\b|"
            r"KING ?AIR|HAWKER|PREMIER|BEECHJET|NEXTANT|\b1900\b|T-?34|MENTOR|\bT-?6[A-Z]?\b|TEXAN|"
            r"T-?1A|JAYHAWK|T-?44|PEGASUS|C-?12|HURON|EXPEDITOR|\bSNB\b|C-?45|D-?18|D18S|STAGGERWING|"
            r"KANSAN|AT-?11|AT-?7\b|VOLPAR|TRADEWIND|SHADOW|AVENGER|LIBERTY"
        ),
    ),
    # SR20/SR22 only; the SF50 Vision Jet is excluded.
    BrandRule("CIRRUS", brand=r"\bCIRRUS\b", keep=r"SR-?20|SR-?22",
              exclude=r"VISION|\bSF-?50\b|\bJET\b"),
    # M10/M18 Mite, M20 family, M22 Mustang.
    BrandRule("MOONEY", brand=r"\bMOONEY\b", keep=r"M[\.\-]?(?:10|18|20|22)"),
    # Rallye/TB piston family; TBM turboprops and TB30 Epsilon excluded by omission.
    BrandRule("SOCATA", brand=r"\bSOCATA\b",
              keep=r"RALLYE|\bST[\.\-]?10\b|TB[\.\-\s]?(?:200|21|20|10|9)(?!\d)"),
    BrandRule(
        "ROBIN",
        brand=r"\bROBIN\b",
        keep=(
            r"\bDR[\.\-\s]?(?:300|315|340|360|380|400|500)(?!\d)|"
            r"\bHR[\.\-\s]?(?:100|200)(?!\d)|"
            r"\bR[\.\-\s]?(?:1180|2160|2112|2100|2120|3000)(?!\d)|"
            r"\bATL\b"
        ),
    ),
    # HK36 Super Dimona is a motorglider, hence excluded.
    BrandRule(
        "DIAMOND",
        brand=r"\bDIAMOND\b",
        keep=r"\bDA[\.\-\s]?(?:20|40|42|50|62)(?!\d)|\bDV[\.\-\s]?20(?!\d)|KATANA|TWIN ?STAR|DIAMOND STAR",
        exclude=r"HK[\.\-\s]?36|DIMONA",
    ),
    # 7-series Champ/Citabria, 8-series Decathlon/Scout, 14-/17-series
    # Cruisair/Viking. Excluded: kit/homebuilt lookalikes.
    BrandRule(
        "BELLANCA_CHAMPION",
        brand=r"\bBELLANCA\b|\bCHAMPION\b",
        keep=(
            r"\b7[A-Z]{1,5}\b|\b8[GK][A-Z]{1,5}\b|"
            r"14-\d{1,2}|17-3[01]|"
            r"CITABRIA|DECATHLON|\bSCOUT\b|VIKING|\bCHAMP\b|CHIEF|\bSEDAN\b|"
            r"CRUISAIR|CRUISEMASTER|PACEMAKER|SKYROCKET|CH-?300|CH-?400|TRAVEL(?:L)?ER"
        ),
        exclude=r"KITFOX|\bRANS\b|VAN'?S|\bRV-?\d|GLASAIR|LANCAIR",
    ),
    # PT-19/23, O-58/L-3/L-16 are WWII military liaison/trainer designations,
    # not civil Champs/Chiefs.
    BrandRule(
        "AERONCA",
        brand=r"\bAERONCA\b",
        keep=(
            r"\b7[A-Z]{1,4}\b|\b11[A-Z]{1,3}\b|\b15AC\b|65-?[A-Z]{1,3}\b|\bK[A-Z]{0,2}\b|\bC-?[23]\b|"
            r"\bLC\b|\bLB\b|CHAMP|CHIEF|\bSEDAN\b|DEFENDER|\bSCOUT\b"
        ),
        exclude=r"\bPT-?19\b|\bPT-?23\b|\bO-?58[A-Z]?\b|\bL-?3[BC]?\b|\bL-?16[A-Z]?\b|GRASSHOPPER",
    ),
    # Auster is the British military derivative, not the civil Taylorcraft line.
    BrandRule(
        "TAYLORCRAFT",
        brand=r"\bTAYLORCRAFT\b",
        keep=(
            r"\bBC\b|\bBC-?12[A-Z0-9\-]*|\bBL\b|\bBL-?\d+[A-Z0-9\-]*|\bBF\b|\bBF-?\d+[A-Z0-9\-]*|"
            r"\bDC-?\d+[A-Z0-9\-]*|\bDCO-?\d+[A-Z0-9\-]*|\bDF-?\d+[A-Z0-9\-]*|\bDL-?\d+[A-Z0-9\-]*|"
            r"\bF-?19\b|\bF-?2[12][A-Z]?\b|\bBCS-?\d*[A-Z0-9\-]*|\b19\b|\b20\b|PLUS ?D|SPORTSMAN|TWOSOME"
        ),
        exclude=r"AUSTER",
    ),
    BrandRule("LUSCOMBE", brand=r"\bLUSCOMBE\b",
              keep=r"\b8[A-Z]{0,2}X?\b|\b11[AE]\b|SILVAIRE|PHANTOM"),
    # "VULTEE" is matched broadly because postwar Stinson 108s were built
    # under that corporate name; the exclude strips Vultee's *own* military
    # BT-13/PBY/Convair lines.
    BrandRule(
        "STINSON",
        brand=r"\bSTINSON\b|VULTEE",
        keep=r"(?<!\d)108(?!\d)|VOYAGER|RELIANT|HW-?75|(?<!\d)10(?!\d)|SR-?\d",
        exclude=r"VALIANT|\bBT-?1[35]\b|\bSNV\b|\bPBY\b|CONVAIR|\bL-?1[3F]?\b|\bAT-?19\b",
    ),
]


def clean_fields(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Validate required columns and return cleaned (manufacturer, model, icao) Series.

    Each Series is uppercased/stripped/NaN-filled; df itself is not mutated.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in input file: {missing}")

    mfr = df["manufacturername"].fillna("").astype(str).str.upper().str.strip()
    model = df["model"].fillna("").astype(str).str.upper().str.strip()
    icao = df["icaoaircrafttype"].fillna("").astype(str).str.upper().str.strip()
    return mfr, model, icao


def valid_icao24_mask(df: pd.DataFrame) -> pd.Series:
    """True where icao24 is a well-formed 6-character hex address.

    The raw OpenSky database contains occasional malformed rows (e.g. a
    truncated 5-char code duplicating an aircraft that also has a valid
    entry). A malformed code can never match real ADS-B traffic downstream,
    so such rows are dead weight in the whitelist and are dropped here.
    """
    icao24 = df["icao24"].fillna("").astype(str).str.strip()
    return icao24.str.match(ICAO24_HEX_PATTERN)


def build_base_mask(mfr: pd.Series, model: pd.Series) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """Apply every BRAND_RULES entry and OR the results together.

    Returns (base_mask, brand_masks): brand_masks records which rows matched
    each brand's manufacturer regex, for reuse by the stricter exclusion
    rules in build_strict_exclude_mask().
    """
    base_mask = pd.Series(False, index=mfr.index)
    brand_masks: Dict[str, pd.Series] = {}

    for rule in BRAND_RULES:
        is_brand = mfr.str.contains(rule.brand, regex=True)
        brand_masks[rule.name] = is_brand

        keep = is_brand & model.str.contains(rule.keep)
        if rule.exclude is not None:
            keep &= ~model.str.contains(rule.exclude)
        base_mask |= keep

    # Strip anything that reads as rotorcraft/glider/etc., regardless of
    # which brand pattern matched it.
    return base_mask & ~model.str.contains(GLOBAL_EXCLUDE), brand_masks


def build_strict_exclude_mask(
    mfr: pd.Series,
    model: pd.Series,
    icao: pd.Series,
    brand_masks: Dict[str, pd.Series],
) -> pd.Series:
    """Stricter exclusions layered on top of build_base_mask(), for a more
    homogeneous light-GA motion-prior training set:

    1. Military-designation Cub / Super Cub variants (L-4/L-18/L-21/C-145).
    2. Rows tagged helicopter/jet/turboprop in `icaoaircrafttype`, unless the
       model is unambiguously one of a small set of allowed piston families
       (the source data mistags some ordinary Cessna/Piper/Cirrus models).
    3. Large business/utility piston twins (Navajo, Duke, Queen Air, etc.).
    """
    # 1) Military-designation Cub / Super Cub variants.
    military_cub_pat = re.compile(r"\bL-?4[A-Z]?\b|\bL-?18[A-Z]?\b|\bL-?21[A-Z]?\b|\bC-?145\b")
    military_cub_exclude = model.str.contains(military_cub_pat)

    # 2) icaoaircrafttype-based exclusion (with per-brand piston exceptions).
    # Codes follow [category][engine count][engine type]: H = helicopter, and
    # the last letter is the engine type (P = piston, T = turboprop, J = jet).
    icao_flagged = (
        icao.str.match(r"^H", na=False)
        | icao.str.contains(r"J$", na=False, regex=True)
        | icao.str.contains(r"T$", na=False, regex=True)
    )

    exception_pat_cessna = re.compile(any_num([150, 152, 172, 177, 180, 182, 185, 206, 207, 210]))
    exception_pat_piper = re.compile(r"PA[\.\-\s]?(?:28|32|34|38|44)(?!\d)")
    exception_pat_cirrus = re.compile(r"SR-?20|SR-?22")
    exception_pat_diamond = re.compile(r"\bDA[\.\-\s]?(?:20|40|42)(?!\d)")
    exception_pat_mooney = re.compile(r"M[\.\-]?20")
    exception_pat_robin = re.compile(r"\bDR[\.\-\s]?400(?!\d)")
    exception_pat_socata = re.compile(r"TB[\.\-\s]?(?:9|10|20|21)(?!\d)")
    exception_pat_beech = re.compile(r"BONANZA|\bBARON\b|MUSKETEER|SUNDOWNER|\bSIERRA\b|DUCHESS|SKIPPER")

    icao_exception = (
        (brand_masks["CESSNA"] & model.str.contains(exception_pat_cessna))
        | (brand_masks["PIPER"] & model.str.contains(exception_pat_piper))
        | (brand_masks["CIRRUS"] & model.str.contains(exception_pat_cirrus))
        | (brand_masks["DIAMOND"] & model.str.contains(exception_pat_diamond))
        | (brand_masks["MOONEY"] & model.str.contains(exception_pat_mooney))
        | (brand_masks["ROBIN"] & model.str.contains(exception_pat_robin))
        | (brand_masks["SOCATA"] & model.str.contains(exception_pat_socata))
        | (brand_masks["BEECH"] & model.str.contains(exception_pat_beech))
    )
    icao_exclude = icao_flagged & ~icao_exception

    # 3) Large business/utility piston twins.
    large_twin_exclude = (
        (brand_masks["PIPER"] & model.str.contains(re.compile(r"PA-?31|NAVAJO|CHIEFTAIN")))
        | (brand_masks["CESSNA"] & model.str.contains(re.compile(any_num([401, 402, 404, 411, 414, 421]))))
        | (
            brand_masks["BEECH"]
            & model.str.contains(re.compile(
                r"\bDUKE\b|\bB60\b|\bA60\b|" + any_num([60]) + r"|QUEEN ?AIR|" + any_num([65, 70, 80, 88])
            ))
        )
    )

    return military_cub_exclude | icao_exclude | large_twin_exclude


def classify(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, int]]:
    """Run the full conventional-GA classification pipeline on df.

    Returns (final_mask, stats), where stats has:
      total_rows, rows_kept_previous, rows_removed_new,
      rows_removed_bad_icao24, final_rows
    """
    mfr, model, icao = clean_fields(df)

    base_mask, brand_masks = build_base_mask(mfr, model)
    new_exclude = build_strict_exclude_mask(mfr, model, icao, brand_masks)

    classified_mask = base_mask & ~new_exclude

    hex_ok = valid_icao24_mask(df)
    final_mask = classified_mask & hex_ok

    stats = {
        "total_rows": len(df),
        "rows_kept_previous": int(base_mask.sum()),
        "rows_removed_new": int((base_mask & new_exclude).sum()),
        "rows_removed_bad_icao24": int((classified_mask & ~hex_ok).sum()),
        "final_rows": int(final_mask.sum()),
    }
    return final_mask, stats
