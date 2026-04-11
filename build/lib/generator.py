"""
DataBridgeEnv — Synthetic payload generator.

Generates (broken_payload, target_schema, mismatch_info) tuples for each task.

Mismatch categories (core 5):
  1. field_name      — camelCase vs snake_case, or completely different name
  2. type_coercion   — "123" instead of 123, "49.99" instead of 49.99, etc.
  3. shape_nesting   — nested obj that should be flat, or vice-versa
  4. array_scalar    — comma-separated string instead of list, or vice-versa
  5. missing_extra   — required field absent, or extra unexpected field present
"""

from __future__ import annotations

import random
import re
import string
from typing import Any, Dict, List, Tuple


STATUS_VALUES = [
    "pending",
    "active",
    "cancelled",
    "completed",
    "refunded",
    "on_hold",
    "processing",
]

ACCOUNT_TYPES = ["premium", "standard", "business"]

CATEGORY_POOL = [
    "office",
    "electronics",
    "furniture",
    "clothing",
    "food",
    "toys",
    "fitness",
    "beauty",
    "garden",
    "books",
]

FIRST_NAME_SYLLABLES = ["al", "be", "ca", "di", "el", "fa", "gi", "ha", "jo", "ka", "li", "ma", "na", "or", "pa", "ra", "sa", "ta", "vi", "za"]
LAST_NAME_SYLLABLES = ["son", "ton", "ford", "well", "berg", "man", "field", "wood", "stone", "brook", "line", "croft", "wald", "mont", "sen"]

STREET_PREFIXES = ["Maple", "Cedar", "Pine", "Oak", "Lake", "Hill", "River", "Sunset", "Elm", "Willow"]
STREET_SUFFIXES = ["St", "Ave", "Rd", "Blvd", "Lane", "Way", "Drive", "Court"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_snake_case(key: str) -> str:
    if not isinstance(key, str):
        return str(key)
    key = key.strip()
    key = key.replace("-", "_").replace(" ", "_")
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    key = re.sub(r"__+", "_", key)
    return key.lower().strip("_")


def _rand_user_id(rng: random.Random) -> int:
    return rng.randint(100, 999999)


def _rand_amount(rng: random.Random) -> float:
    return round(rng.uniform(5.0, 5000.0), 2)


def _rand_name(rng: random.Random) -> Tuple[str, str]:
    first = (rng.choice(FIRST_NAME_SYLLABLES) + rng.choice(FIRST_NAME_SYLLABLES)).capitalize()
    last = (rng.choice(LAST_NAME_SYLLABLES) + rng.choice(LAST_NAME_SYLLABLES)).capitalize()
    return first, last


def _rand_city(rng: random.Random) -> str:
    a = rng.choice(FIRST_NAME_SYLLABLES).capitalize()
    b = rng.choice(LAST_NAME_SYLLABLES)
    c = rng.choice(["ville", "town", "ford", "grove", "bury", "side"])
    return f"{a}{b}{c}"


def _rand_street(rng: random.Random) -> str:
    number = rng.randint(1, 9999)
    return f"{number} {rng.choice(STREET_PREFIXES)} {rng.choice(STREET_SUFFIXES)}"


def _rand_token(rng: random.Random, length: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def _rand_items(rng: random.Random, n: int | None = None) -> List[str]:
    k = n if n is not None else rng.randint(1, 5)
    dynamic_pool = [f"{rng.choice(CATEGORY_POOL)[:3]}_{_rand_token(rng, 4).lower()}" for _ in range(max(8, k + 3))]
    rng.shuffle(dynamic_pool)
    return dynamic_pool[:k]


def _rand_timestamp_iso(rng: random.Random) -> str:
    year = rng.randint(2022, 2025)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"


def _iso_to_epoch(iso: str) -> int:
    import datetime

    dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp())


def _discount_for(account_type: str, total_price: float) -> float:
    if account_type == "premium":
        return round(total_price * (0.15 if total_price >= 250 else 0.05), 2)
    if account_type == "business":
        return round(total_price * 0.08, 2)
    return 0.0


def _format_numeric_as_string(rng: random.Random, value: float, allow_currency: bool = False) -> str:
    style = rng.choice(["plain", "fixed2", "comma", "scientific", "space", "currency"])
    if style == "plain":
        return str(value)
    if style == "fixed2":
        return f"{value:.2f}"
    if style == "comma":
        return f"{value:,.2f}"
    if style == "scientific":
        return f"{value:.2e}"
    if style == "space":
        return f" {value:.2f} "
    if allow_currency:
        return f"${value:.2f}"
    return f"{value:.2f}"


# ---------------------------------------------------------------------------
# TASK 1 — Easy
# mismatch types: field_name + type_coercion
# ---------------------------------------------------------------------------

def _generate_easy(rng: random.Random) -> Dict[str, Any]:
    uid = _rand_user_id(rng)
    oid = f"ord_{_rand_token(rng, 6).lower()}"
    amt = _rand_amount(rng)
    qty = rng.randint(1, 5000)
    stat = rng.choice(STATUS_VALUES)

    broken: Dict[str, Any] = {
        "userId": uid,
        "orderId": oid,
    }
    mismatches: List[str] = ["field_name", "type_coercion"]

    # Type-coercion diversity: mostly parseable but sometimes intentionally tricky.
    amount_roll = rng.random()
    if amount_roll < 0.55:
        amount_as_text = f"{amt:.2f}"
    elif amount_roll < 0.75:
        amount_as_text = str(amt)
    elif amount_roll < 0.85:
        amount_as_text = f"{amt:.2e}"
    elif amount_roll < 0.93:
        amount_as_text = f"{amt:,.2f}"
    else:
        amount_as_text = f"${amt:.2f}"

    quantity_roll = rng.random()
    if quantity_roll < 0.55:
        quantity_style = "plain"
    elif quantity_roll < 0.75:
        quantity_style = "float_like"
    elif quantity_roll < 0.85:
        quantity_style = "scientific"
    elif quantity_roll < 0.93:
        quantity_style = "comma"
    else:
        quantity_style = "unit_suffix"

    if quantity_style == "plain":
        quantity_as_text = str(qty)
    elif quantity_style == "float_like":
        quantity_as_text = f"{float(qty):.1f}"
    elif quantity_style == "comma":
        quantity_as_text = f"{qty:,}"
    elif quantity_style == "scientific":
        quantity_as_text = f"{qty:.1e}"
    else:
        quantity_as_text = f"{qty} units"

    broken["amount"] = amount_as_text
    broken["quantity"] = quantity_as_text
    broken["status"] = stat

    # Optional field_name+type_coercion case for diversity checks.
    include_optional_alias_field = rng.random() < 0.35
    if include_optional_alias_field:
        j_val = rng.randint(1, 999)
        broken["jUnk"] = str(j_val)

    target_schema = {
        "user_id": "int",
        "order_id": "str",
        "amount": "float",
        "quantity": "int",
        "status": "str",
    }

    correct_output = {
        "user_id": uid,
        "order_id": oid,
        "amount": amt,
        "quantity": qty,
        "status": stat,
    }

    if include_optional_alias_field:
        target_schema["j_unk"] = "int"
        correct_output["j_unk"] = int(broken["jUnk"])

    return {
        "broken_payload": broken,
        "target_schema": target_schema,
        "correct_output": correct_output,
        "mismatch_types": sorted(set(mismatches)),
        "schema_examples": None,
    }


# ---------------------------------------------------------------------------
# TASK 2 — Medium
# mismatch types: shape_nesting + array_scalar + missing_extra + type_coercion
# ---------------------------------------------------------------------------

def _generate_medium(rng: random.Random) -> Dict[str, Any]:
    uid = _rand_user_id(rng)
    fname, lname = _rand_name(rng)
    items = _rand_items(rng)
    price = _rand_amount(rng)
    category = rng.choice(CATEGORY_POOL)
    account_type = "premium"
    discount_pct = rng.choice([5, 10, 12, 15])
    price_cents = int(round(price * 100))
    discount_value = round(price * (discount_pct / 100), 2)

    mismatches: List[str] = ["shape_nesting", "array_scalar", "missing_extra", "type_coercion"]

    broken: Dict[str, Any] = {
        "user": {
            "id": str(uid),
            "name": f"{fname} {lname}",
        },
        "account_type": account_type,
        "discount_pct": str(discount_pct),
        "category": category,
        "email": f"{fname.lower()}.{lname.lower()}@example.com",
        "items": ",".join(items),
    }

    # Type mismatch with semantic signal.
    if rng.random() < 0.7:
        broken["price"] = str(price_cents)
        broken["price_minor_unit"] = 100
    else:
        broken["price"] = f"{price:.2f}"
        broken["price_minor_unit"] = 1

    # missing_extra mismatch: optional noisy extra field and always-missing required discount
    if rng.random() < 0.6:
        broken["debug_flag"] = rng.choice([True, False])

    target_schema = {
        "user_id": "int",
        "user_name": "str",
        "email": "str",
        "items": "list[str]",
        "price": "float",
        "account_type": "str",
        "discount": "float",
        "category": "str",
    }

    correct_output = {
        "user_id": uid,
        "user_name": f"{fname} {lname}",
        "email": f"{fname.lower()}.{lname.lower()}@example.com",
        "items": items,
        "price": price,
        "account_type": account_type,
        "discount": discount_value,
        "category": category,
    }

    return {
        "broken_payload": broken,
        "target_schema": target_schema,
        "correct_output": correct_output,
        "mismatch_types": sorted(set(mismatches)),
        "schema_examples": None,
    }


# ---------------------------------------------------------------------------
# TASK 3 — Hard
# mismatch types: all 5, schema inferred from examples
# ---------------------------------------------------------------------------

def _make_hard_pair(seed_id: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rng = random.Random(seed_id)

    uid = _rand_user_id(rng)
    fname, lname = _rand_name(rng)
    items = _rand_items(rng, n=rng.randint(1, 4))
    price = _rand_amount(rng)
    price_cents = int(round(price * 100))
    iso_ts = _rand_timestamp_iso(rng)
    epoch = _iso_to_epoch(iso_ts)
    tags = _rand_items(rng, n=2)

    account_type_raw = rng.choice(["premium", "Premium", " premium ", "standard", "business"])
    account_type = account_type_raw.strip().lower()
    discount = _discount_for(account_type, price)

    price_mode = rng.choice(["cents", "dollars"])
    if price_mode == "cents":
        price_raw = f"{price_cents:,}" if rng.random() < 0.5 else str(price_cents)
    else:
        price_raw = _format_numeric_as_string(rng, price, allow_currency=False)

    price_minor_unit = rng.choice([1, 10, 100, 1000])
    if price_mode == "cents" and rng.random() < 0.5:
        price_minor_unit = 100
    if price_mode == "dollars" and rng.random() < 0.5:
        price_minor_unit = 1

    full_name_raw = f"{fname} {lname}"
    if rng.random() < 0.5:
        full_name_raw = f"  {fname}  {lname}  "

    broken = {
        "userId": f"{uid:,}" if rng.random() < 0.5 else str(uid),
        "fullName": full_name_raw,
        "totalPrice": price_raw,
        "accountType": account_type_raw,
        "address": {
            "street": _rand_street(rng),
            "city": _rand_city(rng),
        },
        "orderItems": ",".join(items),
        "tags": tags,
        "createdAt": f" {iso_ts} " if rng.random() < 0.5 else iso_ts,
        "uiLocale": rng.choice(["en-US", "EN-us", " en-US "]),
        "priceMinorUnit": price_minor_unit,
    }

    edge_case = seed_id % 3
    if edge_case == 0:
        broken["totalPrice"] = "1,20,000"
        broken["priceMinorUnit"] = 100
        broken["amount"] = "1,20,000"
        price = 1200.0
    elif edge_case == 1:
        broken["totalPrice"] = "500"
        broken["priceMinorUnit"] = 1000
        broken["amount"] = "500"
        price = 500.0
    else:
        broken.pop("userId", None)
        broken["user"] = {"id": str(uid)}
        broken["accountType"] = "premium"
        account_type = "premium"

    discount = _discount_for(account_type, price)

    generated_output = {
        "userId": uid,
        "fullName": f"{fname} {lname}",
        "userEmail": f"{fname.lower()}.{lname.lower()}@example.com",
        "totalPrice": price,
        "streetAddress": broken["address"]["street"],
        "cityName": broken["address"]["city"],
        "orderItems": items,
        "itemTags": tags,
        "accountType": account_type,
        "discountValue": discount,
        "createdAt": epoch,
    }
    if account_type == "premium" and price >= 420:
        generated_output["handlingTier"] = "priority"

    correct = {to_snake_case(k): v for k, v in generated_output.items()}
    return broken, correct


def _generate_hard(rng: random.Random) -> Dict[str, Any]:
    examples = []
    # Use independent seeds for examples so hard-task schema inference sees diverse patterns.
    for seed in rng.sample(range(1000), 3):
        b, c = _make_hard_pair(seed)
        examples.append({"broken_input": b, "correct_output": c})

    uid = _rand_user_id(rng)
    fname, lname = _rand_name(rng)
    items = _rand_items(rng)
    price = _rand_amount(rng)
    price_cents = int(round(price * 100))
    iso_ts = _rand_timestamp_iso(rng)
    epoch = _iso_to_epoch(iso_ts)
    tags = _rand_items(rng, n=rng.randint(1, 3))

    account_type_raw = rng.choice(["premium", "Premium", " premium ", "standard", "business"])
    account_type = account_type_raw.strip().lower()

    price_mode = rng.choice(["cents", "dollars"])
    if price_mode == "cents":
        price_raw = f"{price_cents:,}" if rng.random() < 0.5 else str(price_cents)
    else:
        price_raw = _format_numeric_as_string(rng, price, allow_currency=False)

    price_minor_unit = rng.choice([1, 10, 100, 1000])
    if price_mode == "cents" and rng.random() < 0.5:
        price_minor_unit = 100
    if price_mode == "dollars" and rng.random() < 0.5:
        price_minor_unit = 1

    full_name_raw = f"{fname} {lname}"
    if rng.random() < 0.5:
        full_name_raw = f"  {fname}  {lname}  "

    broken: Dict[str, Any] = {
        "userId": f"{uid:,}" if rng.random() < 0.5 else str(uid),
        "fullName": full_name_raw,
        "totalPrice": price_raw,
        "accountType": account_type_raw,
        "address": {
            "street": _rand_street(rng),
            "city": _rand_city(rng),
        },
        "orderItems": ",".join(items),
        "createdAt": f" {iso_ts} " if rng.random() < 0.5 else iso_ts,
        "tags": tags,
        "priceMinorUnit": price_minor_unit,
    }

    edge_case = rng.randint(0, 2)
    if edge_case == 0:
        broken["totalPrice"] = "1,20,000"
        broken["priceMinorUnit"] = 100
        broken["amount"] = "1,20,000"
        price = 1200.0
    elif edge_case == 1:
        broken["totalPrice"] = "500"
        broken["priceMinorUnit"] = 1000
        broken["amount"] = "500"
        price = 500.0
    else:
        broken.pop("userId", None)
        broken["user"] = {"id": str(uid)}
        broken["accountType"] = "premium"
        account_type = "premium"

    discount = _discount_for(account_type, price)

    generated_output = {
        "userId": uid,
        "fullName": f"{fname} {lname}",
        "userEmail": f"{fname.lower()}.{lname.lower()}@example.com",
        "totalPrice": price,
        "streetAddress": broken["address"]["street"],
        "cityName": broken["address"]["city"],
        "orderItems": items,
        "accountType": account_type,
        "discountValue": discount,
        "createdAt": epoch,
        "itemTags": tags,
    }
    if account_type == "premium" and price >= 420:
        generated_output["handlingTier"] = "priority"

    correct_output = {to_snake_case(k): v for k, v in generated_output.items()}

    mismatches = ["field_name", "type_coercion", "shape_nesting", "array_scalar", "missing_extra"]

    return {
        "broken_payload": broken,
        "target_schema": None,
        "correct_output": correct_output,
        "mismatch_types": mismatches,
        "schema_examples": examples,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TASK_GENERATORS = {
    "easy": _generate_easy,
    "medium": _generate_medium,
    "hard": _generate_hard,
}


def generate_episode(task_id: str, seed: int = 42) -> Dict[str, Any]:
    """
    Generate a full episode dict for the given task_id.

    Returns:
        {
          broken_payload: dict,
          target_schema:  dict | None,
          correct_output: dict,
          mismatch_types: list[str],
          schema_examples: list | None,
        }
    """
    if task_id not in TASK_GENERATORS:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_GENERATORS)}")
    rng = random.Random(seed)
    return TASK_GENERATORS[task_id](rng)
