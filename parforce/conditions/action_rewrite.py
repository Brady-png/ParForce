"""Rewrite a first-frame action as a short sequence of grounded controls."""

from pathlib import Path
import json
import re

POLICY = Path(__file__).with_name("policies") / "action-rewrite.txt"
ROUTES = {"rigid", "liquid_pour", "cloth_grip", "cloth_wind"}
DIRECTIONS = {"left", "right", "up", "down", "away", "toward_camera", "toward_target", "inward", "across", "into", "none"}
STRENGTHS = {"gentle", "moderate", "firm"}
SLOW_POUR = re.compile(r"\b(slow(?:ly)?|low flow(?: rate)?|slight tilt)\b", re.I)
BELOW_RIM = re.compile(r"\bbefore\b[^.]{0,90}\b(?:rim|brim)\b", re.I)
SIMPLE_GARMENT_FOLD = re.compile(
    r"\s*(?:please\s+)?fold\s+(?:(?:the|this)\s+)?(?:\w+[- ]+){0,2}"
    r"(?:t-?shirt|shirt|skirt|garment|clothes)\s*[.!]?\s*(?:fixed camera[.!]?\s*)?",
    re.I,
)
ROUTE_ACTIONS = {
    "rigid": {"push", "pull", "strike", "release", "gravity"},
    "liquid_pour": {"pour"},
    "cloth_grip": {"fold", "pull", "lift", "hold", "release", "return"},
    "cloth_wind": {"blow", "hold"},
}


def read_rewrite(raw, allowed_routes=ROUTES, original_prompt=""):
    """Check the model's typed rewrite before geometry work starts."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.partition("\n")[2].rsplit("```", 1)[0].strip()
    plan = json.loads(raw)
    fields = {"route", "strength", "anchors", "phases", "stop_condition"}
    if not isinstance(plan, dict) or not {"route", "phases"} <= plan.keys() <= fields:
        raise ValueError("Expected route, strength, anchors, and phases")
    route = plan["route"]
    external = re.split(r"\bexternal force\s*:", original_prompt, flags=re.I)[-1]
    if "strength" not in plan:
        degree = re.search(r"\b(gentl\w*|slow(?:ly)?|firm\w*|strong\w*|hard|sharp\w*|brisk\w*)\b", external, re.I)
        default = "moderate"
        if degree:
            default = "gentle" if degree[0].lower().startswith(("gentl", "slow")) else "firm"
        plan["strength"] = default
    if route in ("rigid", "liquid_pour"):
        plan.setdefault("anchors", [])
    if route == "liquid_pour" and SLOW_POUR.search(external):
        plan["strength"] = "gentle"
    if route == "liquid_pour" and BELOW_RIM.search(external):
        plan["stop_condition"] = "before_rim"
    elif "stop_condition" in plan:
        raise ValueError("Stop conditions must match an explicit instruction in the input")
    if route not in allowed_routes or plan["strength"] not in STRENGTHS:
        raise ValueError("Action does not match the supported route or strength")
    if route == "cloth_grip" and not plan.get("anchors") and SIMPLE_GARMENT_FOLD.fullmatch(external):
        plan["anchors"] = []
        plan["phases"] = [[dict(op="fold", target=target, directions=[direction], destination=None)]
                          for target, direction in (("left edge", "right"), ("right edge", "left"),
                                                    ("lower hem", "up"))]
    anchors = plan["anchors"]
    phases = plan["phases"]
    if not isinstance(anchors, list) or len(anchors) > 4 or not all(clean_label(s) for s in anchors):
        raise ValueError("Expected short stationary attachment labels")
    if not isinstance(phases, list) or not 1 <= len(phases) <= 4:
        raise ValueError("Expected one to four ordered phases")
    for phase in phases:
        if not isinstance(phase, list) or not 1 <= len(phase) <= 3:
            raise ValueError("Expected one to three simultaneous actions per phase")
        for action in phase:
            if not isinstance(action, dict) or set(action) != {"op", "target", "directions", "destination"}:
                raise ValueError("Invalid action record")
            if route == "cloth_grip" and action["op"] == "lower" and re.search(
                    r"\b(?:back|starting position|initial position)\b", external, re.I):
                action["op"] = "return"
            if route == "cloth_grip" and action["op"] == "move" and re.search(
                    r"\b(?:grasp|hold|grip|grab)\w*\b", external, re.I):
                action["op"] = "pull"
            directions = action["directions"]
            if isinstance(directions, str):
                directions = action["directions"] = [directions]
            if isinstance(directions, list) and action["op"] == "lift" and "up" not in directions:
                directions = action["directions"] = ["up", *[d for d in directions if d != "none"]]
            if action["op"] not in ROUTE_ACTIONS[route] or not clean_label(action["target"]):
                raise ValueError("Unsupported action or missing target")
            if (not isinstance(directions, list) or not 1 <= len(directions) <= 3
                    or any(d not in DIRECTIONS for d in directions)
                    or len(set(directions)) != len(directions)
                    or ("none" in directions and directions != ["none"])):
                raise ValueError("Invalid direction list")
            destination = action["destination"]
            if destination is not None and not clean_label(destination):
                raise ValueError("Invalid destination")
            if any(d in directions for d in ("toward_target", "into")) and destination is None:
                raise ValueError("A named destination is required")
            if action["op"] in ("hold", "release"):
                action["directions"] = ["none"]
    if route == "cloth_grip":
        for phase in phases:
            merge_cloth_controls(phase)
    if route == "liquid_pour" and (len(phases) != 1 or len(phases[0]) != 1):
        raise ValueError("A pouring instruction has one pour into the named receiving vessel")
    if route == "cloth_wind" and sum(a["op"] == "blow" for phase in phases for a in phase) != 1:
        raise ValueError("A wind instruction has one continuous wind action")
    if route == "cloth_grip" and re.search(r"\b(fixed|stationary)\b", external, re.I):
        held = [a["target"] for a in phases[0] if a["op"] == "hold"]
        moved = {a["target"] for phase in phases for a in phase if a["op"] not in ("hold", "release")}
        destinations = {a["destination"] for phase in phases for a in phase if a["op"] in ("fold", "lift", "pull")}
        new_anchors = [part for part in held if part in destinations and part not in moved]
        if new_anchors or anchors:
            plan["anchors"] = list(dict.fromkeys([*anchors, *new_anchors]))
            plan["phases"] = [kept for phase in phases if (kept := [
                a for a in phase if not (a["op"] == "hold" and a["target"] in plan["anchors"])
            ])]
    return plan


def merge_cloth_controls(phase):
    """Combine simultaneous lift and pull commands for the same grip."""
    moving = {}
    for action in phase[:]:
        if action["op"] not in {"lift", "pull", "fold"}:
            continue
        target = action["target"]
        key = re.sub(r"[ _-]+", " ", target).casefold()
        if key not in moving:
            moving[key] = action
            continue
        first = moving[key]
        destinations = {x for x in (first["destination"], action["destination"]) if x is not None}
        directions = list(dict.fromkeys(d for d in first["directions"] + action["directions"] if d != "none"))
        if len(destinations) > 1 or len(directions) > 3:
            raise ValueError("One moving grip cannot target different destinations in the same phase")
        first["op"] = "fold" if "fold" in {first["op"], action["op"]} else "pull"
        first["directions"] = directions
        first["destination"] = next(iter(destinations), None)
        phase.remove(action)


def clean_label(value):
    return (isinstance(value, str) and value == value.strip() and 1 <= len(value) <= 72
            and 1 <= len(value.split()) <= 10 and not any(c in value for c in "\r\n:;.!?"))


def describe_rewrite(plan):
    phases = []
    for phase in plan["phases"]:
        simultaneous = []
        for move in phase:
            directions = " + ".join(d for d in move["directions"] if d != "none")
            phrase = f"{move['op']} {move['target']}"
            if directions:
                phrase += f" [{directions}]"
            if move["destination"]:
                phrase += f" ({move['destination']})"
            simultaneous.append(phrase)
        phases.append(" and ".join(simultaneous))
    summary = " → ".join(phases)
    if plan["route"] == "liquid_pour" and plan["strength"] == "gentle":
        summary = "gently " + summary
    if plan.get("stop_condition") == "before_rim":
        summary += " → stop before the liquid reaches the rim"
    if plan["anchors"]:
        summary += f". Keep {', '.join(plan['anchors'])} fixed."
    return summary


def rewrite_case(model, processor, first, source, receipt_dir, allowed_routes=ROUTES):
    """Use the offline VLM and preserve the input, response, and normalized plan."""
    from .firstview_v2 import digest
    from .material_agent import model_request
    from .run import write_json

    receipt_dir.mkdir(parents=True, exist_ok=True)
    policy = POLICY.read_text()
    identity = {"prompt": source["prompt"], "frame_sha256": digest(source["image"]),
                "rewrite_policy_sha256": digest(POLICY)}
    receipt = receipt_dir / "REWRITE.json"
    if receipt.is_file():
        previous = json.loads(receipt.read_text())
        if all(previous.get(k) == v for k, v in identity.items()):
            normalized = read_rewrite(json.dumps(previous["plan"]), allowed_routes, source["prompt"])
            if normalized != previous["plan"]:
                previous.update(plan=normalized, display=describe_rewrite(normalized))
                write_json(receipt, previous)
            return previous
        raise ValueError("Rewrite receipt belongs to a different action, first frame, or policy")
    task = f"{policy}\nAllowed routes: {', '.join(sorted(allowed_routes))}.\nOriginal input:\n{source['prompt']}"
    fault = ""
    first = first.copy()
    first.thumbnail((512, 384))
    for attempt in range(1, 4):
        raw = model_request(model, processor, [first], task + fault, 320,
                            receipt_dir / f"REWRITE_TOKENS_{attempt}.json")
        write_json(receipt_dir / f"REWRITE_RESPONSE_{attempt}.json", {"raw": raw, "request": task + fault})
        try:
            plan = read_rewrite(raw, allowed_routes, source["prompt"])
        except (TypeError, ValueError) as error:
            fault = f"\nThe previous response failed validation. {error}. Return schema-valid JSON only."
            continue
        record = {**identity, "plan": plan, "display": describe_rewrite(plan), "attempt": attempt}
        write_json(receipt, record)
        return record
    raise ValueError(f"No valid action rewrite: {fault.strip()}")


def compile_rigid_action(perception, source, rewrite):
    """Apply a single rewritten intervention to grounded rigid-body observations."""
    from .firstview_v2 import digest

    plan = rewrite["plan"]
    if plan["route"] != "rigid" or plan["anchors"]:
        raise ValueError("This rigid solver expects a single intervention and a static support graph")
    steps = [move for phase in plan["phases"] for move in phase]
    if len(steps) > 2 or (len(steps) == 2 and steps[-1]["op"] != "release"):
        raise ValueError("This rigid solver can drive one body, then release it")
    move = steps[0]
    row = {**perception, **source, "first_frame_sha256": digest(source["image"]),
           "material": "rigid", "action_rewrite": rewrite}
    directions = move["directions"]
    if move["op"] == "gravity":
        row.update(actuation_type="gravity", axis=0., contact_target_actor=None)
    elif move["op"] in ("push", "pull", "strike"):
        if move["op"] != "push":
            row["requested_actuation"] = move["op"]
        if directions in (["left"], ["right"]):
            sign = -1. if directions[0] == "left" else 1.
            external = re.split(r"\bexternal force\s*:", source["prompt"], flags=re.I)[-1]
            explicit = re.search(r"\b(?:to|toward|towards) (?:the )?(left|right)\b(?! die| cup| ball| edge| side)", external, re.I)
            if explicit and directions[0] != explicit.group(1).lower():
                raise ValueError("Rewritten direction conflicts with the original action")
            row.update(axis=sign)
        elif directions != ["toward_target"] or not (
                row.get("contact_target_actor") is not None or row.get("axis")):
            raise ValueError("The rewritten direction needs a resolved horizontal direction or contact")
        row["actuation_type"] = "push"
    else:
        raise ValueError("Missing supported rigid control")
    row["grounding_note"] = "Uses supplied first-frame perception and its calibrated actuator profile"
    return row
