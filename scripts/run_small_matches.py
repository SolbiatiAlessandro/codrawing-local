"""Start three user-authorized two-round matches, with fresh canvases and swapped targets.

Each round is one hosted episode. Stable idempotency keys prevent duplicate
episodes on rerun. Stop admissions on any rejection; never buy credits or retry
with a different key. Artifacts exclude credentials and realized game tokens.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import jsonschema
from coworld.api_client import CoworldApiClient

COWORLD = "cow_acedc9da-d152-4321-ba9a-2b282c2d9815"
POLICY = "agent-bedrock-sonnet46:v9"
MATCHES = [
    ("flags", "french flag", "italian flag"),
    ("sun-moon", "sun", "crescent moon"),
    ("hearts", "red heart", "blue heart"),
]
OUT = Path("small-matches-status.json")


def make_plan(manifest):
    base = next(v["game_config"] for v in manifest["variants"] if v["id"] == "flags")
    assert len(base["players"]) == 6
    assert [t["slots"] for t in base["teams"]] == [[0, 1, 2], [3, 4, 5]]
    plan = []
    for name, target_a, target_b in MATCHES:
        for round_number, targets in enumerate(((target_a, target_b), (target_b, target_a)), 1):
            config = copy.deepcopy(base)
            config.update(width=24, height=16, max_turns=50, seed=1)
            for team, target in zip(config["teams"], targets):
                team["target"] = target
                team["region"] = {"x": 0, "y": 0, "width": 24, "height": 16}
            jsonschema.validate(dict(config, tokens=[f"validation-{s}" for s in range(6)]),
                                manifest["game"]["config_schema"])
            body = {
                "idempotency_key": f"botpaint-20260910-easy-{name}-24x16-round{round_number}",
                "coworld_id": COWORLD,
                "variant_id": "flags",
                "game_config_overrides": {k: config[k] for k in ("width", "height", "max_turns", "seed", "teams")},
                "roster": [{"player": {"policy_ref": POLICY}, "slot": s} for s in range(6)],
                "num_episodes": 1,
                "notes": f"Easy shared-canvas {name} match, round {round_number}/2. Fresh 24x16 canvas, 3v3, 50 turns. Targets swap between rounds.",
            }
            plan.append({"match": name, "round": round_number, "targets": list(targets), "request": body,
                         "phase": "not_submitted"})
    return plan


def save(plan):
    OUT.write_text(json.dumps({"canvas": [24, 16], "turns_per_round": 50, "plan": plan}, indent=2) + "\n")


def summary(result):
    episodes = []
    for episode in result.get("episodes", []):
        row = {k: episode.get(k) for k in ("id", "status", "episode_id", "replay_url", "live_url", "error_type", "error")}
        config = episode.get("game_config") or {}
        row.update(canvas=[config.get("width"), config.get("height")], max_turns=config.get("max_turns"),
                   targets=[t.get("target") for t in config.get("teams", [])])
        episodes.append(row)
    return {"request_id": result["id"], "status": result["status"], "episodes": episodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-local", type=Path)
    args = parser.parse_args()
    if args.validate_local:
        plan = make_plan(json.loads(args.validate_local.read_text()))
        for match in MATCHES:
            first, second = [p for p in plan if p["match"] == match[0]]
            assert first["targets"] == second["targets"][::-1]
        assert len({p["request"]["idempotency_key"] for p in plan}) == 6
        print("Validated six rounds: 24x16, 3v3, 50 turns, fresh state, swapped targets; no network calls.")
        return
    with CoworldApiClient.from_login(server_url="https://softmax.com/api") as c:
        manifest = c._get(f"/v2/coworlds/{COWORLD}", dict)["manifest"]
        plan = make_plan(manifest)
        save(plan)
        for item in plan:
            response = c._http_client.post("/v2/experience-requests", headers=c._headers(), json=item["request"], timeout=120)
            if not response.is_success:
                try:
                    error = response.json().get("detail")
                except (ValueError, AttributeError):
                    error = "Non-JSON error response."
                item.update(phase="rejected", http_status=response.status_code, detail=error)
                save(plan)
                print("ADMISSION_REJECTED", json.dumps({k: item[k] for k in ("match", "round", "http_status", "detail")}), flush=True)
                break
            result = response.json()
            item.update(phase="admitted", cost_preview=result.get("cost_preview"), **summary(result))
            save(plan)
            print("ADMITTED", json.dumps({k: item.get(k) for k in ("match", "round", "request_id", "status", "cost_preview", "episodes")}), flush=True)
        admitted = [p for p in plan if p["phase"] == "admitted"]
        if not admitted:
            raise SystemExit(1)
        for _ in range(320):
            active = False
            for item in admitted:
                result = c._get(f"/v2/experience-requests/{item['request_id']}", dict)
                item.update(summary(result))
                for ep in item["episodes"]:
                    if ep.get("replay_url") and ep.get("episode_id"):
                        session = c.create_replay_session(coworld_id=COWORLD, episode_id=ep["episode_id"], replay_uri=ep["replay_url"])
                        ep.update(viewer_url=session.viewer_url, viewer_ready=session.ready)
                active |= item["status"] not in ("completed", "failed", "cancelled")
                print("STATUS", json.dumps({k: item.get(k) for k in ("match", "round", "request_id", "status", "episodes")}), flush=True)
            save(plan)
            if not active:
                raise SystemExit(0 if len(admitted) == 6 and all(p["status"] == "completed" for p in admitted) else 1)
            time.sleep(30)
        raise SystemExit("Polling ended; inspect saved request IDs. Do not create replacements.")


if __name__ == "__main__":
    main()
