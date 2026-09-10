"""List existing BOTPAINT replays and resolve only their static hosted viewers.

The saved inventory contains public game metadata, never replay configs or tokens.
This does not create games, modify policies, or start replay containers.
"""
import json
from collections import Counter
from pathlib import Path

from coworld.api_client import CoworldApiClient

inventory = []
with CoworldApiClient.from_login(server_url="https://softmax.com/api") as client:
    coworlds = []
    cursor = None
    while True:
        params = {"mine": "true", "limit": 500}
        if cursor:
            params["cursor"] = cursor
        r = client._http_client.get("/v2/coworlds", headers=client._headers(), params=params)
        r.raise_for_status()
        for row in r.json():
            if any(word in row.get("name", "").lower() for word in ("codrawing", "coplace", "co/place", "botpaint")):
                coworlds.append(row)
        cursor = r.headers.get("X-Next-Cursor")
        if not cursor:
            break
    print("MATCHED_COWORLDS", [(x["id"], x["name"], x["version"]) for x in coworlds], flush=True)
    seen = set()
    for coworld in coworlds:
        cow_id = coworld["id"]
        detail = client._get(f"/v2/coworlds/{cow_id}", dict)
        static_viewer = (detail.get("manifest", {}).get("game", {}).get("replay_viewer") or {}).get("bundle")
        cursor = None
        statuses = Counter()
        while True:
            page = client.list_coworld_episode_requests(cow_id, limit=100, cursor=cursor)
            for summary in page.entries:
                statuses[summary.status] += 1
                if not summary.replay_url or summary.id in seen:
                    continue
                seen.add(summary.id)
                ep = client.get_episode_request(summary.id)
                if not ep.episode_id or ep.coworld_id != cow_id:
                    continue
                item = {
                    "request_id": ep.id,
                    "episode_id": str(ep.episode_id),
                    "coworld_id": cow_id,
                    "coworld_name": coworld["name"],
                    "version": coworld["version"],
                    "status": ep.status,
                    "created_at": ep.created_at.isoformat(),
                    "replay_url": ep.replay_url,
                    "policy_refs": sorted({getattr(p, "label", "unknown") for p in ep.participants}),
                    "viewer_url": None,
                }
                if static_viewer:
                    session = client.create_replay_session(coworld_id=cow_id, episode_id=ep.episode_id, replay_uri=ep.replay_url)
                    item["viewer_url"] = session.viewer_url
                    item["viewer_ready"] = session.ready
                # Public replay fetched without authentication; no config or seat token is emitted.
                replay_response = client._http_client.get(ep.replay_url, timeout=30)
                item["replay_http_status"] = replay_response.status_code
                if replay_response.status_code == 200:
                    replay = replay_response.json()
                    config = replay.get("config", {})
                    teams = config.get("teams", [])
                    item.update({
                        "targets": [t.get("target") for t in teams],
                        "team_sizes": [len(t.get("slots", [])) for t in teams],
                        "regions": [t.get("region") for t in teams],
                        "canvas": [config.get("width"), config.get("height")],
                        "max_turns": config.get("max_turns"),
                        "frames": len(replay.get("frames", [])),
                    })
                inventory.append(item)
                print("REPLAY", json.dumps(item), flush=True)
            cursor = page.next_cursor
            if not cursor:
                break
        print("COWORLD_STATUS", cow_id, dict(statuses), flush=True)
Path("hosted-replay-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
print("TOTAL_REPLAYS", len(inventory), flush=True)
print("Existing replays only; no game or model run submitted.", flush=True)
