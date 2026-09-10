"""Resolve the platform's static viewer for the existing BOTPAINT flag replay."""
from uuid import UUID
from coworld.api_client import CoworldApiClient

COWORLD = "cow_acedc9da-d152-4321-ba9a-2b282c2d9815"
EPISODE_REQUEST = "ereq_eaadd10c-23f1-404b-ad55-437820b220ca"
EPISODE = "7d8fdb54-ce90-4667-b270-f2a3095de06c"

with CoworldApiClient.from_login(server_url="https://softmax.com/api") as client:
    coworld = client._get(f"/v2/coworlds/{COWORLD}", dict)
    viewer = coworld.get("manifest", {}).get("game", {}).get("replay_viewer", {})
    if not viewer or not viewer.get("bundle"):
        raise SystemExit("No static viewer confirmed; no replay container or game started.")
    episode = client.get_episode_request(EPISODE_REQUEST)
    if episode.coworld_id != COWORLD or str(episode.episode_id) != EPISODE or not episode.replay_url:
        raise SystemExit("Episode identity or replay unavailable; nothing started.")
    session = client.create_replay_session(
        coworld_id=COWORLD, episode_id=UUID(EPISODE), replay_uri=episode.replay_url
    )
    print("STATIC_VIEWER_READY", session.ready, flush=True)
    print("VIEWER_URL", session.viewer_url, flush=True)
    print("Existing episode only; no new game submitted.", flush=True)
