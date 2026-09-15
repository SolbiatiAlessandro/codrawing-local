"""Submit the two BOTPAINT league policies under two distinct players, with sanitized evidence.

`status` only reads. `submit` creates the second player if needed and submits each policy
that has no live submission yet. Never prints tokens, headers or raw responses.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from coworld.api_client import CoworldApiClient

SERVER = 'https://softmax.com/api'
LEAGUE = 'league_3f6f5062-ce72-44dd-bc9e-7bd2c8682f43'
# (policy name, player display name; None = the account's default player)
PLAN = [('botpaint-sonnet5', None), ('botpaint-flashlite', 'BOTPAINT Flash Lite')]
OUT = Path('botpaint-submissions-status.json')
state = {'checked_at': datetime.now(timezone.utc).isoformat(), 'operations': [], 'league_id': LEAGUE}


def save():
    OUT.write_text(json.dumps(state, indent=2, default=str) + '\n')


def request(api, method, path, body=None, params=None):
    response = api._http_client.request(method, path, headers=api._headers(), json=body, params=params, timeout=120)
    item = {'method': method, 'path': path, 'status': response.status_code}
    if response.status_code >= 400:
        try:
            detail = response.json().get('detail')
            if isinstance(detail, str) and len(detail) < 500 and not any(x in detail.lower() for x in ('bearer', 'secret', 'token=')):
                item['detail'] = detail
        except ValueError:
            pass
    state['operations'].append(item)
    save()
    print(json.dumps(item), flush=True)
    if response.status_code >= 400:
        return None
    return response.json() if response.content else {}


def player_record(p):
    return {k: p.get(k) for k in ('id', 'name', 'is_default', 'disabled_at')}


def submission_record(s):
    pv = s.get('policy_version') or {}
    pl = s.get('player') or {}
    return {
        'id': s.get('id'), 'status': s.get('status'), 'created_at': s.get('created_at'),
        'policy': pv.get('name') or pv.get('policy_name'), 'policy_version_id': pv.get('id'),
        'player_id': pl.get('id'), 'player_name': pl.get('name'), 'auto_champion': s.get('auto_champion'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'submit'])
    args = parser.parse_args()
    with CoworldApiClient.from_login(server_url=SERVER) as api:
        players = [p for p in (request(api, 'GET', '/players') or []) if not p.get('disabled_at')]
        state['players'] = [player_record(p) for p in players]
        versions = {}
        for name, _ in PLAN:
            pv = api.lookup_policy_version(name=name, version=None)
            versions[name] = None if pv is None else {'id': str(pv.id), 'version': getattr(pv, 'version', None)}
        state['policy_versions'] = versions
        page = api.list_submissions(league_id=LEAGUE, mine=True, limit=50)
        existing = [submission_record(s.model_dump()) for s in page.entries]
        state['existing_submissions'] = existing
        save()
        print(json.dumps({'players': state['players'], 'policy_versions': versions, 'existing': existing}, default=str), flush=True)
        if args.action == 'status':
            state['phase'] = 'status_only'; save(); return

        default = next((p for p in players if p.get('is_default')), players[0] if players else None)
        if default is None:
            raise SystemExit('No player on this account; nothing submitted')
        results = []
        for name, player_name in PLAN:
            pv = versions.get(name)
            if not pv:
                results.append({'policy': name, 'skipped': 'policy not found'}); continue
            if player_name is None:
                player = default
            else:
                player = next((p for p in players if p.get('name') == player_name), None)
                if player is None:
                    others = [p for p in players if not p.get('is_default')]
                    if others:
                        player = others[0]
                    else:
                        created = request(api, 'POST', '/players', {'name': player_name})
                        if created is None:
                            results.append({'policy': name, 'skipped': 'player creation rejected'}); continue
                        player = created; players.append(created)
                        state['players'] = [player_record(p) for p in players]
            already = [s for s in existing if s['policy_version_id'] == pv['id'] and s['status'] not in ('rejected', 'withdrawn', 'retired')]
            if already:
                results.append({'policy': name, 'player_id': player['id'], 'skipped': 'already submitted', 'submission_id': already[0]['id']}); continue
            data = request(api, 'POST', '/v2/league-submissions', {
                'league_id': LEAGUE, 'policy_version_id': pv['id'], 'player_id': player['id'],
                'auto_champion': 'always', 'notes': f'BOTPAINT league entry: {name}',
            })
            results.append({'policy': name, 'player_id': player['id'], 'player_name': player.get('name'),
                            'submission': None if data is None else submission_record(data)})
            state['results'] = results; save()
        state['results'] = results
        state['phase'] = 'submitted' if any(r.get('submission') for r in results) else 'nothing_submitted'
        save()
        print(json.dumps(results, default=str), flush=True)


if __name__ == '__main__':
    main()
