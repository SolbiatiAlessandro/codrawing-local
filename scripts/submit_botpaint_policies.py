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
PLAN = [('botpaint-bedrock-sonnet46', None), ('botpaint-bedrock-haiku45', '@lessandro-forum-power-user')]
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
        'policy': pv.get('name') or pv.get('policy_name'), 'policy_version_id': str(pv.get('id')) if pv.get('id') else None,
        'player_id': pl.get('id'), 'player_name': pl.get('name'), 'auto_champion': s.get('auto_champion'),
    }


def _pick(d, keys):
    return {k: d.get(k) for k in keys if k in d} if isinstance(d, dict) else d


def rounds_report(api):
    """Read-only: champions, ladder, recent rounds and their episodes (sanitized)."""
    mem = request(api, 'GET', '/v2/league-policy-memberships', params={'league_id': LEAGUE, 'active_only': 'true', 'limit': 20})
    entries = mem.get('entries', mem) if isinstance(mem, dict) else (mem or [])
    state['memberships'] = [
        {'id': m.get('id'), 'status': m.get('status'), 'is_champion': m.get('is_champion') or m.get('champion'),
         'division_id': m.get('division_id'), 'player': _pick(m.get('player') or {}, ('id', 'name')),
         'policy_version_id': str((m.get('policy_version') or {}).get('id') or m.get('policy_version_id')),
         'rating': m.get('rating') or m.get('elo'), 'created_at': m.get('created_at')}
        for m in entries if isinstance(m, dict)]
    ladder = request(api, 'GET', f'/v2/leagues/{LEAGUE}/division-ladder')
    if isinstance(ladder, dict):
        state['division_ladder_keys'] = list(ladder)
        state['division_ladder'] = _pick(ladder, ('enabled', 'paused', 'status', 'divisions', 'next_round_at', 'last_round_at', 'round_interval_minutes'))
    rounds = request(api, 'GET', '/v2/rounds', params={'league_id': LEAGUE, 'limit': 10})
    rentries = rounds.get('entries', rounds) if isinstance(rounds, dict) else (rounds or [])
    out = []
    for r in rentries:
        if not isinstance(r, dict):
            continue
        rec = _pick(r, ('id', 'status', 'division_id', 'round_number', 'created_at', 'started_at', 'completed_at', 'settled_at', 'planned_at'))
        eps = request(api, 'GET', f"/v2/rounds/{r.get('id')}/episodes", params={'limit': 10})
        eentries = eps.get('entries', eps) if isinstance(eps, dict) else (eps or [])
        rec['episodes'] = [_pick(e, ('id', 'episode_id', 'status', 'created_at', 'completed_at', 'replay_url', 'variant_id', 'seats')) for e in eentries if isinstance(e, dict)]
        out.append(rec)
    state['rounds'] = out
    state['phase'] = 'rounds_report'
    save()
    print(json.dumps({'memberships': state['memberships'], 'ladder': state.get('division_ladder'), 'rounds': out}, default=str), flush=True)


import os
import re

SECRET_RE = re.compile(r'(sk-or-[A-Za-z0-9_-]+|Bearer\s+\S+|ply_[0-9a-f-]{36}\.\S+|token=\S+)')


def scrub(text):
    return SECRET_RE.sub('[REDACTED]', text)


def diagnose(api):
    """Read-only: why did a round's episodes fail? Sanitized excerpts only."""
    round_id = os.environ.get('ROUND_ID')
    ereq_ids = [x.strip() for x in os.environ.get('EPISODE_REQUEST_IDS', '').split(',') if x.strip()]
    if not round_id and not ereq_ids:
        raise SystemExit('ROUND_ID or EPISODE_REQUEST_IDS is required for diagnose')
    state['round'] = None; state['commissioner_logs'] = None; state['membership_events'] = None
    if round_id:
        rnd = request(api, 'GET', f'/v2/rounds/{round_id}')
        state['round'] = {k: v for k, v in (rnd or {}).items() if any(s in k for s in ('id', 'status', 'number', 'reason', 'error', 'message', 'created', 'completed'))}
        logs = request(api, 'GET', f'/v2/rounds/{round_id}/commissioner-logs')
        state['commissioner_logs'] = scrub(json.dumps(logs, default=str))[-4000:] if logs is not None else None
        ev = request(api, 'GET', '/v2/policy-membership-events', params={'league_id': LEAGUE, 'limit': 20})
        state['membership_events'] = scrub(json.dumps(ev, default=str))[-3000:] if ev is not None else None
        reqs = request(api, 'GET', f'/v2/rounds/{round_id}/episode-requests', params={'limit': 10})
        entries = reqs.get('entries', reqs) if isinstance(reqs, dict) else (reqs or [])
    else:
        entries = [{'id': i} for i in ereq_ids]
    out = []
    for r in entries:
        rid = r.get('id')
        full = request(api, 'GET', f'/v2/episode-requests/{rid}') or {}
        rec = {k: v for k, v in full.items() if any(s in k.lower() for s in ('id', 'status', 'reason', 'error', 'fail', 'message', 'created', 'completed', 'started', 'job', 'variant', 'phase', 'summary'))}
        rec['keys'] = list(full)
        stats = request(api, 'GET', f'/v2/episode-requests/{rid}/episode-stats')
        rec['stats'] = scrub(json.dumps(stats, default=str))[:1500] if stats is not None else None
        rec['policy_logs'] = {}
        for pv_id in sorted(set(full.get('policy_version_ids') or [])):
            for agent_idx in range(8):
                resp = api._http_client.request('GET', f'/v2/episode-requests/{rid}/{pv_id}/policy-logs/{agent_idx}', headers=api._headers(), timeout=120)
                state['operations'].append({'method': 'GET', 'path': f'/v2/episode-requests/{rid}/{pv_id[:8]}/policy-logs/{agent_idx}', 'status': resp.status_code})
                if resp.status_code < 400 and resp.content:
                    text = resp.text
                    rec['policy_logs'][f'{pv_id[:8]}/{agent_idx}'] = {'bytes': len(text), 'head': scrub(text[:1200]), 'tail': scrub(text[-2500:])}
        out.append(rec)
        save()
    state['episode_requests'] = out
    state['phase'] = 'diagnose'
    save()
    print(json.dumps({'round': state['round'], 'episode_requests': out, 'commissioner_logs': state['commissioner_logs'], 'membership_events': state['membership_events']}, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'submit', 'rounds', 'diagnose'])
    args = parser.parse_args()
    with CoworldApiClient.from_login(server_url=SERVER) as api:
        if args.action == 'rounds':
            rounds_report(api); return
        if args.action == 'diagnose':
            diagnose(api); return
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
            rec = {'policy': name, 'player_id': player['id'], 'player_name': player.get('name'),
                   'submission': None if data is None else submission_record(data)}
            # One champion per player: promote the new membership so the previous entry is benched.
            mid = (data or {}).get('league_policy_membership_id')
            if mid:
                promoted = request(api, 'POST', f'/v2/league-policy-memberships/{mid}/champion', {})
                rec['champion'] = {'membership_id': mid, 'ok': promoted is not None,
                                   'status': promoted.get('status') if isinstance(promoted, dict) else None}
            results.append(rec)
            state['results'] = results; save()
        state['results'] = results
        state['phase'] = 'submitted' if any(r.get('submission') for r in results) else 'nothing_submitted'
        save()
        print(json.dumps(results, default=str), flush=True)


if __name__ == '__main__':
    main()
