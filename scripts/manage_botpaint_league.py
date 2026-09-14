"""Create/inspect the owner-authorized BOTPAINT league with sanitized evidence."""
import argparse
from types import SimpleNamespace
import json
from pathlib import Path
from datetime import datetime, timezone
from coworld.api_client import CoworldApiClient
from coworld.upload import CoworldUploadClient

SERVER = 'https://softmax.com/api'
OUT = Path('botpaint-league-status.json')
state = {'checked_at': datetime.now(timezone.utc).isoformat(), 'operations': []}

def save():
    OUT.write_text(json.dumps(state, indent=2, default=str)+'\n')

def request(client, method, path, body=None):
    response = client._http_client.request(method, path, headers=client._headers(), json=body, timeout=90)
    item={'method':method,'path':path,'status':response.status_code}
    if response.status_code >= 400:
        try:
            detail=response.json().get('detail')
            # Only report short route errors; never dump credentials, headers or a whole response.
            if isinstance(detail,str) and len(detail)<500 and not any(x in detail.lower() for x in ('bearer','secret','token=')):
                item['detail']=detail
        except ValueError:
            pass
    state['operations'].append(item);save();print(json.dumps(item),flush=True)
    if response.status_code >= 400:
        return None
    return response.json() if response.content else {}

def seed_record(seed):
    return {k:getattr(seed,k) for k in ('id','coworld_name','league_key','league_name','default_variant_id','enabled','league_id')}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inspect','create'])
    args=parser.parse_args()
    with CoworldUploadClient.from_login(server_url=SERVER) as upload, CoworldApiClient.from_login(server_url=SERVER) as api:
        cow=upload.find_canonical_coworld('botpaint')
        if cow is None:
            raise SystemExit('No canonical BOTPAINT version; creation stopped')
        state['coworld']={k:getattr(cow,k) for k in ('id','name','version','canonical','manifest_hash')}
        state['variants']=[{'id':v['id'],'width':v['game_config'].get('width'),'height':v['game_config'].get('height'),'seats':len(v['game_config'].get('players',[])),'turns':v['game_config'].get('max_turns'),'targets':[t['target'] for t in v['game_config'].get('teams',[])]} for v in cow.manifest['variants']]
        save();print(json.dumps(state['coworld']),flush=True)
        raw_seeds=request(upload,'GET','/v2/coworld-league-seeds')
        seeds=[SimpleNamespace(**s) for s in (raw_seeds or []) if s['coworld_name']=='botpaint']
        leagues=[l for l in api.list_leagues() if l.game.coworld_name=='botpaint']
        state['seeds']=[seed_record(s) for s in seeds]
        state['leagues']=[{'id':l.id,'name':l.name,'slug':l.slug,'public':l.public,'hidden':l.hidden,'disabled_at':l.disabled_at,'commissioner_key':l.commissioner_key} for l in leagues]
        save()
        if len(seeds)>1 or len(leagues)>1:
            raise SystemExit('Multiple BOTPAINT leagues; no mutation performed')
        if not seeds and not leagues and args.action=='create':
            data=request(upload,'POST','/v2/coworld-league-seeds',{'coworld_name':'botpaint','league_key':'default','league_name':'BOTPAINT','template':'commissioner_driven','enabled':True,'overrides':{'commissioner_key':'platform'}})
            if data is None:
                state['phase']='creation_rejected';save();raise SystemExit('League creation rejected; see sanitized status')
            seed=SimpleNamespace(**data)
            seeds=[seed];state['seeds']=[seed_record(seed)];state['created']=True;save()
        league_id=seeds[0].league_id if seeds else (leagues[0].id if leagues else None)
        if not league_id:
            state['phase']='no_materialized_league';save();print(json.dumps(state,default=str),flush=True);return
        state['league_id']=league_id
        state['league_url']=f'https://softmax.com/observatory/v2?detail=league:{league_id}'
        data=request(api,'GET',f'/v2/leagues/{league_id}')
        if data:
            state['league']={k:data.get(k) for k in ('id','name','slug','public','hidden','commissioner_key','disabled_at')}
        data=request(api,'GET',f'/v2/leagues/{league_id}/settings')
        if data:
            # Configuration fields only. Secret references and owner identity are intentionally excluded.
            state['settings_keys']=list(data)
            settings=data.get('settings',data)
            state['ladder']=settings.get('ladder')
            state['round_interval_minutes']=settings.get('round_interval_minutes')
            state['effective_ladder_config']=data.get('effective_ladder_config')
        data=request(api,'GET',f'/v2/leagues/{league_id}/locks')
        if data:
            state['locks']=data
        state['phase']='league_created_or_existing';save();print(json.dumps(state,default=str),flush=True)

if __name__=='__main__':
    try:
        main()
    finally:
        save()
