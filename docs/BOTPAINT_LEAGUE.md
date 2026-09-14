# BOTPAINT league

[Open BOTPAINT on Softmax](https://softmax.com/observatory/v2?detail=league:league_3f6f5062-ce72-44dd-bc9e-7bd2c8682f43).

## Verified September 14, 2026

The public league was created. Its submissions are unlocked, but **matches are not running**. The account can create an owner league, read settings and read submission locks. Creating the Competition division returned HTTP 403: `User is not a softmax team member`. No scheduling settings were changed, no bot was submitted and no experience request was created.

- League: `league_3f6f5062-ce72-44dd-bc9e-7bd2c8682f43`
- Seed: `lseed_6a3c2467-fa74-4d4d-910f-787e96f5aef1`
- Coworld: `botpaint:0.1.0`, `cow_1bb10b94-b130-4029-b50f-990a3fb38f8c`, canonical.
- Public: true; hidden: false; submissions locked: false; ladder enabled: false.
- The hosted release's main viewer and atelier background match the latest local showcase source byte for byte. A duplicate visual-only upload is unnecessary.

## Softmax team activation handoff

Suggested initial format: existing `flags-arena` variant, 24×16 shared canvas, 3v3, 80 turns. The variant is already uploaded. The two submitted policies should each control a complete team. Each pair must play both target assignments.

1. Set this league's default variant to `flags-arena` through the team-authorized seed settings. The current implicit default is the eight-seat split-fruit variant, so verify the resulting six-seat format before scheduling.
2. Declare one Competition division:

   `PUT /v2/leagues/league_3f6f5062-ce72-44dd-bc9e-7bd2c8682f43/divisions`

   Body: `{"divisions":[{"name":"Competition","level":1,"type":"competition","hidden":false}]}`

3. Read existing league settings and merge the ladder proposal below, substituting the returned division ID. Preserve sibling settings; POST replaces the settings document. Save with `enabled: false`, then verify the effective configuration.
4. Confirm automatic-round funding and an initial cadence, enable scheduling, admit two real bot submissions and prove one completed paired round. Verify six seats, contiguous team slots and both side assignments in the actual episode plan. No other person's bot or account access has been verified by this setup.

```json
{
  "ladder": {
    "enabled": false,
    "scheduler": {
      "strategy": "team_pair",
      "team_layout": "blocks",
      "insufficient_players": "do_not_run"
    },
    "fulfillment": {"allowed_failures": 0.0, "retry_times": 2},
    "ranking": {
      "algorithm": "elo",
      "initial_rating": 1500.0,
      "k_factor": 32.0,
      "round_scoring_rule": "mean"
    },
    "divisions": [{
      "division_id": "REPLACE_WITH_CREATED_DIVISION_ID",
      "name": "Competition",
      "disqualify_after_consecutive_failures": 3
    }]
  }
}
```

The proposed 80-turn format is the existing hosted variant. The earlier local showcases used 50 turns and a different scorer. This document does not claim identical scoring or a completed league match. Additional pairs and shorter rounds can be published after activation works.

## Bot submission

For a policy already uploaded to Softmax:

```sh
coworld submit YOUR_POLICY --league league_3f6f5062-ce72-44dd-bc9e-7bd2c8682f43 --no-open-browser
```

The API reports submissions unlocked, but placement and completed matches still require the division/scheduler setup above. See the [official player upload instructions](https://github.com/Metta-AI/coworld/blob/4c26e519040ffe7182104f3417035020acea607b/COOKBOOK.md#upload-and-submit-a-player) for new policies.

## Message to Softmax (draft, not sent)

> I created the public BOTPAINT league: https://softmax.com/observatory/v2?detail=league:league_3f6f5062-ce72-44dd-bc9e-7bd2c8682f43
>
> Owner creation worked, but creating its Competition division returns 403, “User is not a softmax team member.” Could you finish the league setup? We want the `flags-arena` variant (24×16 shared canvas, 3v3), `team_pair` with `team_layout: blocks`, both side assignments, and open bot submissions. Please also confirm how automatic league matches are funded. The separate manual XP-credit reset does not appear to block league creation.

## Evidence

- [Successful owner creation](https://github.com/SolbiatiAlessandro/codrawing-local/actions/runs/34877848133).
- [Rejected division setup](https://github.com/SolbiatiAlessandro/codrawing-local/actions/runs/34877953672).
- [Official platform ladder guide](https://github.com/Metta-AI/coworld/blob/4c26e519040ffe7182104f3417035020acea607b/src/coworld/docs/PLATFORM_LADDER_LEAGUE.md).
- [Official seating rules](https://github.com/Metta-AI/coworld/blob/4c26e519040ffe7182104f3417035020acea607b/src/coworld/docs/LADDER_SEATING.md).

The branch workflow reuses the existing GitHub-held Softmax credential only on the ephemeral runner. It never requests elevated/team privileges. The first diagnostic run stopped on the team-only seed listing; the successful owner creation used the documented independent POST route. Further runs locate the existing public league and do not duplicate it.
