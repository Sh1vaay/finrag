# Upgrade Playbook

Cookbook for "the maintainer pushed a change, how do I get it?" — one entry per common scenario. Most paths reduce to `finrag upgrade`; a few require an extra step (corpus re-ingest, CLI republish, breaking-change negotiation).

The premise: **the project is architected so the same user command works for ~90% of realistic future changes**.

## The default workflow

```bash
finrag upgrade
```

What that runs:

1. `git status` in `~/.finrag/repo` (or current project root) — refuses if working tree is dirty (override with `--force`)
2. `git pull --ff-only` (still useful for `scripts/`, `docs/`, `.env.example` updates)
3. `docker compose pull` — pulls the pre-built API image matching your CLI version, plus refreshes qdrant / postgres / redis pinned images
4. `docker compose up -d --force-recreate api` (restarts the api with the new image; preserves volumes)
5. Polls `/v1/health` until 200 (or 6 min timeout)

Total wall time: ~90 seconds on first pull, ~10 seconds on subsequent (image cached locally).

Data preservation: all named volumes survive (`pg_data`, `qdrant_data`, `redis_data`, `hf_cache`) plus the host-mounted `cost_logs/` and `logs/`. Chat history, ingested corpora, cost trail all intact.

### Building from source instead of pulling

For dev work, pre-release verification, or if remote registry is unreachable:

```bash
finrag upgrade --build
# or
BUILD_FROM_SOURCE=1 finrag upgrade
```

This runs `docker compose build api` locally. Useful when you've modified `src/` and want to test before tagging a release.

### After upgrade — clean up the old local image

Each pull leaves the previous version's image in your local Docker cache (~2.5 GB per version). To free disk:

```bash
docker image prune -a -f --filter "until=24h"
```

This keeps containers / images used in the last 24h, evicts the older ones. Run periodically.

## Recipes by change type

### "I tweaked a prompt template"

```bash
finrag upgrade
```

That's it. The new prompt ships with the rebuilt api image. Zero data migration.

### "I changed the generator model (e.g. claude-sonnet-4-6 → claude-sonnet-5)"

If the change is in `src/services/llm_factory.py`:

```bash
finrag upgrade
```

If you want to override without rebuilding (faster):

```bash
# Edit .env in your cloned repo:
GENERATOR_MODEL=claude-sonnet-5
# Then:
docker compose -f compose.minimal.yml restart api
```

The api will pick up the env change on restart. Verify with `finrag status` → "Backend semver" + the boot banner (`docker compose logs api | grep "generator:"`).

### "I added a new LangGraph node"

```bash
finrag upgrade
```

Node graph changes ship with the api image rebuild. The CLI is unaffected — it sees the same SSE events. If the new node should surface progress in the CLI, the maintainer also added it to `_NODE_LABELS` in `src/api/routes/chat.py` so the spinner label updates.

### "I swapped the embedding model" (DESTRUCTIVE — re-ingest required)

Cross-embedding queries return garbage (incompatible vector spaces). The boot banner's fingerprint check warns when the live `EMBEDDING_PROVIDER` doesn't match what's stored in the Qdrant collection sentinel:

```
WARNING: collection 'financial_docs' was created with voyage-finance-2 (1024-dim)
WARNING: current EMBEDDING_PROVIDER is openai (text-embedding-3-small, 1536-dim)
WARNING: queries will return garbage or fail with VectorDimensionError.
WARNING: re-ingest the corpus with the new embedding to fix.
```

Fix:

```bash
finrag upgrade

# Drop the stale collection and re-seed
docker compose -f compose.minimal.yml exec api python -c "
from qdrant_client import QdrantClient
from src.config.settings import settings
QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT).delete_collection(settings.QDRANT_COLLECTION)
"
docker compose -f compose.minimal.yml exec api python scripts/seed_qdrant.py --sample
```

Time: ~60s for the 8-PDF sample.

### "I added a new endpoint (additive, backwards-compatible)"

```bash
finrag upgrade
```

Old CLI keeps working — it ignores endpoints it doesn't call. To use the new endpoint from the CLI, the maintainer also added a slash command or subcommand and republished the CLI:

```bash
pip install --upgrade finrag-agent
```

### "I added a new RBAC role"

```bash
finrag upgrade
```

Per [src/services/roles_service.py](../src/services/roles_service.py), roles are DB-first with a fallback to `ROLE_PERMISSIONS` in `src/config/rbac_config.py`. New roles can be added via:

- Editing `ROLE_PERMISSIONS` and rebuilding (ships with the api image)
- OR using `POST /v1/admin/roles` while the api is running (dynamic, no rebuild)

Approval hierarchy (`CAN_APPROVE_FOR`) lives in source code — to grant a new role approval authority, edit `src/config/rbac_config.py` and `finrag upgrade`.

### "I'm doing a breaking API change"

(Rare. The plan is to keep `/v1/*` stable.)

The right pattern:

1. Add new endpoints under `/v2/` prefix while keeping `/v1/` working.
2. Update the CLI to support BOTH versions; bump CLI semver.
3. The CLI's `Accept` header (`application/vnd.finrag.v1+json`) gives the backend a chance to disambiguate.
4. Eventually deprecate v1 over one or two minor versions.
5. Users on the old CLI get a clear error message pointing to `pip install --upgrade finrag-agent`.

For the user, the upgrade command stays the same:

```bash
finrag upgrade
pip install --upgrade finrag-agent
```

### "I'm upgrading cache directory volume permissions"

If cache volume permissions are corrupted, wipe and rebuild:

```bash
cd ~/.finrag/repo
git pull
docker compose -f compose.minimal.yml down -v   # -v removes named volumes
docker compose -f compose.minimal.yml up -d --build
```

`down -v` also wipes Qdrant (your seeded corpus), Postgres (HITL state), and Redis (result cache). The wizard's `_seed_sample_corpus` step re-seeds the sample 8 PDFs automatically on the next `finrag setup` (~60s, ~$0.0005 in embeddings).

Verify the fix landed by inspecting the volume permissions after the rebuild:

```bash
docker exec repo-api-1 stat -c '%U:%G %a' /home/appuser/.cache/huggingface
# Should print: appuser:appuser 755
```

### "I want to roll back an upgrade"

The upgrade command doesn't snapshot — rollback is manual:

```bash
cd ~/.finrag/repo
git log --oneline -10               # find the prior commit
git checkout <prior-commit-sha>
docker compose -f compose.minimal.yml build api
docker compose -f compose.minimal.yml up -d --force-recreate api
```

Data volumes are unaffected (rollback only changes code, not data).

## What `upgrade` does NOT do

- **Doesn't migrate Postgres schemas other than what alembic runs at api startup.** If a release adds a new alembic revision, it runs automatically on the next api boot. If a release REMOVES a column you care about, that's a destructive change — you must back up before upgrading.
- **Doesn't touch your `.env`.** New required env vars in a release won't be auto-set. The boot will fail with a clear error (`startup model_validator`) and you edit `.env` then restart.
- **Doesn't auto-rotate secrets.** `JWT_SECRET`, Langfuse defaults — your responsibility per [docs/deploy.md](deploy.md) production hardening checklist.
- **Doesn't push to GitHub.** It pulls from the upstream repo. If you've forked or made local commits, `finrag upgrade --force` is required to bypass the dirty-tree check.

## Diagnostics if upgrade fails

```bash
# 1. Container logs
docker compose -f compose.minimal.yml logs api | tail -50

# 2. Boot banner — what actually loaded?
docker compose -f compose.minimal.yml logs api | grep -A 15 "Pipeline boot"

# 3. Health probe directly
curl -v http://localhost:8000/v1/health

# 4. Embedding fingerprint mismatch?
docker compose -f compose.minimal.yml logs api | grep -i fingerprint

# 5. Cost log still being written?
ls -la cost_logs/cost_log.jsonl

# 6. Full restart from scratch
finrag down
finrag setup     # NOT setup --full unless you started full
```

## Maintainer-side: what changes require a CLI re-publish

| Maintainer change | Requires CLI republish? |
|---|---|
| Prompt template tweak | No |
| Reranker model swap | No |
| Generator model swap | No |
| New backend endpoint (additive) | Only if CLI should expose it |
| New SSE event type | No (CLI ignores unknown types) |
| New CLI slash command | Yes |
| Different rendering / colour | Yes |
| API breaking change (rare) | Yes, in lockstep |

The CLI versions independently from the backend on PyPI. Backend changes ship via container image rebuild + git pull. CLI changes ship via `pip install --upgrade`.
