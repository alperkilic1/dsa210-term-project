#!/usr/bin/env bash
# GitHub contributor widget cache-bust script
# Repo: alperkilic1/dsa210-term-project
# Problem: Stale contributor (alperw2002) appeared in sidebar widget despite
#          having zero commits and no collaborator access.
#
# FINDINGS (tested 2026-04-21):
# +--------------------------+--------+------------------------------------------+
# | Technique                | Works? | Evidence                                 |
# +--------------------------+--------+------------------------------------------+
# | Topic add/remove         | NO     | Widget unchanged after toggle             |
# | Description edit/restore | NO     | Widget unchanged                          |
# | Branch create/delete     | NO     | Widget unchanged                          |
# | Archive + unarchive      | NO     | Widget unchanged (repo was read-only 3s)  |
# | Default branch swap      | NO     | Widget unchanged (risky for CI/clones)    |
# | Empty commits            | NO     | 4 empty commits, widget unchanged         |
# | Stats endpoint hammering | NO     | All returned 202; widget uses diff source |
# | Push real file change    | YES*   | Widget updated after push + revert combo  |
# +--------------------------+--------+------------------------------------------+
# * The cache bust happened after the COMBINATION of:
#   1. Multiple pushes (empty + file change) creating git event volume
#   2. Stats endpoint polling (triggering async recompute)
#   3. The revert commit (another push event)
#   Total latency: ~15 minutes from first push to widget update.
#
# Root cause: The /contributors_list fragment is a Rails server-side cache
# keyed on repo + collaborator state. It is NOT the same as /contributors API.
# The API was correct the whole time; only the HTML widget was stale.
# Git push events eventually trigger a background job that rebuilds the fragment.
#
# SAFE: Does not rename repo, make private, or break clone URLs.

set -euo pipefail
REPO="alperkilic1/dsa210-term-project"

echo "[1/4] Pushing cache-bust commit..."
DIR=$(mktemp -d)
git clone --depth=1 "https://github.com/$REPO.git" "$DIR" 2>/dev/null
git -C "$DIR" commit --allow-empty -m "chore: trigger contributor cache rebuild"
git -C "$DIR" push origin main

echo "[2/4] Polling stats endpoints to trigger async recompute..."
for endpoint in stats/contributors stats/commit_activity stats/code_frequency; do
  for _ in 1 2 3; do
    gh api "repos/$REPO/$endpoint" >/dev/null 2>&1 || true
    sleep 2
  done
done

echo "[3/4] Waiting 60s for background job..."
sleep 60

echo "[4/4] Verifying..."
WIDGET=$(curl -s "https://github.com/$REPO/contributors_list?current_repository=dsa210-term-project&deferred=true")
COUNT=$(echo "$WIDGET" | grep -o 'title="[0-9]*"' | grep -o '[0-9]*')
HAS_GHOST=$(echo "$WIDGET" | grep -c 'alperw2002' || true)

echo "Widget contributor count: $COUNT"
echo "alperw2002 present: $HAS_GHOST"

if [ "$HAS_GHOST" -eq 0 ] && [ "$COUNT" -eq 1 ]; then
  echo "SUCCESS: Widget shows only alperkilic1"
else
  echo "STILL CACHED: May need more time or GitHub Support ticket"
  echo "  Open: https://support.github.com/request"
  echo "  Subject: Stale contributor widget cache for $REPO"
fi

rm -rf "$DIR"
