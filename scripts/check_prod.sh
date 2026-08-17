#!/usr/bin/env bash
# Post-idle production check for the Kernel.
#
# Run this after the Kernel has been left alone for a while, before trusting it
# again. It checks liveness, deep DB access, and that auth is actually enforced.
#
#   scripts/check_prod.sh                          # reads .env
#   KERNEL_URL=https://staging... scripts/check_prod.sh
#
# The secret is read from .env (gitignored, where it already lives) unless it is
# already exported. Exit code is non-zero if any check fails, so CI or a cron can
# use it — in CI, export KERNEL_API_SECRET instead of shipping a .env.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pull KERNEL_API_SECRET / KERNEL_URL from .env without sourcing the file: it is
# a config file, not a script, and may legitimately contain characters that
# would run as code. Only these two keys are read, and only if not already set.
read_env() {
  [ -f "$ROOT/.env" ] || return 0
  local value
  value="$(grep -E "^[[:space:]]*$1=" "$ROOT/.env" | tail -1 | cut -d= -f2- | sed 's/^["'\'']//; s/["'\'']$//')"
  printf '%s' "$value"
}

KERNEL_API_SECRET="${KERNEL_API_SECRET:-$(read_env KERNEL_API_SECRET)}"
KERNEL_URL="${KERNEL_URL:-$(read_env KERNEL_URL)}"
KERNEL_URL="${KERNEL_URL:-https://bluestift-kernel-production.up.railway.app}"
SECRET="$KERNEL_API_SECRET"
FAILED=0

say()  { printf '\n\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=1; }

code_of() { curl -sS -o /dev/null -w '%{http_code}' -m 20 "$@" 2>/dev/null; }
body_of() { curl -sS -m 20 "$@" 2>/dev/null; }

say "1. Liveness — GET /health"
if [ "$(code_of "$KERNEL_URL/health")" = "200" ]; then
  ok "$(body_of "$KERNEL_URL/health")"
else
  bad "no 200 from /health — the service is down or the URL moved."
  echo "        Railway free instances sleep; check the deploy before anything else."
fi

say "2. Deep health — GET /ready"
READY_CODE="$(code_of "$KERNEL_URL/ready")"
READY_BODY="$(body_of "$KERNEL_URL/ready")"
case "$READY_CODE" in
  200) ok "$READY_BODY" ;;
  503)
    bad "degraded: $READY_BODY"
    echo "        The shared DB likely lost the kernel schema or the service_role"
    echo "        grants (a RAYA-side setup can reset them). Re-run:"
    echo "          migrations/009_shared_db_hardening.sql"
    ;;
  *) bad "unexpected status $READY_CODE from /ready" ;;
esac

say "3. Auth is enforced — POST /analyze with no secret"
NOAUTH="$(code_of -X POST "$KERNEL_URL/analyze" -H 'content-type: application/json' -d '{}')"
if [ "$NOAUTH" = "401" ]; then
  ok "401 without a secret, as expected."
elif [ "$NOAUTH" = "422" ]; then
  bad "422 — the request was ACCEPTED without a secret (only the body was invalid)."
  echo "        KERNEL_API_SECRET is not set on the Kernel. Set it on Railway."
else
  bad "expected 401, got $NOAUTH"
fi

say "4. The secret this shell holds is the right one"
if [ -z "$SECRET" ]; then
  printf '  \033[33mskip\033[0m  no KERNEL_API_SECRET in the environment or .env; cannot check.\n'
else
  AUTHED="$(code_of -X POST "$KERNEL_URL/load_profile" \
    -H "authorization: Bearer $SECRET" -H 'content-type: application/json' \
    -d '{"user_id":"00000000-0000-0000-0000-000000000000"}')"
  if [ "$AUTHED" = "401" ]; then
    bad "401 with the secret — app and Kernel hold DIFFERENT secrets."
  elif [ "$AUTHED" = "200" ]; then
    ok "200 with the secret; app and Kernel agree."
  else
    bad "expected 200, got $AUTHED (auth passed, but the route erred)."
  fi
fi

echo
if [ "$FAILED" = "0" ]; then
  printf '\033[32mAll checks passed.\033[0m\n'
else
  printf '\033[31mSome checks failed — see above.\033[0m\n'
fi
exit "$FAILED"
