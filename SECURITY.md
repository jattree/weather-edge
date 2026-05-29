# Security Policy

Weather Edge can place orders that move real funds on Polymarket via a Polygon
wallet key, so security reports are taken seriously even though the project is
sunset (see the status notice in the [README](README.md)).

## Reporting a vulnerability

Please report privately, **not** in a public issue or pull request:

- Use GitHub's private vulnerability reporting: the repo's **Security** tab,
  then **"Report a vulnerability"** (GitHub Security Advisories).

Include enough detail to reproduce: the affected file or path, the conditions,
and the impact. This is a solo, best-effort project with no SLA, but legitimate
reports will be acknowledged and serious issues addressed as time allows.

## Supported versions

Only the latest `main` is supported. The `v1.0-as-it-died` tag is preserved for
historical reference and will not receive fixes.

## Scope and the one true secret

- The only true secret in this system is `POLYMARKET_PRIVATE_KEY` (the Polygon
  wallet key). It controls real funds, is read from the environment, and must
  never be committed. `.env` is gitignored. Never paste it anywhere public.
- API keys (Anthropic, Gemini, Open-Meteo, GribStream, Polygonscan) are lower
  impact but should also live only in `.env`.
- The dashboard binds to `127.0.0.1` by default and has **no authentication**.
  Do not expose it to untrusted networks. Its CSRF and Host-allowlist
  protections assume a localhost bind; if you bind elsewhere, put it behind your
  own auth.

## If you run this

- Use a dedicated wallet with limited funds, and run in paper mode until you
  understand the behavior.
- Rotate any key you suspect has been exposed.
