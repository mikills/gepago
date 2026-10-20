# Contributing

Thanks for helping improve `gepago`.

## Local checks

Run the full verification suite before opening a PR:

```bash
make verify
```

This runs formatting, tests, `go vet`, `staticcheck`, and `diago`.

If `diago` is not installed, run the standard checks first:

```bash
make fmt
make test
make vet
make staticcheck
```

## Provider E2E tests

Provider tests that call APIs require credentials and the `e2e` build tag:

```bash
OPENAI_API_KEY=... go test -tags=e2e ./providers/openai -run TestOptimizerE2E -count=1 -v
ANTHROPIC_API_KEY=... go test -tags=e2e ./providers/claude -run TestProposerE2E -count=1 -v
```

Never commit API keys, reports, `.diago` output, or local notes.

## Releases

The root module and `programs/crucible` are tagged separately, as `vX.Y.Z` and
`programs/crucible/vX.Y.Z`. When crucible requires a root version that is not tagged yet,
`go.work` replaces that version with the workspace module so the workspace builds. Release in
this order:

1. Tag the root module: `make release VERSION=vX.Y.Z`.
2. In `programs/crucible`, run `GOWORK=off go mod tidy` to add the new version's `go.sum`
   lines, remove the `replace` for it from `go.work`, and commit both.
3. Tag crucible: `make release-crucible VERSION=vA.B.C`. The target and the release-crucible
   workflow test crucible with `GOWORK=off` on that commit, so they fail until step 2 is in.

## Design guidelines

- Keep the root package provider-neutral and independent of the optional agent runtime.
- Put provider integrations under `providers/*`.
- Keep comments concise and focused on non-obvious behaviour.
- Prefer table-driven tests or target tests with `t.Run` scenario names over long test function names.
- Avoid generated artefacts in commits unless they are intentional documentation assets.
