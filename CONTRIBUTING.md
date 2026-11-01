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

The root module and the programs `programs/crucible` and `programs/classify` are tagged
separately, as `vX.Y.Z`, `programs/crucible/vX.Y.Z` and `programs/classify/vX.Y.Z`. When a
program requires a root version that is not tagged yet, `go.work` replaces that version with
the workspace module so the workspace builds. Release in this order:

1. Tag the root module: `make release VERSION=vX.Y.Z`.
2. In each program that requires the new version, run `GOWORK=off go mod tidy` to add its
   `go.sum` lines. Remove the `replace` for the version from `go.work`, and commit them all.
3. Tag the programs: `make release-crucible VERSION=vA.B.C` and
   `make release-classify VERSION=vA.B.C`. The targets and their release workflows test each
   program with `GOWORK=off` on that commit, so they fail until step 2 is in.

## Design guidelines

- Keep the root package provider-neutral and independent of the optional agent runtime.
- Put provider integrations under `providers/*`.
- Keep comments concise and focused on non-obvious behaviour.
- Prefer table-driven tests or target tests with `t.Run` scenario names over long test function names.
- Avoid generated artefacts in commits unless they are intentional documentation assets.
