.PHONY: fmt modernize test vet staticcheck diago wasm-build wasm-test verify check release release-major release-crucible

GOFILES := $(shell find . -name '*.go' \
	-not -path './testdata/*' \
	-not -path './usecases/*' \
	-not -path './programs/kiln/train/repos/quote/v3/*')
# go run tool@version ignores go.mod and builds the tool with the go on PATH, which cannot read
# code that needs a newer Go. Unless GOTOOLCHAIN is set, every go command and tool here runs on
# at least the toolchain go.mod selects, and gofmt is that toolchain's.
ifeq ($(origin GOTOOLCHAIN),undefined)
export GOTOOLCHAIN := $(shell go env GOVERSION)+auto
endif
GOFMT := $(shell go env GOROOT)/bin/gofmt
GOLINES := go run github.com/segmentio/golines@v0.13.0
MODERNIZE := go run golang.org/x/tools/go/analysis/passes/modernize/cmd/modernize@v0.52.0
STATICCHECK := go run honnef.co/go/tools/cmd/staticcheck@v0.8.2
MAX_LEN ?= 120
DIAGO_OUTPUT ?= .diago/audit.txt
WORK_MODULES := . ./programs/crucible
WASM_PACKAGE := packages/gepago/wasm

modernize:
	@set -e; for module in $(WORK_MODULES); do \
		(cd $$module && $(MODERNIZE) -mapsloop -fix ./...); \
	done

fmt: modernize
	$(GOLINES) --max-len=$(MAX_LEN) -w .
	$(GOFMT) -w $(GOFILES)

test:
	@set -e; for module in $(WORK_MODULES); do \
		(cd $$module && go test ./... -count=1); \
	done

vet:
	@set -e; for module in $(WORK_MODULES); do \
		(cd $$module && go vet ./...); \
	done

staticcheck:
	@set -e; for module in $(WORK_MODULES); do \
		(cd $$module && $(STATICCHECK) ./...); \
	done

diago:
	@set -e; for module in $(WORK_MODULES); do \
		(cd $$module && diago -target ./... -race -coverage -deps -output $(DIAGO_OUTPUT)); \
	done

wasm-build:
	cd $(WASM_PACKAGE) && npm run build

wasm-test:
	cd $(WASM_PACKAGE) && npm test

verify: fmt test vet staticcheck diago wasm-test

check: fmt test vet staticcheck
	git diff --exit-code

# Release tags and pushes a semver tag. Default bumps minor: vX.Y.0 -> vX.(Y+1).0
# Examples:
#   make release                 # bump minor
#   make release MAJOR=1         # bump major
#   make release-major           # bump major
#   make release MINOR=3         # set minor on current major, patch=0
#   make release VERSION=v1.2.3  # explicit version
#   make release DRY_RUN=1       # print without tagging/pushing
release: test
	@set -euo pipefail; \
	latest="$$(git describe --tags --match 'v[0-9]*' --abbrev=0 2>/dev/null || true)"; \
	if [[ -z "$$latest" ]]; then latest="v0.0.0"; fi; \
	version="$(VERSION)"; \
	if [[ -z "$$version" ]]; then \
		base="$${latest#v}"; \
		IFS=. read -r major minor patch <<<"$$base"; \
		major="$${major:-0}"; minor="$${minor:-0}"; patch="$${patch:-0}"; \
		if [[ -n "$(MAJOR)" ]]; then \
			major=$$((major + 1)); minor=0; patch=0; \
		elif [[ -n "$(MINOR)" ]]; then \
			minor="$(MINOR)"; patch=0; \
		else \
			minor=$$((minor + 1)); patch=0; \
		fi; \
		version="v$${major}.$${minor}.$${patch}"; \
	fi; \
	if [[ ! "$$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$$ ]]; then \
		echo "invalid version: $$version (expected vX.Y.Z)" >&2; exit 1; \
	fi; \
	echo "latest tag: $$latest"; \
	echo "release tag: $$version"; \
	if [[ -n "$(DRY_RUN)" ]]; then \
		echo "dry run: git tag $$version && git push origin $$version"; \
		exit 0; \
	fi; \
	git tag "$$version"; \
	git push origin "$$version"

release-major:
	$(MAKE) release MAJOR=1

# Release the nested Crucible module. Nested Go modules use a repository-path
# tag, so Crucible v0.1.0 is tagged as programs/crucible/v0.1.0.
# Examples:
#   make release-crucible VERSION=v0.1.0
#   make release-crucible VERSION=v0.1.0 DRY_RUN=1
release-crucible: test
	cd programs/crucible && GOWORK=off go test ./... -count=1
	@set -euo pipefail; \
	latest="$$(git tag --list 'programs/crucible/v[0-9]*' --sort=-version:refname | head -n 1)"; \
	if [[ -z "$$latest" ]]; then latest="programs/crucible/v0.0.0"; fi; \
	version="$(VERSION)"; \
	if [[ ! "$$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$$ ]]; then \
		echo "invalid Crucible version: $$version (expected vX.Y.Z)" >&2; exit 1; \
	fi; \
	tag="programs/crucible/$$version"; \
	echo "latest Crucible tag: $$latest"; \
	echo "release tag: $$tag"; \
	if git rev-parse -q --verify "refs/tags/$$tag" >/dev/null; then \
		echo "tag already exists: $$tag" >&2; exit 1; \
	fi; \
	if [[ -n "$(DRY_RUN)" ]]; then \
		echo "dry run: git tag $$tag && git push origin $$tag"; \
		exit 0; \
	fi; \
	git tag "$$tag"; \
	git push origin "$$tag"
