.PHONY: fmt modernize test vet staticcheck diago verify check

GOFILES := $(shell find . -name '*.go' -not -path './testdata/*' -not -path './usecases/*')
GOLINES := go run github.com/segmentio/golines@v0.13.0
MODERNIZE := go run golang.org/x/tools/go/analysis/passes/modernize/cmd/modernize@v0.45.0
STATICCHECK := go run honnef.co/go/tools/cmd/staticcheck@v0.7.0
MAX_LEN ?= 120
DIAGO_OUTPUT ?= .diago/audit.txt

modernize:
	$(MODERNIZE) -mapsloop -fix ./...

fmt: modernize
	$(GOLINES) --max-len=$(MAX_LEN) -w .
	gofmt -w $(GOFILES)

test:
	go test ./... -count=1

vet:
	go vet ./...

staticcheck:
	$(STATICCHECK) ./...

diago:
	diago -target ./... -race -coverage -deps -output $(DIAGO_OUTPUT)

verify: fmt test vet staticcheck diago

check: fmt test vet staticcheck
	git diff --exit-code
