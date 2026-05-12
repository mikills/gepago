package gepa

import (
	"context"
	"testing"
)

type routerTestModel struct {
	name string
}

func (m routerTestModel) Generate(context.Context, string) (string, error) {
	return m.name, nil
}

func TestModelRouterFor(t *testing.T) {
	defaultModel := routerTestModel{name: "default"}
	judgeModel := routerTestModel{name: "judge"}
	router := ModelRouter{Default: defaultModel}.With(RoleJudge, judgeModel)

	if got := router.For(RoleJudge); got != judgeModel {
		t.Fatalf("expected judge model, got %#v", got)
	}
	if got := router.For(RoleTask); got != defaultModel {
		t.Fatalf("expected default model, got %#v", got)
	}
}

func TestModelRouterWithDoesNotMutateOriginal(t *testing.T) {
	defaultModel := routerTestModel{name: "default"}
	judgeModel := routerTestModel{name: "judge"}
	original := ModelRouter{Default: defaultModel}
	updated := original.With(RoleJudge, judgeModel)

	if got := original.For(RoleJudge); got != defaultModel {
		t.Fatalf("expected original to fall back to default, got %#v", got)
	}
	if got := updated.For(RoleJudge); got != judgeModel {
		t.Fatalf("expected updated router to use judge model, got %#v", got)
	}
}
