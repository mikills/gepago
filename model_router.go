package gepa

import "maps"

// ModelRole identifies a language-model responsibility within a GEPA programme.
type ModelRole string

const (
	// RoleTask is the model being optimised for the task itself.
	RoleTask ModelRole = "task"
	// RoleReflector is the model used to turn feedback into improved candidates.
	RoleReflector ModelRole = "reflector"
	// RoleJudge is the model used for LLM-as-judge metrics or semantic review.
	RoleJudge ModelRole = "judge"
	// RoleRepair is the model used to repair malformed model output.
	RoleRepair ModelRole = "repair"
	// RolePlanner is the model used to choose what should be improved next.
	RolePlanner ModelRole = "planner"
	// RoleImplementer is the model used to produce concrete implementation edits.
	RoleImplementer ModelRole = "implementer"
	// RoleReporter is the model used to summarize a run for humans.
	RoleReporter ModelRole = "reporter"
)

// ModelRouter maps GEPA responsibilities to language models while preserving a default fallback.
type ModelRouter struct {
	Default LanguageModel
	Models  map[ModelRole]LanguageModel
}

// For returns the model configured for role, falling back to Default.
func (r ModelRouter) For(role ModelRole) LanguageModel {
	if r.Models != nil {
		if model := r.Models[role]; model != nil {
			return model
		}
	}
	return r.Default
}

// With returns a copy of the router with role bound to model.
func (r ModelRouter) With(role ModelRole, model LanguageModel) ModelRouter {
	models := make(map[ModelRole]LanguageModel, len(r.Models)+1)
	maps.Copy(models, r.Models)
	models[role] = model
	r.Models = models
	return r
}
