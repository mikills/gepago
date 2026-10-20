package gepa

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
)

// pipelineFingerprintVersion is hashed into every fingerprint, so changing what a fingerprint
// covers changes every fingerprint instead of letting two different shapes match.
const pipelineFingerprintVersion = 1

// Fingerprint returns a stable hash of the pipeline's shape as 64 lower-case hexadecimal
// digits, or "" for a Pipeline not created by Build. The same shape has the same fingerprint
// in every run and process. Pipeline is a Fingerprinter, so CompiledProgram.Artifact and
// ProgramRegistry.TrainAndSave store the fingerprint on the artefacts they produce, and
// LoadCompiledProgram refuses an artefact saved for another shape.
//
// The shape is:
//   - the pipeline inputs, by name and type;
//   - the top-level steps, in order;
//   - each step's name and its reads and writes, by name and type, for nested steps too;
//   - the structure of composite steps: the arms of a Branch with their labels, the steps of a
//     Parallel, the primary and secondary steps of a Fallback, and the fallback step of a
//     WithPolicy;
//   - for a NewProgramStep whose programme is a Fingerprinter, such as a nested Pipeline, that
//     programme's fingerprint.
//
// Everything else is left out because it does not change which trained candidate fits the
// pipeline: candidate components and their text (instructions, demos and other prompts), the
// Go type and code of each step, a programme that is not a Fingerprinter, merge functions,
// Defaults, Budget, AllowExtraInputs, the timeout, retries and error action of a WithPolicy
// apart from its fallback step, and WithCache, which adds nothing to the step it wraps. Reads,
// writes and the steps of a Parallel are taken in name order, so declaring them in another
// order keeps the fingerprint. A named type is identified by its package path and name, not by
// its definition, so changing the fields of a named struct type keeps the fingerprint; give
// the artefact and the loader a ProgramVersion when such a change matters.
func (p Pipeline) Fingerprint() string {
	if p.Validate() != nil {
		return ""
	}
	shape := pipelineShape{Version: pipelineFingerprintVersion, Inputs: keyShapes(p.inputs)}
	for _, node := range p.steps {
		shape.Steps = append(shape.Steps, shapeOf(node))
	}
	// the shape holds only strings, integers, slices and pointers, which always encode.
	data, _ := json.Marshal(shape)
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// pipelineShape is what a pipeline fingerprint hashes, in its JSON encoding.
type pipelineShape struct {
	Version int         `json:"version"`
	Inputs  []keyShape  `json:"inputs"`
	Steps   []stepShape `json:"steps"`
}

type keyShape struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// stepShape is the shape of one step. Kind is "step" for a step that is not a composite;
// Program is the fingerprint of a Fingerprinter programme behind NewProgramStep; Fallbacks
// holds the fallback steps of the policies wrapping the step, innermost first.
type stepShape struct {
	Kind      string       `json:"kind"`
	Name      string       `json:"name"`
	Reads     []keyShape   `json:"reads"`
	Writes    []keyShape   `json:"writes"`
	Program   string       `json:"program,omitempty"`
	Children  []childShape `json:"children,omitempty"`
	Fallbacks []stepShape  `json:"fallbacks,omitempty"`
}

type childShape struct {
	Label string    `json:"label,omitempty"`
	Step  stepShape `json:"step"`
}

func shapeOf(node *stepNode) stepShape {
	shape := stepShape{Kind: "step", Name: node.name, Reads: keyShapes(node.reads), Writes: keyShapes(node.writes)}
	step := node.step
	if component, ok := step.(componentComposite); ok {
		step = component.compositeStep
	}
	switch s := step.(type) {
	case *policyStep:
		shape = shapeOf(s.child)
		if s.fallback != nil {
			shape.Fallbacks = append(shape.Fallbacks, shapeOf(s.fallback))
		}
	case *cacheStep:
		shape = shapeOf(s.child)
	case *branchStep:
		shape.Kind = "branch"
		for _, label := range s.labels {
			shape.Children = append(shape.Children, childShape{Label: label, Step: shapeOf(s.arms[label])})
		}
	case *parallelStep:
		shape.Kind = "parallel"
		for _, child := range s.steps {
			shape.Children = append(shape.Children, childShape{Step: shapeOf(child)})
		}
		slices.SortFunc(shape.Children, func(a, b childShape) int { return strings.Compare(a.Step.Name, b.Step.Name) })
	case *fallbackStep:
		shape.Kind = "fallback"
		shape.Children = []childShape{
			{Label: "primary", Step: shapeOf(s.primary)},
			{Label: "secondary", Step: shapeOf(s.secondary)},
		}
	case programStep:
		shape.Program = programFingerprint(s.program)
	case compositeStep:
		// a composite this function does not know yet still contributes its children, under
		// the labels Build gives them.
		shape.Kind = fmt.Sprintf("%T", s)
		for _, child := range s.children() {
			shape.Children = append(shape.Children, childShape{Label: child.label, Step: shapeOf(child.node)})
		}
	}
	return shape
}

func keyShapes(keys []KeyRef) []keyShape {
	shapes := make([]keyShape, 0, len(keys))
	for _, key := range keys {
		shapes = append(shapes, keyShape{Name: key.Name, Type: typeIdentity(key.Type)})
	}
	slices.SortFunc(shapes, func(a, b keyShape) int {
		return cmp.Or(strings.Compare(a.Name, b.Name), strings.Compare(a.Type, b.Type))
	})
	return shapes
}

// typeIdentity names typ the same way in every process: a named type by its package path and
// name, which for an instantiated generic type includes the package paths of its type
// arguments, and an unnamed type by its structure.
func typeIdentity(typ reflect.Type) string {
	if typ == nil {
		return ""
	}
	if typ.Name() != "" {
		if typ.PkgPath() == "" {
			return typ.Name()
		}
		return typ.PkgPath() + "." + typ.Name()
	}
	switch typ.Kind() {
	case reflect.Pointer:
		return "*" + typeIdentity(typ.Elem())
	case reflect.Slice:
		return "[]" + typeIdentity(typ.Elem())
	case reflect.Array:
		return fmt.Sprintf("[%d]%s", typ.Len(), typeIdentity(typ.Elem()))
	case reflect.Map:
		return "map[" + typeIdentity(typ.Key()) + "]" + typeIdentity(typ.Elem())
	case reflect.Chan:
		return chanPrefix(typ.ChanDir()) + typeIdentity(typ.Elem())
	case reflect.Func:
		return funcIdentity(typ)
	case reflect.Struct:
		return structIdentity(typ)
	case reflect.Interface:
		return interfaceIdentity(typ)
	}
	return typ.String()
}

func chanPrefix(dir reflect.ChanDir) string {
	switch dir {
	case reflect.RecvDir:
		return "<-chan "
	case reflect.SendDir:
		return "chan<- "
	}
	return "chan "
}

func funcIdentity(typ reflect.Type) string {
	params := make([]string, 0, typ.NumIn())
	for i := range typ.NumIn() {
		param := typeIdentity(typ.In(i))
		if typ.IsVariadic() && i == typ.NumIn()-1 {
			param = "..." + typeIdentity(typ.In(i).Elem())
		}
		params = append(params, param)
	}
	results := make([]string, 0, typ.NumOut())
	for i := range typ.NumOut() {
		results = append(results, typeIdentity(typ.Out(i)))
	}
	return "func(" + strings.Join(params, ", ") + ") (" + strings.Join(results, ", ") + ")"
}

func structIdentity(typ reflect.Type) string {
	fields := make([]string, 0, typ.NumField())
	for i := range typ.NumField() {
		field := typ.Field(i)
		name := field.Name
		if field.PkgPath != "" {
			name = field.PkgPath + "." + name
		}
		if field.Anonymous {
			name = "embedded " + name
		}
		fields = append(fields, fmt.Sprintf("%s %s %q", name, typeIdentity(field.Type), field.Tag))
	}
	return "struct{" + strings.Join(fields, "; ") + "}"
}

func interfaceIdentity(typ reflect.Type) string {
	methods := make([]string, 0, typ.NumMethod())
	for i := range typ.NumMethod() {
		method := typ.Method(i)
		name := method.Name
		if method.PkgPath != "" {
			name = method.PkgPath + "." + name
		}
		methods = append(methods, name+" "+typeIdentity(method.Type))
	}
	return "interface{" + strings.Join(methods, "; ") + "}"
}
