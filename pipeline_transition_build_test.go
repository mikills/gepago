package gepa

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestBuildIsLinearInStepsAndTransitions builds pipelines of growing size whose transitions skip
// or rewind over most of their steps, and bounds work, which counts the iterations of Build's
// pass over the transitions and its pass over the steps and every fact the passes write (see
// transitionFacts). Build records the steps a rule skips or rewinds as the bounds of difference
// arrays, so the count stays between the steps plus the transitions and a small multiple of
// them; walking the steps between each rule's From and To to write a fact, or copying a range's
// keys once per rule, would make it quadratic. The last subtest holds the rule that makes the
// count complete whether or not a loop counts its own iterations: only the methods of
// transitionFacts that count write its facts, each without a loop. A count makes the check
// independent of the machine's load.
func TestBuildIsLinearInStepsAndTransitions(t *testing.T) {
	name := func(i int) string { return fmt.Sprintf("s%d", i) }
	// the forward rules are declared from the last step back, so the first rule that skips over a
	// step is declared late and a search through the rules for one is long.
	shapes := []struct {
		name        string
		transitions func(n int) ([]Transition, map[string]int)
	}{
		{
			// each step may end the run, so every rule skips over the steps after its From, and every
			// step but the first may not write.
			name: "an End rule per step",
			transitions: func(n int) ([]Transition, map[string]int) {
				rules := make([]Transition, n)
				for i := range rules {
					rules[i] = Transition{From: name(n - 1 - i), When: amountIs(1), To: End}
				}
				return rules, nil
			},
		},
		{
			name: "an OnError rule per step to the last",
			transitions: func(n int) ([]Transition, map[string]int) {
				rules := make([]Transition, n-1)
				for i := range rules {
					rules[i] = Transition{From: name(n - 2 - i), OnError: true, To: name(n - 1)}
				}
				return rules, nil
			},
		},
		{
			name: "a rewind of every step",
			transitions: func(n int) ([]Transition, map[string]int) {
				rules := make([]Transition, n)
				for i := range rules {
					rules[i] = Transition{From: name(n - 1), When: amountIs(i), To: name(0)}
				}
				return rules, map[string]int{name(0): 2}
			},
		},
	}
	for _, shape := range shapes {
		t.Run(shape.name, func(t *testing.T) {
			for _, n := range []int{100, 1000, 10000} {
				// each step writes a key of its own, which no step reads, so the steps that may not
				// write need no Default.
				steps := make([]Step, n)
				for i := range steps {
					key := Key[int]{Name: fmt.Sprintf("k%d", i)}
					steps[i] = NewStep(name(i), nil, refs(key.Ref()), func(_ context.Context, s State) error {
						key.Set(s, i)
						return nil
					})
				}
				transitions, maxVisits := shape.transitions(n)
				var work int
				_, err := PipelineSpec{Steps: steps, Transitions: transitions, MaxVisits: maxVisits}.build(&work)
				require.NoError(t, err)

				// a rule counts its iteration, its rule and its skip; a step its iteration and, for
				// its one key, the key of its range, its writer and its mark as unwritten.
				size := n + len(transitions)
				t.Logf("%d steps, %d transitions: %d iterations and facts", n, len(transitions), work)
				require.GreaterOrEqual(t, work, size, "each pass counts its iterations")
				require.LessOrEqual(t, work, 4*size, "%d steps", n)
			}
		})
	}
	t.Run("only the counting methods write the facts", func(t *testing.T) {
		require.Empty(t, uncountedFactWrites(t))
	})
}

// factWriters are the methods of transitionFacts that may write its facts.
var factWriters = []string{"mark", "skip", "cover", "enter", "addRangeKey", "markUnwritten", "addWriter"}

// uncountedFactWrites parses the package's source files and returns each write of a fact that
// the count could miss: of a field of transitionFacts other than work and rules, or of the keys
// of a rewind range or rule, outside factWriters; and each loop in, or method without a count
// of, one of factWriters. It sees direct writes, not those through an alias.
func uncountedFactWrites(t *testing.T) []string {
	t.Helper()
	fset := token.NewFileSet()
	files, err := filepath.Glob("*.go")
	require.NoError(t, err)
	fields := map[string]bool{"keys": true}
	var funcs []*ast.FuncDecl
	for _, path := range files {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, path, nil, 0)
		require.NoError(t, err)
		for _, decl := range file.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok {
				funcs = append(funcs, fn)
			}
		}
		ast.Inspect(file, func(node ast.Node) bool {
			spec, ok := node.(*ast.TypeSpec)
			if !ok || spec.Name.Name != "transitionFacts" {
				return true
			}
			for _, field := range spec.Type.(*ast.StructType).Fields.List {
				for _, ident := range field.Names {
					fields[ident.Name] = ident.Name != "work" && ident.Name != "rules"
				}
			}
			return false
		})
	}
	require.True(t, fields["skips"], "the fields of transitionFacts are found")

	var problems []string
	for _, fn := range funcs {
		writer := isFactWriter(fn)
		counts := false
		ast.Inspect(fn.Body, func(node ast.Node) bool {
			var targets []ast.Expr
			switch node := node.(type) {
			case *ast.AssignStmt:
				targets = node.Lhs
			case *ast.IncDecStmt:
				targets = []ast.Expr{node.X}
				if selector, ok := node.X.(*ast.SelectorExpr); ok && selector.Sel.Name == "work" {
					counts = true
				}
			case *ast.CallExpr:
				if writesFirstArgument(node) {
					targets = node.Args[:1]
				}
			case *ast.ForStmt, *ast.RangeStmt:
				if writer {
					problems = append(problems, fmt.Sprintf("%s: %s loops", fset.Position(node.Pos()), fn.Name.Name))
				}
			}
			for _, target := range targets {
				if field := factField(target, fields); field != "" && !writer {
					problems = append(problems, fmt.Sprintf(
						"%s: %s writes %s outside the counting methods",
						fset.Position(target.Pos()),
						fn.Name.Name,
						field,
					))
				}
			}
			return true
		})
		if writer && !counts {
			problems = append(problems, fmt.Sprintf("%s: %s does not count", fset.Position(fn.Pos()), fn.Name.Name))
		}
	}
	return problems
}

// isFactWriter reports whether fn is one of factWriters.
func isFactWriter(fn *ast.FuncDecl) bool {
	if fn.Recv == nil || len(fn.Recv.List) != 1 || !slices.Contains(factWriters, fn.Name.Name) {
		return false
	}
	star, ok := fn.Recv.List[0].Type.(*ast.StarExpr)
	if !ok {
		return false
	}
	ident, ok := star.X.(*ast.Ident)
	return ok && ident.Name == "transitionFacts"
}

// writesFirstArgument reports whether call writes into its first argument: delete, clear, copy,
// maps.Copy, maps.Insert or maps.DeleteFunc.
func writesFirstArgument(call *ast.CallExpr) bool {
	if len(call.Args) == 0 {
		return false
	}
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		return fun.Name == "delete" || fun.Name == "clear" || fun.Name == "copy"
	case *ast.SelectorExpr:
		pkg, ok := fun.X.(*ast.Ident)
		return ok && pkg.Name == "maps" && slices.Contains([]string{"Copy", "Insert", "DeleteFunc"}, fun.Sel.Name)
	}
	return false
}

// factField returns the fact that a write to target writes into, or "" when it writes none.
func factField(target ast.Expr, fields map[string]bool) string {
	for {
		switch expr := target.(type) {
		case *ast.SelectorExpr:
			if fields[expr.Sel.Name] {
				return expr.Sel.Name
			}
			target = expr.X
		case *ast.IndexExpr:
			target = expr.X
		case *ast.StarExpr:
			target = expr.X
		case *ast.ParenExpr:
			target = expr.X
		default:
			return ""
		}
	}
}
