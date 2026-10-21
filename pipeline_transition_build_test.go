package gepa

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"maps"
	"os"
	"os/exec"
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
// keys once per rule, would make it quadratic. The count misses work a loop does without
// counting it, so the last subtests check the source for the likely ways to do that (see
// factProblems): a fact written outside the methods of transitionFacts that count, a loop over
// the facts outside the passes and the methods that word an error, and recursion or goto in a
// function that uses them. A count makes the check independent of the machine's load.
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

	source := loadPackageSource(t)
	t.Run("the source check finds nothing in the package", func(t *testing.T) {
		require.Empty(t, source.factProblems(t, ""))
	})
	t.Run("the source check finds what the count could miss", func(t *testing.T) {
		tests := []struct {
			name string
			// mutation is a file of the package whose functions take the place of those with
			// the same names.
			mutation string
			want     []string
		}{
			{
				name: "a loop over the rules that writes nothing",
				mutation: `
func (f *transitionFacts) mayNotWrite(i int) bool {
	for _, rule := range f.rules {
		if !rule.rewind && rule.skips(i, len(f.onErrorBy)) {
			return true
		}
	}
	return f.onErrorBy[i] >= 0
}`,
				want: []string{
					"mutation.go:4:2: transitionFacts.mayNotWrite loops over transitionFacts.rules; " + loopAdvice,
				},
			},
			{
				name: "a loop over the steps that reads a fact of each",
				mutation: `
func (f *transitionFacts) mayNotWrite(i int) bool {
	skipping := 0
	for k := 0; k <= i; k++ {
		skipping += f.skips[k]
	}
	return skipping > 0 || f.onErrorBy[i] >= 0
}`,
				want: []string{
					"mutation.go:5:2: transitionFacts.mayNotWrite loops over transitionFacts.skips; " + loopAdvice,
				},
			},
			{
				// a method of transitionFacts may read any fact, a getter's or reach's among them.
				name: "loops that read the facts through a method",
				mutation: `
func (f *transitionFacts) skipsAt(k int) int { return f.skips[k] }

func (f *transitionFacts) mayNotWrite(i int) bool {
	skipping := 0
	for k := 0; k <= i; k++ {
		skipping += f.skipsAt(k)
	}
	return skipping > 0 || f.onErrorBy[i] >= 0
}

func (b *pipelineBuilder) bothCommit(key string) bool {
	earlier := b.flow.writtenBy[key]
	for k := range b.index {
		if b.flow.reach(k) < 0 {
			return true
		}
	}
	return earlier.reach <= b.index || b.flow.rewinds > earlier.rewinds
}`,
				want: []string{
					"mutation.go:7:2: transitionFacts.mayNotWrite calls transitionFacts.skipsAt in a loop; " +
						loopAdvice,
					"mutation.go:15:2: pipelineBuilder.bothCommit calls transitionFacts.reach in a loop; " +
						loopAdvice,
				},
			},
			{
				name: "a walk by recursion or goto",
				mutation: `
func skippedThrough(f *transitionFacts, k int) int {
	if k < 0 {
		return 0
	}
	return f.skips[k] + skippedThrough(f, k-1)
}

func (f *transitionFacts) skip(start, end int) {
	f.work++
	k := start
next:
	if k < end {
		f.skips[k]++
		f.skips[k+1]--
		k++
		goto next
	}
}`,
				want: []string{
					"mutation.go:7:22: skippedThrough calls itself; " + loopAdvice,
					"mutation.go:18:3: transitionFacts.skip uses goto; " + loopAdvice,
				},
			},
			{
				name: "a helper that walks the facts it is passed",
				mutation: `
func (f *transitionFacts) skip(start, end int) {
	f.work++
	if start < end {
		walkSkips(f.skips, start, end)
	}
}

func walkSkips(skips []int, start, end int) {
	for k := start; k < end; k++ {
		skips[k]++
		skips[k+1]--
	}
}`,
				want: []string{
					"mutation.go:6:13: transitionFacts.skip passes transitionFacts.skips to walkSkips, " +
						"where a loop over it goes uncounted; " + loopAdvice,
				},
			},
			{
				name: "facts returned or kept elsewhere",
				mutation: `
func (f *transitionFacts) skipCounts() []int {
	return f.skips
}

func keepRules(b *pipelineBuilder, table *transitionTable) {
	table.rules = b.flow.rules
}`,
				want: []string{
					"mutation.go:4:9: transitionFacts.skipCounts returns transitionFacts.skips, " +
						"where a loop over it goes uncounted; " + loopAdvice,
					"mutation.go:8:2: keepRules stores transitionFacts.rules in table.rules, " +
						"where a loop over it goes uncounted; " + loopAdvice,
				},
			},
			{
				name: "writes outside the counting methods",
				mutation: `
import "maps"

func (b *pipelineBuilder) leave(node *stepNode) {
	if b.flow.current != nil {
		maps.Copy(b.flow.current.keys, node.writeTypes)
	}
}

func (b *pipelineBuilder) unskip() {
	skips := b.flow.skips
	skips[b.index] = 0
}`,
				want: []string{
					"mutation.go:7:13: pipelineBuilder.leave writes rewindRange.keys outside the counting methods; " +
						writeAdvice,
					"mutation.go:13:2: pipelineBuilder.unskip writes transitionFacts.skips outside the counting " +
						"methods; " + writeAdvice,
				},
			},
			{
				name: "counting methods that loop or do not count",
				mutation: `
import "reflect"

func (f *transitionFacts) skip(start, end int) {
	f.work++
	for k := start; k < end; k++ {
		f.skips[k]++
		f.skips[k+1]--
	}
}

func (f *transitionFacts) addRangeKey(key string, typ reflect.Type) {
	if f.current != nil {
		f.current.keys[key] = typ
	}
}`,
				want: []string{
					"mutation.go:7:2: counting method transitionFacts.skip loops; " + loopAdvice,
					"mutation.go:13:1: counting method transitionFacts.addRangeKey does not add to work; " +
						countAdvice,
				},
			},
			{
				name: "a method that words an error and does not count",
				mutation: `
import "fmt"

func (f *transitionFacts) unwrittenReason(i int) string {
	for _, rule := range f.rules {
		if !rule.rewind && rule.skips(i, len(f.onErrorBy)) {
			return fmt.Sprintf("transition %d skips over it", rule.index)
		}
	}
	return fmt.Sprintf("its OnError transition %d goes forward", f.onErrorBy[i])
}`,
				want: []string{
					"mutation.go:6:2: transitionFacts.unwrittenReason loops over transitionFacts.rules without " +
						"adding to work; " + countAdvice,
				},
			},
			{
				// fields of other types are told apart from the facts by type, not by name.
				name: "fields named like facts, and other ways to count",
				mutation: `
import (
	"reflect"
	"slices"
	"strings"
)

func WithAmendable(step Step, keys ...KeyRef) Step {
	a := &amendableStep{child: newStepNode(step)}
	a.keys = slices.Clone(keys)
	a.types, a.keysErr = stateKeyTypes(a.keys)
	return asComponentStep(a, a.child)
}

type runTally struct {
	skips   []int
	amended []bool
}

func (r *runTally) tick(i int) {
	for k := range r.skips {
		r.skips[k]++
	}
	r.amended[i] = true
}

func (r *runTally) skipped(k int) int {
	if k < 0 {
		return 0
	}
	return r.skips[k] + r.skipped(k-1)
}

func (f *transitionFacts) skip(start, end int) {
	f.work += 1
	if start < end {
		f.skips[start]++
		f.skips[end]--
	}
}

func (f *transitionFacts) addRangeKey(key string, typ reflect.Type) {
	const entries = 1
	f.work += entries
	if f.current != nil {
		f.current.keys[key] = typ
	}
}`,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				// the problems in the package's own source are the previous subtest's.
				var got []string
				for _, problem := range source.factProblems(t, tt.mutation) {
					if strings.HasPrefix(problem, "mutation.go:") {
						got = append(got, problem)
					}
				}
				require.Equal(t, tt.want, got)
			})
		}
	})
}

// what to do about each problem the source check finds.
const (
	writeAdvice = "write facts through the counting methods"
	loopAdvice  = "loops over facts belong in the two passes"
	countAdvice = "add to work with f.work++ or f.work += n"
)

// factWriters are the methods of transitionFacts that may write its facts, and factLoopers
// those that may loop over them, counting each iteration, to word an error. The outermost loops
// of factPasses are Build's two passes, over the transitions and over the steps, which may read
// the facts of the rule or step they are at; a loop nested in them is checked as any other.
var (
	factWriters = []string{"mark", "skip", "cover", "enter", "addRangeKey", "markUnwritten", "addWriter"}
	factLoopers = []string{"unwrittenReason", "overlapError"}
	factPasses  = []string{"pipelineBuilder.checkTransitions", "PipelineSpec.build"}
)

// packageSource is the package's non-test source, parsed, and the export data of what it
// imports, so a test can type-check it with some of its functions replaced.
type packageSource struct {
	path    string
	fset    *token.FileSet
	files   []*ast.File
	exports map[string]string
}

// loadPackageSource asks the go command for the package's files, as the build sees them, and for
// the export data of its dependencies, which the go command compiles if it must.
func loadPackageSource(t *testing.T) packageSource {
	t.Helper()
	var stderr bytes.Buffer
	cmd := exec.Command("go", "list", "-export", "-deps", "-json=ImportPath,Dir,Export,GoFiles,DepOnly", ".")
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	require.NoError(t, err, stderr.String())

	source := packageSource{fset: token.NewFileSet(), exports: map[string]string{}}
	decoder := json.NewDecoder(bytes.NewReader(out))
	for decoder.More() {
		var pkg struct {
			ImportPath, Dir, Export string
			GoFiles                 []string
			DepOnly                 bool
		}
		require.NoError(t, decoder.Decode(&pkg))
		source.exports[pkg.ImportPath] = pkg.Export
		if pkg.DepOnly {
			continue
		}
		source.path = pkg.ImportPath
		for _, name := range pkg.GoFiles {
			src, err := os.ReadFile(filepath.Join(pkg.Dir, name))
			require.NoError(t, err)
			file, err := parser.ParseFile(source.fset, name, src, parser.SkipObjectResolution)
			require.NoError(t, err)
			source.files = append(source.files, file)
		}
	}
	require.NotEmpty(t, source.files, "the package's files are listed")
	return source
}

// factProblems type-checks the package, with the functions mutation declares, when it is not
// empty, in place of those of the same names, and returns what could make Build's work miss
// work it does, each with the function, the field and what to do:
//
//   - a write of a fact outside factWriters: of a field of transitionFacts other than work and
//     rules, which the pass over transitions sets once, or of the keys of a rewindRange or a
//     transitionRule, directly or through a local variable taken from the facts;
//   - a method of factWriters that loops or does not add to work;
//   - outside factLoopers and the passes, a loop that walks the facts: a range over, or a for
//     whose header uses, a field of transitionFacts or a value taken from one, or a loop whose
//     body indexes a slice of the facts or names a method of transitionFacts other than
//     factWriters and factLoopers, which may read any fact; and such a loop of factLoopers that
//     does not add to work;
//   - outside factLoopers, a slice, map or pointer taken from the facts that is passed to a
//     function other than len, cap, delete or append's first argument, returned, or stored
//     anywhere but a local variable or the facts, since a loop over it there would go unseen;
//   - in a function that names a field or a method of transitionFacts, a goto or a call of the
//     function itself, either of which can walk the facts without a loop.
//
// Selectors are resolved by type, so a field of another type is never taken for a fact. The
// check guards against the likely regressions and is not a proof. It does not see a walk that
// reaches the facts in other ways: through recursion across functions; through a function
// value, an interface or a closure; through a method called on a value taken from them, such
// as a rule's skips, which the check does not follow; or through another name for a slice the
// facts were given, as table holds target and rules. Nor does it see a loop over something
// else, such as the keys a step reads, that looks the facts up in a map by each element, which
// is linear only when what it walks is.
func (s packageSource) factProblems(t *testing.T, mutation string) []string {
	t.Helper()
	files := s.files
	if mutation != "" {
		file, err := parser.ParseFile(s.fset, "mutation.go", "package gepa\n"+mutation, parser.SkipObjectResolution)
		require.NoError(t, err)
		files = withFuncs(files, file)
	}
	var errs []error
	config := types.Config{
		Importer: importer.ForCompiler(s.fset, "gc", func(path string) (io.ReadCloser, error) {
			return os.Open(s.exports[path])
		}),
		// a replaced function may leave an import or a variable unused, which does not matter here.
		Error: func(err error) {
			if typeErr, ok := err.(types.Error); !ok || !typeErr.Soft {
				errs = append(errs, err)
			}
		},
	}
	info := &types.Info{
		Types:      map[ast.Expr]types.TypeAndValue{},
		Defs:       map[*ast.Ident]types.Object{},
		Uses:       map[*ast.Ident]types.Object{},
		Selections: map[*ast.SelectorExpr]*types.Selection{},
	}
	pkg, _ := config.Check(s.path, s.fset, files, info)
	require.Empty(t, errs, "the package type-checks")

	c := newFactChecker(t, s.fset, pkg, info)
	for _, file := range files {
		for _, decl := range file.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok && fn.Body != nil {
				c.checkFunc(fn)
			}
		}
	}
	return c.problems
}

// withFuncs returns files with the functions and methods that file declares in place of those
// with the same names, and file added.
func withFuncs(files []*ast.File, file *ast.File) []*ast.File {
	replaced := map[string]bool{}
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			replaced[funcName(fn)] = true
		}
	}
	out := make([]*ast.File, 0, len(files)+1)
	for _, original := range files {
		copied := *original
		copied.Decls = slices.DeleteFunc(slices.Clone(original.Decls), func(decl ast.Decl) bool {
			fn, ok := decl.(*ast.FuncDecl)
			return ok && replaced[funcName(fn)]
		})
		out = append(out, &copied)
	}
	return append(out, file)
}

// funcName names fn as the source check reports it: its receiver's type and its name for a
// method, its name for a function.
func funcName(fn *ast.FuncDecl) string {
	if fn.Recv == nil || len(fn.Recv.List) == 0 {
		return fn.Name.Name
	}
	recv := fn.Recv.List[0].Type
	if star, ok := recv.(*ast.StarExpr); ok {
		recv = star.X
	}
	switch generic := recv.(type) {
	case *ast.IndexExpr:
		recv = generic.X
	case *ast.IndexListExpr:
		recv = generic.X
	}
	return types.ExprString(recv) + "." + fn.Name.Name
}

// factChecker applies the rules of factProblems to the functions of a type-checked package.
// facts holds the fields of transitionFacts and written those that only factWriters write, the
// keys of a rewindRange and a transitionRule among them, each with the name problems give it.
type factChecker struct {
	fset     *token.FileSet
	pkg      *types.Package
	info     *types.Info
	facts    map[*types.Var]string
	written  map[*types.Var]string
	work     *types.Var
	problems []string
}

// funcFacts is what the rules need to know of the function they check: its name, whether it is
// one of factWriters or factLoopers, its outermost loops when it is one of factPasses, and its
// local variables that hold a value taken from the facts, each with the field of
// transitionFacts it was taken from.
type funcFacts struct {
	name    string
	writer  bool
	looper  bool
	passes  map[ast.Node]bool
	tainted map[types.Object]*types.Var
}

func newFactChecker(t *testing.T, fset *token.FileSet, pkg *types.Package, info *types.Info) *factChecker {
	t.Helper()
	c := &factChecker{
		fset:    fset,
		pkg:     pkg,
		info:    info,
		facts:   map[*types.Var]string{},
		written: map[*types.Var]string{},
	}
	fields := func(typeName string) *types.Struct {
		obj := pkg.Scope().Lookup(typeName)
		require.NotNil(t, obj, "type %s is declared", typeName)
		s, ok := obj.Type().Underlying().(*types.Struct)
		require.True(t, ok, "type %s is a struct", typeName)
		return s
	}
	for field := range fields("transitionFacts").Fields() {
		name := "transitionFacts." + field.Name()
		c.facts[field] = name
		switch field.Name() {
		case "work":
			c.work = field
		case "rules":
		default:
			c.written[field] = name
		}
	}
	for _, typeName := range []string{"rewindRange", "transitionRule"} {
		for field := range fields(typeName).Fields() {
			if field.Name() == "keys" {
				c.written[field] = typeName + ".keys"
			}
		}
	}
	require.NotNil(t, c.work, "transitionFacts counts its work")
	require.Contains(t, slices.Collect(maps.Values(c.written)), "transitionFacts.skips")
	return c
}

func (c *factChecker) report(node ast.Node, format string, args ...any) {
	c.problems = append(c.problems, fmt.Sprintf("%s: ", c.fset.Position(node.Pos()))+fmt.Sprintf(format, args...))
}

// checkFunc applies the rules to fn and the function literals in it.
func (c *factChecker) checkFunc(fn *ast.FuncDecl) {
	self, touches := c.info.Defs[fn.Name], c.touches(fn.Body)
	f := &funcFacts{
		name:    funcName(fn),
		writer:  c.factsMethod(fn, factWriters),
		looper:  c.factsMethod(fn, factLoopers),
		passes:  map[ast.Node]bool{},
		tainted: c.taint(fn.Body),
	}
	if slices.Contains(factPasses, f.name) {
		ast.Inspect(fn.Body, func(node ast.Node) bool {
			switch node.(type) {
			case *ast.ForStmt, *ast.RangeStmt:
				f.passes[node] = true
				return false
			case *ast.FuncLit:
				return false
			}
			return true
		})
	}
	ast.Inspect(fn.Body, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.IncDecStmt:
			c.checkWrite(f, node.X)
		case *ast.AssignStmt:
			for i, lhs := range node.Lhs {
				c.checkWrite(f, lhs)
				if node.Tok == token.ASSIGN {
					c.checkStore(f, lhs, valueAt(node.Rhs, i))
				}
			}
		case *ast.CallExpr:
			c.checkCall(f, node)
			if callee := c.callee(node); touches && callee != nil && callee == self {
				c.report(node, "%s calls itself; %s", f.name, loopAdvice)
			}
		case *ast.BranchStmt:
			if touches && node.Tok == token.GOTO {
				c.report(node, "%s uses goto; %s", f.name, loopAdvice)
			}
		case *ast.ReturnStmt:
			for _, result := range node.Results {
				if from := c.escapes(result, f.tainted); from != nil && !f.looper {
					c.report(result, "%s returns %s, where a loop over it goes uncounted; %s",
						f.name, c.facts[from], loopAdvice)
				}
			}
		case *ast.RangeStmt, *ast.ForStmt:
			c.checkLoop(f, node.(ast.Stmt))
		}
		return true
	})
	if f.writer && !c.counts(fn.Body) {
		c.report(fn, "counting method %s does not add to work; %s", f.name, countAdvice)
	}
}

// factsMethod reports whether fn is a method of transitionFacts named in names.
func (c *factChecker) factsMethod(fn *ast.FuncDecl, names []string) bool {
	return slices.Contains(names, fn.Name.Name) && c.onFacts(c.info.Defs[fn.Name])
}

// onFacts reports whether obj is a method of transitionFacts.
func (c *factChecker) onFacts(obj types.Object) bool {
	method, ok := obj.(*types.Func)
	if !ok || method.Signature().Recv() == nil {
		return false
	}
	recv := method.Signature().Recv().Type()
	if pointer, ok := recv.(*types.Pointer); ok {
		recv = pointer.Elem()
	}
	return types.Identical(recv, c.pkg.Scope().Lookup("transitionFacts").Type())
}

// touches reports whether body selects a field or a method of transitionFacts.
func (c *factChecker) touches(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(node ast.Node) bool {
		if selector, ok := node.(*ast.SelectorExpr); ok {
			found = c.facts[c.field(selector)] != "" || c.onFacts(c.info.Uses[selector.Sel])
		}
		return !found
	})
	return found
}

// callee returns the function or method that call names, or nil when it names none.
func (c *factChecker) callee(call *ast.CallExpr) *types.Func {
	fun := ast.Unparen(call.Fun)
	if selector, ok := fun.(*ast.SelectorExpr); ok {
		fun = selector.Sel
	}
	if ident, ok := fun.(*ast.Ident); ok {
		if fn, ok := c.info.Uses[ident].(*types.Func); ok {
			return fn.Origin()
		}
	}
	return nil
}

// taint returns the local variables of body that hold a value taken from the facts, each with
// the field of transitionFacts it was taken from.
func (c *factChecker) taint(body *ast.BlockStmt) map[types.Object]*types.Var {
	tainted := map[types.Object]*types.Var{}
	mark := func(expr ast.Expr, from *types.Var) bool {
		ident, ok := ast.Unparen(expr).(*ast.Ident)
		if !ok || from == nil {
			return false
		}
		obj := c.info.ObjectOf(ident)
		if obj == nil || tainted[obj] != nil {
			return false
		}
		tainted[obj] = from
		return true
	}
	for changed := true; changed; {
		changed = false
		ast.Inspect(body, func(node ast.Node) bool {
			switch node := node.(type) {
			case *ast.AssignStmt:
				for i, lhs := range node.Lhs {
					changed = mark(lhs, c.takenFrom(valueAt(node.Rhs, i), tainted)) || changed
				}
			case *ast.ValueSpec:
				for i, name := range node.Names {
					changed = mark(name, c.takenFrom(valueAt(node.Values, i), tainted)) || changed
				}
			case *ast.RangeStmt:
				from := c.takenFrom(node.X, tainted)
				changed = mark(node.Key, from) || changed
				changed = mark(node.Value, from) || changed
			}
			return true
		})
	}
	return tainted
}

// valueAt returns the value assigned to the i-th of the names on the left of an assignment
// whose right holds values: the one value of a call or a comma-ok form, the i-th otherwise.
func valueAt(values []ast.Expr, i int) ast.Expr {
	switch {
	case len(values) == 1:
		return values[0]
	case i < len(values):
		return values[i]
	}
	return nil
}

// takenFrom returns the field of transitionFacts that node uses, directly or through a tainted
// local variable, or nil when it uses none.
func (c *factChecker) takenFrom(node ast.Node, tainted map[types.Object]*types.Var) *types.Var {
	if node == nil {
		return nil
	}
	var from *types.Var
	ast.Inspect(node, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.SelectorExpr:
			if field := c.field(node); c.facts[field] != "" {
				from = field
			}
		case *ast.Ident:
			from = tainted[c.info.Uses[node]]
		}
		return from == nil
	})
	return from
}

// field returns the field that selector selects, or nil when it selects none.
func (c *factChecker) field(selector *ast.SelectorExpr) *types.Var {
	if selection := c.info.Selections[selector]; selection != nil && selection.Kind() == types.FieldVal {
		field, _ := selection.Obj().(*types.Var)
		return field
	}
	return nil
}

// factAt returns the field a write to target writes into: a field of transitionFacts or the keys
// of a rewind range or rule that target selects, or, for a write through a local variable taken
// from the facts that shares their memory, the field it was taken from. It returns nil when
// target is not in the facts.
func (c *factChecker) factAt(target ast.Expr, tainted map[types.Object]*types.Var) *types.Var {
	through := false
	for {
		switch expr := ast.Unparen(target).(type) {
		case *ast.SelectorExpr:
			if field := c.field(expr); c.facts[field] != "" || c.written[field] != "" {
				return field
			}
			target, through = expr.X, true
		case *ast.IndexExpr:
			target, through = expr.X, true
		case *ast.SliceExpr:
			target, through = expr.X, true
		case *ast.StarExpr:
			target, through = expr.X, true
		case *ast.Ident:
			if local, ok := c.info.Uses[expr].(*types.Var); ok && through && holdsReference(local.Type()) {
				return tainted[local]
			}
			return nil
		default:
			return nil
		}
	}
}

// escapes returns the field of transitionFacts that expr's value shares memory with, or nil when
// it shares none.
func (c *factChecker) escapes(expr ast.Expr, tainted map[types.Object]*types.Var) *types.Var {
	if !holdsReference(c.info.TypeOf(expr)) {
		return nil
	}
	return c.takenFrom(expr, tainted)
}

// holdsReference reports whether a value of type t shares memory with its copies: a slice, map,
// pointer or channel, or an array or struct that holds one.
func holdsReference(t types.Type) bool {
	if t == nil {
		return false
	}
	switch t := t.Underlying().(type) {
	case *types.Slice, *types.Map, *types.Pointer, *types.Chan:
		return true
	case *types.Array:
		return holdsReference(t.Elem())
	case *types.Struct:
		for field := range t.Fields() {
			if holdsReference(field.Type()) {
				return true
			}
		}
	}
	return false
}

// checkWrite refuses a write of a fact outside factWriters.
func (c *factChecker) checkWrite(f *funcFacts, target ast.Expr) {
	if f.writer {
		return
	}
	if fact := c.written[c.factAt(target, f.tainted)]; fact != "" {
		c.report(target, "%s writes %s outside the counting methods; %s", f.name, fact, writeAdvice)
	}
}

// checkStore refuses, outside factLoopers, a value taken from the facts stored anywhere but a
// local variable or the facts.
func (c *factChecker) checkStore(f *funcFacts, lhs, value ast.Expr) {
	if f.looper {
		return
	}
	if ident, ok := ast.Unparen(lhs).(*ast.Ident); ok {
		if v, ok := c.info.Uses[ident].(*types.Var); !ok || v.Parent() != c.pkg.Scope() {
			return
		}
	} else if c.factAt(lhs, f.tainted) != nil {
		return
	}
	if from := c.escapes(value, f.tainted); from != nil {
		c.report(lhs, "%s stores %s in %s, where a loop over it goes uncounted; %s",
			f.name, c.facts[from], types.ExprString(lhs), loopAdvice)
	}
}

// checkCall refuses a call outside factWriters that writes into a fact, and one outside
// factLoopers that is passed a value taken from the facts. A method called on such a value is
// not followed.
func (c *factChecker) checkCall(f *funcFacts, call *ast.CallExpr) {
	if callee, ok := c.info.Types[call.Fun]; ok && callee.IsType() {
		return
	}
	args := call.Args
	if c.writesFirstArgument(call) && !f.writer {
		if fact := c.written[c.factAt(args[0], f.tainted)]; fact != "" {
			c.report(args[0], "%s writes %s outside the counting methods; %s", f.name, fact, writeAdvice)
			args = args[1:]
		}
	}
	if ident, ok := ast.Unparen(call.Fun).(*ast.Ident); ok {
		if builtin, ok := c.info.Uses[ident].(*types.Builtin); ok {
			switch builtin.Name() {
			case "len", "cap", "delete":
				return
			case "append":
				args = args[1:]
			}
		}
	}
	if f.looper {
		return
	}
	for _, arg := range args {
		if from := c.escapes(arg, f.tainted); from != nil {
			c.report(arg, "%s passes %s to %s, where a loop over it goes uncounted; %s",
				f.name, c.facts[from], types.ExprString(call.Fun), loopAdvice)
		}
	}
}

// writesFirstArgument reports whether call writes into its first argument: delete, clear, copy,
// maps.Copy, maps.Insert or maps.DeleteFunc.
func (c *factChecker) writesFirstArgument(call *ast.CallExpr) bool {
	if len(call.Args) == 0 {
		return false
	}
	switch fun := ast.Unparen(call.Fun).(type) {
	case *ast.Ident:
		builtin, ok := c.info.Uses[fun].(*types.Builtin)
		return ok && slices.Contains([]string{"delete", "clear", "copy"}, builtin.Name())
	case *ast.SelectorExpr:
		fn, ok := c.info.Uses[fun.Sel].(*types.Func)
		return ok && fn.Pkg() != nil && fn.Pkg().Path() == "maps" &&
			slices.Contains([]string{"Copy", "Insert", "DeleteFunc"}, fn.Name())
	}
	return false
}

// checkLoop refuses any loop in a method of factWriters and, outside factLoopers and the passes,
// a loop that walks the facts or calls a method that reads them; a loop of factLoopers that does
// either must add to work.
func (c *factChecker) checkLoop(f *funcFacts, loop ast.Stmt) {
	if f.writer {
		c.report(loop, "counting method %s loops; %s", f.name, loopAdvice)
		return
	}
	var from *types.Var
	var body *ast.BlockStmt
	switch loop := loop.(type) {
	case *ast.RangeStmt:
		from, body = c.takenFrom(loop.X, f.tainted), loop.Body
	case *ast.ForStmt:
		body = loop.Body
		for _, part := range []ast.Node{loop.Init, loop.Cond, loop.Post} {
			if from == nil {
				from = c.takenFrom(part, f.tainted)
			}
		}
	}
	if from == nil {
		from = c.indexedFact(body, f.tainted)
	}
	walk := ""
	if from != nil {
		walk = "loops over " + c.facts[from]
	} else if reader := c.readerIn(loop); reader != "" {
		walk = "calls " + reader + " in a loop"
	}
	switch {
	case walk == "", f.passes[loop]:
	case !f.looper:
		c.report(loop, "%s %s; %s", f.name, walk, loopAdvice)
	case !c.counts(body):
		c.report(loop, "%s %s without adding to work; %s", f.name, walk, countAdvice)
	}
}

// readerIn returns the method of transitionFacts other than factWriters and factLoopers that
// loop names, outside the loops in it, or "" when it names none. Such a method may read any of
// the facts, so a call of it is taken for a read.
func (c *factChecker) readerIn(loop ast.Stmt) string {
	reader := ""
	ast.Inspect(loop, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			return node == loop
		case *ast.SelectorExpr:
			name := node.Sel.Name
			if c.onFacts(c.info.Uses[node.Sel]) && !slices.Contains(factWriters, name) &&
				!slices.Contains(factLoopers, name) {
				reader = "transitionFacts." + name
			}
		}
		return reader == ""
	})
	return reader
}

// indexedFact returns the field of transitionFacts whose slice or array body indexes, outside
// the loops in it, which are checked on their own, or nil when it indexes none.
func (c *factChecker) indexedFact(body *ast.BlockStmt, tainted map[types.Object]*types.Var) *types.Var {
	var from *types.Var
	ast.Inspect(body, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			return false
		case *ast.IndexExpr:
			if typ := c.info.TypeOf(node.X); typ != nil {
				if _, lookup := typ.Underlying().(*types.Map); !lookup {
					from = c.takenFrom(node.X, tainted)
				}
			}
		}
		return from == nil
	})
	return from
}

// counts reports whether node adds to work: work++, work += 1 or work += n.
func (c *factChecker) counts(node ast.Node) bool {
	found := false
	ast.Inspect(node, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.IncDecStmt:
			found = found || node.Tok == token.INC && c.isWork(node.X)
		case *ast.AssignStmt:
			found = found || node.Tok == token.ADD_ASSIGN && c.isWork(node.Lhs[0])
		}
		return !found
	})
	return found
}

func (c *factChecker) isWork(expr ast.Expr) bool {
	selector, ok := ast.Unparen(expr).(*ast.SelectorExpr)
	return ok && c.field(selector) == c.work
}
