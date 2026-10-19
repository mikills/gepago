package gepa

import (
	"errors"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"sync"
)

// toolName is the name every provider accepts for a tool or a middleware.
var toolName = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// ToolRegistry holds tools and chat middleware by name. Pipelines take it as PipelineSpec.Tools
// and agents as Agent.ToolRegistry; their steps and agents then name the tools they call, and
// the names are checked when the pipeline is built or the agent validated. Names are unique
// and never removed, so a name found once keeps meaning the same tool.
//
// The zero value is an empty registry ready for use. A ToolRegistry is safe for concurrent
// use and must not be copied after first use.
type ToolRegistry struct {
	mu         sync.RWMutex
	tools      map[string]registeredTool
	middleware map[string]ChatMiddleware
}

// registeredTool is a provider with the schema it reported when registered.
type registeredTool struct {
	provider ToolProvider
	schema   Tool
}

// NewToolRegistry returns a registry holding providers, registered in order.
func NewToolRegistry(providers ...ToolProvider) (*ToolRegistry, error) {
	registry := &ToolRegistry{}
	for _, provider := range providers {
		if err := registry.Register(provider); err != nil {
			return nil, err
		}
	}
	return registry, nil
}

// Register adds a tool under the name its schema gives. It rejects a nil provider, a schema
// that fails Tool.Validate, a name that is not 1 to 64 letters, digits, '_' or '-', a name
// already registered, and a provider whose Validate method fails. A panic in the provider's
// Schema or Validate method is returned as an error.
func (r *ToolRegistry) Register(provider ToolProvider) (err error) {
	if r == nil {
		return errors.New("tool registry is nil")
	}
	if isNil(provider) {
		return errors.New("register tool: provider is nil")
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("register tool: provider panicked: %v", recovered)
		}
	}()
	if validator, ok := provider.(interface{ Validate() error }); ok {
		if err := validator.Validate(); err != nil {
			return fmt.Errorf("register tool: %w", err)
		}
	}
	schema := provider.Schema()
	if err := schema.Validate(); err != nil {
		return fmt.Errorf("register tool %q: %w", schema.Name, err)
	}
	if !toolName.MatchString(schema.Name) {
		return fmt.Errorf("register tool %q: name must be 1 to 64 letters, digits, '_' or '-'", schema.Name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[schema.Name]; exists {
		return fmt.Errorf("register tool %q: a tool with this name is already registered", schema.Name)
	}
	if r.tools == nil {
		r.tools = map[string]registeredTool{}
	}
	r.tools[schema.Name] = registeredTool{provider: provider, schema: schema}
	return nil
}

// Lookup returns the tool registered under name.
func (r *ToolRegistry) Lookup(name string) (ToolProvider, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	tool, ok := r.tools[name]
	return tool.provider, ok
}

// Schemas returns the schema of every registered tool, as each reported it when registered,
// sorted by name.
func (r *ToolRegistry) Schemas() []Tool {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	schemas := make([]Tool, 0, len(r.tools))
	for _, name := range slices.Sorted(maps.Keys(r.tools)) {
		schemas = append(schemas, cloneTool(r.tools[name].schema))
	}
	return schemas
}

// schema returns the schema the named tool reported when registered.
func (r *ToolRegistry) schema(name string) Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return cloneTool(r.tools[name].schema)
}

// providers returns every registered tool, sorted by name.
func (r *ToolRegistry) providers() []ToolProvider {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	providers := make([]ToolProvider, 0, len(r.tools))
	for _, name := range slices.Sorted(maps.Keys(r.tools)) {
		providers = append(providers, r.tools[name].provider)
	}
	return providers
}

// RegisterMiddleware adds chat middleware under name, which must be 1 to 64 letters, digits,
// '_' or '-' and not already used by other middleware. Tools and middleware have separate
// names.
func (r *ToolRegistry) RegisterMiddleware(name string, middleware ChatMiddleware) error {
	if r == nil {
		return errors.New("tool registry is nil")
	}
	if !toolName.MatchString(name) {
		return fmt.Errorf("register chat middleware %q: name must be 1 to 64 letters, digits, '_' or '-'", name)
	}
	if middleware == nil {
		return fmt.Errorf("register chat middleware %q: middleware is nil", name)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.middleware[name]; exists {
		return fmt.Errorf("register chat middleware %q: middleware with this name is already registered", name)
	}
	if r.middleware == nil {
		r.middleware = map[string]ChatMiddleware{}
	}
	r.middleware[name] = middleware
	return nil
}

// LookupMiddleware returns the chat middleware registered under name.
func (r *ToolRegistry) LookupMiddleware(name string) (ChatMiddleware, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	middleware, ok := r.middleware[name]
	return middleware, ok
}

// WrapChat wraps client in the named middleware. The first name is the outermost wrapper: it
// sees each request first and each response last. An unknown name, a nil client, and
// middleware that panics or returns a nil client are errors.
func (r *ToolRegistry) WrapChat(client ChatClient, names ...string) (wrapped ChatClient, err error) {
	if isNil(client) {
		return nil, errors.New("wrap chat client: client is nil")
	}
	if err := r.checkMiddleware(names); err != nil {
		return nil, err
	}
	wrapped = client
	for i := len(names) - 1; i >= 0; i-- {
		middleware, _ := r.LookupMiddleware(names[i])
		if wrapped, err = applyMiddleware(names[i], middleware, wrapped); err != nil {
			return nil, err
		}
	}
	return wrapped, nil
}

// checkMiddleware checks that every name is registered middleware and none repeats.
func (r *ToolRegistry) checkMiddleware(names []string) error {
	seen := map[string]bool{}
	for _, name := range names {
		if seen[name] {
			return fmt.Errorf("chat middleware %q is listed twice", name)
		}
		seen[name] = true
		if _, ok := r.LookupMiddleware(name); !ok {
			if r == nil {
				return fmt.Errorf("chat middleware %q: no tool registry holds middleware", name)
			}
			return fmt.Errorf("chat middleware %q is not registered", name)
		}
	}
	return nil
}

func applyMiddleware(name string, middleware ChatMiddleware, next ChatClient) (wrapped ChatClient, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			wrapped, err = nil, fmt.Errorf("chat middleware %q panicked: %v", name, recovered)
		}
	}()
	wrapped = middleware(next)
	if isNil(wrapped) {
		return nil, fmt.Errorf("chat middleware %q returned a nil client", name)
	}
	return wrapped, nil
}

func cloneTool(tool Tool) Tool {
	tool.Parameters = slices.Clone(tool.Parameters)
	return tool
}
