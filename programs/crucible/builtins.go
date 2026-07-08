package crucible

import (
	"embed"
	"encoding/json"
	"fmt"
)

//go:embed built-ins/manifest.json
var builtInManifestFS embed.FS

// BuiltIn describes a runnable Crucible built-in program.
type BuiltIn struct {
	Name         string   `json:"name"`
	Path         string   `json:"path"`
	Description  string   `json:"description"`
	Capabilities []string `json:"capabilities,omitempty"`
}

type builtInManifest struct {
	BuiltIns []BuiltIn `json:"built_ins"`
}

// BuiltIns returns the registered built-in evaluation programs bundled with Crucible.
func BuiltIns() ([]BuiltIn, error) {
	manifest, err := loadBuiltInManifest()
	if err != nil {
		return nil, err
	}
	return manifest.BuiltIns, nil
}

// BuiltInInfo returns one built-in by name.
func BuiltInInfo(name string) (BuiltIn, bool, error) {
	builtIns, err := BuiltIns()
	if err != nil {
		return BuiltIn{}, false, err
	}
	for _, builtIn := range builtIns {
		if builtIn.Name == name {
			return builtIn, true, nil
		}
	}
	return BuiltIn{}, false, nil
}

func loadBuiltInManifest() (builtInManifest, error) {
	data, err := builtInManifestFS.ReadFile("built-ins/manifest.json")
	if err != nil {
		return builtInManifest{}, err
	}
	var manifest builtInManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return builtInManifest{}, fmt.Errorf("decode built-ins manifest: %w", err)
	}
	return manifest, nil
}
