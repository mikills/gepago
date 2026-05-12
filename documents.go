package gepa

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Document is loaded text plus source metadata for extraction workflows.
type Document struct {
	ID       string         `json:"id"`
	Path     string         `json:"path"`
	Name     string         `json:"name"`
	Text     string         `json:"text"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// DocumentLoader loads a document from a path into text and metadata.
type DocumentLoader interface {
	LoadDocument(ctx context.Context, path string) (Document, error)
}

// TextFileDocumentLoader loads plain text files with extension and size checks.
type TextFileDocumentLoader struct {
	MaxBytes          int64
	AllowedExtensions []string
}

// LoadDocument reads a text file and returns its absolute path, text, and metadata.
func (l TextFileDocumentLoader) LoadDocument(ctx context.Context, path string) (Document, error) {
	if err := validateDocumentPath(ctx, path, l.MaxBytes, l.AllowedExtensions); err != nil {
		return Document{}, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Document{}, err
	}
	if err := ctx.Err(); err != nil {
		return Document{}, err
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return Document{}, err
	}
	return textDocumentFromBytes(path, abs, data), nil
}

func validateDocumentPath(ctx context.Context, path string, maxBytes int64, allowedExtensions []string) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("document path is required")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if err := validateDocumentFile(path, info, maxBytes); err != nil {
		return err
	}
	return validateDocumentExtension(path, allowedExtensions)
}

func validateDocumentFile(path string, info os.FileInfo, maxBytes int64) error {
	if info.IsDir() {
		return fmt.Errorf("document path %q is a directory", path)
	}
	if maxBytes > 0 && info.Size() > maxBytes {
		return fmt.Errorf("document %q is %d bytes, max is %d", path, info.Size(), maxBytes)
	}
	return nil
}

func textDocumentFromBytes(path string, abs string, data []byte) Document {
	return Document{
		ID:       documentIDFromPath(path),
		Path:     abs,
		Name:     filepath.Base(path),
		Text:     string(data),
		Metadata: map[string]any{"bytes": len(data), "extension": documentExtension(path)},
	}
}

func validateDocumentExtension(path string, allowed []string) error {
	ext := documentExtension(path)
	if ext == "" {
		return fmt.Errorf("document %q has no supported extension", path)
	}
	allowedExtensions := allowed
	if len(allowedExtensions) == 0 {
		allowedExtensions = defaultDocumentExtensions
	}
	for _, candidate := range allowedExtensions {
		if strings.EqualFold(ext, strings.TrimSpace(candidate)) {
			return nil
		}
	}
	return fmt.Errorf("document %q has unsupported extension %q", path, ext)
}

func documentExtension(path string) string {
	name := strings.ToLower(filepath.Base(path))
	for _, ext := range compoundDocumentExtensions {
		if strings.HasSuffix(name, ext) {
			return ext
		}
	}
	return strings.ToLower(filepath.Ext(name))
}

func documentIDFromPath(path string) string {
	name := filepath.Base(path)
	ext := documentExtension(path)
	if ext == "" {
		return strings.TrimSuffix(name, filepath.Ext(name))
	}
	return strings.TrimSuffix(name, name[len(name)-len(ext):])
}

var defaultDocumentExtensions = []string{".parsed.txt", ".txt", ".md", ".json", ".csv"}
var compoundDocumentExtensions = []string{".parsed.txt"}

// DocumentExample is an Example payload for document extraction tasks.
type DocumentExample struct {
	Document Document `json:"document"`
	Expected string   `json:"expected"`
	Label    string   `json:"label,omitempty"`
}

// LoadDocumentExamples loads document cases into optimisation examples.
func LoadDocumentExamples(ctx context.Context, loader DocumentLoader, cases []DocumentExampleCase) ([]Example, error) {
	if loader == nil {
		return nil, errors.New("document loader is required")
	}
	examples := make([]Example, 0, len(cases))
	for _, item := range cases {
		doc, err := loader.LoadDocument(ctx, item.Path)
		if err != nil {
			return nil, err
		}
		id := item.ID
		if strings.TrimSpace(id) == "" {
			id = doc.ID
		}
		examples = append(
			examples,
			Example{ID: id, Input: DocumentExample{Document: doc, Expected: item.Expected, Label: item.Label}},
		)
	}
	return examples, nil
}

// DocumentExampleCase describes one document fixture and expected extraction target.
type DocumentExampleCase struct {
	ID       string `json:"id"`
	Path     string `json:"path"`
	Expected string `json:"expected"`
	Label    string `json:"label,omitempty"`
}
