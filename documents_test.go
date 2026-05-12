package gepa

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestTextFileDocumentLoader(t *testing.T) {
	t.Run("loads parsed text file", func(t *testing.T) {
		path := filepath.Join("testdata", "documents", "apple-2023-10k-balance-sheet.parsed.txt")
		doc, err := TextFileDocumentLoader{MaxBytes: 2_000_000}.LoadDocument(context.Background(), path)
		if err != nil {
			t.Fatalf("LoadDocument() error = %v", err)
		}
		if doc.Name != "apple-2023-10k-balance-sheet.parsed.txt" {
			t.Fatalf("Name = %q", doc.Name)
		}
		if len(doc.Text) == 0 || doc.ID == "" || doc.Path == "" {
			t.Fatalf("document not populated: %#v", doc)
		}
		if doc.ID != "apple-2023-10k-balance-sheet" || doc.Metadata["extension"] != ".parsed.txt" {
			t.Fatalf("document identity metadata = id %q metadata %#v", doc.ID, doc.Metadata)
		}
	})

	t.Run("respects max bytes", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "doc.txt")
		if err := os.WriteFile(path, []byte("hello"), 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
		_, err := TextFileDocumentLoader{MaxBytes: 2}.LoadDocument(context.Background(), path)
		if err == nil {
			t.Fatal("LoadDocument() error = nil, want max bytes error")
		}
	})

	t.Run("loads public SEC fixtures", func(t *testing.T) {
		cases := []DocumentExampleCase{
			{
				ID:       "apple",
				Path:     filepath.Join("testdata", "documents", "apple-2023-10k-balance-sheet.parsed.txt"),
				Expected: "29508",
			},
			{
				ID:       "microsoft",
				Path:     filepath.Join("testdata", "documents", "microsoft-2023-10k-balance-sheet.parsed.txt"),
				Expected: "48688",
			},
			{
				ID:       "tesla",
				Path:     filepath.Join("testdata", "documents", "tesla-2023-10k-balance-sheet.parsed.txt"),
				Expected: "3508",
			},
		}
		examples, err := LoadDocumentExamples(context.Background(), TextFileDocumentLoader{}, cases)
		if err != nil {
			t.Fatalf("LoadDocumentExamples() error = %v", err)
		}
		if len(examples) != 3 {
			t.Fatalf("examples len = %d", len(examples))
		}
	})

	t.Run("rejects unsupported extensions", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "doc.pdf")
		if err := os.WriteFile(path, []byte("not parsed text"), 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
		_, err := TextFileDocumentLoader{}.LoadDocument(context.Background(), path)
		if err == nil {
			t.Fatal("LoadDocument() error = nil, want unsupported extension error")
		}
	})

	t.Run("allows configured extensions", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "doc.fixture")
		if err := os.WriteFile(path, []byte("fixture text"), 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
		doc, err := TextFileDocumentLoader{
			AllowedExtensions: []string{".fixture"},
		}.LoadDocument(
			context.Background(),
			path,
		)
		if err != nil {
			t.Fatalf("LoadDocument() error = %v", err)
		}
		if doc.Metadata["extension"] != ".fixture" {
			t.Fatalf("extension metadata = %#v", doc.Metadata)
		}
	})
}

func TestLoadDocumentExamples(t *testing.T) {
	t.Run("loads cases into examples", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "doc.txt")
		if err := os.WriteFile(path, []byte("accounts receivable 123"), 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
		examples, err := LoadDocumentExamples(
			context.Background(),
			TextFileDocumentLoader{},
			[]DocumentExampleCase{{ID: "ar", Path: path, Expected: "123", Label: "accounts receivable"}},
		)
		if err != nil {
			t.Fatalf("LoadDocumentExamples() error = %v", err)
		}
		if len(examples) != 1 || examples[0].ID != "ar" {
			t.Fatalf("examples = %#v", examples)
		}
		input, ok := examples[0].Input.(DocumentExample)
		if !ok || input.Expected != "123" || input.Document.Text == "" {
			t.Fatalf("input = %#v", examples[0].Input)
		}
	})
}
