package gepa

import "testing"

func TestSplitExamples(t *testing.T) {
	examples := []Example{{ID: "a"}, {ID: "b"}, {ID: "c"}, {ID: "d"}}
	split, err := SplitExamples(examples, SplitConfig{ValidationSize: 2})
	if err != nil {
		t.Fatalf("SplitExamples() error = %v", err)
	}
	if len(split.Validation) != 2 || len(split.Train) != 2 {
		t.Fatalf("split = %#v", split)
	}
	if split.Validation[0].ID != "a" || split.Train[0].ID != "c" {
		t.Fatalf("split order = %#v", split)
	}

	ratioSplit, err := SplitExamples(examples, SplitConfig{ValidationRatio: 0.2})
	if err != nil {
		t.Fatalf("SplitExamples() ratio error = %v", err)
	}
	if len(ratioSplit.Validation) != 1 || len(ratioSplit.Train) != 3 {
		t.Fatalf("ratio split = %#v", ratioSplit)
	}
}
