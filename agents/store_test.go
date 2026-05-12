package agents

import (
	"encoding/json"
	"testing"
)

func TestStore(t *testing.T) {
	t.Run("copies input and returns copies", func(t *testing.T) {
		input := map[string]int{"a": 1}
		store := NewStore[string, int](input)
		input["a"] = 2
		if got := store.Get("a"); got != 1 {
			t.Fatalf("Get(a) = %d, want 1", got)
		}
		all := store.GetAll()
		all["a"] = 3
		if got := store.Get("a"); got != 1 {
			t.Fatalf("Get(a) after mutating copy = %d, want 1", got)
		}
	})

	t.Run("get or set only calls setter once", func(t *testing.T) {
		store := NewStore[string, int](nil)
		calls := 0
		first := store.GetOrSet("a", func() int { calls++; return 10 })
		second := store.GetOrSet("a", func() int { calls++; return 20 })
		if first != 10 || second != 10 || calls != 1 {
			t.Fatalf("first=%d second=%d calls=%d", first, second, calls)
		}
	})

	t.Run("limit only blocks new keys", func(t *testing.T) {
		store := NewStore[string, int](nil)
		if !store.SetIfLessThanLimit("a", 1, 1) {
			t.Fatal("first set was blocked")
		}
		if store.SetIfLessThanLimit("b", 2, 1) {
			t.Fatal("new key beyond limit was allowed")
		}
		if !store.SetIfLessThanLimit("a", 3, 1) {
			t.Fatal("existing key update was blocked")
		}
	})

	t.Run("json round trip", func(t *testing.T) {
		store := NewStore[string, int](map[string]int{"a": 1})
		data, err := json.Marshal(store)
		if err != nil {
			t.Fatalf("Marshal() error = %v", err)
		}
		var decoded Store[string, int]
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatalf("Unmarshal() error = %v", err)
		}
		if got := decoded.Get("a"); got != 1 {
			t.Fatalf("decoded Get(a) = %d, want 1", got)
		}
	})
}
