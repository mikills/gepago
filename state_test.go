package gepa

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type stateObligation struct {
	Amount   int    `json:"amount"`
	Deadline string `json:"deadline"`
}

func TestKeyGetReturnsTypedValue(t *testing.T) {
	key := Key[stateObligation]{Name: "obligation"}
	state := NewState(nil)
	key.Set(state, stateObligation{Amount: 10, Deadline: "2026-06-30"})

	got, err := key.Get(state)
	require.NoError(t, err)
	require.Equal(t, stateObligation{Amount: 10, Deadline: "2026-06-30"}, got)
}

func TestKeyGetDecodesModelValues(t *testing.T) {
	state := NewState(Prediction{
		"obligation": map[string]any{"amount": json.Number("125000"), "deadline": "2026-06-30"},
		"count":      json.Number("3"),
	})

	obligation, err := Key[stateObligation]{Name: "obligation"}.Get(state)
	require.NoError(t, err)
	require.Equal(t, stateObligation{Amount: 125000, Deadline: "2026-06-30"}, obligation)

	count, err := Key[int]{Name: "count"}.Get(state)
	require.NoError(t, err)
	require.Equal(t, 3, count)
}

func TestKeyGetReportsDrift(t *testing.T) {
	state := NewState(Prediction{
		"renamed":      map[string]any{"amount_usd": 10, "deadline": "2026-06-30"},
		"reshaped":     map[string]any{"amount": "ten", "deadline": "2026-06-30"},
		"dropped":      map[string]any{"amount": 10},
		"recased":      map[string]any{"AMOUNT": 10, "deadline": "2026-06-30"},
		"null field":   map[string]any{"amount": nil, "deadline": "2026-06-30"},
		"null element": []any{1, nil},
		"nested":       []any{map[string]any{"amount": 1}},
		"fraction":     2.5,
		"null":         nil,
	})

	tests := []struct {
		name    string
		get     func() error
		wantErr string
	}{
		{
			name:    "missing key",
			get:     func() error { _, err := Key[int]{Name: "absent"}.Get(state); return err },
			wantErr: `state key "absent" is missing`,
		},
		{
			name: "renamed field",
			get: func() error {
				_, err := Key[stateObligation]{Name: "renamed"}.Get(state)
				return err
			},
			wantErr: `state key "renamed": decode as gepa.stateObligation: ` +
				`field "amount_usd" is not in gepa.stateObligation`,
		},
		{
			name: "dropped field",
			get: func() error {
				_, err := Key[stateObligation]{Name: "dropped"}.Get(state)
				return err
			},
			wantErr: `state key "dropped": decode as gepa.stateObligation: required field "deadline" is missing`,
		},
		{
			name: "recased field",
			get: func() error {
				_, err := Key[stateObligation]{Name: "recased"}.Get(state)
				return err
			},
			wantErr: `state key "recased": decode as gepa.stateObligation: field "AMOUNT" is not in gepa.stateObligation`,
		},
		{
			name: "null field",
			get: func() error {
				_, err := Key[stateObligation]{Name: "null field"}.Get(state)
				return err
			},
			wantErr: `state key "null field": decode as gepa.stateObligation: field "amount" is null`,
		},
		{
			name:    "null element",
			get:     func() error { _, err := Key[[]int]{Name: "null element"}.Get(state); return err },
			wantErr: `state key "null element": decode as []int: field "[1]" is null`,
		},
		{
			name: "nested dropped field",
			get: func() error {
				_, err := Key[[]stateObligation]{Name: "nested"}.Get(state)
				return err
			},
			wantErr: `state key "nested": decode as []gepa.stateObligation: required field "[0].deadline" is missing`,
		},
		{
			name: "reshaped field",
			get: func() error {
				_, err := Key[stateObligation]{Name: "reshaped"}.Get(state)
				return err
			},
			wantErr: `state key "reshaped": decode as gepa.stateObligation: ` +
				`json: cannot unmarshal string into Go struct field stateObligation.amount of type int`,
		},
		{
			name:    "lossy number",
			get:     func() error { _, err := Key[int]{Name: "fraction"}.Get(state); return err },
			wantErr: `state key "fraction": decode as int: json: cannot unmarshal number 2.5 into Go value of type int`,
		},
		{
			name:    "null value",
			get:     func() error { _, err := Key[string]{Name: "null"}.Get(state); return err },
			wantErr: `state key "null" is null, want string`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.EqualError(t, tt.get(), tt.wantErr)
		})
	}
}

func TestKeyGetAllowsNullForNillableTypes(t *testing.T) {
	state := NewState(Prediction{"items": nil})

	got, err := Key[[]string]{Name: "items"}.Get(state)
	require.NoError(t, err)
	require.Nil(t, got)
}

func TestNewStateCopiesValues(t *testing.T) {
	values := Prediction{"document": "cash 10"}
	state := NewState(values)
	Key[string]{Name: "document"}.Set(state, "changed")

	require.Equal(t, "cash 10", values["document"])
}

func TestStepScopeLimitsAccess(t *testing.T) {
	scope := &stepScope{
		reads:   map[string]reflect.Type{"document": reflect.TypeFor[string]()},
		writes:  map[string]reflect.Type{"amount": reflect.TypeFor[int]()},
		pending: map[string]any{},
	}
	state := State{values: Prediction{"document": "cash 10", "secret": "x"}, scope: scope}

	_, err := Key[string]{Name: "secret"}.Get(state)
	require.EqualError(t, err, `state key "secret" is not a declared read`)

	Key[int]{Name: "amount"}.Set(state, 10)
	amount, err := Key[int]{Name: "amount"}.Get(state)
	require.NoError(t, err)
	require.Equal(t, 10, amount)
	require.Equal(t, Prediction{"document": "cash 10", "amount": 10}, state.Prediction())

	Key[string]{Name: "document"}.Set(state, "overwritten")
	require.EqualError(t, scope.err, `state key "document" is not a declared write`)
	require.Equal(t, "cash 10", state.values["document"])
}

type validatedObligation struct {
	Amount   int    `json:"amount"`
	Currency string `json:"currency"`
}

func (o validatedObligation) Validate() error {
	if o.Currency == "" {
		return errors.New("currency is required")
	}
	return nil
}

type pointerValidated struct {
	Name string `json:"name"`
}

func (p *pointerValidated) Validate() error {
	if p.Name == "" {
		return errors.New("name is required")
	}
	return nil
}

func TestKeyGetValidatesValues(t *testing.T) {
	key := Key[validatedObligation]{Name: "obligation"}

	decoded := NewState(Prediction{"obligation": map[string]any{"amount": 10, "currency": ""}})
	_, err := key.Get(decoded)
	require.EqualError(t, err, `state key "obligation": currency is required`)

	typed := NewState(nil)
	key.Set(typed, validatedObligation{Amount: 10})
	_, err = key.Get(typed)
	require.EqualError(t, err, `state key "obligation": currency is required`)

	key.Set(typed, validatedObligation{Amount: 10, Currency: "GBP"})
	got, err := key.Get(typed)
	require.NoError(t, err)
	require.Equal(t, validatedObligation{Amount: 10, Currency: "GBP"}, got)
}

func TestKeyGetValidatesPointerReceivers(t *testing.T) {
	state := NewState(Prediction{"value": map[string]any{"name": ""}, "absent": nil})

	_, err := Key[pointerValidated]{Name: "value"}.Get(state)
	require.EqualError(t, err, `state key "value": name is required`)

	_, err = Key[*pointerValidated]{Name: "value"}.Get(state)
	require.EqualError(t, err, `state key "value": name is required`)

	got, err := Key[*pointerValidated]{Name: "absent"}.Get(state)
	require.NoError(t, err)
	require.Nil(t, got)
}

type obligationBase struct {
	ID string `json:"id"`
}

type flexibleObligation struct {
	obligationBase
	Amount int       `json:"amount"`
	Due    time.Time `json:"due"`
	Note   string    `json:"note,omitempty"`
}

func TestKeyGetDecodesOptionalEmbeddedAndCustomFields(t *testing.T) {
	state := NewState(Prediction{
		"obligation": map[string]any{"id": "o-1", "amount": 10, "due": "2026-06-30T00:00:00Z"},
	})

	got, err := Key[flexibleObligation]{Name: "obligation"}.Get(state)
	require.NoError(t, err)
	require.Equal(t, "o-1", got.ID)
	require.Equal(t, 10, got.Amount)
	require.Equal(t, time.Date(2026, 6, 30, 0, 0, 0, 0, time.UTC), got.Due)
	require.Empty(t, got.Note)
}

func TestKeyGetReturnsCopies(t *testing.T) {
	key := Key[map[string][]int]{Name: "scores"}
	state := NewState(nil)
	key.Set(state, map[string][]int{"a": {1}})

	first, err := key.Get(state)
	require.NoError(t, err)
	first["a"][0] = 99
	first["b"] = []int{2}

	second, err := key.Get(state)
	require.NoError(t, err)
	require.Equal(t, map[string][]int{"a": {1}}, second)
}

func TestDeepCopyHandlesCycles(t *testing.T) {
	original := map[string]any{"name": "loop"}
	original["self"] = original

	copied := deepCopy(original).(map[string]any)

	self := copied["self"].(map[string]any)
	require.Equal(t, reflect.ValueOf(copied).Pointer(), reflect.ValueOf(self).Pointer())
	require.NotEqual(t, reflect.ValueOf(original).Pointer(), reflect.ValueOf(copied).Pointer())
	require.Equal(t, "loop", self["name"])
}
