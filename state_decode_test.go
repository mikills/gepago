package gepa

import (
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// TestDecodeV2RejectsWhatTheDeletedChecksRejected proves that encoding/json/v2, with the options
// decodeJSONValue passes, refuses on its own what checkJSONShape used to look for, and that Get
// still words an unknown member as before.
func TestDecodeV2RejectsWhatTheDeletedChecksRejected(t *testing.T) {
	tests := []struct {
		name    string
		typ     reflect.Type
		data    string
		unknown bool
	}{
		{
			name:    "unknown member",
			typ:     reflect.TypeFor[stateObligation](),
			data:    `{"amount_usd":1,"deadline":"d"}`,
			unknown: true,
		},
		{
			name:    "case-mismatched name",
			typ:     reflect.TypeFor[stateObligation](),
			data:    `{"Amount":1,"deadline":"d"}`,
			unknown: true,
		},
		{
			name:    "nested unknown member",
			typ:     reflect.TypeFor[[]stateObligation](),
			data:    `[{"amount":1,"deadline":"d","note":"x"}]`,
			unknown: true,
		},
		{name: "lossy number", typ: reflect.TypeFor[int](), data: `2.5`},
		{name: "quoted number", typ: reflect.TypeFor[int](), data: `"3"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := jsonv2.Unmarshal([]byte(tt.data), reflect.New(tt.typ).Interface(), decodeOptions)
			require.Error(t, err)
			require.Equal(t, tt.unknown, errors.Is(err, jsonv2.ErrUnknownName))
		})
	}

	nested := NewState(Prediction{
		"list": []any{map[string]any{"amount": 1, "deadline": "d", "note": "x"}},
		"map":  map[string]any{"0": map[string]any{"amount": 1, "deadline": "d", "note": "x"}},
	})
	_, err := Key[[]stateObligation]{Name: "list"}.Get(nested)
	require.EqualError(
		t,
		err,
		`state key "list": decode as []gepa.stateObligation: field "[0].note" is not in gepa.stateObligation`,
	)
	_, err = Key[map[string]*stateObligation]{Name: "map"}.Get(nested)
	require.EqualError(
		t,
		err,
		`state key "map": decode as map[string]*gepa.stateObligation: field "0.note" is not in gepa.stateObligation`,
	)
}

// TestDecodeV2AcceptsWhatTheKeptChecksRefuse proves that encoding/json/v2 accepts each value
// checkJSONShape still refuses, or refuses it without the detail Get reports, so each check
// stays.
func TestDecodeV2AcceptsWhatTheKeptChecksRefuse(t *testing.T) {
	tests := []struct {
		name    string
		typ     reflect.Type
		data    string
		v2Err   bool
		wantErr string
	}{
		{
			name:    "missing required field",
			typ:     reflect.TypeFor[stateObligation](),
			data:    `{"amount":1}`,
			wantErr: `required field "deadline" is missing`,
		},
		{
			name:    "null into a field that cannot hold it",
			typ:     reflect.TypeFor[stateObligation](),
			data:    `{"amount":null,"deadline":"d"}`,
			wantErr: `field "amount" is null`,
		},
		{
			name:    "null element",
			typ:     reflect.TypeFor[[]int](),
			data:    `[1,null]`,
			wantErr: `field "[1]" is null`,
		},
		{
			name:    "float that underflows",
			typ:     reflect.TypeFor[float32](),
			data:    `1e-50`,
			wantErr: `value: number 1e-50 is too small for float32 and would decode as 0`,
		},
		{
			name:    "quoted float that underflows",
			typ:     reflect.TypeFor[quotedFloat](),
			data:    `{"small":"1e-50","ratio":null}`,
			wantErr: `field "small": number 1e-50 is too small for float32 and would decode as 0`,
		},
		{
			name:    `the string "null" in a ",string" field`,
			typ:     reflect.TypeFor[quotedInt](),
			data:    `{"n":"null","s":"\"x\""}`,
			wantErr: `field "n" is the string "null", which a ",string" field decodes as no value`,
		},
		{
			name:    `a leading zero in a ",string" field`,
			typ:     reflect.TypeFor[quotedInt](),
			data:    `{"n":"01","s":"\"x\""}`,
			wantErr: `field "n" is "01", which is not a JSON number`,
		},
		{
			name:    `a trailing point in a ",string" field`,
			typ:     reflect.TypeFor[quotedFloat](),
			data:    `{"small":"1.","ratio":null}`,
			wantErr: `field "small" is "1.", which is not a JSON number`,
		},
		{
			name:    `an infinity in a ",string" field`,
			typ:     reflect.TypeFor[quotedFloat](),
			data:    `{"small":"-Inf","ratio":null}`,
			wantErr: `field "small" is "-Inf", which is not a JSON number`,
		},
		{
			name:    `a hexadecimal float in a ",string" field`,
			typ:     reflect.TypeFor[quotedFloat](),
			data:    `{"small":"0x1p-2","ratio":null}`,
			wantErr: `field "small" is "0x1p-2", which is not a JSON number`,
		},
		{
			name:    "array of the wrong length",
			typ:     reflect.TypeFor[[2]int](),
			data:    `[1]`,
			v2Err:   true,
			wantErr: `value has 1 elements, want 2`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := jsonv2.Unmarshal([]byte(tt.data), reflect.New(tt.typ).Interface(), decodeOptions)
			if tt.v2Err {
				require.Error(t, err)
				require.NotContains(t, err.Error(), "elements, want", "v2 does not say how many elements it found")
			} else {
				require.NoError(t, err)
			}

			_, err = decodeJSONValue(tt.typ, json.RawMessage(tt.data))
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

type quotedScalars struct {
	S string `json:"s,string"`
	B bool   `json:"b,string"`
}

type durationField struct {
	D time.Duration `json:"d"`
}

type byteArrayField struct {
	B [4]byte `json:"b"`
}

type namedByte uint8

type namedBytesField struct {
	B []namedByte `json:"b"`
}

type bytesField struct {
	B []byte `json:"b"`
}

// TestDecodeOptionsKeepEncodingJSONForms proves that each encoding/json option in decodeOptions
// is needed: without it v2 refuses a form that json.Marshal writes and gepago's JSON Schema
// describes.
func TestDecodeOptionsKeepEncodingJSONForms(t *testing.T) {
	tests := []struct {
		name string
		data string
		out  any
		want any
	}{
		{
			name: `",string" on string and bool fields`,
			data: `{"s":"\"x\"","b":"true"}`,
			out:  &quotedScalars{},
			want: &quotedScalars{S: "x", B: true},
		},
		{
			name: "time.Duration as nanoseconds",
			data: `{"d":1500}`,
			out:  &durationField{},
			want: &durationField{D: 1500},
		},
		{
			name: "[N]byte as an array",
			data: `{"b":[1,2,3,4]}`,
			out:  &byteArrayField{},
			want: &byteArrayField{B: [4]byte{1, 2, 3, 4}},
		},
		{
			name: "[]~byte as base64",
			data: `{"b":"aGk="}`,
			out:  &namedBytesField{},
			want: &namedBytesField{B: []namedByte{'h', 'i'}},
		},
		{name: "[]byte as an array", data: `{"b":[104,105]}`, out: &bytesField{}, want: &bytesField{B: []byte("hi")}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fresh := reflect.New(reflect.TypeOf(tt.out).Elem()).Interface()
			require.Error(t, jsonv2.Unmarshal([]byte(tt.data), fresh, jsonv2.RejectUnknownMembers(true)))

			require.NoError(t, jsonv2.Unmarshal([]byte(tt.data), tt.out, decodeOptions))
			require.Equal(t, tt.want, tt.out)
		})
	}
}

type decodedKinds struct {
	At     time.Time     `json:"at"`
	Wait   time.Duration `json:"wait"`
	Amount json.Number   `json:"amount"`
	Blob   []byte        `json:"blob"`
}

// decodedKindsCopy has decodedKinds' JSON form but another Go type, so Get converts it through
// JSON.
type decodedKindsCopy struct {
	At     time.Time     `json:"at"`
	Wait   time.Duration `json:"wait"`
	Amount json.Number   `json:"amount"`
	Blob   []byte        `json:"blob"`
}

// TestDecodeTimeDurationNumberAndBytes decodes time.Time, time.Duration, json.Number and []byte
// fields through State as v0.8.0 did, apart from the stricter time and base64 forms that
// TestDecodeDiffersFromV080 lists.
func TestDecodeTimeDurationNumberAndBytes(t *testing.T) {
	at := time.Date(2026, 6, 30, 10, 0, 0, 500_000_000, time.FixedZone("", 3600))
	want := decodedKinds{At: at, Wait: 1500 * time.Millisecond, Amount: "3.50", Blob: []byte("hi")}
	reply := func(field string, value any) map[string]any {
		values := map[string]any{
			"at":     "2026-06-30T10:00:00.5+01:00",
			"wait":   json.Number("1500000000"),
			"amount": json.Number("3.50"),
			"blob":   "aGk=",
		}
		values[field] = value
		return values
	}
	key := Key[decodedKinds]{Name: "k"}
	get := func(value any) (decodedKinds, error) { return key.Get(NewState(Prediction{"k": value})) }

	t.Run("model reply", func(t *testing.T) {
		got, err := get(reply("amount", json.Number("3.50")))
		require.NoError(t, err)
		require.True(t, want.At.Equal(got.At))
		got.At = want.At
		require.Equal(t, want, got)
	})
	t.Run("another Go type with the same JSON form", func(t *testing.T) {
		got, err := get(decodedKindsCopy(want))
		require.NoError(t, err)
		require.True(t, want.At.Equal(got.At))
		got.At = want.At
		require.Equal(t, want, got)
	})
	t.Run("typed value", func(t *testing.T) {
		state := NewState(nil)
		key.Set(state, want)
		got, err := key.Get(state)
		require.NoError(t, err)
		require.Equal(t, want, got)
	})
	t.Run("typed predict output", func(t *testing.T) {
		got, err := decodeTypedValue[decodedKinds](reply("blob", nil))
		require.NoError(t, err)
		require.Equal(t, time.Duration(1500*time.Millisecond), got.Wait)
		require.Nil(t, got.Blob)
	})

	accepted := []struct {
		name  string
		field string
		value any
		check func(t *testing.T, got decodedKinds)
	}{
		{
			name:  "a JSON number as a string",
			field: "amount",
			value: "3",
			check: func(t *testing.T, got decodedKinds) { require.Equal(t, json.Number("3"), got.Amount) },
		},
		{
			name:  "bytes as an array of numbers",
			field: "blob",
			value: []any{json.Number("104"), json.Number("105")},
			check: func(t *testing.T, got decodedKinds) { require.Equal(t, []byte("hi"), got.Blob) },
		},
		{
			name:  "null bytes",
			field: "blob",
			value: nil,
			check: func(t *testing.T, got decodedKinds) { require.Nil(t, got.Blob) },
		},
		{
			name:  "empty bytes",
			field: "blob",
			value: "",
			check: func(t *testing.T, got decodedKinds) { require.Equal(t, []byte{}, got.Blob) },
		},
	}
	for _, tt := range accepted {
		t.Run(tt.name, func(t *testing.T) {
			got, err := get(reply(tt.field, tt.value))
			require.NoError(t, err)
			tt.check(t, got)
		})
	}

	const prefix = `state key "k": decode as gepa.decodedKinds: `
	refused := []struct {
		name    string
		field   string
		value   any
		wantErr string
	}{
		{
			name:  "a time that is not RFC 3339",
			field: "at",
			value: "2026-06-30",
			wantErr: `json: cannot unmarshal string into Go struct field decodedKinds.at of type time.Time: ` +
				`parsing time "2026-06-30" as "2006-01-02T15:04:05Z07:00": cannot parse "" as "T"`,
		},
		{
			name:    "a time as a number",
			field:   "at",
			value:   json.Number("1"),
			wantErr: `json: cannot unmarshal number into Go struct field decodedKinds.at of type time.Time`,
		},
		{name: "a null time", field: "at", value: nil, wantErr: `field "at" is null`},
		{
			name:    "a duration as text",
			field:   "wait",
			value:   "1.5s",
			wantErr: `json: cannot unmarshal string into Go struct field decodedKinds.wait of type time.Duration`,
		},
		{
			name:    "a fractional duration",
			field:   "wait",
			value:   json.Number("1.5"),
			wantErr: `json: cannot unmarshal number 1.5 into Go struct field decodedKinds.wait of type time.Duration`,
		},
		{
			name:    "a string that is not a JSON number",
			field:   "amount",
			value:   "x",
			wantErr: `json: cannot unmarshal string "x" into Go struct field decodedKinds.amount of type json.Number: invalid syntax`,
		},
		{
			name:    "a boolean as a JSON number",
			field:   "amount",
			value:   true,
			wantErr: `json: cannot unmarshal bool into Go struct field decodedKinds.amount of type json.Number`,
		},
		{
			name:    "bytes that are not base64",
			field:   "blob",
			value:   "aGk",
			wantErr: `json: cannot unmarshal string into Go struct field decodedKinds.blob of type []uint8: illegal base64 data at input byte 0`,
		},
	}
	for _, tt := range refused {
		t.Run(tt.name, func(t *testing.T) {
			_, err := get(reply(tt.field, tt.value))
			require.EqualError(t, err, prefix+tt.wantErr)
		})
	}

	state := NewState(
		Prediction{"wait": json.Number("42"), "amount": 2.5, "blob": "aGk=", "at": "2026-06-30T10:00:00Z"},
	)
	wait, err := Key[time.Duration]{Name: "wait"}.Get(state)
	require.NoError(t, err)
	require.Equal(t, time.Duration(42), wait)
	amount, err := Key[json.Number]{Name: "amount"}.Get(state)
	require.NoError(t, err)
	require.Equal(t, json.Number("2.5"), amount)
	blob, err := Key[[]byte]{Name: "blob"}.Get(state)
	require.NoError(t, err)
	require.Equal(t, []byte("hi"), blob)
	got, err := Key[time.Time]{Name: "at"}.Get(state)
	require.NoError(t, err)
	require.Equal(t, time.Date(2026, 6, 30, 10, 0, 0, 0, time.UTC), got)
}

// conflictingNames and stringSlice are built at run time because vet and staticcheck refuse
// their tags, which are the misuse under test.
var (
	conflictingNames = reflect.StructOf([]reflect.StructField{
		{Name: "A", Type: reflect.TypeFor[int](), Tag: `json:"x"`},
		{Name: "B", Type: reflect.TypeFor[int](), Tag: `json:"x"`},
		{Name: "C", Type: reflect.TypeFor[int](), Tag: `json:"c"`},
	})
	stringSlice = reflect.StructOf([]reflect.StructField{
		{Name: "A", Type: reflect.TypeFor[[]int](), Tag: `json:"a,string"`},
	})
)

var errRefusedLabel = errors.New("label refused")

type refusingLabel string

func (l *refusingLabel) UnmarshalText(text []byte) error {
	if string(text) == "bad" {
		return errRefusedLabel
	}
	*l = refusingLabel(text)
	return nil
}

type labelled struct {
	Label refusingLabel `json:"label"`
}

type nestedObligations struct {
	Items []stateObligation `json:"items"`
}

// TestDecodeDiffersFromV080 pins each way decoding through encoding/json/v2 differs from
// v0.8.0, which decoded with encoding/json on Go 1.25.
func TestDecodeDiffersFromV080(t *testing.T) {
	tests := []struct {
		name    string
		typ     reflect.Type
		value   any
		wantErr string
	}{
		{
			// v0.8.0 accepted a loose RFC 3339 time, such as a one-digit hour.
			name:  "a time with a one-digit hour",
			typ:   reflect.TypeFor[time.Time](),
			value: "2026-06-30T0:00:00Z",
			wantErr: `json: cannot unmarshal string into Go value of type time.Time: ` +
				`parsing time "2026-06-30T0:00:00Z" as "2006-01-02T15:04:05Z07:00": cannot parse "0" as "15"`,
		},
		{
			// v0.8.0 accepted a comma before the fraction of a second.
			name:  "a time with a decimal comma",
			typ:   reflect.TypeFor[time.Time](),
			value: "2026-06-30T00:00:00,5Z",
			wantErr: `json: cannot unmarshal string into Go value of type time.Time: ` +
				`parsing time "2026-06-30T00:00:00,5Z" as "2006-01-02T15:04:05Z07:00": cannot parse "," as "."`,
		},
		{
			// v0.8.0 skipped line breaks in base64.
			name:    "base64 with a line break",
			typ:     reflect.TypeFor[[]byte](),
			value:   "aG\nk=",
			wantErr: `json: cannot unmarshal string into Go value of type []uint8: illegal character '\n' at offset 2`,
		},
		{
			// v0.8.0 kept the last of the duplicates.
			name:    "duplicate names in raw JSON",
			typ:     reflect.TypeFor[stateObligation](),
			value:   json.RawMessage(`{"amount":1,"amount":2,"deadline":"d"}`),
			wantErr: `jsontext: duplicate object member name "amount"`,
		},
		{
			// v0.8.0 decoded a lone surrogate as U+FFFD.
			name:  `a lone surrogate in a ",string" field`,
			typ:   reflect.TypeFor[quotedInt](),
			value: map[string]any{"n": "1", "s": `"\ud800"`},
			wantErr: "json: cannot unmarshal string into Go struct field quotedInt.s of type string: " +
				"jsontext: invalid surrogate pair `\\ud800\"` in string",
		},
		{
			// v0.8.0 dropped both fields and decoded the rest.
			name:  "two fields with one JSON name",
			typ:   conflictingNames,
			value: map[string]any{"c": 1},
			wantErr: "json: cannot unmarshal object into Go value of type " + conflictingNames.String() +
				`: Go struct fields A and B conflict over JSON object name "x"`,
		},
		{
			// v0.8.0 ignored ",string" on a field that is not a boolean, number or string.
			name:    `",string" on a slice`,
			typ:     stringSlice,
			value:   map[string]any{"a": []any{1}},
			wantErr: "json: cannot unmarshal JSON into Go struct field .a of type []int: invalid use of `string` tag option",
		},
		{
			// v0.8.0 returned the method's error alone.
			name:    "an UnmarshalText error",
			typ:     reflect.TypeFor[labelled](),
			value:   map[string]any{"label": "bad"},
			wantErr: `json: cannot unmarshal string into Go struct field labelled.label of type gepa.refusingLabel: label refused`,
		},
		{
			// v0.8.0 named the innermost struct and left out indexes: stateObligation.items.amount.
			name:  "a value of the wrong type in a list",
			typ:   reflect.TypeFor[nestedObligations](),
			value: map[string]any{"items": []any{map[string]any{"amount": "ten", "deadline": "d"}}},
			wantErr: `json: cannot unmarshal string into Go struct field ` +
				`nestedObligations.items.0.amount of type int`,
		},
		{
			// v0.8.0 reported the unknown member first.
			name:    "a value of the wrong type before an unknown member",
			typ:     reflect.TypeFor[stateObligation](),
			value:   map[string]any{"amount": "ten", "deadline": "d", "note": "x"},
			wantErr: `json: cannot unmarshal string into Go struct field stateObligation.amount of type int`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := decodeJSONValue(tt.typ, tt.value)
			require.EqualError(t, err, tt.wantErr)
		})
	}

	_, err := decodeJSONValue(reflect.TypeFor[labelled](), map[string]any{"label": "bad"})
	require.ErrorIs(t, err, errRefusedLabel, "a method's error is still reachable")
}

// TestDecodeErrorWording checks how decodeError words each kind of encoding/json/v2 error.
func TestDecodeErrorWording(t *testing.T) {
	typ := reflect.TypeFor[*stateObligation]()
	for name, err := range map[string]error{
		"not from v2":       errors.New("boom"),
		"a syntax error":    &jsontext.SyntacticError{ByteOffset: 1, Err: errors.New("bad")},
		"without a Go type": &jsonv2.SemanticError{JSONKind: '"'},
	} {
		t.Run(name, func(t *testing.T) {
			require.Same(t, err, decodeError(typ, err))
		})
	}

	tests := []struct {
		name    string
		err     *jsonv2.SemanticError
		wantErr string
	}{
		{
			name:    "a Go type refused before any value",
			err:     &jsonv2.SemanticError{GoType: typ.Elem(), Err: errors.New("bad type")},
			wantErr: `json: cannot unmarshal JSON into Go value of type gepa.stateObligation: bad type`,
		},
		{
			name: "a quoted number that does not fit",
			err: &jsonv2.SemanticError{
				JSONKind:    '"',
				JSONPointer: "/amount",
				JSONValue:   jsontext.Value(`"300"`),
				GoType:      reflect.TypeFor[int8](),
				Err:         strconv.ErrRange,
			},
			wantErr: `json: cannot unmarshal number 300 into Go struct field stateObligation.amount of type int8`,
		},
		{
			name: "a number for a string",
			err: &jsonv2.SemanticError{
				JSONKind:  '0',
				JSONValue: jsontext.Value(`7`),
				GoType:    reflect.TypeFor[string](),
			},
			wantErr: `json: cannot unmarshal number 7 into Go value of type string`,
		},
		{
			name: "a string another error refused",
			err: &jsonv2.SemanticError{
				JSONKind:    '"',
				JSONPointer: "/deadline",
				JSONValue:   jsontext.Value(`"x"`),
				GoType:      reflect.TypeFor[time.Time](),
				Err:         errors.New("bad time"),
			},
			wantErr: `json: cannot unmarshal string "x" into Go struct field stateObligation.deadline of type time.Time: ` +
				`bad time`,
		},
		{
			name: "an unknown member",
			err: &jsonv2.SemanticError{
				JSONKind:    '"',
				JSONPointer: "/a~1b",
				GoType:      typ.Elem(),
				Err:         jsonv2.ErrUnknownName,
			},
			wantErr: `field "a/b" is not in gepa.stateObligation`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.EqualError(t, decodeError(typ, tt.err), tt.wantErr)
		})
	}
}
