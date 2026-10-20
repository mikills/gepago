package gepa

import (
	"context"
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
// is needed: turned off, v2 refuses a form that json.Marshal writes and gepago's JSON Schema
// describes, or accepts a form encoding/json refused.
func TestDecodeOptionsKeepEncodingJSONForms(t *testing.T) {
	tests := []struct {
		name string
		data string
		// without turns off the option under test.
		without jsonv2.Options
		out     any
		// want is nil when decodeOptions refuses data.
		want any
	}{
		{
			name:    `",string" on string and bool fields`,
			data:    `{"s":"\"x\"","b":"true"}`,
			without: json.StringifyWithLegacySemantics(false),
			out:     &quotedScalars{},
			want:    &quotedScalars{S: "x", B: true},
		},
		{
			name:    "time.Duration as nanoseconds",
			data:    `{"d":1500}`,
			without: json.FormatDurationAsNano(false),
			out:     &durationField{},
			want:    &durationField{D: 1500},
		},
		{
			// either byte option keeps this form.
			name:    "[N]byte as an array",
			data:    `{"b":[1,2,3,4]}`,
			without: jsonv2.JoinOptions(json.FormatByteArrayAsArray(false), json.FormatBytesWithLegacySemantics(false)),
			out:     &byteArrayField{},
			want:    &byteArrayField{B: [4]byte{1, 2, 3, 4}},
		},
		{
			name:    "[N]byte not as base64",
			data:    `{"b":"AQIDBA=="}`,
			without: json.FormatByteArrayAsArray(false),
			out:     &byteArrayField{},
		},
		{
			name:    "[]~byte as base64",
			data:    `{"b":"aGk="}`,
			without: json.FormatBytesWithLegacySemantics(false),
			out:     &namedBytesField{},
			want:    &namedBytesField{B: []namedByte{'h', 'i'}},
		},
		{
			name:    "[]byte as an array",
			data:    `{"b":[104,105]}`,
			without: json.FormatBytesWithLegacySemantics(false),
			out:     &bytesField{},
			want:    &bytesField{B: []byte("hi")},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fresh := reflect.New(reflect.TypeOf(tt.out).Elem()).Interface()
			errWithout := jsonv2.Unmarshal([]byte(tt.data), fresh, jsonv2.JoinOptions(decodeOptions, tt.without))
			err := jsonv2.Unmarshal([]byte(tt.data), tt.out, decodeOptions)
			if tt.want == nil {
				require.Error(t, err)
				require.NoError(t, errWithout, "without the option v2 accepts the form")
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, tt.out)
			require.Error(t, errWithout, "without the option v2 refuses the form")
		})
	}

	_, err := Key[byteArrayField]{Name: "k"}.Get(NewState(Prediction{"k": json.RawMessage(`{"b":"AQIDBA=="}`)}))
	require.EqualError(
		t,
		err,
		`state key "k": decode as gepa.byteArrayField: `+
			`json: cannot unmarshal string into Go struct field byteArrayField.b of type [4]uint8`,
	)
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

type obligationsByKey struct {
	Items map[string][]*stateObligation `json:"items"`
}

type stateCounts struct {
	Counts map[string]int `json:"counts"`
}

type stateIntMap struct {
	M map[int]string `json:"m"`
}

// TestDecodeErrorNamesTheInnermostStruct checks that a type error names the innermost struct
// and the JSON names of the struct fields that lead to it, without indexes or map keys, as
// v0.8.0 did, whatever the root.
func TestDecodeErrorNamesTheInnermostStruct(t *testing.T) {
	bad := `{"amount":"ten","deadline":"d"}`
	tests := []struct {
		name    string
		typ     reflect.Type
		data    string
		wantErr string
	}{
		{
			name:    "slice root",
			typ:     reflect.TypeFor[[]stateObligation](),
			data:    `[{"amount":1,"deadline":"d"},` + bad + `]`,
			wantErr: `json: cannot unmarshal string into Go struct field stateObligation.amount of type int`,
		},
		{
			name:    "pointer to a slice root",
			typ:     reflect.TypeFor[*[]stateObligation](),
			data:    `[` + bad + `]`,
			wantErr: `json: cannot unmarshal string into Go struct field stateObligation.amount of type int`,
		},
		{
			name:    "map root",
			typ:     reflect.TypeFor[map[string]stateObligation](),
			data:    `{"k":` + bad + `}`,
			wantErr: `json: cannot unmarshal string into Go struct field stateObligation.amount of type int`,
		},
		{
			name:    "slice root without a struct",
			typ:     reflect.TypeFor[[]int](),
			data:    `[1,"s"]`,
			wantErr: `json: cannot unmarshal string into Go value of type int`,
		},
		{
			name:    "map root without a struct",
			typ:     reflect.TypeFor[map[string]int](),
			data:    `{"a":"s"}`,
			wantErr: `json: cannot unmarshal string into Go value of type int`,
		},
		{
			name:    "map key",
			typ:     reflect.TypeFor[map[int]string](),
			data:    `{"1.0":"a"}`,
			wantErr: `json: cannot unmarshal number 1.0 into Go value of type int`,
		},
		{
			name:    "list in a struct",
			typ:     reflect.TypeFor[nestedObligations](),
			data:    `{"items":[` + bad + `]}`,
			wantErr: `json: cannot unmarshal string into Go struct field stateObligation.items.amount of type int`,
		},
		{
			name:    "map of lists of pointers in a struct",
			typ:     reflect.TypeFor[obligationsByKey](),
			data:    `{"items":{"k":[{"amount":1,"deadline":"d"},` + bad + `]}}`,
			wantErr: `json: cannot unmarshal string into Go struct field stateObligation.items.amount of type int`,
		},
		{
			name:    "map value in a struct",
			typ:     reflect.TypeFor[stateCounts](),
			data:    `{"counts":{"a":"s"}}`,
			wantErr: `json: cannot unmarshal string into Go struct field stateCounts.counts of type int`,
		},
		{
			name:    "map key in a struct",
			typ:     reflect.TypeFor[stateIntMap](),
			data:    `{"m":{"1.0":"a"}}`,
			wantErr: `json: cannot unmarshal number 1.0 into Go struct field stateIntMap.m of type int`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := decodeJSONValue(tt.typ, json.RawMessage(tt.data))
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

// stateCode is a map key that decodes itself from text, "null" included.
type stateCode int

func (c *stateCode) UnmarshalText(text []byte) error {
	if string(text) == "null" {
		*c = -1
		return nil
	}
	n, err := strconv.Atoi(string(text))
	*c = stateCode(n)
	return err
}

// TestDecodeRefusesMapKeysV080Refused checks that Get refuses the object keys v0.8.0 refused
// and encoding/json/v2 decodes: "null" for an integer key, which v2 decodes as 0, and any key
// of a type that is not a string, an integer or a text unmarshaler, such as a float.
func TestDecodeRefusesMapKeysV080Refused(t *testing.T) {
	floats := reflect.TypeFor[map[float64]string]()
	const floatKeys = `value: map key type float64 cannot be decoded from a JSON object key`
	tests := []struct {
		name    string
		typ     reflect.Type
		data    string
		wantErr string
	}{
		{
			name:    "null for an int key",
			typ:     reflect.TypeFor[map[int]string](),
			data:    `{"null":"a"}`,
			wantErr: `value has the key "null", which is not a number for int`,
		},
		{
			name:    "null for a uint key",
			typ:     reflect.TypeFor[map[uint8]string](),
			data:    `{"1":"a","null":"b"}`,
			wantErr: `value has the key "null", which is not a number for uint8`,
		},
		{
			name:    "null for an int key in a struct",
			typ:     reflect.TypeFor[stateIntMap](),
			data:    `{"m":{"null":"a"}}`,
			wantErr: `field "m" has the key "null", which is not a number for int`,
		},
		{name: "hexadecimal float key", typ: floats, data: `{"0x1p-2":"a"}`, wantErr: floatKeys},
		{name: "NaN key", typ: floats, data: `{"NaN":"a"}`, wantErr: floatKeys},
		{name: "decimal float key", typ: floats, data: `{"1.5":"a"}`, wantErr: floatKeys},
		{name: "no keys for a float key", typ: floats, data: `{}`, wantErr: floatKeys},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.NoError(
				t,
				jsonv2.Unmarshal([]byte(tt.data), reflect.New(tt.typ).Interface(), decodeOptions),
				"v2 decodes it on its own",
			)
			_, err := decodeJSONValue(tt.typ, json.RawMessage(tt.data))
			require.EqualError(t, err, tt.wantErr)
		})
	}

	_, err := decodeJSONValue(reflect.TypeFor[map[bool]string](), json.RawMessage(`{"true":"a"}`))
	require.EqualError(t, err, `value: map key type bool cannot be decoded from a JSON object key`)

	accepted := []struct {
		name string
		typ  reflect.Type
		data string
		want any
	}{
		{"int keys", reflect.TypeFor[map[int]string](), `{"1":"a","-2":"b"}`, map[int]string{1: "a", -2: "b"}},
		{"null for a string key", reflect.TypeFor[map[string]int](), `{"null":1}`, map[string]int{"null": 1}},
		{
			"null for a key that decodes itself",
			reflect.TypeFor[map[stateCode]string](),
			`{"null":"a","2":"b"}`,
			map[stateCode]string{-1: "a", 2: "b"},
		},
		{"null for float keys", reflect.TypeFor[*map[float64]string](), `null`, (*map[float64]string)(nil)},
	}
	for _, tt := range accepted {
		t.Run(tt.name, func(t *testing.T) {
			got, err := decodeJSONValue(tt.typ, json.RawMessage(tt.data))
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}

// celsius decodes itself from {"c": degrees} with UnmarshalJSONFrom, so its JSON form is not
// its fields'.
type celsius struct {
	Deg float64 `json:"deg"`
}

func (c *celsius) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var reading struct {
		C float64 `json:"c"`
	}
	if err := jsonv2.UnmarshalDecode(dec, &reading); err != nil {
		return err
	}
	c.Deg = reading.C
	return nil
}

type celsiusReadings struct {
	Readings []celsius `json:"readings"`
}

// TestDecodeUsesUnmarshalJSONFrom checks that a type with encoding/json/v2's UnmarshalJSONFrom
// method decodes as the method defines, as one with UnmarshalJSON does, and that its schema is
// unconstrained.
func TestDecodeUsesUnmarshalJSONFrom(t *testing.T) {
	got, err := Key[celsius]{Name: "t"}.Get(NewState(Prediction{"t": map[string]any{"c": 21.5}}))
	require.NoError(t, err)
	require.Equal(t, celsius{Deg: 21.5}, got)

	readings, err := Key[celsiusReadings]{Name: "r"}.Get(
		NewState(Prediction{"r": map[string]any{"readings": []any{map[string]any{"c": 1}}}}),
	)
	require.NoError(t, err)
	require.Equal(t, celsiusReadings{Readings: []celsius{{Deg: 1}}}, readings)

	_, err = Key[celsius]{Name: "t"}.Get(NewState(Prediction{"t": map[string]any{"deg": 21.5}}))
	require.EqualError(
		t,
		err,
		`state key "t": decode as gepa.celsius: field "deg" is not in struct { C float64 "json:\"c\"" }`,
		"the method refuses the form of celsius' fields",
	)

	schema, err := JSONSchemaFor[celsius]()
	require.NoError(t, err)
	require.JSONEq(t, `{}`, string(schema))
}

type stateKind string

// stateKinded embeds a type that is not a struct without giving it a JSON name.
type stateKinded struct {
	stateKind
	Name string `json:"name"`
}

type stateRef struct {
	Ref string `json:"ref"`
}

// stateEmbedsWithOptions embeds a struct with a tag option.
type stateEmbedsWithOptions struct {
	stateRef `       json:",omitempty"`
	Name     string `json:"name"`
}

// stamp writes itself as text, so a struct that embeds it cannot take its fields in v2.
type stamp struct {
	At string `json:"at"`
}

func (s stamp) MarshalText() ([]byte, error) { return []byte(s.At), nil }

type stamped struct {
	stamp
	Name string `json:"name"`
}

// stateOpaque has no exported fields.
type stateOpaque struct{ n int }

type stateHoldsOpaque struct {
	Opaque stateOpaque `json:"opaque"`
}

// structWithTag returns a struct of one field, built at run time because vet or staticcheck
// refuse the tags under test.
func structWithTag(name string, typ reflect.Type, tag reflect.StructTag) reflect.Type {
	return reflect.StructOf([]reflect.StructField{{Name: name, Type: typ, Tag: tag}})
}

// TestDecodeRefusesStructsJSONV2Refuses pins each kind of struct that v0.8.0 decoded and
// encoding/json/v2 refuses by its definition: Get refuses every object for it, naming the
// field and the type, and JSONSchemaFor, NewTool and TypedPredict refuse it before any call.
func TestDecodeRefusesStructsJSONV2Refuses(t *testing.T) {
	require.Empty(t, stateKinded{}.stateKind)
	require.Zero(t, stateOpaque{}.n)
	unexportedTagged := reflect.StructOf([]reflect.StructField{
		{Name: "Name", Type: reflect.TypeFor[string](), Tag: `json:"name"`},
		{Name: "secret", PkgPath: "github.com/mikills/gepago", Type: reflect.TypeFor[string](), Tag: `json:"secret"`},
	})
	str, at := reflect.TypeFor[string](), reflect.TypeFor[time.Time]()
	id := map[string]any{"id": "x"}
	tests := []struct {
		name  string
		typ   reflect.Type
		value map[string]any
		// reason is v2's error for the type.
		reason string
	}{
		{
			name:   "unexported field with a json tag",
			typ:    unexportedTagged,
			value:  map[string]any{"name": "n"},
			reason: "unexported Go struct field secret cannot have non-ignored `json:\"secret\"` tag",
		},
		{
			name:   "embedded non-struct without a JSON name",
			typ:    reflect.TypeFor[stateKinded](),
			value:  map[string]any{"name": "n"},
			reason: "embedded Go struct field stateKind of non-struct type must be explicitly given a JSON name",
		},
		{
			name:   "embedded struct with omitempty",
			typ:    reflect.TypeFor[stateEmbedsWithOptions](),
			value:  map[string]any{"ref": "r", "name": "n"},
			reason: "Go struct field stateRef cannot have any options other than `embed` specified",
		},
		{
			name:  "space before an option",
			typ:   structWithTag("ID", str, `json:"id, omitempty"`),
			value: id,
			reason: "Go struct field ID has malformed `json` tag: " +
				"invalid character ' ' at start of option (expecting Unicode letter)",
		},
		{
			name:   "option in the wrong case",
			typ:    structWithTag("ID", str, `json:"id,omitEmpty"`),
			value:  id,
			reason: "Go struct field ID has invalid appearance of `omitEmpty` tag option; specify `omitempty` instead",
		},
		{
			name:   "repeated option",
			typ:    structWithTag("ID", str, `json:"id,omitempty,omitempty"`),
			value:  id,
			reason: "Go struct field ID has duplicate appearance of `omitempty` tag option",
		},
		{
			name:   "trailing comma",
			typ:    structWithTag("ID", str, `json:"id,"`),
			value:  id,
			reason: "Go struct field ID has malformed `json` tag: invalid trailing ',' character",
		},
		{
			name:   "format option",
			typ:    structWithTag("At", at, `json:"at,format:RFC3339"`),
			value:  map[string]any{"at": "2026-06-30T10:00:00Z"},
			reason: "Go struct field At has unsupported `format` tag option",
		},
		{
			name:   "two fields with one JSON name",
			typ:    conflictingNames,
			value:  map[string]any{"c": 1},
			reason: `Go struct fields A and B conflict over JSON object name "x"`,
		},
		{
			name:   "embedded struct with a marshal method",
			typ:    reflect.TypeFor[stamped](),
			value:  map[string]any{"at": "a", "name": "n"},
			reason: "embedded Go struct field stamp of type gepa.stamp must not implement marshal or unmarshal methods",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := decodeJSONValue(tt.typ, tt.value)
			require.EqualError(
				t,
				err,
				"json: cannot unmarshal object into Go value of type "+tt.typ.String()+": "+tt.reason,
			)

			_, _, err = newSchemaBuilder().typeSchema(tt.typ, "")
			require.EqualError(t, err, tt.typ.String()+" cannot be decoded from JSON: "+tt.reason)
		})
	}

	const noExportedFields = "gepa.stateOpaque cannot be decoded from JSON: Go struct has no exported fields"
	t.Run("nested struct with no exported fields", func(t *testing.T) {
		_, err := decodeJSONValue(reflect.TypeFor[stateHoldsOpaque](), map[string]any{"opaque": map[string]any{}})
		require.EqualError(
			t,
			err,
			"json: cannot unmarshal object into Go struct field stateHoldsOpaque.opaque of type gepa.stateOpaque: "+
				"Go struct has no exported fields",
		)

		_, err = JSONSchemaFor[stateHoldsOpaque]()
		require.EqualError(t, err, `json schema for gepa.stateHoldsOpaque: field "opaque": `+noExportedFields)
	})

	embeds := "gepa.stateEmbedsWithOptions cannot be decoded from JSON: " +
		"Go struct field stateRef cannot have any options other than `embed` specified"
	kinded := "gepa.stateKinded cannot be decoded from JSON: " +
		"embedded Go struct field stateKind of non-struct type must be explicitly given a JSON name"
	t.Run("tool", func(t *testing.T) {
		tool := NewTool("t", "d", func(context.Context, stateEmbedsWithOptions) (string, error) { return "", nil })
		_, err := NewToolRegistry(tool)
		require.EqualError(t, err, `register tool: tool "t": json schema for gepa.stateEmbedsWithOptions: `+embeds)
	})

	lm := textModel(nil)
	givenInputs := Signature{Inputs: []Field{{Name: "name"}}}
	for name, tt := range map[string]struct {
		validate func() error
		wantErr  string
	}{
		"typed predict output": {
			validate: TypedPredict[stateObligation, stateKinded]{LM: lm}.Validate,
			wantErr:  "typed predict outputs: " + kinded,
		},
		"typed predict output field": {
			validate: TypedPredict[stateObligation, stateHoldsOpaque]{LM: lm}.Validate,
			wantErr:  `typed predict outputs: field "opaque": ` + noExportedFields,
		},
		"typed predict input": {
			validate: TypedPredict[stateEmbedsWithOptions, stateObligation]{LM: lm}.Validate,
			wantErr:  "typed predict inputs: " + embeds,
		},
		"typed predict input with given names": {
			validate: TypedPredict[stateKinded, stateObligation]{Signature: givenInputs, LM: lm}.Validate,
			wantErr:  "typed predict inputs: " + kinded,
		},
	} {
		t.Run(name, func(t *testing.T) {
			require.EqualError(t, tt.validate(), tt.wantErr)
		})
	}
}
