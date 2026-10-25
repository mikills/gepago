package gepa

import (
	"context"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// decodeJSONValue converts value, a value from state, to typ (see decodeValue).
func decodeJSONValue(typ reflect.Type, value any) (any, error) {
	return decodeValue(typ, value, fromState)
}

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
		got, err := decodeTypedValue[decodedKinds](reply("blob", nil), fromText)
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
			wantErr: `json: invalid number literal, trying to unmarshal "\"x\"" into Number`,
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

// conflictingNames, stringSlice and dashNamed are built at run time because vet and staticcheck
// refuse their tags, which are the misuse under test.
var (
	conflictingNames = reflect.StructOf([]reflect.StructField{
		{Name: "A", Type: reflect.TypeFor[int](), Tag: `json:"x"`},
		{Name: "B", Type: reflect.TypeFor[int](), Tag: `json:"x"`},
		{Name: "C", Type: reflect.TypeFor[int](), Tag: `json:"c"`},
	})
	stringSlice = reflect.StructOf([]reflect.StructField{
		{Name: "A", Type: reflect.TypeFor[[]int](), Tag: `json:"a,string"`},
	})
	dashNamed = structWithTag("A", reflect.TypeFor[int](), `json:"-,"`)
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
			name:  `",string" on a slice`,
			typ:   stringSlice,
			value: map[string]any{"a": []any{1}},
			wantErr: "json: cannot unmarshal object into Go value of type " + stringSlice.String() +
				": struct field A of type []int cannot have the `string` tag option, " +
				"which applies only to booleans, numbers and strings",
		},
		{
			// v0.8.0 ignored a tag option encoding/json did not define.
			name:  "a tag option encoding/json ignored",
			typ:   reflect.TypeFor[stateCaseIgnored](),
			value: map[string]any{"name": "x"},
			wantErr: "json: cannot unmarshal object into Go value of type gepa.stateCaseIgnored" +
				": struct field Name of type string cannot have the `case:ignore` tag option, " +
				"which encoding/json ignored and encoding/json/v2 may apply; " +
				"only omitempty, omitzero and string are allowed",
		},
		{
			// v0.8.0 decoded the member "-".
			name:  `the tag json:"-,"`,
			typ:   dashNamed,
			value: map[string]any{"-": 1},
			wantErr: "json: cannot unmarshal object into Go value of type " + dashNamed.String() +
				": struct field A of type int cannot be named \"-\" by the tag `json:\"-,\"`, " +
				"which encoding/json/v2 refuses; v2 names a field \"-\" only with the omitempty, omitzero or string option",
		},
		{
			// v0.8.0 kept the text as it was.
			name:    `a ",string" json.Number holding text that is not a JSON number`,
			typ:     reflect.TypeFor[stateQuotedNumber](),
			value:   map[string]any{"n": "0x10"},
			wantErr: `field "n" is "0x10", which is not a JSON number`,
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

type stateAnything struct {
	V any `json:"v"`
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
		{
			name:    "number inside an interface",
			typ:     reflect.TypeFor[stateAnything](),
			data:    `{"v":{"a":[1e400]}}`,
			wantErr: `json: cannot unmarshal number 1e400 into Go struct field stateAnything.v of type float64`,
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

type stateFloatMap struct {
	Labels map[float64]string `json:"labels"`
}

// TestDecodeRefusesMapKeysV080Refused checks that Get refuses the object keys v0.8.0 refused
// and encoding/json/v2 decodes, with the *json.UnmarshalTypeError v0.8.0 returned: "null" for
// an integer key, which v2 decodes as 0, and any object for a map whose key type is not a
// string, an integer or a text unmarshaler, such as a float or a bool.
func TestDecodeRefusesMapKeysV080Refused(t *testing.T) {
	floats := reflect.TypeFor[map[float64]string]()
	tests := []struct {
		name    string
		typ     reflect.Type
		data    string
		want    json.UnmarshalTypeError
		wantErr string
	}{
		{
			name:    "null for an int key",
			typ:     reflect.TypeFor[map[int]string](),
			data:    `{"null":"a"}`,
			want:    json.UnmarshalTypeError{Value: "number null", Type: reflect.TypeFor[int]()},
			wantErr: `json: cannot unmarshal number null into Go value of type int`,
		},
		{
			name:    "null for a uint key",
			typ:     reflect.TypeFor[map[uint8]string](),
			data:    `{"1":"a","null":"b"}`,
			want:    json.UnmarshalTypeError{Value: "number null", Type: reflect.TypeFor[uint8]()},
			wantErr: `json: cannot unmarshal number null into Go value of type uint8`,
		},
		{
			name: "null for an int key in a struct",
			typ:  reflect.TypeFor[stateIntMap](),
			data: `{"m":{"null":"a"}}`,
			want: json.UnmarshalTypeError{
				Value:  "number null",
				Type:   reflect.TypeFor[int](),
				Struct: "stateIntMap",
				Field:  "m",
			},
			wantErr: `json: cannot unmarshal number null into Go struct field stateIntMap.m of type int`,
		},
		{
			name:    "hexadecimal float key",
			typ:     floats,
			data:    `{"0x1p-2":"a"}`,
			want:    json.UnmarshalTypeError{Value: "object", Type: floats},
			wantErr: `json: cannot unmarshal object into Go value of type map[float64]string`,
		},
		{
			name:    "NaN key",
			typ:     floats,
			data:    `{"NaN":"a"}`,
			want:    json.UnmarshalTypeError{Value: "object", Type: floats},
			wantErr: `json: cannot unmarshal object into Go value of type map[float64]string`,
		},
		{
			name:    "no keys for a float key",
			typ:     floats,
			data:    `{}`,
			want:    json.UnmarshalTypeError{Value: "object", Type: floats},
			wantErr: `json: cannot unmarshal object into Go value of type map[float64]string`,
		},
		{
			name:    "float keys in a struct",
			typ:     reflect.TypeFor[stateFloatMap](),
			data:    `{"labels":{"1.5":"a"}}`,
			want:    json.UnmarshalTypeError{Value: "object", Type: floats, Struct: "stateFloatMap", Field: "labels"},
			wantErr: `json: cannot unmarshal object into Go struct field stateFloatMap.labels of type map[float64]string`,
		},
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
			var typeErr *json.UnmarshalTypeError
			require.ErrorAs(t, err, &typeErr)
			require.Equal(t, tt.want.Value, typeErr.Value)
			require.Equal(t, tt.want.Type, typeErr.Type)
			require.Equal(t, tt.want.Struct, typeErr.Struct)
			require.Equal(t, tt.want.Field, typeErr.Field)
		})
	}

	_, err := decodeJSONValue(reflect.TypeFor[map[bool]string](), json.RawMessage(`{"true":"a"}`))
	require.EqualError(t, err, `json: cannot unmarshal object into Go value of type map[bool]string`)
	var typeErr *json.UnmarshalTypeError
	require.ErrorAs(t, err, &typeErr)
	require.Equal(t, reflect.TypeFor[map[bool]string](), typeErr.Type)

	_, err = decodeJSONValue(floats, json.RawMessage(`{"1.5":null}`))
	require.EqualError(t, err, `field "1.5" is null`, "the values are checked before the keys, as in v0.8.0")

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
	str := reflect.TypeFor[string]()
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

// stateEmbedsTagged has an `embed` field, which encoding/json ignored and v2 applies: v2 reads
// the members of Inner as members of stateEmbedsTagged.
type stateEmbedsTagged struct {
	Inner stateRef `json:",embed"`
	Y     int      `json:"y"`
}

type stateHoldsEmbedsTagged struct {
	V *stateEmbedsTagged `json:"v"`
}

// stateCatchAll's Extra collects every unknown member in v2.
type stateCatchAll struct {
	A     int             `json:"a"`
	Extra json.RawMessage `json:",embed"`
}

// stateCaseIgnored matches its member name in any case in v2.
type stateCaseIgnored struct {
	Name string `json:"name,omitempty,case:ignore"`
}

type stateEmbedsCaseIgnored struct {
	stateCaseIgnored
	Y int `json:"y"`
}

// StateInlined is exported so a struct built at run time can embed it.
type StateInlined struct {
	X int `json:"x"`
}

// optionRefused is the reason a struct is refused for a field with a tag option that
// encoding/json did not define.
func optionRefused(field string, typ reflect.Type, option string) string {
	return fmt.Sprintf(
		"struct field %s of type %s cannot have the `%s` tag option, which encoding/json ignored and "+
			"encoding/json/v2 may apply; only omitempty, omitzero and string are allowed",
		field,
		typ,
		option,
	)
}

// TestDecodeRefusesTagOptionsEncodingJSONIgnored checks that a struct with a json tag that
// encoding/json and encoding/json/v2 read differently is refused up front by JSONSchemaFor,
// NewTool and TypedPredict, and for every object by Get, naming the field and the option: a tag
// option other than omitempty, omitzero and string, which encoding/json ignored and v2 may apply
// (`embed` moves members or collects unknown ones, `case:` changes name matching), and the tag
// `json:"-,"`, which named a field "-" and which v2 refuses. Each value is one v0.8.0 decoded.
func TestDecodeRefusesTagOptionsEncodingJSONIgnored(t *testing.T) {
	str, at := reflect.TypeFor[string](), reflect.TypeFor[time.Time]()
	inlined := reflect.TypeFor[StateInlined]()
	tests := []struct {
		name  string
		typ   reflect.Type
		value map[string]any
		// reason is the refusal of typ.
		reason string
	}{
		{
			name:   "embed on a struct",
			typ:    structWithTag("Inner", inlined, `json:",embed"`),
			value:  map[string]any{"Inner": map[string]any{"x": 1}},
			reason: optionRefused("Inner", inlined, "embed"),
		},
		{
			name:   "embed on a map",
			typ:    structWithTag("Extra", reflect.TypeFor[map[string]int](), `json:",embed"`),
			value:  map[string]any{"Extra": map[string]any{"a": 1}},
			reason: optionRefused("Extra", reflect.TypeFor[map[string]int](), "embed"),
		},
		{
			name:   "embed on a raw message",
			typ:    structWithTag("Extra", reflect.TypeFor[json.RawMessage](), `json:",embed"`),
			value:  map[string]any{"Extra": map[string]any{"q": 1}},
			reason: optionRefused("Extra", reflect.TypeFor[json.RawMessage](), "embed"),
		},
		{
			name: "inline on an embedded struct",
			typ: reflect.StructOf([]reflect.StructField{
				{Name: "StateInlined", Type: inlined, Anonymous: true, Tag: `json:",inline"`},
			}),
			value:  map[string]any{"x": 1},
			reason: optionRefused("StateInlined", inlined, "inline"),
		},
		{
			name:   "unknown",
			typ:    structWithTag("Extra", reflect.TypeFor[map[string]any](), `json:",unknown"`),
			value:  map[string]any{"Extra": map[string]any{"a": 1}},
			reason: optionRefused("Extra", reflect.TypeFor[map[string]any](), "unknown"),
		},
		{
			name:   "case:ignore",
			typ:    structWithTag("Name", str, `json:"name,case:ignore"`),
			value:  map[string]any{"name": "x"},
			reason: optionRefused("Name", str, "case:ignore"),
		},
		{
			name:   "case:strict after a known option",
			typ:    structWithTag("N", reflect.TypeFor[int](), `json:"n,string,case:strict"`),
			value:  map[string]any{"n": "1"},
			reason: optionRefused("N", reflect.TypeFor[int](), "case:strict"),
		},
		{
			name:   "format",
			typ:    structWithTag("At", at, `json:"at,format:RFC3339"`),
			value:  map[string]any{"at": "2026-06-30T10:00:00Z"},
			reason: optionRefused("At", at, "format:RFC3339"),
		},
		{
			name:   "format holding a comma",
			typ:    structWithTag("At", at, `json:"at,format:'2006-01-02, 15:04'"`),
			value:  map[string]any{"at": "2026-06-30T10:00:00Z"},
			reason: optionRefused("At", at, "format:'2006-01-02, 15:04'"),
		},
		{
			name:   "an option neither package defines",
			typ:    structWithTag("N", reflect.TypeFor[int](), `json:"n,foo"`),
			value:  map[string]any{"n": 1},
			reason: optionRefused("N", reflect.TypeFor[int](), "foo"),
		},
		{
			name:   "an option in the wrong case",
			typ:    structWithTag("ID", str, `json:"id,omitEmpty"`),
			value:  map[string]any{"id": "x"},
			reason: optionRefused("ID", str, "omitEmpty"),
		},
		{
			name:   "a space before an option",
			typ:    structWithTag("ID", str, `json:"id, omitempty"`),
			value:  map[string]any{"id": "x"},
			reason: optionRefused("ID", str, " omitempty"),
		},
		{
			name:  `the tag json:"-,"`,
			typ:   dashNamed,
			value: map[string]any{"-": 1},
			reason: "struct field A of type int cannot be named \"-\" by the tag `json:\"-,\"`, which encoding/json/v2 " +
				"refuses; v2 names a field \"-\" only with the omitempty, omitzero or string option",
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

	t.Run("a field named - with an option", func(t *testing.T) {
		dash := structWithTag("A", reflect.TypeFor[int](), `json:"-,omitempty"`)
		got, err := decodeJSONValue(dash, map[string]any{"-": 1})
		require.NoError(t, err)
		require.Equal(t, 1, reflect.ValueOf(got).Field(0).Interface())
		_, _, err = newSchemaBuilder().typeSchema(dash, "")
		require.NoError(t, err)
	})

	embedReason := optionRefused("Inner", reflect.TypeFor[stateRef](), "embed")
	catchAllReason := optionRefused("Extra", reflect.TypeFor[json.RawMessage](), "embed")
	caseReason := optionRefused("Name", str, "case:ignore")
	embeds := "gepa.stateEmbedsTagged cannot be decoded from JSON: " + embedReason
	catchAll := "gepa.stateCatchAll cannot be decoded from JSON: " + catchAllReason
	caseIgnored := "gepa.stateCaseIgnored cannot be decoded from JSON: " + caseReason
	t.Run("schema, tool and typed predict", func(t *testing.T) {
		_, err := JSONSchemaFor[stateEmbedsTagged]()
		require.EqualError(t, err, "json schema for gepa.stateEmbedsTagged: "+embeds)
		_, err = JSONSchemaFor[stateHoldsEmbedsTagged]()
		require.EqualError(t, err, `json schema for gepa.stateHoldsEmbedsTagged: field "v": `+embeds)

		tool := NewTool("t", "d", func(context.Context, stateCaseIgnored) (string, error) { return "", nil })
		_, err = NewToolRegistry(tool)
		require.EqualError(t, err, `register tool: tool "t": json schema for gepa.stateCaseIgnored: `+caseIgnored)

		lm := textModel(nil)
		require.EqualError(
			t,
			TypedPredict[stateObligation, stateCatchAll]{LM: lm}.Validate(),
			"typed predict outputs: "+catchAll,
		)
		require.EqualError(
			t,
			TypedPredict[stateEmbedsCaseIgnored, stateObligation]{LM: lm}.Validate(),
			"typed predict inputs: "+caseIgnored,
		)
		given := Signature{Inputs: []Field{{Name: "v"}}}
		require.EqualError(
			t,
			TypedPredict[stateHolder[stateCatchAll], stateObligation]{Signature: given, LM: lm}.Validate(),
			`typed predict inputs: field "v": `+catchAll,
		)
	})

	// Get refuses each value, which v0.8.0 decoded or refused as an unknown member and v2 alone
	// decodes differently or accepts, even when v2 meets an unknown member first.
	const refused = "json: cannot unmarshal object into Go value of type "
	for name, tt := range map[string]struct {
		get     func(State) error
		value   string
		wantErr string
	}{
		"the object the schema described": {
			get:     getError[stateEmbedsTagged],
			value:   `{"Inner":{"ref":"r"},"y":2}`,
			wantErr: "decode as gepa.stateEmbedsTagged: " + refused + "gepa.stateEmbedsTagged: " + embedReason,
		},
		"the object v2 expects": {
			get:     getError[stateEmbedsTagged],
			value:   `{"ref":"r","y":2}`,
			wantErr: "decode as gepa.stateEmbedsTagged: " + refused + "gepa.stateEmbedsTagged: " + embedReason,
		},
		"a nested object": {
			get:   getError[stateHoldsEmbedsTagged],
			value: `{"v":{"Inner":{"ref":"r"},"y":2}}`,
			wantErr: "decode as gepa.stateHoldsEmbedsTagged: json: cannot unmarshal object into Go struct field " +
				"stateHoldsEmbedsTagged.v of type gepa.stateEmbedsTagged: " + embedReason,
		},
		"a raw message v2 fills with the whole object": {
			get:     getError[stateCatchAll],
			value:   `{"a":1,"Extra":{"q":1}}`,
			wantErr: "decode as gepa.stateCatchAll: " + refused + "gepa.stateCatchAll: " + catchAllReason,
		},
		"an unknown member v2 collects": {
			get:     getError[stateCatchAll],
			value:   `{"a":1,"Extra":{},"zzz":5}`,
			wantErr: "decode as gepa.stateCatchAll: " + refused + "gepa.stateCatchAll: " + catchAllReason,
		},
		"a name in the wrong case": {
			get:     getError[stateCaseIgnored],
			value:   `{"NAME":"x"}`,
			wantErr: "decode as gepa.stateCaseIgnored: " + refused + "gepa.stateCaseIgnored: " + caseReason,
		},
		"an unknown member after a name in the wrong case": {
			get:     getError[stateCaseIgnored],
			value:   `{"NAME":"x","zzz":1}`,
			wantErr: "decode as gepa.stateCaseIgnored: " + refused + "gepa.stateCaseIgnored: " + caseReason,
		},
		"an embedded struct": {
			get:     getError[stateEmbedsCaseIgnored],
			value:   `{"name":"x","y":1}`,
			wantErr: "decode as gepa.stateEmbedsCaseIgnored: " + refused + "gepa.stateCaseIgnored: " + caseReason,
		},
	} {
		t.Run(name, func(t *testing.T) {
			err := tt.get(NewState(Prediction{"k": json.RawMessage(tt.value)}))
			require.EqualError(t, err, `state key "k": `+tt.wantErr)
		})
	}
}

// getError returns the error Key[T].Get returns for the key "k" of s.
func getError[T any](s State) error {
	_, err := Key[T]{Name: "k"}.Get(s)
	return err
}

// TestRefusedDefinitionOnPath checks which struct refusedDefinitionOnPath finds on the way to
// the value an error is about, and where it locates it.
func TestRefusedDefinitionOnPath(t *testing.T) {
	unknownAt := func(pointer string) error {
		return &jsonv2.SemanticError{JSONPointer: jsontext.Pointer(pointer), Err: jsonv2.ErrUnknownName}
	}
	tests := []struct {
		name string
		typ  reflect.Type
		err  error
		// at is where the refusal is located; empty for none.
		at     string
		goType reflect.Type
	}{
		{name: "not a semantic error", typ: reflect.TypeFor[stateCaseIgnored](), err: errors.New("boom")},
		{name: "valid structs", typ: reflect.TypeFor[nestedObligations](), err: unknownAt("/items/0/note")},
		{
			name:   "the top",
			typ:    reflect.TypeFor[stateCaseIgnored](),
			err:    unknownAt("/zzz"),
			at:     "",
			goType: reflect.TypeFor[stateCaseIgnored](),
		},
		{
			name:   "behind a pointer",
			typ:    reflect.TypeFor[stateHoldsEmbedsTagged](),
			err:    unknownAt("/v/Inner"),
			at:     "/v",
			goType: reflect.TypeFor[stateEmbedsTagged](),
		},
		{
			name:   "in a map of slices",
			typ:    reflect.TypeFor[map[string][]stateCatchAll](),
			err:    unknownAt("/a/1/zzz"),
			at:     "/a/1",
			goType: reflect.TypeFor[stateCatchAll](),
		},
		{
			name: "inside a type with its own method",
			typ:  reflect.TypeFor[stateHolder[celsius]](),
			err:  unknownAt("/v/x"),
		},
		{name: "past a number", typ: reflect.TypeFor[stateHolder[int]](), err: unknownAt("/v/x/y")},
		{name: "past an unknown member", typ: reflect.TypeFor[stateHolder[stateCatchAll]](), err: unknownAt("/w/a")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := refusedDefinitionOnPath(tt.typ, tt.err)
			if tt.goType == nil {
				require.NoError(t, err)
				return
			}
			var semantic *jsonv2.SemanticError
			require.ErrorAs(t, err, &semantic)
			require.Equal(t, jsontext.Pointer(tt.at), semantic.JSONPointer)
			require.Equal(t, tt.goType, semantic.GoType)
		})
	}
}

// rawCount decodes itself from a bare JSON number with UnmarshalJSON, so a quoted number is an
// error.
type rawCount int

func (c *rawCount) UnmarshalJSON(data []byte) error {
	n, err := strconv.Atoi(string(data))
	*c = rawCount(n)
	return err
}

// rawSize decodes itself from a bare JSON number with UnmarshalJSON.
type rawSize uint8

func (s *rawSize) UnmarshalJSON(data []byte) error {
	n, err := strconv.ParseUint(string(data), 10, 8)
	*s = rawSize(n)
	return err
}

// rawKey decodes itself with UnmarshalJSON, keeping the JSON text, quotes and all.
type rawKey string

func (k *rawKey) UnmarshalJSON(data []byte) error {
	*k = rawKey(data)
	return nil
}

// rawLabel decodes itself with UnmarshalJSON or UnmarshalText, saying which one ran.
type rawLabel string

func (l *rawLabel) UnmarshalJSON(data []byte) error {
	*l = rawLabel("json " + string(data))
	return nil
}

func (l *rawLabel) UnmarshalText(text []byte) error {
	*l = rawLabel("text " + string(text))
	return nil
}

// fromLevel decodes itself with encoding/json/v2's UnmarshalJSONFrom, as ten times the number.
type fromLevel int

func (l *fromLevel) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var n int
	if err := jsonv2.UnmarshalDecode(dec, &n); err != nil {
		return err
	}
	*l = fromLevel(n * 10)
	return nil
}

// stateQuotedNumber has a ",string" json.Number, which encoding/json read from a JSON number in
// a string, or a JSON string holding one, and which decodeQuotedNumber reads the same way.
type stateQuotedNumber struct {
	N json.Number `json:"n,string"`
}

// stateQuotedCount has a ",string" rawCount, which encoding/json unquoted before calling its
// UnmarshalJSON and v2 decodes with the method, ignoring the option.
type stateQuotedCount struct {
	N rawCount `json:"n,string"`
}

type stateHoldsQuotedCount struct {
	Quoted *stateQuotedCount `json:"quoted"`
}

// stateIntPointer is a named pointer, to which encoding/json did not apply the ",string" option.
type stateIntPointer *int

// TestDecodeRefusesStringOptionV2DoesNotApply checks that a struct with a ",string" field that
// encoding/json/v2 does not decode as v0.8.0 did is refused up front by JSONSchemaFor, NewTool
// and TypedPredict, and for every object by Get: a field that is not a boolean, number or
// string, or an unnamed pointer to one, which v0.8.0 decoded ignoring the option and v2 refuses
// only when the field is present, and a field whose type has its own unmarshal method, which
// v0.8.0 unquoted before calling the method and v2 decodes with the method, ignoring the option.
func TestDecodeRefusesStringOptionV2DoesNotApply(t *testing.T) {
	const notScalar = "which applies only to booleans, numbers and strings"
	tests := []struct {
		name string
		typ  reflect.Type
		tag  reflect.StructTag
		// value is a field value v0.8.0 decoded; nil leaves the field out.
		value  any
		reason string
	}{
		{"slice", reflect.TypeFor[[]string](), `json:"v,string"`, []any{"a"}, notScalar},
		{"array", reflect.TypeFor[[2]int](), `json:"v,string"`, []any{1, 2}, notScalar},
		{"map", reflect.TypeFor[map[string]int](), `json:"v,string"`, map[string]any{"a": 1}, notScalar},
		{"struct", reflect.TypeFor[stateRef](), `json:"v,string"`, map[string]any{"ref": "r"}, notScalar},
		{"bytes", reflect.TypeFor[[]byte](), `json:"v,string"`, "aGk=", notScalar},
		{"double pointer", reflect.TypeFor[**int](), `json:"v,string"`, 1, notScalar},
		{"named pointer", reflect.TypeFor[stateIntPointer](), `json:"v,string"`, 1, notScalar},
		{"interface", reflect.TypeFor[any](), `json:"v,string"`, "x", notScalar},
		{"time", reflect.TypeFor[time.Time](), `json:"v,string"`, "2026-06-30T10:00:00Z", notScalar},
		{"slice left out", reflect.TypeFor[[]string](), `json:"v,omitempty,string"`, nil, notScalar},
		{
			"UnmarshalJSON",
			reflect.TypeFor[rawCount](),
			`json:"v,string"`,
			"5",
			"since gepa.rawCount has its own unmarshal method",
		},
		{
			"pointer to UnmarshalJSON",
			reflect.TypeFor[*rawCount](),
			`json:"v,omitempty,string"`,
			"5",
			"since gepa.rawCount has its own unmarshal method",
		},
		{
			"UnmarshalText",
			reflect.TypeFor[refusingLabel](),
			`json:"v,string"`,
			`"x"`,
			"since gepa.refusingLabel has its own unmarshal method",
		},
		{
			"UnmarshalJSONFrom",
			reflect.TypeFor[fromLevel](),
			`json:"v,string"`,
			"5",
			"since gepa.fromLevel has its own unmarshal method",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			typ := structWithTag("V", tt.typ, tt.tag)
			reason := fmt.Sprintf(
				"struct field V of type %s cannot have the `string` tag option, %s",
				tt.typ,
				tt.reason,
			)
			object := map[string]any{}
			if tt.value != nil {
				object["v"] = tt.value
			}

			_, err := decodeJSONValue(typ, object)
			require.EqualError(t, err, "json: cannot unmarshal object into Go value of type "+typ.String()+": "+reason)

			_, _, err = newSchemaBuilder().typeSchema(typ, "")
			require.EqualError(t, err, typ.String()+" cannot be decoded from JSON: "+reason)
		})
	}

	const reason = "struct field N of type gepa.rawCount cannot have the `string` tag option, " +
		"since gepa.rawCount has its own unmarshal method"
	const quoted = "gepa.stateQuotedCount cannot be decoded from JSON: " + reason
	_, err := JSONSchemaFor[stateQuotedCount]()
	require.EqualError(t, err, "json schema for gepa.stateQuotedCount: "+quoted)
	_, err = JSONSchemaFor[stateHoldsQuotedCount]()
	require.EqualError(t, err, `json schema for gepa.stateHoldsQuotedCount: field "quoted": `+quoted)

	tool := NewTool("t", "d", func(context.Context, stateQuotedCount) (string, error) { return "", nil })
	_, err = NewToolRegistry(tool)
	require.EqualError(t, err, `register tool: tool "t": json schema for gepa.stateQuotedCount: `+quoted)

	lm := textModel(nil)
	require.EqualError(
		t,
		TypedPredict[stateObligation, stateQuotedCount]{LM: lm}.Validate(),
		"typed predict outputs: "+quoted,
	)
	require.EqualError(
		t,
		TypedPredict[stateQuotedCount, stateObligation]{LM: lm}.Validate(),
		"typed predict inputs: "+quoted,
	)

	const held = `decode as gepa.stateHoldsQuotedCount: json: cannot unmarshal object into Go struct field ` +
		"stateHoldsQuotedCount.quoted of type gepa.stateQuotedCount: " + reason
	state := NewState(Prediction{
		"bare":   map[string]any{"n": "1"},
		"held":   map[string]any{"quoted": map[string]any{"n": "1"}},
		"absent": map[string]any{"quoted": nil},
		// an unknown member ahead of n, inside the refused object, does not hide the refusal.
		"unknown": json.RawMessage(`{"quoted":{"zzz":1,"n":"1"}}`),
	})
	_, err = Key[stateQuotedCount]{Name: "bare"}.Get(state)
	require.EqualError(
		t,
		err,
		`state key "bare": decode as gepa.stateQuotedCount: `+
			"json: cannot unmarshal object into Go value of type gepa.stateQuotedCount: "+reason,
	)
	_, err = Key[stateHoldsQuotedCount]{Name: "held"}.Get(state)
	require.EqualError(t, err, `state key "held": `+held)
	_, err = Key[stateHoldsQuotedCount]{Name: "unknown"}.Get(state)
	require.EqualError(t, err, `state key "unknown": `+held)
	got, err := Key[stateHoldsQuotedCount]{Name: "absent"}.Get(state)
	require.NoError(t, err, "a struct that holds no such object decodes, as v2 checks definitions lazily")
	require.Nil(t, got.Quoted)
}

type stateAmount struct {
	Amount json.Number  `json:"amount"`
	Maybe  *json.Number `json:"maybe,omitempty"`
}

type stateAmounts struct {
	Amount json.Number            `json:"amount"`
	Maybe  *json.Number           `json:"maybe,omitempty"`
	List   []json.Number          `json:"list"`
	ByName map[string]json.Number `json:"by_name"`
}

// TestDecodeJSONNumberAsV080 checks that json.Number, which Go 1.27 gave an UnmarshalJSONFrom
// method, keeps the schema and decoding it had in v0.8.0: a string in its schema, so a struct
// that holds one stays strict for TypedPredict and NewTool, and a JSON number or a string that
// holds one decoded to the number's text, plain and inside structs, slices and maps.
func TestDecodeJSONNumberAsV080(t *testing.T) {
	schema, err := JSONSchemaFor[json.Number]()
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"string"}`, string(schema))

	schema, err = JSONSchemaFor[stateAmount]()
	require.NoError(t, err)
	require.JSONEq(t, `{
		"type": "object",
		"properties": {
			"amount": {"type": "string"},
			"maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]}
		},
		"required": ["amount", "maybe"],
		"additionalProperties": false
	}`, string(schema))
	require.True(t, StrictJSONSchema(schema))

	signature, err := TypedPredict[stateRef, stateAmount]{LM: textModel(nil)}.ProgramSignature()
	require.NoError(t, err)
	format, err := signature.Signature.ResponseFormat()
	require.NoError(t, err)
	require.True(t, format.Strict)

	tool := NewTool("t", "d", func(context.Context, stateAmount) (string, error) { return "", nil })
	require.True(t, StrictJSONSchema(tool.Schema().Parameters))

	maybe := json.Number("1e3")
	want := stateAmounts{
		Amount: "3.50",
		Maybe:  &maybe,
		List:   []json.Number{"1", "-2.5"},
		ByName: map[string]json.Number{"a": "0", "b": "7"},
	}
	for name, value := range map[string]any{
		"raw JSON": json.RawMessage(`{"amount":3.50,"maybe":"1e3","list":[1,"-2.5"],"by_name":{"a":0,"b":"7"}}`),
		"model reply": map[string]any{
			"amount":  json.Number("3.50"),
			"maybe":   "1e3",
			"list":    []any{json.Number("1"), "-2.5"},
			"by_name": map[string]any{"a": json.Number("0"), "b": "7"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := Key[stateAmounts]{Name: "k"}.Get(NewState(Prediction{"k": value}))
			require.NoError(t, err)
			require.Equal(t, want, got)
		})
	}
	for value, want := range map[any]json.Number{2.5: "2.5", "7": "7", json.Number("-0"): "-0"} {
		got, err := Key[json.Number]{Name: "k"}.Get(NewState(Prediction{"k": value}))
		require.NoError(t, err)
		require.Equal(t, want, got)
	}

	refused := []struct {
		name    string
		value   string
		wantErr string
	}{
		{
			name:    "a string that is not a number",
			value:   `{"amount":"01","list":null,"by_name":null}`,
			wantErr: `json: invalid number literal, trying to unmarshal "\"01\"" into Number`,
		},
		{
			name:    "an element that is not a number",
			value:   `{"amount":1,"list":["x"],"by_name":null}`,
			wantErr: `json: invalid number literal, trying to unmarshal "\"x\"" into Number`,
		},
		{
			name:    "a boolean",
			value:   `{"amount":true,"list":null,"by_name":null}`,
			wantErr: `json: cannot unmarshal bool into Go struct field stateAmounts.amount of type json.Number`,
		},
	}
	for _, tt := range refused {
		t.Run(tt.name, func(t *testing.T) {
			_, err := decodeJSONValue(reflect.TypeFor[stateAmounts](), json.RawMessage(tt.value))
			require.EqualError(t, err, tt.wantErr)
		})
	}
}

// stateQuotedNumbers holds ",string" json.Numbers, behind a pointer and in structs in a slice
// and a map.
type stateQuotedNumbers struct {
	N      json.Number                  `json:"n,string"`
	Maybe  *json.Number                 `json:"maybe,omitempty,string"`
	List   []stateQuotedNumber          `json:"list"`
	ByName map[string]stateQuotedNumber `json:"by_name"`
}

// TestDecodeQuotedJSONNumberAsV080 checks that a ",string" json.Number, which v0.8.0 decoded,
// is accepted by JSONSchemaFor, NewTool and TypedPredict with v0.8.0's schema, which is that of
// a ",string" string, and that Get decodes a JSON number in a string, or a JSON string holding
// one, which is the form that schema describes, to the number's text as v0.8.0 did, plain and
// inside pointers, slices and maps. Text that is not a JSON number, which v0.8.0 kept, is
// refused, and so is what v0.8.0 refused.
func TestDecodeQuotedJSONNumberAsV080(t *testing.T) {
	type quotedString struct {
		N string `json:"n,string"`
	}
	schema, err := JSONSchemaFor[stateQuotedNumber]()
	require.NoError(t, err)
	stringSchema, err := JSONSchemaFor[quotedString]()
	require.NoError(t, err)
	require.JSONEq(t, string(stringSchema), string(schema))
	var parsed struct {
		Properties map[string]struct{ Pattern string }
	}
	require.NoError(t, json.Unmarshal(schema, &parsed))
	require.Regexp(t, parsed.Properties["n"].Pattern, `"5"`)

	tool := NewTool("t", "d", func(context.Context, stateQuotedNumbers) (string, error) { return "", nil })
	_, err = NewToolRegistry(tool)
	require.NoError(t, err)
	lm := textModel(nil)
	require.NoError(t, TypedPredict[stateObligation, stateQuotedNumbers]{LM: lm}.Validate())
	require.NoError(t, TypedPredict[stateQuotedNumbers, stateObligation]{LM: lm}.Validate())

	for value, want := range map[string]json.Number{
		`"5"`:              "5",
		`"-2.5e3"`:         "-2.5e3",
		`"-0"`:             "-0",
		`"1e400"`:          "1e400",
		`"\"5\""`:          "5",
		`"\"-1.5E+2\""`:    "-1.5E+2",
		`"\"\\u0037\""`:    "7",
		`"0.000000000001"`: "0.000000000001",
	} {
		state := NewState(Prediction{"k": json.RawMessage(`{"n":` + value + `}`)})
		got, err := Key[stateQuotedNumber]{Name: "k"}.Get(state)
		require.NoError(t, err, value)
		require.Equal(t, want, got.N, value)
	}

	maybe := json.Number("7")
	want := stateQuotedNumbers{
		N:      "1",
		Maybe:  &maybe,
		List:   []stateQuotedNumber{{N: "2"}, {N: "3"}},
		ByName: map[string]stateQuotedNumber{"a": {N: "4"}},
	}
	for name, value := range map[string]any{
		"raw JSON": json.RawMessage(`{"n":"1","maybe":"\"7\"","list":[{"n":"2"},{"n":"\"3\""}],"by_name":{"a":{"n":"4"}}}`),
		"model reply": map[string]any{
			"n":       "1",
			"maybe":   "7",
			"list":    []any{map[string]any{"n": "2"}, map[string]any{"n": `"3"`}},
			"by_name": map[string]any{"a": map[string]any{"n": "4"}},
		},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := Key[stateQuotedNumbers]{Name: "k"}.Get(NewState(Prediction{"k": value}))
			require.NoError(t, err)
			require.Equal(t, want, got)
		})
	}
	for _, value := range []string{
		`{"n":"1","list":null,"by_name":null}`,
		`{"n":"1","maybe":null,"list":[],"by_name":{}}`,
	} {
		got, err := Key[stateQuotedNumbers]{Name: "k"}.Get(NewState(Prediction{"k": json.RawMessage(value)}))
		require.NoError(t, err, value)
		require.Nil(t, got.Maybe, value)
	}

	refused := []struct {
		name    string
		value   string
		wantErr string
	}{
		// v0.8.0 kept these.
		{"hexadecimal", `{"n":"0x10"}`, `field "n" is "0x10", which is not a JSON number`},
		{"a leading zero", `{"n":"01"}`, `field "n" is "01", which is not a JSON number`},
		{"no fraction digits", `{"n":"1."}`, `field "n" is "1.", which is not a JSON number`},
		{"infinity", `{"n":"-Inf"}`, `field "n" is "-Inf", which is not a JSON number`},
		{"a trailing space", `{"n":"5 "}`, `field "n" is "5 ", which is not a JSON number`},
		// v0.8.0 refused these too.
		{"a string holding text", `{"n":"\"x\""}`, `field "n" is "\"x\"", which is not a JSON number`},
		{"a string holding a leading zero", `{"n":"\"01\""}`, `field "n" is "\"01\"", which is not a JSON number`},
		{"a boolean in a string", `{"n":"true"}`, `field "n" is "true", which is not a JSON number`},
		{
			"the string null",
			`{"n":"null"}`,
			`field "n" is the string "null", which a ",string" field decodes as no value`,
		},
		{
			"a number without quotes",
			`{"n":5}`,
			"json: cannot unmarshal number into Go struct field stateQuotedNumber.n of type json.Number",
		},
		{"null", `{"n":null}`, `field "n" is null`},
	}
	for _, tt := range refused {
		t.Run(tt.name, func(t *testing.T) {
			_, err := decodeJSONValue(reflect.TypeFor[stateQuotedNumber](), json.RawMessage(tt.value))
			require.EqualError(t, err, tt.wantErr)
		})
	}
	_, err = decodeJSONValue(
		reflect.TypeFor[stateQuotedNumbers](),
		json.RawMessage(`{"n":"1","list":[{"n":"x"}],"by_name":null}`),
	)
	require.EqualError(t, err, `field "list[0].n" is "x", which is not a JSON number`)
}

// TestDecodeQuotedNumber checks decodeQuotedNumber on its own: json.Number's method refuses the
// JSON string holding a number that it accepts, and it refuses what is not a number with the
// error json.Number's method gives.
func TestDecodeQuotedNumber(t *testing.T) {
	var number stateQuotedNumber
	data := []byte(`{"n":"\"5\""}`)
	require.Error(t, jsonv2.Unmarshal(data, &number), "json.Number's own method")
	require.NoError(t, jsonv2.Unmarshal(data, &number, decodeOptions))
	require.Equal(t, json.Number("5"), number.N)

	err := jsonv2.Unmarshal([]byte(`{"n":"0x10"}`), &number, decodeOptions)
	var semantic *jsonv2.SemanticError
	require.ErrorAs(t, err, &semantic)
	require.Equal(t, jsonNumberType, semantic.GoType)
	require.Equal(t, jsontext.Value(`"0x10"`), semantic.JSONValue)
	require.ErrorIs(t, err, strconv.ErrSyntax)

	err = jsonv2.Unmarshal([]byte(`{"n":"\q"}`), &number, decodeOptions)
	var syntactic *jsontext.SyntacticError
	require.ErrorAs(t, err, &syntactic)

	var plain json.Number
	require.NoError(t, jsonv2.Unmarshal([]byte(`"5"`), &plain, decodeOptions), "outside a \",string\" field")
	require.Error(t, jsonv2.Unmarshal([]byte(`"\"5\""`), &plain, decodeOptions))
}

// stateHolder holds a value of T as the JSON field "v".
type stateHolder[T any] struct {
	V T `json:"v"`
}

// stateTree contains itself.
type stateTree struct {
	Children []stateTree `json:"children"`
}

// TestTypedPredictChecksStructsGivenInputsHold checks that TypedPredict with given inputs, like
// one whose inputs are derived and like JSONSchemaFor, refuses an In that holds a struct that
// does not decode, behind a pointer, a slice, an array, a map or another struct, since
// RunCandidate decodes its inputs into In.
func TestTypedPredictChecksStructsGivenInputsHold(t *testing.T) {
	lm := textModel(nil)
	given := Signature{Inputs: []Field{{Name: "v"}}}
	const opaque = "gepa.stateOpaque cannot be decoded from JSON: Go struct has no exported fields"
	tests := []struct {
		name     string
		validate func() error
		schema   func() (json.RawMessage, error)
		wantErr  string
	}{
		{
			name:     "pointer",
			validate: TypedPredict[stateHolder[*stateOpaque], stateObligation]{Signature: given, LM: lm}.Validate,
			schema:   JSONSchemaFor[stateHolder[*stateOpaque]],
			wantErr:  `field "v": ` + opaque,
		},
		{
			name:     "slice",
			validate: TypedPredict[stateHolder[[]stateOpaque], stateObligation]{Signature: given, LM: lm}.Validate,
			schema:   JSONSchemaFor[stateHolder[[]stateOpaque]],
			wantErr:  `field "v[]": ` + opaque,
		},
		{
			name:     "array",
			validate: TypedPredict[stateHolder[[1]stateOpaque], stateObligation]{Signature: given, LM: lm}.Validate,
			schema:   JSONSchemaFor[stateHolder[[1]stateOpaque]],
			wantErr:  `field "v[]": ` + opaque,
		},
		{
			name: "map value",
			validate: TypedPredict[stateHolder[map[string]stateOpaque], stateObligation]{
				Signature: given,
				LM:        lm,
			}.Validate,
			schema:  JSONSchemaFor[stateHolder[map[string]stateOpaque]],
			wantErr: `field "v[]": ` + opaque,
		},
		{
			name: "struct in a struct",
			validate: TypedPredict[stateHolder[stateHoldsOpaque], stateObligation]{
				Signature: given,
				LM:        lm,
			}.Validate,
			schema:  JSONSchemaFor[stateHolder[stateHoldsOpaque]],
			wantErr: `field "v.opaque": ` + opaque,
		},
		{
			name: "map In",
			validate: TypedPredict[map[string]*stateOpaque, stateObligation]{
				Signature: Signature{Inputs: []Field{{Name: "k"}}},
				LM:        lm,
			}.Validate,
			schema:  JSONSchemaFor[map[string]*stateOpaque],
			wantErr: `field "[]": ` + opaque,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.EqualError(t, tt.validate(), "typed predict inputs: "+tt.wantErr)
			_, err := tt.schema()
			require.ErrorContains(t, err, ": "+tt.wantErr)
		})
	}

	for name, validate := range map[string]func() error{
		"structs that decode": TypedPredict[stateHolder[*stateObligation], stateObligation]{
			Signature: given,
			LM:        lm,
		}.Validate,
		"a struct that contains itself": TypedPredict[stateHolder[stateTree], stateObligation]{
			Signature: given,
			LM:        lm,
		}.Validate,
		"a type that decodes itself": TypedPredict[stateHolder[celsius], stateObligation]{
			Signature: given,
			LM:        lm,
		}.Validate,
		"an interface": TypedPredict[stateHolder[any], stateObligation]{Signature: given, LM: lm}.Validate,
	} {
		t.Run(name, func(t *testing.T) {
			require.NoError(t, validate())
		})
	}
}

// StateTerms is exported so encoding/json can set a nil embedded pointer to it.
type StateTerms struct {
	Days int `json:"days"`
}

type stateEmbedsObligation struct {
	stateObligation
	Note string `json:"note"`
}

type stateEmbedsTerms struct {
	*StateTerms
	Note string `json:"note"`
}

type stateEmbedsTwice struct {
	stateEmbedsObligation
}

type stateEmbedsHolder struct {
	stateHolder[stateObligation]
}

// TestDecodeErrorNamesEmbeddedStructs checks that a type error in a field promoted from an
// embedded struct names the embedded structs on the way to it, as v0.8.0 did, and that the
// field stops at a type that decodes itself, whose JSON form is its own.
func TestDecodeErrorNamesEmbeddedStructs(t *testing.T) {
	tests := []struct {
		name    string
		typ     reflect.Type
		data    string
		want    json.UnmarshalTypeError
		wantErr string
	}{
		{
			name: "embedded struct",
			typ:  reflect.TypeFor[stateEmbedsObligation](),
			data: `{"amount":"ten","deadline":"d","note":"n"}`,
			want: json.UnmarshalTypeError{Struct: "stateEmbedsObligation", Field: "stateObligation.amount"},
			wantErr: "json: cannot unmarshal string into Go struct field " +
				"stateEmbedsObligation.stateObligation.amount of type int",
		},
		{
			name:    "embedded pointer",
			typ:     reflect.TypeFor[stateEmbedsTerms](),
			data:    `{"days":"x","note":"n"}`,
			want:    json.UnmarshalTypeError{Struct: "stateEmbedsTerms", Field: "StateTerms.days"},
			wantErr: `json: cannot unmarshal string into Go struct field stateEmbedsTerms.StateTerms.days of type int`,
		},
		{
			name: "embedded twice",
			typ:  reflect.TypeFor[stateEmbedsTwice](),
			data: `{"amount":"ten","deadline":"d","note":"n"}`,
			want: json.UnmarshalTypeError{
				Struct: "stateEmbedsTwice",
				Field:  "stateEmbedsObligation.stateObligation.amount",
			},
			wantErr: "json: cannot unmarshal string into Go struct field " +
				"stateEmbedsTwice.stateEmbedsObligation.stateObligation.amount of type int",
		},
		{
			name:    "struct in an embedded struct",
			typ:     reflect.TypeFor[stateEmbedsHolder](),
			data:    `{"v":{"amount":"ten","deadline":"d"}}`,
			want:    json.UnmarshalTypeError{Struct: "stateObligation", Field: "stateHolder.v.amount"},
			wantErr: `json: cannot unmarshal string into Go struct field stateObligation.stateHolder.v.amount of type int`,
		},
		{
			name:    "type that decodes itself",
			typ:     reflect.TypeFor[celsiusReadings](),
			data:    `{"readings":[{"c":"warm"}]}`,
			want:    json.UnmarshalTypeError{Struct: "celsiusReadings", Field: "readings"},
			wantErr: `json: cannot unmarshal string into Go struct field celsiusReadings.readings of type float64`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := decodeJSONValue(tt.typ, json.RawMessage(tt.data))
			require.EqualError(t, err, tt.wantErr)
			var typeErr *json.UnmarshalTypeError
			require.ErrorAs(t, err, &typeErr)
			require.Equal(t, tt.want.Struct, typeErr.Struct)
			require.Equal(t, tt.want.Field, typeErr.Field)
		})
	}
}

// TestDecodeMapKeysAsV080 checks that map keys decode as v0.8.0 decoded them: a text
// unmarshaler with UnmarshalJSON, given the quoted key, or else UnmarshalText; any other key by
// its kind, as a string or a base-10 integer, whatever other unmarshal method its type has. Such
// a type still decodes values with its method.
func TestDecodeMapKeysAsV080(t *testing.T) {
	tests := []struct {
		name string
		typ  reflect.Type
		data string
		want any
	}{
		{
			name: "json.Number keys hold any text",
			typ:  reflect.TypeFor[map[json.Number]int](),
			data: `{"abc":1,"01":2,"":3}`,
			want: map[json.Number]int{"abc": 1, "01": 2, "": 3},
		},
		{
			name: "a string key with UnmarshalJSON",
			typ:  reflect.TypeFor[map[rawKey]int](),
			data: `{"a":1}`,
			want: map[rawKey]int{"a": 1},
		},
		{
			name: "an int key with UnmarshalJSON",
			typ:  reflect.TypeFor[map[rawCount]int](),
			data: `{"1":1,"-2":2}`,
			want: map[rawCount]int{1: 1, -2: 2},
		},
		{
			name: "a uint key with UnmarshalJSON",
			typ:  reflect.TypeFor[map[rawSize]int](),
			data: `{"7":1}`,
			want: map[rawSize]int{7: 1},
		},
		{
			name: "an int key with UnmarshalJSONFrom",
			typ:  reflect.TypeFor[map[fromLevel]int](),
			data: `{"3":1}`,
			want: map[fromLevel]int{3: 1},
		},
		{
			name: "a text unmarshaler with UnmarshalJSON",
			typ:  reflect.TypeFor[map[rawLabel]int](),
			data: `{"a":1}`,
			want: map[rawLabel]int{`json "a"`: 1},
		},
		{
			name: "a text unmarshaler",
			typ:  reflect.TypeFor[map[stateCode]int](),
			data: `{"4":1}`,
			want: map[stateCode]int{4: 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := decodeJSONValue(tt.typ, json.RawMessage(tt.data))
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}

	key, err := decodeJSONValue(reflect.TypeFor[rawKey](), "a")
	require.NoError(t, err)
	require.Equal(t, rawKey(`"a"`), key, "a value still decodes with the method")
	level, err := decodeJSONValue(reflect.TypeFor[fromLevel](), 3)
	require.NoError(t, err)
	require.Equal(t, fromLevel(30), level)

	refused := []struct {
		name    string
		typ     reflect.Type
		data    string
		wantErr string
	}{
		{
			name:    "null for an int key with UnmarshalJSON",
			typ:     reflect.TypeFor[map[rawCount]int](),
			data:    `{"null":1}`,
			wantErr: `json: cannot unmarshal number null into Go value of type gepa.rawCount`,
		},
		{
			name:    "a fraction for an int key with UnmarshalJSONFrom",
			typ:     reflect.TypeFor[map[fromLevel]int](),
			data:    `{"1.5":1}`,
			wantErr: `json: cannot unmarshal number 1.5 into Go value of type gepa.fromLevel`,
		},
		{
			name:    "a number too big for a uint key",
			typ:     reflect.TypeFor[map[rawSize]int](),
			data:    `{"256":1}`,
			wantErr: `json: cannot unmarshal number 256 into Go value of type gepa.rawSize`,
		},
		{
			name:    "a negative uint key",
			typ:     reflect.TypeFor[map[rawSize]int](),
			data:    `{"-1":1}`,
			wantErr: `json: cannot unmarshal number -1 into Go value of type gepa.rawSize`,
		},
		{
			name:    "a key its UnmarshalText refuses",
			typ:     reflect.TypeFor[map[refusingLabel]int](),
			data:    `{"bad":1}`,
			wantErr: `json: cannot unmarshal string into Go value of type gepa.refusingLabel: label refused`,
		},
		{
			name:    "a struct key with UnmarshalJSONFrom",
			typ:     reflect.TypeFor[map[celsius]int](),
			data:    `{"a":1}`,
			wantErr: `json: cannot unmarshal object into Go value of type map[gepa.celsius]int`,
		},
	}
	for _, tt := range refused {
		t.Run(tt.name, func(t *testing.T) {
			_, err := decodeJSONValue(tt.typ, json.RawMessage(tt.data))
			require.EqualError(t, err, tt.wantErr)
		})
	}

	dec := jsontext.NewDecoder(strings.NewReader("{\"a\xff\":1}"))
	_, err = dec.ReadToken()
	require.NoError(t, err)
	var named rawKey
	require.ErrorContains(t, decodeMethodMapKey[json.Unmarshaler](dec, &named), "invalid UTF-8")

	dec = jsontext.NewDecoder(strings.NewReader(`"a"`))
	require.ErrorIs(t, decodeMethodMapKey[json.Unmarshaler](dec, &named), errors.ErrUnsupported)
}
