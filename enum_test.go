package gepa_test

import (
	"context"
	"encoding/json"
	"testing"

	gepa "github.com/mikills/gepago"
	"github.com/mikills/gepago/gepatest"
	"github.com/stretchr/testify/require"
)

type enumIncident struct {
	Impact string `json:"impact" enum:"outage,degraded,none"`
}

func checkEnumState[T any](t *testing.T, valid, invalid T, field, reason string) {
	t.Helper()
	key := gepa.Key[T]{Name: "details"}
	for _, tc := range []struct {
		name  string
		value T
		valid bool
	}{
		{"valid", valid, true},
		{"invalid", invalid, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := gepa.NewState(nil)
			key.Set(state, tc.value)
			for _, state := range []gepa.State{state, gepa.NewState(gepa.Prediction{"details": tc.value})} {
				got, err := key.Get(state)
				if tc.valid {
					require.NoError(t, err)
					require.Equal(t, tc.value, got)
				} else {
					require.ErrorContains(t, err, `field "`+field+`"`)
					require.ErrorContains(t, err, reason)
				}
			}
			step := gepa.NewStep("write", nil, []gepa.KeyRef{key.Ref()}, func(_ context.Context, s gepa.State) error {
				key.Set(s, tc.value)
				return nil
			})
			pipeline, err := (gepa.PipelineSpec{Steps: []gepa.Step{step}}).Build()
			require.NoError(t, err)
			out, err := pipeline.RunState(context.Background(), gepa.NewState(nil))
			if tc.valid {
				require.NoError(t, err)
				got, err := key.Get(out)
				require.NoError(t, err)
				require.Equal(t, tc.value, got)
			} else {
				require.ErrorContains(t, err, `field "`+field+`"`)
				require.ErrorContains(t, err, reason)
				require.NotContains(t, out.Prediction(), "details", "the failed write must not commit")
			}
			called := false
			read := gepa.NewStep("read", []gepa.KeyRef{key.Ref()}, nil, func(_ context.Context, s gepa.State) error {
				called = true
				_, err := key.Get(s)
				return err
			})
			pipeline, err = (gepa.PipelineSpec{Inputs: []gepa.KeyRef{key.Ref()}, Steps: []gepa.Step{read}}).Build()
			require.NoError(t, err)
			_, err = pipeline.RunState(context.Background(), gepa.NewState(gepa.Prediction{"details": tc.value}))
			if tc.valid {
				require.NoError(t, err)
				require.True(t, called)
			} else {
				require.ErrorContains(t, err, `field "`+field+`"`)
				require.ErrorContains(t, err, reason)
				require.False(t, called, "invalid inputs are refused before running steps")
			}
		})
	}
}

func TestEnumState(t *testing.T) {
	valid, invalid := enumIncident{Impact: "outage"}, enumIncident{Impact: "down"}
	t.Run("struct", func(t *testing.T) { checkEnumState(t, valid, invalid, "impact", "not one of") })
	t.Run("pointer", func(t *testing.T) { checkEnumState(t, &valid, &invalid, "impact", "not one of") })
	t.Run("deep pointer", func(t *testing.T) {
		vp, ip := &valid, &invalid
		vpp, ipp := &vp, &ip
		checkEnumState(t, &vpp, &ipp, "impact", "not one of")
	})
	t.Run("slice", func(t *testing.T) {
		checkEnumState(t, []enumIncident{valid}, []enumIncident{invalid}, "[0].impact", "not one of")
	})
	t.Run("map", func(t *testing.T) {
		checkEnumState(
			t,
			map[string]*enumIncident{"a": &valid},
			map[string]*enumIncident{"a": &invalid},
			"a.impact",
			"not one of",
		)
	})
	t.Run("array", func(t *testing.T) {
		checkEnumState(t, [1]enumIncident{valid}, [1]enumIncident{invalid}, "[0].impact", "not one of")
	})
	t.Run("embedded", func(t *testing.T) {
		type details struct{ *enumIncident }
		checkEnumState(t, details{&valid}, details{&invalid}, "impact", "not one of")
	})
	t.Run("nested", func(t *testing.T) {
		type details struct {
			Child enumIncident `json:"child"`
		}
		checkEnumState(t, details{valid}, details{invalid}, "child.impact", "not one of")
	})
	t.Run("named strings", func(t *testing.T) {
		type impact string
		type impacts []impact
		type details struct {
			Impact impact  `json:"impact" enum:"outage,none"`
			Maybe  *impact `json:"maybe"  enum:"outage,none"`
			Tags   impacts `json:"tags"   enum:"outage,none"`
		}
		v, i := impact("none"), impact("down")
		good := details{Impact: "outage", Maybe: &v, Tags: impacts{"none"}}
		for _, tc := range []struct {
			name  string
			value details
		}{
			{"impact", details{Impact: i}},
			{"maybe", details{Impact: "outage", Maybe: &i}},
			{"tags[1]", details{Impact: "outage", Tags: impacts{"none", i}}},
		} {
			t.Run(tc.name, func(t *testing.T) { checkEnumState(t, good, tc.value, tc.name, "not one of") })
		}
	})
	t.Run("optional and null", func(t *testing.T) {
		type details struct {
			Impact   string                  `json:"impact,omitempty" enum:"outage,none"`
			Zero     string                  `json:"zero,omitzero"    enum:"outage,none"`
			Maybe    *string                 `json:"maybe"            enum:"outage,none"`
			Tags     []string                `json:"tags"             enum:"outage,none"`
			Child    *enumIncident           `json:"child"`
			Children []enumIncident          `json:"children"`
			ByName   map[string]enumIncident `json:"by_name"`
		}
		key := gepa.Key[details]{Name: "v"}
		for _, data := range []string{
			`{"maybe":null,"tags":null,"child":null,"children":null,"by_name":null}`,
			`{"impact":null,"zero":null,"maybe":null,"tags":[],"child":null,"children":[],"by_name":{}}`,
		} {
			var value any
			require.NoError(t, json.Unmarshal([]byte(data), &value))
			got, err := key.Get(gepa.NewState(gepa.Prediction{"v": value}))
			require.NoError(t, err)
			state := gepa.NewState(nil)
			key.Set(state, got)
			_, err = key.Get(state)
			require.NoError(t, err, "decoded nulls and omissions remain valid as typed values")
		}
	})
	t.Run("recursive values", func(t *testing.T) {
		type node struct {
			Next   *node  `json:"next"`
			Impact string `json:"impact" enum:"outage,none"`
		}
		good := &node{Impact: "none"}
		good.Next = good
		bad := &node{Impact: "outage", Next: &node{Impact: "down"}}
		bad.Next.Next = bad
		checkEnumState(t, good, bad, "next.impact", "not one of")
	})
}

func checkEnumDefinition[T any](t *testing.T, reason string, payloads ...string) {
	t.Helper()
	_, err := gepa.JSONSchemaFor[T]()
	require.ErrorContains(t, err, reason)
	type input struct {
		Text string `json:"text"`
	}
	program := gepa.TypedPredict[input, T]{LM: &gepatest.FakeGenerator{}}
	require.ErrorContains(t, program.Validate(), reason)
	tool := gepa.NewTool("details", "Read details", func(context.Context, T) (string, error) { return "", nil })
	require.ErrorContains(t, tool.(interface{ Validate() error }).Validate(), reason)
	key := gepa.Key[T]{Name: "v"}
	for _, payload := range payloads {
		var value any
		require.NoError(t, json.Unmarshal([]byte(payload), &value))
		_, err = key.Get(gepa.NewState(gepa.Prediction{"v": value}))
		require.ErrorContains(t, err, reason, payload)
	}
	var zero T
	_, err = key.Get(gepa.NewState(gepa.Prediction{"v": zero}))
	require.ErrorContains(t, err, reason)
}

func TestEnumDefinition(t *testing.T) {
	type bad struct {
		Impact int `json:"impact" enum:"1,2"`
	}
	const kind = "cannot have an enum tag"
	t.Run("root null", func(t *testing.T) {
		state := gepa.NewState(gepa.Prediction{"v": nil})
		_, err := (gepa.Key[*bad]{Name: "v"}).Get(state)
		require.ErrorContains(t, err, kind)
		_, err = (gepa.Key[[]bad]{Name: "v"}).Get(state)
		require.ErrorContains(t, err, kind)
		_, err = (gepa.Key[map[string]bad]{Name: "v"}).Get(state)
		require.ErrorContains(t, err, kind)
		_, err = (gepa.Key[*enumIncident]{Name: "v"}).Get(state)
		require.NoError(t, err)
		_, err = (gepa.Key[[]enumIncident]{Name: "v"}).Get(state)
		require.NoError(t, err)
	})
	t.Run("omitted struct", func(t *testing.T) {
		type details struct {
			Child bad `json:"child,omitempty"`
		}
		checkEnumDefinition[details](t, kind, `{}`, `{"child":null}`, `{"child":{"impact":1}}`)
	})
	t.Run("null pointer", func(t *testing.T) {
		type details struct {
			Child ***bad `json:"child"`
		}
		checkEnumDefinition[details](t, kind, `{"child":null}`)
	})
	t.Run("empty slice", func(t *testing.T) {
		type details struct {
			Children []bad `json:"children"`
		}
		checkEnumDefinition[details](t, kind, `{"children":null}`, `{"children":[]}`)
	})
	t.Run("empty map", func(t *testing.T) {
		type details struct {
			Children map[string]bad `json:"children"`
		}
		checkEnumDefinition[details](t, kind, `{"children":null}`, `{"children":{}}`)
	})
	t.Run("recursive definition", func(t *testing.T) {
		type details struct {
			Next *details `json:"next"`
			Bad  *bad     `json:"bad"`
		}
		checkEnumDefinition[details](t, kind, `{"next":null,"bad":null}`)
	})
	t.Run("embedded struct", func(t *testing.T) {
		type details struct {
			enumIncident `enum:"a,b"`
		}
		checkEnumDefinition[details](t, kind, `{"impact":"outage"}`)
	})
	t.Run("embedded pointer", func(t *testing.T) {
		type details struct {
			*enumIncident `enum:"a,b"`
		}
		checkEnumDefinition[details](t, kind, `{"impact":"outage"}`)
	})
	t.Run("excluded embedded", func(t *testing.T) {
		type details struct {
			*enumIncident `       json:"-"    enum:"a,b"`
			Text          string `json:"text"`
		}
		_, err := gepa.JSONSchemaFor[details]()
		require.NoError(t, err)
		_, err = (gepa.Key[details]{Name: "v"}).Get(gepa.NewState(gepa.Prediction{"v": map[string]any{"text": "ok"}}))
		require.NoError(t, err)
	})
	t.Run("invalid UTF-8", func(t *testing.T) {
		type details struct {
			Value string "json:\"value\" enum:\"\\xff\""
		}
		checkEnumDefinition[details](t, "invalid UTF-8", `{"value":"\ufffd"}`)
	})
	t.Run("literal text", func(t *testing.T) {
		type details struct {
			Value string "json:\"value\" enum:\"a, b,中文,\\\"quote\\\",line\\n,back\\\\slash\""
		}
		schema, err := gepa.JSONSchemaFor[details]()
		require.NoError(t, err)
		require.True(t, gepa.StrictJSONSchema(schema))
		predict := gepa.Predict{Signature: gepa.Signature{Outputs: []gepa.Field{{Name: "v", Schema: schema}}}}
		for _, text := range []string{"a", " b", "中文", `"quote"`, "line\n", `back\slash`} {
			t.Run(text, func(t *testing.T) {
				data, err := json.Marshal(gepa.Prediction{"v": details{Value: text}})
				require.NoError(t, err)
				_, err = predict.ParseOutput(string(data))
				require.NoError(t, err)
				checkEnumState(t, details{Value: text}, details{Value: "b"}, "value", "not one of")
			})
		}
	})
}

func TestEnumInterface(t *testing.T) {
	type envelope struct {
		Payload any `json:"payload,omitempty"`
	}
	valid, invalid := enumIncident{Impact: "outage"}, enumIncident{Impact: "down"}
	t.Run("field", func(t *testing.T) {
		checkEnumState(t, envelope{valid}, envelope{invalid}, "payload.impact", "not one of")
	})
	t.Run("tagged sibling", func(t *testing.T) {
		type tagged struct {
			Kind    string `json:"kind"    enum:"event"`
			Payload any    `json:"payload"`
		}
		checkEnumState(t, tagged{"event", valid}, tagged{"event", invalid}, "payload.impact", "not one of")
	})
	t.Run("slice", func(t *testing.T) {
		checkEnumState(t, []any{valid}, []any{invalid}, "[0].impact", "not one of")
	})
	t.Run("map", func(t *testing.T) {
		checkEnumState(t, map[string]any{"item": valid}, map[string]any{"item": invalid}, "item.impact", "not one of")
	})
	t.Run("nested", func(t *testing.T) {
		checkEnumState(t, [1]any{map[string]any{"item": &envelope{valid}}},
			[1]any{map[string]any{"item": &envelope{invalid}}}, "[0].item.payload.impact", "not one of")
	})
	t.Run("concrete definitions", func(t *testing.T) {
		type bad struct {
			Count int `json:"count" enum:"1,2"`
		}
		for _, tc := range []struct {
			name  string
			value any
		}{
			{"struct", bad{}},
			{"nil pointer", (*bad)(nil)},
			{"nil slice", []bad(nil)},
			{"empty map", map[string]bad{}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("field", func(t *testing.T) {
					checkEnumState(t, envelope{}, envelope{tc.value}, "payload", "cannot have an enum tag")
				})
				t.Run("slice", func(t *testing.T) {
					checkEnumState(t, []any{nil}, []any{tc.value}, "[0]", "cannot have an enum tag")
				})
				t.Run("map", func(t *testing.T) {
					checkEnumState(t, map[string]any{"item": nil}, map[string]any{"item": tc.value},
						"item", "cannot have an enum tag")
				})
			})
		}
	})
	t.Run("null and optional", func(t *testing.T) {
		type optional struct {
			Impact string `json:"impact,omitempty" enum:"outage,none"`
		}
		for _, tc := range []struct {
			name  string
			value any
		}{
			{"nil", nil},
			{"nil pointer", (*enumIncident)(nil)},
			{"nil slice", []enumIncident(nil)},
			{"empty map", map[string]enumIncident{}},
			{"optional zero", optional{}},
			{"plain", "no tags"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				checkEnumState(t, envelope{tc.value}, envelope{invalid}, "payload.impact", "not one of")
			})
		}
	})
	t.Run("pointer cycle", func(t *testing.T) {
		type node struct {
			Next    any `json:"next"`
			Payload any `json:"payload"`
		}
		good, bad := &node{Payload: valid}, &node{Payload: invalid}
		good.Next, bad.Next = good, bad
		checkEnumState(t, good, bad, "payload.impact", "not one of")
	})
	t.Run("map cycle", func(t *testing.T) {
		good, bad := map[string]any{"z": valid}, map[string]any{"z": invalid}
		good["a"], bad["a"] = good, bad
		checkEnumState(t, good, bad, "z.impact", "not one of")
	})
	t.Run("slice cycle", func(t *testing.T) {
		good, bad := []any{nil, valid}, []any{nil, invalid}
		good[0], bad[0] = good, bad
		checkEnumState(t, good, bad, "[1].impact", "not one of")
	})
}

func BenchmarkEnumGet(b *testing.B) {
	type plain struct {
		Service string   `json:"service"`
		Count   int      `json:"count"`
		Note    string   `json:"note,omitempty" description:"operator note"`
		Tags    []string `json:"tags"`
	}
	key := gepa.Key[plain]{Name: "v"}
	for _, tc := range []struct {
		name  string
		value any
	}{
		{"plain typed", plain{Service: "checkout", Count: 3, Note: "retry", Tags: []string{"api", "web"}}},
		{"plain JSON", map[string]any{"service": "checkout", "count": 3, "note": "retry", "tags": []string{"api", "web"}}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			state := gepa.NewState(gepa.Prediction{"v": tc.value})
			_, err := key.Get(state)
			require.NoError(b, err)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := key.Get(state); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
