package classify_test

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/mikills/gepago/programs/classify"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDecision(t *testing.T) {
	t.Run("abstained", func(t *testing.T) {
		assert.False(t, classify.Decision{Label: "billing", Proposed: "billing", Confidence: 0.9}.Abstained())
		assert.True(
			t,
			classify.Decision{Proposed: "billing", Confidence: 0.4, Abstain: classify.LowConfidence}.Abstained(),
		)
		assert.True(t, classify.Decision{Abstain: classify.ModelFailure}.Abstained())
	})

	t.Run("json", func(t *testing.T) {
		tests := []struct {
			name     string
			decision classify.Decision
			want     string
		}{
			{
				name: "answered",
				decision: classify.Decision{
					Label:      "refund",
					Proposed:   "refund",
					Confidence: 0.9,
					Reason:     "asks for money back",
					Route:      "classify",
				},
				want: `{"label":"refund","proposed":"refund","confidence":0.9,"reason":"asks for money back",` +
					`"route":"classify"}`,
			},
			{
				name: "low confidence",
				decision: classify.Decision{
					Proposed:   "billing",
					Confidence: 0.4,
					Abstain:    classify.LowConfidence,
					Route:      "classify",
				},
				want: `{"proposed":"billing","confidence":0.4,"abstain":"low_confidence","route":"classify"}`,
			},
			{
				name:     "refused",
				decision: classify.Decision{Abstain: classify.Refused},
				want:     `{"confidence":0,"abstain":"refused"}`,
			},
			{
				name: "outputs",
				decision: classify.Decision{
					Label:      "refund",
					Proposed:   "refund",
					Confidence: 0.9,
					Route:      "classify",
					Outputs:    map[string]any{"urgency": "high", "items": []any{"A7"}},
				},
				want: `{"label":"refund","proposed":"refund","confidence":0.9,"route":"classify",` +
					`"outputs":{"urgency":"high","items":["A7"]}}`,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				data, err := json.Marshal(tt.decision)
				require.NoError(t, err)
				require.JSONEq(t, tt.want, string(data))

				var decoded classify.Decision
				require.NoError(t, json.Unmarshal(data, &decoded))
				require.Equal(t, tt.decision, decoded)
			})
		}
	})
}

// shipment is an output value with a Validate method.
type shipment struct {
	Order    string `json:"order"`
	Quantity int    `json:"quantity"`
}

func (s shipment) Validate() error {
	if s.Quantity < 1 {
		return fmt.Errorf("quantity %d is below 1", s.Quantity)
	}
	return nil
}

func TestOutput(t *testing.T) {
	decision := classify.Decision{
		Label:    "refund",
		Proposed: "refund",
		Outputs: map[string]any{
			"shipment": map[string]any{"order": "A7", "quantity": json.Number("2")},
			"empty":    map[string]any{"order": "A7", "quantity": 0},
			"extra":    map[string]any{"order": "A7", "quantity": 2, "carrier": "post"},
			"urgency":  "high",
			"none":     nil,
		},
	}

	t.Run("decodes", func(t *testing.T) {
		got, err := classify.Output[shipment](decision, "shipment")

		require.NoError(t, err)
		require.Equal(t, shipment{Order: "A7", Quantity: 2}, got)
		urgency, err := classify.Output[string](decision, "urgency")
		require.NoError(t, err)
		require.Equal(t, "high", urgency)
		none, err := classify.Output[*shipment](decision, "none")
		require.NoError(t, err)
		require.Nil(t, none)
	})

	t.Run("after json", func(t *testing.T) {
		data, err := json.Marshal(decision)
		require.NoError(t, err)
		var decoded classify.Decision
		require.NoError(t, json.Unmarshal(data, &decoded))

		got, err := classify.Output[shipment](decoded, "shipment")

		require.NoError(t, err)
		require.Equal(t, shipment{Order: "A7", Quantity: 2}, got)
	})

	t.Run("errors", func(t *testing.T) {
		tests := []struct {
			name     string
			decision classify.Decision
			output   string
			read     func(classify.Decision, string) error
			err      string
		}{
			{
				name:     "missing",
				decision: decision,
				output:   "carrier",
				read:     read[shipment],
				err:      `classify decision has no output "carrier"`,
			},
			{
				name:     "abstained",
				decision: classify.Decision{Abstain: classify.Refused},
				output:   "shipment",
				read:     read[shipment],
				err:      `classify decision has no output "shipment"`,
			},
			{
				name:     "wrong type",
				decision: decision,
				output:   "urgency",
				read:     read[int],
				err: `classify decision output: state key "urgency": decode as int: ` +
					`json: cannot unmarshal string into Go value of type int`,
			},
			{
				name:     "unknown field",
				decision: decision,
				output:   "extra",
				read:     read[shipment],
				err: `classify decision output: state key "extra": decode as classify_test.shipment: ` +
					`field "carrier" is not in classify_test.shipment`,
			},
			{
				name:     "invalid",
				decision: decision,
				output:   "empty",
				read:     read[shipment],
				err:      `classify decision output: state key "empty": quantity 0 is below 1`,
			},
			{
				name:     "null",
				decision: decision,
				output:   "none",
				read:     read[shipment],
				err:      `classify decision output: state key "none" is null, want classify_test.shipment`,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := tt.read(tt.decision, tt.output)

				require.EqualError(t, err, tt.err)
			})
		}
	})
}

// read reads the output name of d as T and returns the error.
func read[T any](d classify.Decision, name string) error {
	_, err := classify.Output[T](d, name)
	return err
}
