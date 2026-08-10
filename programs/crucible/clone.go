package crucible

import (
	"maps"
	"reflect"

	gepa "github.com/mikills/gepago"
)

func clonePrediction(input gepa.Prediction) gepa.Prediction {
	if input == nil {
		return nil
	}
	cloned := make(gepa.Prediction, len(input))
	for key, value := range input {
		cloned[key] = cloneJSONValue(value)
	}
	return cloned
}

func cloneJSONValue(value any) any {
	if value == nil {
		return nil
	}
	switch typed := value.(type) {
	case gepa.Prediction:
		return clonePrediction(typed)
	case map[string]any:
		cloned := make(map[string]any, len(typed))
		for key, item := range typed {
			cloned[key] = cloneJSONValue(item)
		}
		return cloned
	case []any:
		cloned := make([]any, len(typed))
		for index, item := range typed {
			cloned[index] = cloneJSONValue(item)
		}
		return cloned
	case []string:
		return append([]string(nil), typed...)
	default:
		return cloneReflectValue(reflect.ValueOf(value)).Interface()
	}
}

func cloneEvalInput(input EvalInput) EvalInput {
	input.Suite = cloneSuite(input.Suite)
	input.Case = cloneEvalCase(input.Case)
	input.Output = cloneSubjectOutput(input.Output)
	return input
}

func clonePairwiseInput(input PairwiseInput) PairwiseInput {
	input.Suite = cloneSuite(input.Suite)
	input.Case = cloneEvalCase(input.Case)
	input.OutputA = cloneSubjectOutput(input.OutputA)
	input.OutputB = cloneSubjectOutput(input.OutputB)
	return input
}

func cloneSuite(input Suite) Suite {
	input.Cases = append([]EvalCase(nil), input.Cases...)
	for index := range input.Cases {
		input.Cases[index] = cloneEvalCase(input.Cases[index])
	}
	input.Evaluators = cloneEvaluatorSpecs(input.Evaluators)
	input.PairwiseEvaluators = cloneEvaluatorSpecs(input.PairwiseEvaluators)
	return input
}

func cloneEvalCase(input EvalCase) EvalCase {
	input.Input = clonePrediction(input.Input)
	input.Expected = clonePrediction(input.Expected)
	input.Expectations = append([]Expectation(nil), input.Expectations...)
	for index := range input.Expectations {
		input.Expectations[index].Value = cloneJSONValue(input.Expectations[index].Value)
	}
	input.Tags = append([]string(nil), input.Tags...)
	input.Constraints = append([]string(nil), input.Constraints...)
	input.Metadata = cloneStringAnyMap(input.Metadata)
	return input
}

func cloneSubjectOutput(input SubjectOutput) SubjectOutput {
	input.Value = clonePrediction(input.Value)
	input.Metadata = cloneStringAnyMap(input.Metadata)
	input.ToolCalls = cloneToolCalls(input.ToolCalls)
	input.Trace = cloneJSONValue(input.Trace)
	return input
}

func cloneEvaluatorSpecs(input []EvaluatorSpec) []EvaluatorSpec {
	cloned := append([]EvaluatorSpec(nil), input...)
	for index := range cloned {
		cloned[index].Fields = append([]string(nil), cloned[index].Fields...)
		cloned[index].RequiredFields = append([]string(nil), cloned[index].RequiredFields...)
		cloned[index].Headers = cloneStringMap(cloned[index].Headers)
		cloned[index].Args = append([]string(nil), cloned[index].Args...)
		cloned[index].Env = append([]string(nil), cloned[index].Env...)
	}
	return cloned
}

func cloneReflectValue(value reflect.Value) reflect.Value {
	if !value.IsValid() {
		return value
	}
	switch value.Kind() {
	case reflect.Interface:
		return cloneReflectInterface(value)
	case reflect.Pointer:
		return cloneReflectPointer(value)
	case reflect.Map:
		return cloneReflectMap(value)
	case reflect.Slice:
		return cloneReflectSlice(value)
	case reflect.Array:
		return cloneReflectArray(value)
	case reflect.Struct:
		return cloneReflectStruct(value)
	default:
		return value
	}
}

func cloneReflectInterface(value reflect.Value) reflect.Value {
	if value.IsNil() {
		return reflect.Zero(value.Type())
	}
	out := reflect.New(value.Type()).Elem()
	out.Set(cloneReflectValue(value.Elem()))
	return out
}

func cloneReflectPointer(value reflect.Value) reflect.Value {
	if value.IsNil() {
		return reflect.Zero(value.Type())
	}
	out := reflect.New(value.Type().Elem())
	out.Elem().Set(cloneReflectValue(value.Elem()))
	return out
}

func cloneReflectMap(value reflect.Value) reflect.Value {
	if value.IsNil() {
		return reflect.Zero(value.Type())
	}
	out := reflect.MakeMapWithSize(value.Type(), value.Len())
	iterator := value.MapRange()
	for iterator.Next() {
		out.SetMapIndex(cloneReflectValue(iterator.Key()), cloneReflectValue(iterator.Value()))
	}
	return out
}

func cloneReflectSlice(value reflect.Value) reflect.Value {
	if value.IsNil() {
		return reflect.Zero(value.Type())
	}
	out := reflect.MakeSlice(value.Type(), value.Len(), value.Len())
	for index := 0; index < value.Len(); index++ {
		out.Index(index).Set(cloneReflectValue(value.Index(index)))
	}
	return out
}

func cloneReflectArray(value reflect.Value) reflect.Value {
	out := reflect.New(value.Type()).Elem()
	for index := 0; index < value.Len(); index++ {
		out.Index(index).Set(cloneReflectValue(value.Index(index)))
	}
	return out
}

func cloneReflectStruct(value reflect.Value) reflect.Value {
	out := reflect.New(value.Type()).Elem()
	out.Set(value)
	for index := 0; index < value.NumField(); index++ {
		if out.Field(index).CanSet() && value.Field(index).CanInterface() {
			out.Field(index).Set(cloneReflectValue(value.Field(index)))
		}
	}
	return out
}

func cloneStringAnyMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	return cloneJSONValue(input).(map[string]any)
}

func cloneStringMap(input map[string]string) map[string]string {
	if input == nil {
		return nil
	}
	cloned := make(map[string]string, len(input))
	maps.Copy(cloned, input)
	return cloned
}

func cloneToolCalls(input []ToolCallSummary) []ToolCallSummary {
	if input == nil {
		return nil
	}
	cloned := append([]ToolCallSummary(nil), input...)
	for index := range cloned {
		cloned[index].ArgumentsValue = cloneStringAnyMap(cloned[index].ArgumentsValue)
	}
	return cloned
}
