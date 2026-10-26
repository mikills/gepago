package gepa

// metadataValues preserves key names and omits their values before any payload copy.
func metadataValues(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	out := make(map[string]any, len(values))
	for key := range values {
		out[key] = nil
	}
	return out
}

func metadataRunContext(rc RunContext) RunContext {
	if rc.Attributes != nil {
		attributes := make(map[string]string, len(rc.Attributes))
		for name := range rc.Attributes {
			attributes[name] = ""
		}
		rc.Attributes = attributes
	}
	return rc
}

// metadataToolCalls copies tool metadata without reading or copying its payload fields.
func metadataToolCalls(records []ToolCallRecord) []ToolCallRecord {
	if records == nil {
		return nil
	}
	metadata := make([]ToolCallRecord, len(records))
	for i, record := range records {
		metadata[i] = ToolCallRecord{
			Step: record.Step, Name: record.Name,
			Usage: record.Usage, UnbudgetedUsage: record.UnbudgetedUsage,
			Duration: record.Duration, Cached: record.Cached,
		}
		if record.Err != "" {
			metadata[i].Err = RedactedText
		}
	}
	return metadata
}
