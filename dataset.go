package gepa

// ReflectiveRecord is one feedback item shown to the proposer for reflection.
type ReflectiveRecord map[string]any

// ReflectiveDataset groups reflection records by candidate component name.
type ReflectiveDataset map[string][]ReflectiveRecord
