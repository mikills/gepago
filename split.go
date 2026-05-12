package gepa

import (
	"errors"
	"math"
	"math/rand"
)

// ExampleSplit contains train and validation partitions for optimisation.
type ExampleSplit struct {
	Train      []Example
	Validation []Example
}

// SplitConfig controls validation size and optional deterministic shuffling.
type SplitConfig struct {
	ValidationSize  int
	ValidationRatio float64
	Seed            int64
	Shuffle         bool
}

// SplitExamples creates the feedback/validation split used by GEPA-style optimisation.
func SplitExamples(examples []Example, config SplitConfig) (ExampleSplit, error) {
	if len(examples) == 0 {
		return ExampleSplit{}, errors.New("examples are required")
	}
	validationSize := config.ValidationSize
	if validationSize == 0 && config.ValidationRatio > 0 {
		validationSize = int(math.Ceil(float64(len(examples)) * config.ValidationRatio))
	}
	if validationSize <= 0 || validationSize >= len(examples) {
		return ExampleSplit{}, errors.New("validation size must be between 1 and len(examples)-1")
	}
	items := append([]Example(nil), examples...)
	if config.Shuffle {
		seed := config.Seed
		if seed == 0 {
			seed = 1
		}
		rand.New(rand.NewSource(seed)).Shuffle(len(items), func(i int, j int) {
			items[i], items[j] = items[j], items[i]
		})
	}
	return ExampleSplit{
		Train:      append([]Example(nil), items[validationSize:]...),
		Validation: append([]Example(nil), items[:validationSize]...),
	}, nil
}
