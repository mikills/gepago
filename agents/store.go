package agents

import (
	"encoding/json"
	"maps"
	"sync"
)

const StoreShrinkThreshold = 200

type Store[K comparable, T any] struct {
	data    map[K]T
	mu      sync.RWMutex
	deleted int64
}

func NewStore[K comparable, T any](data map[K]T) *Store[K, T] {
	s := &Store[K, T]{}
	s.Reset(data)
	return s
}

func (s *Store[K, T]) Reset(newData map[K]T) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(newData) > 0 {
		s.data = make(map[K]T, len(newData))
		maps.Copy(s.data, newData)
	} else {
		s.data = make(map[K]T)
	}
	s.deleted = 0
}

func (s *Store[K, T]) Length() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.data)
}

func (s *Store[K, T]) RemoveAll() {
	s.Reset(nil)
}

func (s *Store[K, T]) Remove(key K) {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.data, key)
	s.deleted++
	if s.deleted >= StoreShrinkThreshold {
		newData := make(map[K]T, len(s.data))
		maps.Copy(newData, s.data)
		s.data = newData
		s.deleted = 0
	}
}

func (s *Store[K, T]) Has(key K) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.data[key]
	return ok
}

func (s *Store[K, T]) Get(key K) T {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.data[key]
}

func (s *Store[K, T]) GetOk(key K) (T, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.data[key]
	return v, ok
}

func (s *Store[K, T]) GetAll() map[K]T {
	s.mu.RLock()
	defer s.mu.RUnlock()

	clone := make(map[K]T, len(s.data))
	maps.Copy(clone, s.data)
	return clone
}

func (s *Store[K, T]) Values() []T {
	s.mu.RLock()
	defer s.mu.RUnlock()

	values := make([]T, 0, len(s.data))
	for _, v := range s.data {
		values = append(values, v)
	}
	return values
}

func (s *Store[K, T]) Set(key K, value T) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.data == nil {
		s.data = make(map[K]T)
	}
	s.data[key] = value
}

func (s *Store[K, T]) SetFunc(key K, fn func(old T) T) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.data == nil {
		s.data = make(map[K]T)
	}
	s.data[key] = fn(s.data[key])
}

func (s *Store[K, T]) GetOrSet(key K, setFunc func() T) T {
	s.mu.RLock()
	v, ok := s.data[key]
	s.mu.RUnlock()
	if ok {
		return v
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	v, ok = s.data[key]
	if !ok {
		v = setFunc()
		if s.data == nil {
			s.data = make(map[K]T)
		}
		s.data[key] = v
	}
	return v
}

func (s *Store[K, T]) SetIfLessThanLimit(key K, value T, maxAllowedElements int) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.data == nil {
		s.data = make(map[K]T)
	}
	_, ok := s.data[key]
	if !ok && len(s.data) >= maxAllowedElements {
		return false
	}
	s.data[key] = value
	return true
}

func (s *Store[K, T]) UnmarshalJSON(data []byte) error {
	raw := map[K]T{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.data == nil {
		s.data = make(map[K]T)
	}
	maps.Copy(s.data, raw)
	return nil
}

func (s *Store[K, T]) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.GetAll())
}
