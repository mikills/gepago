package crucible

import (
	"fmt"
	"io"
)

const maxHTTPResponseBytes = 1 << 20

func readHTTPResponse(body io.Reader) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(body, maxHTTPResponseBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxHTTPResponseBytes {
		return nil, fmt.Errorf("http response exceeds %d bytes", maxHTTPResponseBytes)
	}
	return data, nil
}
