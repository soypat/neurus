package neurus

import (
	"math/rand"
	"testing"
)

func TestNetworkOptimized_small(t *testing.T) {

	nn := NewNetworkOptimized([]int{1, 1},
		func() ActivationFunc { return new(Sigmd) },
		&MeanSquaredError{}, rand.NewSource(1))

}
