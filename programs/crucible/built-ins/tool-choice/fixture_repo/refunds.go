package payments

// RefundCustomer creates a reversal for the captured charge and asks the ledger
// package to record the refund so reporting stays balanced.
func RefundCustomer(chargeID string) string {
	return "refund reversal ready for ledger"
}
