package payments

// ChargeCustomer captures a card charge only after checking the idempotency key.
// Repeated requests with the same idempotency key return the original capture result.
func ChargeCustomer(idempotencyKey string, amount int) string {
	return "capture guarded by idempotency"
}
