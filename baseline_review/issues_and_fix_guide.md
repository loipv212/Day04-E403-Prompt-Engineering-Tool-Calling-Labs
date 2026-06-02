# Baseline Issues And Fix Guide

Baseline command:

```powershell
python grade/scoring.py --module simple_solution.agent.graph --provider mimo
```

Baseline result:

```text
overall_score: 49.62 / 100
total_earned: 645.0 / 1300.0
```

Full output is saved in:

```text
baseline_review/baseline_simple_solution_mimo_result.json
```

## Main Problems From Baseline Output

## 1. Customer fields are empty in saved orders

Common feedback:

```text
root.customer.name: expected ..., got ''
root.customer.phone: expected ..., got ''
root.customer.email: expected ..., got ''
root.customer.shipping_address: expected ..., got ''
```

Impact:

- Saved JSON does not match expected output.
- `order_id` becomes wrong because order ID depends on customer email and phone.
- `save_path` becomes wrong because it depends on `order_id`.

Where to fix:

- `src/agent/graph.py`
- `src/utils/data_store.py`

Fix direction:

- Prompt must require complete customer fields before any save.
- `save_order` tool schema must require explicit customer fields.
- Agent must pass exact user-provided `customer_name`, `customer_phone`, `customer_email`, and `shipping_address` to `save_order`.

## 2. Order IDs are wrong

Common feedback:

```text
root.order_id: expected 'ORD-...', got 'ORD-...'
```

Impact:

- Causes JSON mismatch.
- Causes save path mismatch.

Likely causes:

- Empty or wrong customer email.
- Empty or wrong phone.
- Wrong item quantity or item list.

Where to fix:

- `src/utils/data_store.py`

Fix direction:

- In `save_order`, build deterministic order ID from normalized customer email, normalized phone digits, and sorted item list.
- Match the behavior used by expected fixtures.

## 3. Discount is sometimes wrong

Common feedback:

```text
root.pricing.discount_rate: expected 0.2, got 0.1
root.discount.campaign_code: expected 'FLASH-20', got 'FLASH-10'
```

Impact:

- Pricing fields become wrong.
- Final total becomes wrong.

Likely cause:

- Agent calls `get_discount` with the wrong seed.

Where to fix:

- `src/utils/data_store.py`
- `src/agent/graph.py`

Fix direction:

- `get_discount` should be deterministic.
- Use `customer_email` as `seed_hint`.
- If email is unavailable, fallback to phone, but normal order cases should have email.

## 4. Quantities are lost or defaulted to 1

Common feedback:

```text
root.items[0].quantity: expected 2, got 1
root.items[1].quantity: expected 3, got 1
```

Impact:

- Line totals are wrong.
- Subtotal and final total are wrong.
- Order ID can become wrong.

Where to fix:

- `src/agent/graph.py`

Fix direction:

- Tool schema should use structured items:

```text
[{ "product_id": "...", "quantity": n }]
```

- Prompt must tell the agent to preserve user-requested quantities exactly.

## 5. Agent calls tools when required info is missing

Common feedback:

```text
Tool trace mismatch. Expected subsequence [], got ['list_products', ...]
```

Impact:

- Clarification cases lose tool points.
- Agent may calculate or save too early.

Where to fix:

- `src/agent/graph.py`

Fix direction:

- System prompt must say:

```text
If customer name, phone, email, shipping address, item, or quantity is missing:
ask for the missing fields and stop.
Do not call any tool.
```

## 6. Agent saved an order when email was missing

Case:

```text
clarification_missing_email_only
```

Common feedback:

```text
Order should not have been saved for this case.
Did not ask for the missing email.
Proceeded to create order without clarification.
```

Impact:

- Very large score loss.

Where to fix:

- `src/agent/graph.py`

Fix direction:

- Missing email must be treated as a hard stop.
- The final answer should ask only for the missing email.
- No tool calls should happen.

## 7. Tool order is repeated or noisy

Common feedback:

```text
Expected subsequence ['list_products', 'get_product_details', 'get_discount', 'calculate_order_totals', 'save_order'],
got ['list_products', 'list_products', ...]
```

Impact:

- Tool score can drop.
- Agent behavior is harder to control.

Where to fix:

- `src/agent/graph.py`

Fix direction:

- Prompt should require this order for valid orders:

```text
list_products -> get_product_details -> get_discount -> calculate_order_totals -> save_order
```

- Prompt should avoid repeated calls once enough product IDs/details are available.

## 8. Save path format uses Windows backslashes

Common feedback:

```text
expected 'artifacts/orders/ORD-xxx.json'
got 'artifacts\\orders\\ORD-xxx.json'
```

Impact:

- JSON mismatch on Windows.

Where to fix:

- `src/utils/data_store.py`

Fix direction:

- Use POSIX style for the `save_path` field:

```python
relative_path.as_posix()
```

## Recommended Fix Order

## Step 1. Implement `src/utils/data_store.py`

Implement:

- `__init__`
- `list_products`
- `get_product_details`
- `get_discount`
- `calculate_order_totals`
- `save_order`

Use `simple_solution/utils/data_store.py` as a reference, but make sure `save_path` uses `/`.

## Step 2. Test datastore without LLM

First verify:

- Product search works.
- Detail token works.
- Discount is deterministic.
- Totals are correct.
- Stock failures return error payloads.
- Save order matches expected fixture.

## Step 3. Implement tools in `src/agent/graph.py`

Implement exactly five tools:

```text
list_products
get_product_details
get_discount
calculate_order_totals
save_order
```

Use the Pydantic schemas from `src/core/schemas.py`.

## Step 4. Implement agent runner helpers

Implement:

```text
build_agent()
run_agent()
extract_final_answer()
extract_tool_calls()
extract_saved_order()
```

## Step 5. Write strong system prompt

Prompt must enforce:

- Vietnamese final answer.
- No hallucinated catalog facts, discounts, totals, or paths.
- Clarify before tool use when required fields are missing.
- Refuse fake invoices, manual discounts, stock bypass, or ignoring catalog/policy.
- Follow required tool order for valid orders.
- Save only after validation succeeds.

## Step 6. Run grader for `src`

```powershell
python grade/scoring.py --module src.agent.graph --provider mimo
```

Target:

```text
80+ minimum
90+ strong
```

