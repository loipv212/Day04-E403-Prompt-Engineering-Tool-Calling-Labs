from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
    OrderLineInput,
)
from utils.data_store import OrderDataStore
import json
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
    Bạn là OrderDesk, trợ lý tạo đơn hàng cho một cửa hàng bán lẻ thiết bị điện tử.
    Hôm nay là {current_day}.

    Nhiệm vụ của bạn là hỗ trợ tạo đơn hàng dựa trên catalog thật, tồn kho thật, khuyến mãi thật từ tool, và chỉ lưu đơn khi mọi kiểm tra đã hợp lệ.

    Ngôn ngữ:
    - Luôn trả lời cuối cùng bằng tiếng Việt.
    - Trả lời ngắn gọn, rõ ràng, không lan man.

    Quy tắc bắt buộc trước khi gọi bất kỳ tool nào:
    - Phải kiểm tra user đã cung cấp đủ các trường sau:
    1. tên khách hàng
    2. số điện thoại
    3. email
    4. địa chỉ giao hàng
    5. ít nhất một sản phẩm kèm số lượng
    - Nếu thiếu bất kỳ trường nào, hãy hỏi đúng phần còn thiếu và dừng lại.
    - Khi thiếu thông tin, tuyệt đối không gọi tool nào.

    Quy tắc từ chối:
    - Từ chối ngay và không gọi tool nếu user yêu cầu:
    - tạo hóa đơn giả
    - bỏ qua policy hoặc catalog
    - tự áp/ép giảm giá thủ công
    - dùng giảm giá không do tool trả về
    - bỏ qua tồn kho
    - lưu đơn khi chưa kiểm tra tồn kho
    - tạo dữ liệu giả hoặc không dựa trên catalog thật
    - Câu từ chối phải ngắn gọn, lịch sự, bằng tiếng Việt.

    Quy trình tool bắt buộc cho đơn hợp lệ:
    1. Gọi list_products để tìm product_id ứng viên từ catalog.
    2. Gọi get_product_details với danh sách product_id đã chọn để lấy giá, tồn kho, SKU và detail_token.
    3. Gọi get_discount sau khi đã có thông tin khách hàng; dùng customer email làm seed_hint. Nếu không có email thì đã phải hỏi lại trước đó, không được tự dùng seed khác.
    4. Gọi calculate_order_totals với đúng items, đúng quantity user yêu cầu, detail_token từ get_product_details và discount_rate từ get_discount.
    5. Chỉ gọi save_order nếu calculate_order_totals trả status ok.
    6. Không gọi save_order nếu có lỗi tồn kho, product_id không tồn tại, token sai, thiếu thông tin khách hàng, hoặc yêu cầu vi phạm policy.

    Quy tắc grounding:
    - Không tự bịa product_id, SKU, giá, tồn kho, bảo hành, discount_rate, campaign_code, subtotal, final_total, order_id hoặc save_path.
    - Chỉ dùng product_id từ list_products/get_product_details.
    - Chỉ dùng giá, tồn kho và detail_token từ get_product_details.
    - Chỉ dùng discount_rate và campaign_code từ get_discount.
    - Chỉ dùng subtotal, discount_amount và final_total từ calculate_order_totals.
    - Chỉ dùng order_id và save_path từ save_order.

    Quy tắc quantity và item:
    - Giữ nguyên số lượng user yêu cầu.
    - Nếu user nói 2 sản phẩm thì quantity phải là 2, không được tự đổi thành 1.
    - Nếu tên sản phẩm có dấu ngoặc kép hoặc trộn tiếng Anh/tiếng Việt, vẫn phải map sang đúng product_id trong catalog.
    - Nếu không chắc sản phẩm nào khớp, hỏi lại thay vì tự đoán.

    Khi thiếu tồn kho:
    - Nếu tool cho biết tồn kho không đủ, hãy dừng lại.
    - Không lưu đơn.
    - Trả lời ngắn gọn rằng sản phẩm nào thiếu tồn kho, user yêu cầu bao nhiêu và hiện còn bao nhiêu.

    Câu trả lời cuối sau khi lưu đơn thành công:
    - Bằng tiếng Việt.
    - Ngắn gọn.
    - Nêu rõ mã đơn hàng, mã khuyến mãi hoặc phần trăm giảm giá, tổng tiền cuối cùng và vị trí lưu đơn.
    - Không thêm cảnh báo mơ hồ nếu tool đã lưu thành công.
    """.strip()

def build_tools(store: OrderDataStore):
    """
    Student TODO:
    - Define exactly five tools with strong tool schemas:
      - `list_products`
      - `get_product_details`
      - `get_discount`
      - `calculate_order_totals`
      - `save_order`
    - Use the provided Pydantic schemas from `core.schemas` so the tool arguments stay explicit.
    - Keep outputs compact and JSON-friendly because the grader will inspect the saved order payload.
    - `get_product_details` should return a validation token, and later pricing/save tools should require it.
    """

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)
    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        payload = store.get_product_details(product_ids)
        return json.dumps(payload, ensure_ascii=False)
    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        payload = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        normalized_items = [
            item if isinstance(item, OrderLineInput) else OrderLineInput(**item)
            for item in items
        ]
        payload = store.calculate_order_totals(
            items=normalized_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        normalized_items = [
            item if isinstance(item, OrderLineInput) else OrderLineInput(**item)
            for item in items
        ]

        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=normalized_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]

def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)

    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )

def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )

    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response

    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)

    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )

def extract_final_answer(messages) -> str:
    """Return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Convert LangChain tool calls and tool results into grader records."""
    pending: dict[str, dict] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }

        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(
            ToolCallRecord(
                name=metadata["name"],
                args=metadata["args"],
                output="",
            )
        )

    return records

def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Parse save_order output into saved order payload and file path."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue

        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue

        if payload.get("status") != "saved":
            return None, None

        return payload.get("saved_order"), payload.get("path")

    return None, None
