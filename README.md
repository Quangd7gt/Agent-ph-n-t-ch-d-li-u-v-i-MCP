# Olist MCP Agent

Dự án này xây dựng một agent phân tích dữ liệu thương mại điện tử Olist bằng PostgreSQL, MCP Server, Antigravity và mô hình Gemma chạy cục bộ. Mục tiêu là để người dùng hỏi bằng tiếng Việt như:

```text
top sản phẩm năm 2018
doanh thu tháng thấp nhất năm 2017
giao hàng trễ ảnh hưởng thế nào đến đánh giá của khách hàng?
đơn e481f51cbdc54678b7cc49136f2d6af7 được thanh toán bằng phương thức nào?
```

Luồng xử lý chính:

```text
CSV trong data/
  -> load_olist.py nạp vào PostgreSQL schema raw
  -> sql/001_create_analytics_views.sql tạo schema analytics
  -> agent/agent.py phân tích dữ liệu và sinh nhận xét
  -> model.py gọi Gemma local nếu cần sinh diễn giải
  -> agent_api.py giữ Gemma chạy nền qua HTTP API
  -> server.py expose MCP tools cho Antigravity
```

## 1. Yêu cầu môi trường

Cần cài trước:

- Python 3.10 trở lên.
- PostgreSQL đang chạy local.
- Antigravity nếu muốn gọi qua MCP.
- NVIDIA GPU nếu muốn chạy Gemma bằng CUDA.
- Hugging Face token có quyền đọc model `google/gemma-2b-it`.

Gemma 2B có thể chạy CPU, nhưng rất chậm. Với GPU RTX 3050 Laptop 4GB, model thường chiếm khoảng 3.5-4GB VRAM khi đã được load.

## 2. Cấu trúc thư mục

```text
.
├── agent/
│   ├── agent.py               # Logic chính của OlistAgent
│   ├── system_prompt.txt      # System prompt cho agent phân tích dữ liệu
│   ├── config.py
│   └── templates/
├── data/                      # Các file CSV Olist
├── sql/
│   └── 001_create_analytics_views.sql
├── load_olist.py              # Nạp CSV vào PostgreSQL schema raw
├── server.py                  # MCP server cho Antigravity
├── agent_api.py               # FastAPI giữ agent/Gemma chạy nền
├── model.py                   # Load Hugging Face Gemma local
├── run_gemma_report.py        # Script chạy report trực tiếp
├── requirements.txt
├── .env.example
└── README.md
```

## 3. Tạo môi trường Python

Trong thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu muốn chạy GPU, PyTorch phải là bản CUDA. Kiểm tra bằng:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kết quả mong muốn:

```text
True
NVIDIA GeForce RTX 3050 Laptop GPU
```

Nếu `torch.cuda.is_available()` là `False`, dự án sẽ rơi về CPU dù trong `.env` đặt `GEMMA_DEVICE=cuda`.

## 4. Cấu hình `.env`

Tạo file `.env` từ mẫu:

```powershell
Copy-Item .env.example .env
```

Cấu hình tối thiểu:

```env
PGHOST=localhost
PGPORT=5432
PGDATABASE=olist_db
PGUSER=postgres
PGPASSWORD=your_postgres_password

RAW_SCHEMA=raw
DEFAULT_SCHEMA=analytics
MAX_RETURN_ROWS=200

GEMMA_MODEL=google/gemma-2b-it
GEMMA_DEVICE=cuda
GEMMA_MAX_NEW_TOKENS=120
HUGGINGFACE_TOKEN=your_huggingface_token

AGENT_API_URL=http://127.0.0.1:8000/ask
AGENT_API_TIMEOUT=600
AGENT_API_PRELOAD_GEMMA=true
```

Lưu ý:

- Không commit `.env` lên Git.
- Với Gemma, bạn cần vào Hugging Face, đăng nhập, chấp nhận điều khoản của model `google/gemma-2b-it`, sau đó tạo token loại read.
- Nếu máy thiếu VRAM hoặc muốn test nhanh, đổi `GEMMA_DEVICE=cpu`, nhưng tốc độ sẽ chậm hơn nhiều.

## 5. Chuẩn bị database PostgreSQL

Tạo database:

```powershell
createdb -h localhost -p 5432 -U postgres olist_db
```

Nếu `createdb` không có trong PATH, có thể tạo bằng pgAdmin hoặc psql:

```powershell
psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE olist_db;"
```

Nếu database đã tồn tại thì bỏ qua bước này.

## 6. Nạp dữ liệu CSV vào PostgreSQL

Đặt đầy đủ các file CSV trong thư mục `data/`:

```text
olist_customers_dataset.csv
olist_geolocation_dataset.csv
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
product_category_name_translation.csv
```

Chạy:

```powershell
python load_olist.py
```

Script này sẽ:

- Đọc các file CSV từ `DATA_DIR`, mặc định là `./data`.
- Tạo schema `raw` nếu chưa có.
- Tạo schema `analytics` nếu chưa có.
- Ghi dữ liệu vào các bảng raw:
  - `raw.orders`
  - `raw.order_items`
  - `raw.order_payments`
  - `raw.order_reviews`
  - `raw.products`
  - `raw.customers`
  - `raw.sellers`
  - `raw.category_translation`
  - `raw.geolocation`

Sau khi chạy thành công, terminal sẽ hiện:

```text
All Olist CSV files were loaded into Postgres raw schema.
Next step: run sql/001_create_analytics_views.sql
```

## 7. Tạo analytics views

Chạy file SQL:

```powershell
psql -h localhost -p 5432 -U postgres -d olist_db -f sql/001_create_analytics_views.sql
```

Các view analytics là lớp dữ liệu sạch để agent phân tích. Một số view quan trọng:

- `analytics.fct_orders`: mỗi dòng là một đơn hàng.
- `analytics.fct_order_items`: mỗi dòng là một sản phẩm trong đơn hàng.
- `analytics.order_payments_summary`: tổng hợp thanh toán theo đơn.
- `analytics.order_reviews_summary`: tổng hợp đánh giá theo đơn.

Kiểm tra nhanh:

```powershell
python -c "from server import ping; print(ping())"
```

Nếu thành công, kết quả có dạng:

```text
{'ok': True, 'database': 'olist_db', 'user': 'postgres', 'version': 'PostgreSQL ...'}
```

## 8. Business rules của dự án

Agent phải tuân thủ các quy tắc sau khi phân tích:

- Doanh thu gộp dùng `order_gross_value`.
- Chỉ tính doanh thu cho trạng thái đơn hợp lệ:
  - `delivered`
  - `shipped`
  - `invoiced`
  - `processing`
- Không cộng trực tiếp `payment_value_total` từ bảng order items vì sẽ bị lặp theo từng item.
- `analytics.fct_orders` có grain là một dòng cho một `order_id`.
- `analytics.fct_order_items` có grain là một dòng cho một `order_id` + `order_item_id`.
- Khách hàng quay lại là `customer_unique_id` có ít nhất hai đơn hàng khác nhau.
- Giao hàng trễ khi `order_delivered_customer_date > order_estimated_delivery_date`.

Tool `get_business_rules` dùng để trả về các quy tắc này cho Antigravity. Tool `business_rules_agent` dùng để hỏi đáp về business rules bằng ngôn ngữ tự nhiên.

## 9. Chạy Gemma Agent API

Nên chạy Gemma qua API riêng để model chỉ load một lần, tránh Antigravity gọi MCP rồi load lại model nhiều lần.

Mở terminal riêng:

```powershell
.\.venv\Scripts\Activate.ps1
python agent_api.py
```

Kiểm tra API:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/status
```

Kết quả mong muốn:

```json
{
  "ok": true,
  "model_loaded": true,
  "cuda_available": true,
  "resolved_device": "cuda"
}
```

Kiểm tra GPU:

```powershell
nvidia-smi
```

Nếu Gemma đã load lên GPU, bạn sẽ thấy một tiến trình `python.exe` chiếm VRAM. Với Gemma 2B, VRAM có thể gần 4GB. Trong Task Manager, nên xem GPU NVIDIA, không phải AMD Radeon tích hợp.

## 10. Chạy MCP server trực tiếp

MCP server chạy bằng stdio, thường do Antigravity tự khởi động. Để test nhanh các hàm Python:

```powershell
python -c "from server import ping; print(ping())"
python -c "from server import gemma_runtime_status; print(gemma_runtime_status())"
```

Nếu `ping` chạy được nhưng Antigravity vẫn treo, vấn đề thường nằm ở MCP session hoặc Antigravity agent session, không phải PostgreSQL.

## 11. Cấu hình MCP trong Antigravity

File cấu hình thường nằm ở:

```text
C:\Users\<your_user>\.gemini\antigravity\mcp_config.json
```

Ví dụ:

```json
{
  "mcpServers": {
    "olist-postgres-mcp": {
      "command": "D:\\Thuctap\\olist-mcp-agent\\olist-mcp-agent\\.venv\\Scripts\\python.exe",
      "args": [
        "server.py"
      ],
      "cwd": "D:\\Thuctap\\olist-mcp-agent\\olist-mcp-agent",
      "env": {
        "PGHOST": "localhost",
        "PGPORT": "5432",
        "PGDATABASE": "olist_db",
        "PGUSER": "postgres",
        "PGPASSWORD": "your_postgres_password",
        "DEFAULT_SCHEMA": "analytics",
        "MAX_RETURN_ROWS": "200",
        "AGENT_API_URL": "http://127.0.0.1:8000/ask",
        "AGENT_API_TIMEOUT": "600"
      }
    }
  }
}
```

Sau khi sửa config:

1. Mở Manage MCPs.
2. Bấm Refresh.
3. Bật server `olist-postgres-mcp`.
4. Test tool `ping`.
5. Test tool `gemma_agent`.

## 12. Nên bật những tool nào?

Chế độ khuyến nghị để demo agent:

- `ping`
- `gemma_agent`
- `gemma_runtime_status`
- `business_rules_agent`

Chế độ debug hoặc phát triển thêm:

- `get_business_rules`
- `get_system_prompt`
- `list_tables`
- `describe_table`
- `sample_rows`
- `get_schema_summary`
- `validate_query`
- `run_select_query`
- `revenue_by_month`
- `top_categories`
- `delivery_delay_summary`
- `repeat_customer_rate`

Không nên bật tất cả mọi tool khi demo nếu mục tiêu là chứng minh `agent.py` xử lý chính. Khi quá nhiều tool được bật, Antigravity/Gemini có thể tự chọn tool SQL trực tiếp, tự viết SQL, hoặc gọi nhiều tool phụ thay vì gọi `gemma_agent`.

## 13. Cách hỏi trong Antigravity

Nếu muốn chắc chắn câu hỏi đi qua `agent.py`, hãy hỏi rõ:

```text
Use only MCP tool olist-postgres-mcp/gemma_agent.
Pass exactly:
question="top sản phẩm năm 2018"
output_path=""
Do not rewrite my question. Do not create or edit files.
```

Hoặc ngắn hơn:

```text
Use MCP tool gemma_agent with question="top sản phẩm năm 2018" and output_path="".
```

Ví dụ câu hỏi nên test:

```text
top sản phẩm năm 2018
top danh mục sản phẩm năm 2017
doanh thu tháng thấp nhất năm 2017
doanh thu quý 1, quý 2, quý 3 năm 2018 khác nhau thế nào?
giao hàng trễ ảnh hưởng thế nào đến đánh giá của khách hàng?
bang nào là thị trường quan trọng nhất nếu xét cả doanh thu, số đơn và khách hàng?
nếu muốn cải thiện trải nghiệm khách hàng, nên ưu tiên danh mục hoặc khu vực nào?
đơn e481f51cbdc54678b7cc49136f2d6af7 được thanh toán bằng phương thức nào?
```

## 14. Quy tắc tạo file báo cáo

Mặc định `gemma_agent` trả lời trong chat và không tạo file.

Nếu muốn tạo report, câu hỏi phải nói rõ muốn tạo file hoặc báo cáo, ví dụ:

```text
Use MCP tool gemma_agent with question="tạo báo cáo HTML top sản phẩm năm 2018" and output_path="report_top_products_2018.html".
```

Nếu Antigravity tự truyền `output_path` nhưng câu hỏi không yêu cầu tạo file, `server.py` sẽ bỏ qua `output_path` để tránh tạo file ngoài ý muốn.

## 15. Vai trò của các file chính

`load_olist.py`

- Nạp dữ liệu từ CSV vào PostgreSQL.
- Chỉ xử lý tầng raw data.

`sql/001_create_analytics_views.sql`

- Tạo views phân tích trong schema `analytics`.
- Gom logic SQL nền tảng như order, item, payment, review.

`agent/agent.py`

- Là nơi chứa logic chính của agent.
- Nhận câu hỏi tự nhiên.
- Nhận diện intent.
- Chạy các hàm phân tích như:
  - `analyze_top_products`
  - `analyze_favorite_products`
  - `analyze_revenue_by_month`
  - `analyze_top_categories`
  - `analyze_monthly_revenue_extremes`
  - `analyze_quarterly_revenue`
  - `analyze_delivery_review_impact`
  - `analyze_state_market_importance`
  - `analyze_customer_experience_priorities`
  - `analyze_order_payment`
  - `analyze_order_shipping`
  - `analyze_order_products`
  - `analyze_order_sellers`
  - `analyze_order_review`
  - `analyze_order_customer`
  - `analyze_order_detail`
- Gọi Gemma để sinh diễn giải khi cần.

`model.py`

- Load tokenizer và model Gemma từ Hugging Face.
- Chọn CUDA nếu khả dụng.
- Sinh text từ prompt.

`agent_api.py`

- Chạy FastAPI server.
- Giữ một instance `OlistAgent` chạy nền.
- Giúp model Gemma không phải load lại mỗi lần Antigravity gọi MCP.

`server.py`

- Expose MCP tools cho Antigravity.
- Tool chính cho người dùng là `gemma_agent`.
- Các tool SQL trực tiếp dùng để debug hoặc hỗ trợ phân tích nhanh.

`agent/system_prompt.txt`

- Chứa vai trò, quy tắc, business rules và format trả lời của agent.
- `agent.py` đọc file này khi cần tạo prompt cho Gemma.

## 16. Cách thêm một phân tích mới

Ví dụ muốn thêm câu hỏi: "Danh mục nào có tỷ lệ giao hàng trễ cao nhất?"

Thêm trong `agent/agent.py`:

1. Viết hàm phân tích dữ liệu:

```python
def analyze_late_delivery_by_category(self, year: int | None = None, top_n: int = 10):
    ...
```

2. Viết hàm tóm tắt an toàn khi Gemma lỗi:

```python
def safe_late_delivery_by_category_analysis(self, result: dict[str, Any]) -> str:
    ...
```

3. Thêm intent trong `detect_intent`.

4. Thêm nhánh xử lý trong `answer_question`.

5. Nếu muốn Antigravity gọi riêng, thêm MCP tool trong `server.py`. Nếu chỉ muốn agent tổng xử lý, không cần thêm tool riêng.

## 17. Xử lý lỗi thường gặp

### `ping` lỗi kết nối database

Kiểm tra PostgreSQL đang chạy và `.env` đúng:

```powershell
python -c "from server import ping; print(ping())"
```

Nếu lỗi password hoặc database không tồn tại, sửa `PGDATABASE`, `PGUSER`, `PGPASSWORD`.

### Antigravity treo khi gọi `gemma_agent`

Kiểm tra Agent API trước:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/status
```

Nếu API không phản hồi, chạy lại:

```powershell
python agent_api.py
```

Sau đó Refresh MCP trong Antigravity.

### GPU không chạy

Kiểm tra:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

Nếu `cuda_available` là `False`, cần cài đúng PyTorch CUDA hoặc kiểm tra driver NVIDIA.

Nếu `cuda_available` là `True` nhưng GPU utilization thấp, vẫn có thể bình thường. Với LLM nhỏ, lúc sinh text GPU có thể tăng theo từng đợt ngắn. Chỉ số quan trọng hơn là `Dedicated GPU memory`: nếu model đã load, VRAM sẽ tăng rõ ràng.

### Gemma báo thiếu quyền Hugging Face

Lỗi thường gặp:

```text
Cannot load Gemma from Hugging Face.
```

Cách xử lý:

1. Đăng nhập Hugging Face.
2. Mở trang model `google/gemma-2b-it`.
3. Chấp nhận điều khoản sử dụng.
4. Tạo access token loại read.
5. Đặt `HUGGINGFACE_TOKEN=...` trong `.env`.
6. Khởi động lại `agent_api.py`.

### Antigravity tự sửa câu hỏi

Antigravity/Gemini là lớp điều phối tool. Nó có thể viết lại câu hỏi trước khi gọi MCP. Nếu muốn giữ nguyên câu hỏi, dùng prompt:

```text
Use only MCP tool olist-postgres-mcp/gemma_agent.
Pass exactly my original question as the question argument.
Do not rewrite, expand, summarize, or create files.
question="..."
output_path=""
```

### Antigravity tự tạo file

Trong prompt yêu cầu rõ:

```text
Do not create or edit files. Return the answer in chat only.
```

Ngoài ra, `server.py` đã có guard để bỏ qua `output_path` nếu câu hỏi không yêu cầu tạo file hoặc báo cáo.

## 18. Dừng Agent API

Nếu đang chạy trong terminal, bấm:

```text
Ctrl+C
```

Nếu bị kẹt cổng 8000:

```powershell
netstat -ano | Select-String ":8000"
Stop-Process -Id <PID> -Force
```

## 19. Quy trình chạy từ đầu

Tóm tắt toàn bộ quy trình:

```powershell
# 1. Tạo môi trường
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Cấu hình .env
Copy-Item .env.example .env

# 3. Tạo database
createdb -h localhost -p 5432 -U postgres olist_db

# 4. Nạp CSV vào schema raw
python load_olist.py

# 5. Tạo analytics views
psql -h localhost -p 5432 -U postgres -d olist_db -f sql/001_create_analytics_views.sql

# 6. Test database
python -c "from server import ping; print(ping())"

# 7. Chạy Agent API/Gemma
python agent_api.py

# 8. Kiểm tra API
Invoke-RestMethod http://127.0.0.1:8000/status

# 9. Refresh MCP trong Antigravity và gọi gemma_agent
```

Sau khi hoàn tất, câu hỏi người dùng trong Antigravity nên được chuyển qua tool `gemma_agent`, rồi `agent.py` sẽ phân tích dữ liệu trong PostgreSQL và dùng Gemma local để diễn giải khi cần.
