# Olist MCP Agent Starter

Project tối thiểu để chạy MCP server cho bộ dữ liệu Olist trong VS Code.

## 1) Chuẩn bị

- Python 3.10+
- PostgreSQL 14+ hoặc mới hơn
- Bộ CSV Olist đặt trong thư mục `data/`

Các file CSV cần có:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_products_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_sellers_dataset.csv`
- `product_category_name_translation.csv`
- `olist_geolocation_dataset.csv`

## 2) Cài môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## 3) Tạo database và user

Trong `psql`:

```sql
CREATE DATABASE olist_db;
CREATE USER toolbox_user WITH PASSWORD 'change_me';
GRANT CONNECT ON DATABASE olist_db TO toolbox_user;
```

## 4) Load dữ liệu thô

Đặt các CSV vào thư mục `data/`, sau đó chạy:

```bash
python load_olist.py
```

Script sẽ tạo schema `raw` và load toàn bộ CSV vào đó.

## 5) Tạo analytics views

```bash
psql -h localhost -U postgres -d olist_db -f sql/001_create_analytics_views.sql
```

Sau đó cấp quyền đọc cho MCP user:

```sql
GRANT USAGE ON SCHEMA analytics TO toolbox_user;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO toolbox_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT SELECT ON TABLES TO toolbox_user;
```

## 6) Chạy MCP server

```bash
python server.py
```

Server chạy bằng `stdio` để các MCP client như Antigravity gọi được.

## 7) Các tool chính

- `ping()`
- `list_tables()`
- `describe_table(table_name)`
- `sample_rows(table_name, limit)`
- `get_schema_summary()`
- `validate_query(query)`
- `run_select_query(query, row_limit)`
- `revenue_by_month(year)`
- `top_categories(start_date, end_date, limit)`
- `delivery_delay_summary(start_date, end_date)`
- `repeat_customer_rate(start_date, end_date)`

## 8) Lưu ý về grain

- `analytics.fct_orders`: 1 dòng trên mỗi `order_id`
- `analytics.fct_order_items`: 1 dòng trên mỗi `order_id + order_item_id`

Dùng `fct_orders` cho revenue/order metrics ở cấp đơn hàng.
Dùng `fct_order_items` cho category/product/seller analysis.

## 9) Nối với Antigravity

Sửa file `antigravity_mcp_config.json`:
- thay `cwd` bằng đường dẫn tuyệt đối đến thư mục project
- chỉnh `env` cho khớp với máy của bạn

Sau đó import config này vào phần MCP settings của Antigravity hoặc copy các trường tương đương vào cấu hình MCP của nó.
