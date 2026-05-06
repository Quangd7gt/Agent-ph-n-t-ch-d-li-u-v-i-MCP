CREATE SCHEMA IF NOT EXISTS analytics;

CREATE OR REPLACE VIEW analytics.dim_products AS
SELECT
    p.product_id,
    p.product_category_name,
    ct.product_category_name_english,
    p.product_name_lenght,
    p.product_description_lenght,
    p.product_photos_qty,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm
FROM raw.products p
LEFT JOIN raw.category_translation ct
    ON ct.product_category_name = p.product_category_name;

CREATE OR REPLACE VIEW analytics.dim_customers AS
SELECT
    customer_id,
    customer_unique_id,
    customer_zip_code_prefix,
    customer_city,
    customer_state
FROM raw.customers;

CREATE OR REPLACE VIEW analytics.dim_sellers AS
SELECT
    seller_id,
    seller_zip_code_prefix,
    seller_city,
    seller_state
FROM raw.sellers;

CREATE OR REPLACE VIEW analytics.order_payments_summary AS
SELECT
    order_id,
    SUM(payment_value) AS payment_value_total,
    COUNT(*) AS payment_row_count,
    STRING_AGG(DISTINCT payment_type, ', ' ORDER BY payment_type) AS payment_types,
    MAX(payment_installments) AS max_installments
FROM raw.order_payments
GROUP BY order_id;

CREATE OR REPLACE VIEW analytics.order_reviews_summary AS
SELECT
    order_id,
    AVG(review_score)::numeric(10,2) AS review_score_avg,
    COUNT(*) AS review_count
FROM raw.order_reviews
GROUP BY order_id;

CREATE OR REPLACE VIEW analytics.fct_order_items AS
SELECT
    oi.order_id,
    oi.order_item_id,
    o.customer_id,
    c.customer_unique_id,
    oi.product_id,
    oi.seller_id,
    dp.product_category_name,
    dp.product_category_name_english,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    oi.shipping_limit_date,
    oi.price,
    oi.freight_value,
    (oi.price + oi.freight_value) AS line_total,
    ors.review_score_avg,
    ors.review_count,
    dc.customer_city,
    dc.customer_state,
    ds.seller_city,
    ds.seller_state
FROM raw.order_items oi
JOIN raw.orders o
    ON o.order_id = oi.order_id
LEFT JOIN analytics.dim_customers dc
    ON dc.customer_id = o.customer_id
LEFT JOIN raw.customers c
    ON c.customer_id = o.customer_id
LEFT JOIN analytics.dim_sellers ds
    ON ds.seller_id = oi.seller_id
LEFT JOIN analytics.dim_products dp
    ON dp.product_id = oi.product_id
LEFT JOIN analytics.order_reviews_summary ors
    ON ors.order_id = oi.order_id;

CREATE OR REPLACE VIEW analytics.fct_orders AS
SELECT
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    COUNT(DISTINCT oi.order_item_id) AS item_count,
    COUNT(DISTINCT oi.product_id) AS distinct_product_count,
    COUNT(DISTINCT oi.seller_id) AS distinct_seller_count,
    COALESCE(SUM(oi.price), 0)::numeric(14,2) AS item_price_total,
    COALESCE(SUM(oi.freight_value), 0)::numeric(14,2) AS freight_total,
    COALESCE(SUM(oi.price + oi.freight_value), 0)::numeric(14,2) AS order_gross_value,
    ops.payment_value_total,
    ops.payment_row_count,
    ops.payment_types,
    ops.max_installments,
    ors.review_score_avg,
    ors.review_count,
    dc.customer_city,
    dc.customer_state,
    CASE
        WHEN o.order_delivered_customer_date IS NULL THEN NULL
        WHEN o.order_estimated_delivery_date IS NULL THEN NULL
        WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN TRUE
        ELSE FALSE
    END AS is_delayed_delivery,
    CASE
        WHEN o.order_delivered_customer_date IS NULL THEN NULL
        WHEN o.order_purchase_timestamp IS NULL THEN NULL
        ELSE EXTRACT(EPOCH FROM (o.order_delivered_customer_date - o.order_purchase_timestamp)) / 86400.0
    END AS delivery_days
FROM raw.orders o
LEFT JOIN raw.order_items oi
    ON oi.order_id = o.order_id
LEFT JOIN analytics.order_payments_summary ops
    ON ops.order_id = o.order_id
LEFT JOIN analytics.order_reviews_summary ors
    ON ors.order_id = o.order_id
LEFT JOIN analytics.dim_customers dc
    ON dc.customer_id = o.customer_id
LEFT JOIN raw.customers c
    ON c.customer_id = o.customer_id
GROUP BY
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    ops.payment_value_total,
    ops.payment_row_count,
    ops.payment_types,
    ops.max_installments,
    ors.review_score_avg,
    ors.review_count,
    dc.customer_city,
    dc.customer_state;
