-- 1. Order counts by status
SELECT status, COUNT(*) AS order_records
FROM orders
GROUP BY status
ORDER BY order_records DESC;

-- 2. Total sales from completed order records
SELECT ROUND(SUM(order_amount), 2) AS total_sales_inr
FROM orders
WHERE status = 'completed';

-- 3. Top 10 products by completed-order quantity
SELECT p.product_name,
       SUM(o.quantity) AS units_sold
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.product_id, p.product_name
ORDER BY units_sold DESC
LIMIT 10;

-- 4. Sales by city
SELECT c.city,
       ROUND(SUM(o.order_amount), 2) AS sales_inr
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.city
ORDER BY sales_inr DESC;

-- 5. Average delivery time by city
SELECT c.city,
       ROUND(AVG(o.delivery_minutes), 2) AS avg_delivery_minutes
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
  AND o.delivery_minutes IS NOT NULL
GROUP BY c.city
ORDER BY avg_delivery_minutes;

-- 6. Customers with the most completed order records
SELECT c.customer_id,
       COUNT(*) AS completed_order_records,
       ROUND(SUM(o.order_amount), 2) AS total_spend_inr
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id
ORDER BY completed_order_records DESC
LIMIT 10;
