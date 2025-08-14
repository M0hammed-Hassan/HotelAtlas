import sqlite3
import pandas as pd

df = pd.read_excel("datasets/segmented_customers_dataset.xlsx")

conn = sqlite3.connect("customer_segments.db")
cursor = conn.cursor()

# Save to table
df.to_sql("hotel_bookings", conn, if_exists="replace", index=False)


# Query1: Basic aggregation
query1 = """
SELECT 
    MarketSegment_Direct,
    COUNT(*) AS total_bookings,
    AVG(LodgingRevenue) AS avg_lodging_revenue
FROM hotel_bookings
GROUP BY MarketSegment_Direct;
"""
print("\n--- Basic Aggregations ---")
print(pd.read_sql(query1, conn))

# Query2: Top 5 by LodgingRevenue
query2 = """
SELECT 
    rowid AS customer_id,
    LodgingRevenue
FROM hotel_bookings
ORDER BY LodgingRevenue DESC
LIMIT 5;
"""
print("\n--- Top 5 Customers by LodgingRevenue ---")
print(pd.read_sql(query2, conn))

# Query3: Window function: ranking within MarketSegment_Direct
query3 = """
SELECT 
    rowid AS customer_id,
    MarketSegment_Direct,
    LodgingRevenue,
    RANK() OVER (PARTITION BY MarketSegment_Direct ORDER BY LodgingRevenue DESC) AS rank_in_segment
FROM hotel_bookings;
"""
print("\n--- Ranking Within MarketSegment_Direct ---")
print(pd.read_sql(query3, conn).head(10))

# Query4: Running total of OtherRevenue
query4 = """
SELECT 
    rowid,
    DaysSinceCreation,
    OtherRevenue,
    SUM(OtherRevenue) OVER (ORDER BY DaysSinceCreation) AS running_total_other_revenue
FROM hotel_bookings;
"""
print("\n--- Running Total of OtherRevenue ---")
print(pd.read_sql(query4, conn).head(10))

# Query5: Multi-table join: create lookup table and join
cursor.execute("DROP TABLE IF EXISTS market_segment_lookup;")
cursor.execute("""
CREATE TABLE market_segment_lookup (
    segment_flag INTEGER,
    segment_name TEXT
);
""")
cursor.executemany(
    """
INSERT INTO market_segment_lookup (segment_flag, segment_name) VALUES (?, ?);
""",
    [(0, "Not Direct"), (1, "Direct Booking")],
)

query5 = """
SELECT 
    hb.rowid,
    hb.LodgingRevenue,
    hb.MarketSegment_Direct,
    msl.segment_name
FROM hotel_bookings hb
JOIN market_segment_lookup msl
ON hb.MarketSegment_Direct = msl.segment_flag;
"""
print("\n--- Join with Lookup Table ---")
print(pd.read_sql(query5, conn).head(10))

# Query6: Optimization with indexes
cursor.execute(
    "CREATE INDEX IF NOT EXISTS idx_market_segment_direct ON hotel_bookings(MarketSegment_Direct);"
)
cursor.execute(
    "CREATE INDEX IF NOT EXISTS idx_days_since_creation ON hotel_bookings(DaysSinceCreation);"
)

print("\n✅ Indexes created.")

# Check query plan before/after index
query_plan = (
    "EXPLAIN QUERY PLAN SELECT * FROM hotel_bookings WHERE MarketSegment_Direct = 1;"
)
print("\n--- Query Plan ---")
for row in cursor.execute(query_plan):
    print(row)

conn.close()
