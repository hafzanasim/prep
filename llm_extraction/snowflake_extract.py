import snowflake.connector
import os

def fetch_reports_from_snowflake(empi_id):
    conn = snowflake.connector.connect(
        user=os.getenv("HAFZANASIM"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("YYB34419"),
        warehouse="COMPUTE_WH",
        database="RADIOLOGYPREP",
        schema="INFORMATION_SCHEMA"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT report_text, timestamp FROM clinical_reports WHERE empi_id = %s", (empi_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows
