import psycopg2

conn = psycopg2.connect(
    dbname="face_db",
    user="postgres",
    password="325600",
    host="localhost",
    port="5432"
)

def insert_user(name, embedding):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO employees (name, embedding) VALUES (%s, %s)",
        (name, embedding)
    )
    conn.commit()
    cur.close()

def get_all_users():
    cur = conn.cursor()
    cur.execute("SELECT name, embedding FROM employees")
    rows = cur.fetchall()
    cur.close()
    return rows