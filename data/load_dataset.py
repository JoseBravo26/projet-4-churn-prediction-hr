import pandas as pd
import psycopg2

DB_NAME = "churn_predictor_db"
DB_USER = "churn_app"
DB_PASSWORD = "secure_password_123"   # luego lo pasas a .env
DB_HOST = "localhost"
DB_PORT = "5432"

CSV_PATH = "dataset_final.csv"   # asegúrate de que el archivo está en la misma carpeta

def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )

def main():
    # 1. Leer CSV
    df = pd.read_csv(CSV_PATH)

    # 2. Mapear columnas del CSV a columnas de employees
    df_mapped = pd.DataFrame({
        "age": df["age"],
        "genre": df["genre"].map({0: "M", 1: "F"}),   # ajusta si tu codificación es distinta
        "salaire": df["revenu_mensuel"],
        "anciennete": df["annees_dans_l_entreprise"],
        "satisfaction": df["satisfaccion_media"],
        "turnover": df["a_quitte_l_entreprise"].astype(bool),
    })

    # 3. Insertar en PostgreSQL
    conn = get_connection()
    cur = conn.cursor()

    insert_query = """
        INSERT INTO employees (age, genre, salaire, anciennete, satisfaction, turnover)
        VALUES (%s, %s, %s, %s, %s, %s);
    """

    count = 0
    for row in df_mapped.itertuples(index=False):
        cur.execute(
            insert_query,
            (
                int(row.age),
                row.genre,
                float(row.salaire),
                float(row.anciennete),
                float(row.satisfaction),
                bool(row.turnover),
            ),
        )
        count += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"Insertadas {count} filas desde dataset_final.csv en employees.")

if __name__ == "__main__":
    main()
