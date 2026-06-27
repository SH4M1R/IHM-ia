"""
Recomendador de productos - Tienda Mass
========================================
Producción: Render Free
- Los CSVs se descargan automáticamente desde Spring Boot al llamar /reentrenar
- Los modelos se guardan en /tmp (efímeros pero suficientes para Free tier)
- Spring Boot URL se configura via variable de entorno SPRINGBOOT_URL
"""

import pandas as pd
import numpy as np
import joblib
import os
import requests
import io
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import implicit
from flask import Flask, request, jsonify
from flask_cors import CORS

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
# En Render: configura esta variable de entorno con la URL de tu Spring Boot
# Ej: https://mass-backend.onrender.com
SPRINGBOOT_URL = os.environ.get("SPRINGBOOT_URL", "http://localhost:8080")

# En Render Free el filesystem es efímero → usamos /tmp para los modelos
MODELS_DIR     = os.environ.get("MODELS_DIR", "/tmp/modelos")

MIN_SUPPORT    = 0.05
MIN_CONFIDENCE = 0.3
MIN_LIFT       = 1.2
ALS_FACTORS    = 50
ALS_ITERATIONS = 20
TOP_N          = 5

os.makedirs(MODELS_DIR, exist_ok=True)

# Estado global de los modelos (en memoria mientras el proceso vive)
_reglas                 = None
_artefactos_colaborativo = None


# ══════════════════════════════════════════════
# 0. DESCARGA DE CSVs DESDE SPRING BOOT
# ══════════════════════════════════════════════
def descargar_csv_desde_springboot(endpoint: str) -> pd.DataFrame:
    """
    Descarga un CSV desde el endpoint de Spring Boot y lo retorna como DataFrame.
    endpoint: p.ej. "/api/exportar/market-basket"
    """
    url = f"{SPRINGBOOT_URL}{endpoint}"
    print(f"  → Descargando {url} ...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        print(f"  ✓ {len(df)} filas descargadas desde {endpoint}")
        return df
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"No se pudo descargar {url}: {e}")


# ══════════════════════════════════════════════
# 1. MARKET BASKET — Productos que van juntos
# ══════════════════════════════════════════════
def entrenar_market_basket(df: pd.DataFrame):
    print("\n[1/2] Entrenando Market Basket (Apriori)...")

    df["nombreProducto"] = df["nombreProducto"].str.strip().str.lower()

    mapa = {}
    for _, row in df[["nombreProducto", "idProducto", "precio"]].drop_duplicates("nombreProducto").iterrows():
        mapa[row["nombreProducto"]] = {
            "idProducto": int(row["idProducto"]),
            "precio": float(row["precio"]),
        }
    joblib.dump(mapa, f"{MODELS_DIR}/mapa_productos.pkl")
    print(f"  ✓ Mapa de {len(mapa)} productos guardado.")

    canastas = df.groupby("idVenta")["nombreProducto"].apply(list).tolist()

    te = TransactionEncoder()
    te_array = te.fit(canastas).transform(canastas)
    df_basket = pd.DataFrame(te_array, columns=te.columns_)

    frecuentes = apriori(df_basket, min_support=MIN_SUPPORT, use_colnames=True)
    if frecuentes.empty:
        print("  ⚠ Pocos datos para Apriori. Reduciendo min_support a 0.01...")
        frecuentes = apriori(df_basket, min_support=0.01, use_colnames=True)

    reglas = association_rules(frecuentes, metric="lift", min_threshold=MIN_LIFT)
    reglas = reglas[reglas["confidence"] >= MIN_CONFIDENCE]
    reglas = reglas.sort_values("lift", ascending=False)

    reglas.to_pickle(f"{MODELS_DIR}/reglas_basket.pkl")
    print(f"  ✓ {len(reglas)} reglas encontradas y guardadas.")
    return reglas


def recomendar_productos_juntos(productos_en_carrito: list, reglas: pd.DataFrame, top_n=TOP_N) -> list:
    if reglas is None or reglas.empty:
        return []

    mapa_path = f"{MODELS_DIR}/mapa_productos.pkl"
    mapa = joblib.load(mapa_path) if os.path.exists(mapa_path) else {}

    set_carrito = set(p.strip().lower() for p in productos_en_carrito)

    recomendaciones = {}
    for _, row in reglas.iterrows():
        antecedentes = set(row["antecedents"])
        consecuentes = set(row["consequents"])

        if antecedentes.issubset(set_carrito) or antecedentes & set_carrito:
            for prod in consecuentes:
                if prod not in set_carrito:
                    if prod not in recomendaciones or recomendaciones[prod]["lift"] < row["lift"]:
                        info = mapa.get(prod, {})
                        recomendaciones[prod] = {
                            "producto": prod,
                            "idProducto": info.get("idProducto", None),
                            "precio": info.get("precio", None),
                            "confianza": round(row["confidence"], 3),
                            "lift": round(row["lift"], 3),
                            "soporte": round(row["support"], 3),
                        }

    ordenados = sorted(recomendaciones.values(), key=lambda x: x["lift"], reverse=True)
    return ordenados[:top_n]


# ══════════════════════════════════════════════
# 2. FILTRADO COLABORATIVO — Preferencias por usuario
# ══════════════════════════════════════════════
def entrenar_colaborativo(df: pd.DataFrame):
    print("\n[2/2] Entrenando Filtrado Colaborativo (ALS)...")

    df["idUsuario"]  = df["idUsuario"].astype(str)
    df["idProducto"] = df["idProducto"].astype(str)

    le_user = LabelEncoder()
    le_prod = LabelEncoder()

    df["user_idx"] = le_user.fit_transform(df["idUsuario"])
    df["prod_idx"] = le_prod.fit_transform(df["idProducto"])

    interacciones = df.groupby(["user_idx", "prod_idx"])["cantidad"].sum().reset_index()

    n_users = df["user_idx"].nunique()
    n_prods = df["prod_idx"].nunique()

    matriz = csr_matrix(
        (interacciones["cantidad"].values,
         (interacciones["user_idx"].values, interacciones["prod_idx"].values)),
        shape=(n_users, n_prods)
    )

    modelo = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERATIONS,
        regularization=0.1,
        random_state=42
    )
    modelo.fit(matriz)

    joblib.dump({
        "modelo": modelo,
        "matriz": matriz,
        "le_user": le_user,
        "le_prod": le_prod,
        "df_productos": df[["idProducto", "nombreProducto", "categoria", "precio"]].drop_duplicates(),
    }, f"{MODELS_DIR}/colaborativo.pkl")

    print(f"  ✓ Modelo ALS entrenado con {n_users} usuarios y {n_prods} productos.")
    return modelo, matriz, le_user, le_prod, df


def recomendar_para_usuario(id_usuario: str, artefactos: dict, top_n=TOP_N) -> list:
    modelo   = artefactos["modelo"]
    matriz   = artefactos["matriz"]
    le_user  = artefactos["le_user"]
    le_prod  = artefactos["le_prod"]
    df_prods = artefactos["df_productos"]

    if id_usuario not in le_user.classes_:
        return []

    user_idx = le_user.transform([id_usuario])[0]
    ids_recomendados, scores = modelo.recommend(
        user_idx, matriz[user_idx], N=top_n, filter_already_liked_items=True
    )

    resultados = []
    for prod_idx, score in zip(ids_recomendados, scores):
        id_prod = le_prod.inverse_transform([prod_idx])[0]
        info = df_prods[df_prods["idProducto"] == id_prod]
        if not info.empty:
            resultados.append({
                "idProducto": int(id_prod),
                "nombreProducto": info.iloc[0]["nombreProducto"],
                "categoria": info.iloc[0]["categoria"],
                "precio": float(info.iloc[0]["precio"]),
                "score": round(float(score), 4),
            })

    return resultados


# ══════════════════════════════════════════════
# 3. LÓGICA DE ENTRENAMIENTO COMPLETO
# ══════════════════════════════════════════════
def ejecutar_entrenamiento_completo():
    """
    Descarga los CSVs desde Spring Boot y reentrena ambos modelos.
    Actualiza las variables globales en memoria.
    """
    global _reglas, _artefactos_colaborativo

    df_basket    = descargar_csv_desde_springboot("/api/exportar/market-basket/raw")
    df_historial = descargar_csv_desde_springboot("/api/exportar/historial-usuario/raw")

    print(f"\n  Market basket: {len(df_basket)} filas, {df_basket['idVenta'].nunique()} ventas únicas")
    print(f"  Historial: {len(df_historial)} filas, {df_historial['idUsuario'].nunique()} usuarios únicos")

    # Entrenar
    errores = []

    try:
        _reglas = entrenar_market_basket(df_basket)
    except Exception as e:
        errores.append(f"Market Basket: {e}")
        print(f"  ⚠ Error en Market Basket: {e}")

    try:
        entrenar_colaborativo(df_historial)
        _artefactos_colaborativo = joblib.load(f"{MODELS_DIR}/colaborativo.pkl")
    except Exception as e:
        errores.append(f"Colaborativo: {e}")
        print(f"  ⚠ Error en Colaborativo: {e}")

    return errores


# ══════════════════════════════════════════════
# 4. API FLASK
# ══════════════════════════════════════════════
def crear_api():
    app = Flask(__name__)
    CORS(app)

    @app.route("/salud", methods=["GET"])
    def salud():
        return jsonify({
            "estado": "ok",
            "servicio": "Recomendador Mass IA",
            "modelos": {
                "basket": _reglas is not None and not _reglas.empty,
                "colaborativo": _artefactos_colaborativo is not None,
            }
        })

    @app.route("/recomendar/carrito", methods=["POST"])
    def recomendar_carrito():
        body = request.get_json()
        productos_carrito = body.get("productosEnCarrito", [])
        id_usuario = str(body.get("idUsuario", ""))

        recomendaciones_basket  = recomendar_productos_juntos(productos_carrito, _reglas)
        recomendaciones_usuario = []

        if id_usuario and _artefactos_colaborativo:
            recomendaciones_usuario = recomendar_para_usuario(id_usuario, _artefactos_colaborativo)

        nombres_ya_recomendados = {r["producto"] for r in recomendaciones_basket}
        personalizadas_filtradas = [
            r for r in recomendaciones_usuario
            if r["nombreProducto"] not in nombres_ya_recomendados
            and r["nombreProducto"] not in productos_carrito
        ]

        return jsonify({
            "porCarrito": recomendaciones_basket,
            "personalizadas": personalizadas_filtradas[:TOP_N],
        })

    @app.route("/recomendar/usuario/<id_usuario>", methods=["GET"])
    def recomendar_usuario(id_usuario):
        if not _artefactos_colaborativo:
            return jsonify({"error": "Modelo colaborativo no disponible. Llama a /reentrenar primero."}), 503
        resultados = recomendar_para_usuario(str(id_usuario), _artefactos_colaborativo)
        return jsonify({"recomendaciones": resultados})

    @app.route("/reentrenar", methods=["POST"])
    def reentrenar():
        """
        Descarga los CSVs desde Spring Boot y reentrena los modelos.
        Llamar desde Spring Boot o manualmente cuando haya nuevos datos.
        """
        try:
            errores = ejecutar_entrenamiento_completo()
            if errores:
                return jsonify({
                    "resultado": "Entrenamiento parcial",
                    "advertencias": errores
                }), 207
            return jsonify({"resultado": "Modelos reentrenados correctamente"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


# ══════════════════════════════════════════════
# 5. APP — expuesta a nivel de módulo para gunicorn
# ══════════════════════════════════════════════
app = crear_api()  # gunicorn busca esta variable al importar el módulo


# ══════════════════════════════════════════════
# 6. MAIN — solo se ejecuta con `python recomendador_mass.py`
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Recomendador Tienda Mass — Iniciando (Producción)")
    print("=" * 55)
    print(f"  Spring Boot URL : {SPRINGBOOT_URL}")
    print(f"  Modelos dir     : {MODELS_DIR}")

    # Intentar entrenar al arrancar si Spring Boot ya está disponible
    # Si falla, el servicio igual arranca y se puede llamar a /reentrenar después
    print("\n  Intentando entrenamiento inicial...")
    try:
        errores = ejecutar_entrenamiento_completo()
        if errores:
            print(f"  ⚠ Entrenamiento parcial: {errores}")
        else:
            print("  ✓ Modelos listos.")
    except Exception as e:
        print(f"  ⚠ No se pudo entrenar al arrancar: {e}")
        print("  → El servicio arranca igual. Llama a POST /reentrenar cuando Spring Boot esté disponible.")

    print("\n  Endpoints disponibles:")
    print("  POST /reentrenar              ← descarga CSVs de Spring Boot y reentrena")
    print("  POST /recomendar/carrito")
    print("  GET  /recomendar/usuario/<id>")
    print("  GET  /salud\n")

    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)