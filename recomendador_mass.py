"""
Recomendador de productos - Tienda Mass
========================================
Requiere 2 CSVs:
  - market_basket.csv     → idVenta, idProducto, nombreProducto, categoria, precio
  - historial_usuario.csv → idUsuario, idVenta, fecha, idProducto, nombreProducto, categoria, precio, cantidad

Instalación:
  pip install pandas scikit-learn mlxtend implicit scipy numpy joblib flask flask-cors

Uso:
  python recomendador_mass.py
  → Entrena modelos y levanta API en http://localhost:5001
"""

import pandas as pd
import numpy as np
import joblib
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import implicit
from flask import Flask, request, jsonify
from flask_cors import CORS

MARKET_BASKET_CSV    = "market_basket.csv"
HISTORIAL_CSV        = "historial_usuario.csv"
MODELS_DIR           = "modelos"
MIN_SUPPORT          = 0.05   # 5% de ventas deben contener el itemset (baja si tienes pocos datos)
MIN_CONFIDENCE       = 0.3    # 30% de confianza mínima
MIN_LIFT             = 1.2    # Lift mínimo (> 1 = correlación positiva)
ALS_FACTORS          = 50     # Factores latentes para colaborativo
ALS_ITERATIONS       = 20
TOP_N                = 5      # Cuántas recomendaciones devolver

os.makedirs(MODELS_DIR, exist_ok=True)


def entrenar_market_basket(df: pd.DataFrame):
    """
    Usa Apriori para encontrar asociaciones entre productos.
    Input: df con columnas [idVenta, idProducto, nombreProducto]
    """
    print("\n[1/2] Entrenando Market Basket (Apriori)...")

    # Construir canasta: cada venta es una lista de productos
    canastas = df.groupby("idVenta")["nombreProducto"].apply(list).tolist()

    te = TransactionEncoder()
    te_array = te.fit(canastas).transform(canastas)
    df_basket = pd.DataFrame(te_array, columns=te.columns_)

    # Apriori
    frecuentes = apriori(df_basket, min_support=MIN_SUPPORT, use_colnames=True)

    if frecuentes.empty:
        print("  ⚠ Pocos datos para Apriori. Reduciendo min_support a 0.01...")
        frecuentes = apriori(df_basket, min_support=0.01, use_colnames=True)

    reglas = association_rules(frecuentes, metric="lift", min_threshold=MIN_LIFT)
    reglas = reglas[reglas["confidence"] >= MIN_CONFIDENCE]
    reglas = reglas.sort_values("lift", ascending=False)

    # Guardar
    reglas.to_pickle(f"{MODELS_DIR}/reglas_basket.pkl")
    print(f"  ✓ {len(reglas)} reglas encontradas y guardadas.")
    return reglas


def recomendar_productos_juntos(productos_en_carrito: list, reglas: pd.DataFrame, top_n=TOP_N) -> list:
    """
    Dado una lista de nombres de productos en el carrito,
    devuelve productos recomendados para agregar.
    """
    if reglas is None or reglas.empty:
        return []

    recomendaciones = {}
    set_carrito = set(productos_en_carrito)

    for _, row in reglas.iterrows():
        antecedentes = set(row["antecedents"])
        consecuentes = set(row["consequents"])

        # Si algún producto del carrito coincide con los antecedentes
        if antecedentes.issubset(set_carrito) or antecedentes & set_carrito:
            for prod in consecuentes:
                if prod not in set_carrito:
                    if prod not in recomendaciones or recomendaciones[prod]["lift"] < row["lift"]:
                        recomendaciones[prod] = {
                            "producto": prod,
                            "confianza": round(row["confidence"], 3),
                            "lift": round(row["lift"], 3),
                            "soporte": round(row["support"], 3),
                        }

    ordenados = sorted(recomendaciones.values(), key=lambda x: x["lift"], reverse=True)
    return ordenados[:top_n]


def entrenar_colaborativo(df: pd.DataFrame):
    """
    Usa ALS (Alternating Least Squares) implícito para aprender
    preferencias de usuarios basado en frecuencia de compra.
    """
    print("\n[2/2] Entrenando Filtrado Colaborativo (ALS)...")

    # Pivot: filas=usuarios, columnas=productos, valores=cantidad comprada
    df["idUsuario"] = df["idUsuario"].astype(str)
    df["idProducto"] = df["idProducto"].astype(str)

    le_user = LabelEncoder()
    le_prod = LabelEncoder()

    df["user_idx"] = le_user.fit_transform(df["idUsuario"])
    df["prod_idx"] = le_prod.fit_transform(df["idProducto"])

    # Sumar cantidades por usuario-producto
    interacciones = df.groupby(["user_idx", "prod_idx"])["cantidad"].sum().reset_index()

    n_users = df["user_idx"].nunique()
    n_prods = df["prod_idx"].nunique()

    # Matriz sparse usuario x producto
    matriz = csr_matrix(
        (interacciones["cantidad"].values,
         (interacciones["user_idx"].values, interacciones["prod_idx"].values)),
        shape=(n_users, n_prods)
    )

    # Entrenar ALS
    modelo = implicit.als.AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERATIONS,
        regularization=0.1,
        random_state=42
    )
    modelo.fit(matriz)

    # Guardar todo lo necesario para inferencia
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
    """
    Dado un idUsuario, devuelve productos recomendados basados en su historial.
    """
    modelo    = artefactos["modelo"]
    matriz    = artefactos["matriz"]
    le_user   = artefactos["le_user"]
    le_prod   = artefactos["le_prod"]
    df_prods  = artefactos["df_productos"]

    if id_usuario not in le_user.classes_:
        return []  # Usuario sin historial

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


def crear_api(reglas, artefactos_colaborativo):
    app = Flask(__name__)
    CORS(app)  # Permite llamadas desde Spring Boot / Next.js

    @app.route("/salud", methods=["GET"])
    def salud():
        return jsonify({"estado": "ok", "servicio": "Recomendador Mass IA"})

    @app.route("/recomendar/carrito", methods=["POST"])
    def recomendar_carrito():
        body = request.get_json(force=True, silent=True) or {}
        productos_carrito = body.get("productosEnCarrito", [])
        id_usuario = str(body.get("idUsuario", ""))

        # Recomendaciones por market basket
        recomendaciones_basket = recomendar_productos_juntos(productos_carrito, reglas)

        # Recomendaciones personalizadas (si hay usuario)
        recomendaciones_usuario = []
        if id_usuario and artefactos_colaborativo:
            recomendaciones_usuario = recomendar_para_usuario(id_usuario, artefactos_colaborativo)

        # Combinar: primero basket (más contextuales), luego personalizadas
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
        """Recomendaciones solo por historial del usuario."""
        if not artefactos_colaborativo:
            return jsonify({"error": "Modelo colaborativo no disponible"}), 503
        resultados = recomendar_para_usuario(str(id_usuario), artefactos_colaborativo)
        return jsonify({"recomendaciones": resultados})

    @app.route("/reentrenar", methods=["POST"])
    def reentrenar():
        """
        Vuelve a entrenar con los CSVs actuales.
        Llamar desde Spring Boot cuando haya nuevas ventas.
        """
        try:
            df_basket    = pd.read_csv(MARKET_BASKET_CSV)
            df_historial = pd.read_csv(HISTORIAL_CSV)
            entrenar_market_basket(df_basket)
            entrenar_colaborativo(df_historial)
            return jsonify({"resultado": "Modelos reentrenados correctamente"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    print("=" * 50)
    print("  Recomendador Tienda Mass — Iniciando")
    print("=" * 50)

    # ── Cargar CSVs ──
    if not os.path.exists(MARKET_BASKET_CSV):
        raise FileNotFoundError(f"No se encontró {MARKET_BASKET_CSV}")
    if not os.path.exists(HISTORIAL_CSV):
        raise FileNotFoundError(f"No se encontró {HISTORIAL_CSV}")

    df_basket    = pd.read_csv(MARKET_BASKET_CSV)
    df_historial = pd.read_csv(HISTORIAL_CSV)

    print(f"\n📦 Market basket: {len(df_basket)} filas, {df_basket['idVenta'].nunique()} ventas únicas")
    print(f"👤 Historial: {len(df_historial)} filas, {df_historial['idUsuario'].nunique()} usuarios únicos")

    # ── Entrenar ──
    reglas = None
    artefactos_colaborativo = None

    try:
        reglas = entrenar_market_basket(df_basket)
    except Exception as e:
        print(f"  ⚠ Error en Market Basket: {e}")

    try:
        modelo, matriz, le_user, le_prod, df_h = entrenar_colaborativo(df_historial)
        artefactos_colaborativo = joblib.load(f"{MODELS_DIR}/colaborativo.pkl")
    except Exception as e:
        print(f"  ⚠ Error en Colaborativo: {e}")

    print("\n Modelos listos. Levantando API en http://localhost:5001")
    print("   Endpoints disponibles:")
    print("   POST /recomendar/carrito")
    print("   GET  /recomendar/usuario/<id>")
    print("   POST /reentrenar")
    print("   GET  /salud\n")

    app = crear_api(reglas, artefactos_colaborativo)
    app.run(host="0.0.0.0", port=5001, debug=False)