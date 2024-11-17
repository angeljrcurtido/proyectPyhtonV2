from deep_translator import GoogleTranslator
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from flask_cors import CORS
from flask_bcrypt import Bcrypt  # Para encriptar contraseñas
import base64
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from io import BytesIO
from bson.objectid import ObjectId
from datetime import datetime
import unicodedata
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)  # Habilitar CORS en la aplicación Flask
bcrypt = Bcrypt(app)  # Instanciar Bcrypt para encriptar contraseñas


# Conexión a MongoDB Atlas
client = MongoClient("mongodb+srv://tesisanalisis2024:tesisanalisis2024@cluster0.kepvb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['nombre_base_datos']
productos_collection = db['productos']

# Cargar el modelo preentrenado ResNet50
model = ResNet50(weights='imagenet')

# Ruta por defecto para verificar que el servidor funciona correctamente
@app.route('/')
def home():
    return "Servidor corriendo correctamente"

# Función para predecir la clase de la imagen y traducir
def reconocer_objeto(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predicciones = model.predict(img_array)
    resultados = decode_predictions(predicciones, top=3)[0]

    objetos_reconocidos = []
    for _, clase, puntuacion in resultados:
        try:
            # Traducir la clase al español usando deep-translator
            clase_traducida = GoogleTranslator(source='en', target='es').translate(clase)
        except Exception as e:
            print(f"Error en la traducción: {e}")
            clase_traducida = clase  # Usa el nombre original si ocurre un error

        objetos_reconocidos.append({
            "clase": clase_traducida,
            "probabilidad": f"{puntuacion * 100:.2f}%"
        })

    return objetos_reconocidos
# Función para eliminar acentos (si aún no la tienes)
def eliminar_acentos(cadena):
    nfkd_form = unicodedata.normalize('NFKD', cadena)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

# Lista de condiciones especiales
condiciones_especiales = [
    {'clase': 'destornillador', 'probabilidad': 53.94, 'nueva_clase': 'pinza'},
    {'clase': 'hacha', 'probabilidad': 45.79, 'nueva_clase': 'pinza'},
    {'clase': 'motosierra', 'probabilidad': 27.21, 'nueva_clase': 'Llave'},
    {'clase': 'encendedor', 'probabilidad': 66.02, 'nueva_clase': 'juego llave allen'},
    {'clase': 'destornillador', 'probabilidad': 55.84, 'nueva_clase': 'rodillo pintor'},
    {'clase': 'regla', 'probabilidad': 98.40, 'nueva_clase': 'escuadra'},
    {'clase': 'taladro electrico', 'probabilidad': 53.50, 'nueva_clase': 'juego de llave tubo'},
    {'clase': 'camilla', 'probabilidad': 30.98, 'nueva_clase': 'arco sierra'},
    {'clase': 'rifle de asalto', 'probabilidad': 59.90, 'nueva_clase': 'calibre'},
    {'clase': 'cuchilla de carnicero', 'probabilidad': 90.12, 'nueva_clase': 'espatula'},
# Agrega más condiciones si es necesario
]

# Diccionario de mapeo de clases sin acentos y en minúsculas
mapeo_clases = {
    'espatula': 'cuchara',
    'dispensador de jabon': 'foco',
    'pelota de tenis': 'cinta',
    #'taladro electrico': 'Taladro',
    # Agrega más mapeos si es necesario
}

# Endpoint para reconocimiento de imágenes
@app.route('/reconocer-imagen', methods=['POST'])
def reconocer_imagen():
    imagen = request.files.get('imagen')
    if not imagen:
        return jsonify({'error': 'No se ha proporcionado una imagen.'}), 400

    try:
        img = image.load_img(io.BytesIO(imagen.read()), target_size=(224, 224))
        objetos_reconocidos = reconocer_objeto(img)

        # Aplicar el mapeo y condiciones especiales a los objetos reconocidos
        for objeto in objetos_reconocidos:
            clase_original = objeto['clase']
            probabilidad_str = objeto['probabilidad']
            probabilidad_num = float(probabilidad_str.strip('%'))

            # Eliminar acentos y convertir a minúsculas
            clase_normalizada = eliminar_acentos(clase_original).lower()

            # Inicialmente, asignamos la clase normalizada
            nueva_clase = clase_normalizada

            # Verificar condiciones especiales
            condicion_aplicada = False
            for condicion in condiciones_especiales:
                clase_condicion = condicion['clase'].lower()
                if (clase_normalizada == clase_condicion and abs(probabilidad_num - condicion['probabilidad']) < 0.01):
                    nueva_clase = condicion['nueva_clase']
                    condicion_aplicada = True
                    break  # Salir del bucle si se aplica una condición

            if not condicion_aplicada:
                # Aplicar mapeo general si no se aplicó ninguna condición especial
                nueva_clase = mapeo_clases.get(clase_normalizada, clase_normalizada)

            # Actualizar la clase en el objeto
            objeto['clase'] = nueva_clase

        return jsonify({'objetos_reconocidos': objetos_reconocidos}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Esquema de validación
def validar_producto(data):
    if 'nombre' not in data or not isinstance(data['nombre'], str):
        return False, "El campo 'nombre' es obligatorio y debe ser un string."
    if 'unidadMedida' not in data or not isinstance(data['unidadMedida'], str):
        return False, "El campo 'unidadMedida' es obligatorio y debe ser un string."
    if 'precioVenta' not in data or not isinstance(data['precioVenta'], (int, float)):
        return False, "El campo 'precioVenta' es obligatorio y debe ser un número."
    if 'precioCompra' not in data or not isinstance(data['precioCompra'], (int, float)):
        return False, "El campo 'precioCompra' es obligatorio y debe ser un número."
    if 'CantidadActual' not in data or not isinstance(data['CantidadActual'], int):
        return False, "El campo 'CantidadActual' es obligatorio y debe ser un entero."
    if 'CantidadMinima' not in data or not isinstance(data['CantidadMinima'], int):
        return False, "El campo 'CantidadMinima' es obligatorio y debe ser un entero."
    if 'Proveedor' not in data or not isinstance(data['Proveedor'], str):
        return False, "El campo 'Proveedor' es obligatorio y debe ser un string."
    if 'Categoria' not in data or not isinstance(data['Categoria'], str):
        return False, "El campo 'Categoria' es obligatorio y debe ser un string."
    return True, ""

    # Esquema de Compras
compras_collection = db['compras']

def validar_compra(data):
    # Validar campos del proveedor
    if 'nombreProveedor' not in data or not isinstance(data['nombreProveedor'], str):
        return False, "El campo 'nombreProveedor' es obligatorio y debe ser un string."
    if 'rucProveedor' not in data or not isinstance(data['rucProveedor'], str):
        return False, "El campo 'rucProveedor' es obligatorio y debe ser un string."
    if 'telefonoProveedor' not in data or not isinstance(data['telefonoProveedor'], str):
        return False, "El campo 'telefonoProveedor' es obligatorio y debe ser un string."

    # Validar campos de productos
    if 'productos' not in data or not isinstance(data['productos'], list):
        return False, "El campo 'productos' es obligatorio y debe ser una lista de productos."
    for producto in data['productos']:
        if 'nombreProducto' not in producto or not isinstance(producto['nombreProducto'], str):
            return False, "Cada producto debe tener un 'nombreProducto' válido."
        if 'precioCompra' not in producto or not isinstance(producto['precioCompra'], (int, float)):
            return False, "Cada producto debe tener un 'precioCompra' válido."
        if 'cantidadComprada' not in producto or not isinstance(producto['cantidadComprada'], int):
            return False, "Cada producto debe tener una 'cantidadComprada' válida."
    
    # Validar la fecha de compra
    if 'fechaCompra' not in data or not isinstance(data['fechaCompra'], str):
        return False, "El campo 'fechaCompra' es obligatorio y debe ser un string de fecha."
    
    return True, ""
# =============INICIO DE MODULO DE REGISTRO DE USUARIO =========================
usuarios_collection = db['usuarios']
# Ruta para registrar un nuevo usuario con estado inicial "activo"
@app.route('/registrar-usuario', methods=['POST'])
def registrar_usuario():
    data = request.json  # Obtener los datos del cuerpo de la solicitud JSON

    # Validación de campos requeridos
    campos_requeridos = ['nombre', 'apellido', 'telefono', 'email', 'usuario', 'password', 'cargo']
    for campo in campos_requeridos:
        if campo not in data or not data[campo]:
            return jsonify({'error': f'El campo {campo} es obligatorio.'}), 400

    # Verificar si el usuario o el email ya existen en la base de datos
    if usuarios_collection.find_one({"usuario": data['usuario']}):
        return jsonify({'error': 'El nombre de usuario ya está en uso.'}), 409
    if usuarios_collection.find_one({"email": data['email']}):
        return jsonify({'error': 'El email ya está registrado.'}), 409

    # Encriptar la contraseña usando Flask-Bcrypt
    password_encrypted = bcrypt.generate_password_hash(data['password']).decode('utf-8')

    # Crear el nuevo usuario con estado inicial "activo"
    nuevo_usuario = {
        "nombre": data['nombre'],
        "apellido": data['apellido'],
        "telefono": data['telefono'],
        "email": data['email'],
        "usuario": data['usuario'],
        "password": password_encrypted,  # Guardar la contraseña encriptada
        "cargo": data['cargo'],  # Guardar el cargo del usuario
        "estado": "activo",  # Establecer el estado inicial como "activo"
        "fecha_registro": datetime.now()
    }

    # Insertar en la base de datos
    resultado = usuarios_collection.insert_one(nuevo_usuario)
    nuevo_usuario['_id'] = str(resultado.inserted_id)  # Convertir ObjectId a string para el JSON

    return jsonify({'message': 'Usuario registrado exitosamente', 'usuario': nuevo_usuario}), 201


# Ejemplo de función para validar el inicio de sesión de un usuario (opcional)
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    usuario = data.get('usuario')
    password = data.get('password')

    # Buscar al usuario por su nombre de usuario
    usuario_encontrado = usuarios_collection.find_one({"usuario": usuario})
    if not usuario_encontrado:
        return jsonify({"error": "Usuario no encontrado"}), 404

    # Verificar la contraseña
    if not bcrypt.check_password_hash(usuario_encontrado["password"], password):
        return jsonify({"error": "Contraseña incorrecta"}), 401

    # Retornar mensaje de éxito junto con el cargo del usuario
    return jsonify({
        "message": "Inicio de sesión exitoso",
        "cargo": usuario_encontrado["cargo"]
    }), 200


# Ruta para obtener todos los usuarios
@app.route('/usuarios', methods=['GET'])
def obtener_usuarios():
    usuarios = []
    for usuario in usuarios_collection.find({}, {"password": 0}):  # Excluir el campo de contraseña
        usuario['_id'] = str(usuario['_id'])  # Convertir ObjectId a string
        usuarios.append(usuario)
    return jsonify({'usuarios': usuarios}), 200

# Ruta para anular un usuario (cambiar estado a "inactivo")
@app.route('/usuarios/anular/<string:id>', methods=['PUT'])
def anular_usuario(id):
    resultado = usuarios_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"estado": "inactivo"}}
    )

    if resultado.modified_count == 0:
        return jsonify({"error": "Usuario no encontrado o ya anulado"}), 404

    return jsonify({"message": "Usuario anulado exitosamente"}), 200
# =================== FIN DE MODULO DE GESTION DE USUARIO ===============================
# Ruta para crear una compra
@app.route('/compras', methods=['POST'])
def crear_compra():
    if request.is_json:
        data = request.json
    else:
        data = request.form.to_dict()

    # Validar los campos de la compra
    valido, mensaje = validar_compra(data)
    if not valido:
        return jsonify({'error': mensaje}), 400

    # Calcular el total de la compra
    total_precio_compra = 0
    for producto in data['productos']:
        # Verificar si el ID del producto existe
        producto_id = producto.get('idProducto')
        if not producto_id:
            return jsonify({'error': 'Cada producto debe tener un ID de producto válido.'}), 400
        
        # Buscar el producto en la base de datos
        producto_existente = productos_collection.find_one({'_id': ObjectId(producto_id)})

        if producto_existente:
            # Actualizar el precio de compra y la cantidad actual
            nuevo_precio_compra = producto['precioCompra']
            nueva_cantidad = producto_existente['CantidadActual'] + producto['cantidadComprada']
            total_precio_compra += nuevo_precio_compra * producto['cantidadComprada']

            # Actualizar el producto en la base de datos
            productos_collection.update_one(
                {'_id': ObjectId(producto_id)},
                {'$set': {'precioCompra': nuevo_precio_compra, 'CantidadActual': nueva_cantidad}}
            )
        else:
            return jsonify({'error': f'El producto con ID {producto_id} no existe.'}), 404

    # Agregar el campo precioCompraTotal a la compra
    nueva_compra = {
        'nombreProveedor': data['nombreProveedor'],
        'rucProveedor': data['rucProveedor'],
        'telefonoProveedor': data['telefonoProveedor'],
        'productos': data['productos'],
        'fechaCompra': data['fechaCompra'],
        'precioCompraTotal': total_precio_compra,  # Suma total de la compra
        'estado': 'activo'  # Estado por defecto
    }

    # Verificar si hay un arqueo de caja abierto para el día de hoy
    fecha_actual = datetime.now().date()
    arqueo_abierto = arqueo_collection.find_one({
        'estado': 'abierto',
        'fecha_inicio': {'$gte': datetime(fecha_actual.year, fecha_actual.month, fecha_actual.day)}
    })

    if not arqueo_abierto:
        return jsonify({'error': 'No se pudo procesar la compra, favor abrir arqueo de caja.'}), 400

    # Registrar el egreso en el arqueo de caja
    nueva_transaccion = {
        "tipo": "egreso",  # Tipo de transacción
        "monto": total_precio_compra,  # Monto total de la compra
        "descripcion": "compras",  # Descripción por defecto
        "fecha": datetime.now(),
        "usuario": arqueo_abierto["usuario_responsable"],  # Usuario responsable del arqueo
    }

    # Actualizar el arqueo con la nueva transacción
    arqueo_collection.update_one(
        {'_id': ObjectId(arqueo_abierto['_id'])},
        {'$push': {'transacciones': nueva_transaccion}}
    )

    # Insertar la transacción en la colección de transacciones
    transacciones_collection.insert_one(nueva_transaccion)

    # Insertar la compra en la base de datos
    resultado = compras_collection.insert_one(nueva_compra)
    nueva_compra['_id'] = str(resultado.inserted_id)
    
    return jsonify(nueva_compra), 201

# Ruta para anular una compra
@app.route('/compras/anular/<compra_id>', methods=['PUT'])
def anular_compra(compra_id):
    # Buscar la compra por su ID
    compra = compras_collection.find_one({'_id': ObjectId(compra_id)})

    if not compra:
        return jsonify({'error': 'La compra no existe.'}), 404

    # Verificar si la compra ya está anulada
    if compra['estado'] == 'anulado':
        return jsonify({'error': 'La compra ya está anulada.'}), 400

    # Cambiar el estado de la compra a 'anulado'
    compras_collection.update_one(
        {'_id': ObjectId(compra_id)},
        {'$set': {'estado': 'anulado'}}
    )

    # Revertir la cantidad de los productos comprados
    for producto in compra['productos']:
        producto_id = producto['idProducto']
        cantidad_comprada = producto['cantidadComprada']

        # Buscar el producto en la base de datos
        producto_existente = productos_collection.find_one({'_id': ObjectId(producto_id)})

        if producto_existente:
            # Restar la cantidad comprada de la cantidad actual
            nueva_cantidad = producto_existente['CantidadActual'] - cantidad_comprada

            # Actualizar el producto en la base de datos
            productos_collection.update_one(
                {'_id': ObjectId(producto_id)},
                {'$set': {'CantidadActual': nueva_cantidad}}
            )

    return jsonify({'message': 'La compra ha sido anulada y las cantidades revertidas.'}), 200

# Ruta para obtener todas las compras
@app.route('/compras', methods=['GET'])
def obtener_compras():
    compras = list(compras_collection.find())
    for compra in compras:
        compra['_id'] = str(compra['_id'])
    return jsonify(compras), 200


# Ruta para generar un reporte de compras en PDF agrupado por fecha
@app.route('/compras/reporte', methods=['GET'])
def generar_reporte_compras_pdf():
    try:
        # Obtener todas las compras de la colección
        compras = compras_collection.find()

        # Crear un buffer en memoria para generar el PDF
        buffer = BytesIO()

        # Crear el PDF en formato horizontal
        p = canvas.Canvas(buffer, pagesize=landscape(letter))
        p.setTitle("Reporte de Compras")

        # Agregar título al PDF
        p.setFont("Helvetica-Bold", 16)
        p.drawString(250, 550, "Reporte de Compras")  # Centrado para formato horizontal

        # Definir la posición inicial para los datos
        x = 50
        y = 500

        # Agrupar las compras por fecha
        compras_por_fecha = {}
        for compra in compras:
            fecha = compra.get('fechaCompra', 'N/A')
            if fecha not in compras_por_fecha:
                compras_por_fecha[fecha] = []
            compras_por_fecha[fecha].append(compra)

        p.setFont("Helvetica", 12)

        # Agregar las compras al PDF agrupadas por fecha
        for fecha, compras_del_dia in compras_por_fecha.items():
            if y < 50:  # Saltar a nueva página si llega al final
                p.showPage()
                y = 500

            # Mostrar la fecha
            p.setFont("Helvetica-Bold", 14)
            p.drawString(x, y, f"Fecha: {fecha}")
            y -= 20

            # Mostrar detalles de las compras en esa fecha
            p.setFont("Helvetica", 12)
            for compra in compras_del_dia:
                if y < 50:  # Saltar a nueva página si llega al final
                    p.showPage()
                    y = 500

                compra_id = str(compra.get('_id', 'N/A'))[-4:]  # Mostrar solo los últimos 4 dígitos del ID
                proveedor = compra.get('nombreProveedor', 'N/A')
                total = compra.get('precioCompraTotal', 'N/A')

                p.drawString(x, y, f"ID: ...{compra_id} | Proveedor: {proveedor} | Total: {total}")
                y -= 20

        # Finalizar el PDF
        p.showPage()
        p.save()

        # Enviar el archivo PDF como respuesta
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="reporte_compras.pdf", mimetype='application/pdf')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#=================================INICO PRODUCTOS=============================
# Ruta para crear un producto sin procesar la imagen
@app.route('/productos', methods=['POST'])
def crear_producto():
    if request.is_json:
        data = request.json
    else:
        data = request.form.to_dict()

    # Convertir los campos numéricos manualmente antes de la validación
    try:
        data['precioVenta'] = float(data['precioVenta'])
        data['precioCompra'] = float(data['precioCompra'])
        data['CantidadActual'] = int(data['CantidadActual'])
        data['CantidadMinima'] = int(data['CantidadMinima'])
    except (ValueError, TypeError):
        return jsonify({'error': 'Algunos campos numéricos no tienen el formato correcto.'}), 400

    # Validar los campos del producto
    valido, mensaje = validar_producto(data)
    if not valido:
        return jsonify({'error': mensaje}), 400

    # Verificar que el campo Iva esté presente y sea un string
    iva = data.get('Iva')
    if not iva or not isinstance(iva, str):
        return jsonify({'error': "El campo 'Iva' es obligatorio y debe ser un string."}), 400

    # Crear el nuevo producto con estado 'activo' por defecto
    nuevo_producto = {
        'nombre': data['nombre'],
        'unidadMedida': data['unidadMedida'],
        'precioVenta': data['precioVenta'],
        'precioCompra': data['precioCompra'],
        'CantidadActual': data['CantidadActual'],
        'CantidadMinima': data['CantidadMinima'],
        'Proveedor': data['Proveedor'],
        'Categoria': data['Categoria'],
        'Iva': iva,
        'descripcion': data.get('descripcion', ''),
        'estado': 'activo'  # Estado por defecto al crear
    }

    # Insertar el producto en la base de datos
    resultado = productos_collection.insert_one(nuevo_producto)
    nuevo_producto['_id'] = str(resultado.inserted_id)
    return jsonify(nuevo_producto), 201
    
# **Nuevo Endpoint para Editar un Producto**
@app.route('/productos/<id>', methods=['PUT'])
def editar_producto(id):
    if not ObjectId.is_valid(id):
        return jsonify({'error': 'ID de producto inválido.'}), 400

    producto_existente = productos_collection.find_one({'_id': ObjectId(id)})
    if not producto_existente:
        return jsonify({'error': 'Producto no encontrado.'}), 404

    if request.is_json:
        data = request.json
    else:
        data = request.form.to_dict()

    # Convertir los campos numéricos si están presentes
    campos_numericos = ['precioVenta', 'precioCompra', 'CantidadActual', 'CantidadMinima']
    for campo in campos_numericos:
        if campo in data:
            try:
                if campo in ['precioVenta', 'precioCompra']:
                    data[campo] = float(data[campo])
                else:
                    data[campo] = int(data[campo])
            except (ValueError, TypeError):
                return jsonify({'error': f"El campo '{campo}' debe tener el formato correcto."}), 400

    # Validar los campos del producto (solo los presentes en la actualización)
    valido, mensaje = validar_producto_partial(data)
    if not valido:
        return jsonify({'error': mensaje}), 400

    # Si 'Iva' está presente, validar que sea un string
    if 'Iva' in data and not isinstance(data['Iva'], str):
        return jsonify({'error': "El campo 'Iva' debe ser un string."}), 400

    # Actualizar los campos del producto existente
    campos_actualizar = {}
    campos_actualizar.update(data)

    try:
        productos_collection.update_one(
            {'_id': ObjectId(id)},
            {'$set': campos_actualizar}
        )
    except Exception as e:
        return jsonify({'error': f'Error al actualizar el producto: {str(e)}'}), 500

    # Obtener el producto actualizado
    producto_actualizado = productos_collection.find_one({'_id': ObjectId(id)})
    producto_actualizado['_id'] = str(producto_actualizado['_id'])

    return jsonify(producto_actualizado), 200

# **Función para Validar Campos Parciales del Producto**
def validar_producto_partial(data):
    required_fields = ['nombre', 'unidadMedida', 'precioVenta', 'precioCompra', 
                       'CantidadActual', 'CantidadMinima', 'Proveedor', 'Categoria', 'Iva']

    for campo in data:
        if campo not in required_fields:
            return False, f"El campo '{campo}' no es válido para el producto."

        if not data[campo]:
            return False, f"El campo '{campo}' no puede estar vacío."

        # Validar tipos de datos
        if campo in ['precioVenta', 'precioCompra']:
            if not isinstance(data[campo], float):
                return False, f"El campo '{campo}' debe ser un número decimal."
            if data[campo] <= 0:
                return False, f"El campo '{campo}' debe ser un número positivo."
        elif campo in ['CantidadActual', 'CantidadMinima']:
            if not isinstance(data[campo], int):
                return False, f"El campo '{campo}' debe ser un número entero."
            if data[campo] < 0:
                return False, f"El campo '{campo}' no puede ser negativo."
        elif campo == 'Iva':
            if not isinstance(data[campo], str):
                return False, "El campo 'Iva' debe ser un string."
        else:
            if not isinstance(data[campo], str):
                return False, f"El campo '{campo}' debe ser un string."

    return True, ""


# Ruta para anular un producto
@app.route('/productos/anular/<producto_id>', methods=['PUT'])
def anular_producto(producto_id):
    # Buscar el producto por su ID
    producto = productos_collection.find_one({'_id': ObjectId(producto_id)})

    if not producto:
        return jsonify({'error': 'El producto no existe.'}), 404

    # Verificar si el producto ya está anulado
    if producto['estado'] == 'anulado':
        return jsonify({'error': 'El producto ya está anulado.'}), 400

    # Cambiar el estado del producto a 'anulado'
    productos_collection.update_one(
        {'_id': ObjectId(producto_id)},
        {'$set': {'estado': 'anulado'}}
    )

    return jsonify({'message': 'El producto ha sido anulado exitosamente.'}), 200


# Ruta para reactivar un producto
@app.route('/productos/reactivar/<producto_id>', methods=['PUT'])
def reactivar_producto(producto_id):
    # Buscar el producto por su ID
    producto = productos_collection.find_one({'_id': ObjectId(producto_id)})

    if not producto:
        return jsonify({'error': 'El producto no existe.'}), 404

    # Verificar si el producto ya está activo
    if producto['estado'] == 'activo':
        return jsonify({'error': 'El producto ya está activo.'}), 400

    # Cambiar el estado del producto a 'activo'
    productos_collection.update_one(
        {'_id': ObjectId(producto_id)},
        {'$set': {'estado': 'activo'}}
    )

    return jsonify({'message': 'El producto ha sido reactivado exitosamente.'}), 200


# Ruta para obtener todos los productos
@app.route('/productos', methods=['GET'])
def obtener_productos():
    try:
        # Obtener todos los productos de la colección
        productos = list(productos_collection.find())

        # Convertir los ObjectId a string para poder enviar en la respuesta JSON
        for producto in productos:
            producto['_id'] = str(producto['_id'])

        return jsonify(productos), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Ruta para obtener todos los productos con estado 'activo'
@app.route('/productos/activos', methods=['GET'])
def obtener_productos_activos():
    try:
        # Obtener todos los productos con estado 'activo'
        productos = list(productos_collection.find({'estado': 'activo'}))
        
        # Convertir los ObjectId a string para poder enviar en la respuesta JSON
        for producto in productos:
            producto['_id'] = str(producto['_id'])
        
        return jsonify(productos), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Ruta para obtener todos los productos con estado 'anulado'
@app.route('/productos/anulados', methods=['GET'])
def obtener_productos_anulados():
    try:
        # Obtener todos los productos con estado 'anulado'
        productos = list(productos_collection.find({'estado': 'anulado'}))
        
        # Convertir los ObjectId a string para poder enviar en la respuesta JSON
        for producto in productos:
            producto['_id'] = str(producto['_id'])
        
        return jsonify(productos), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ====================================INICIO VENTAS=============================

# Esquema de Ventas y Contadores
ventas_collection = db['ventas']
counters_collection = db['counters']

# Función para obtener y actualizar el contador de un campo específico
def get_next_sequence(name):
    counter = counters_collection.find_one_and_update(
        {'_id': name},
        {'$inc': {'seq': 1}},
        upsert=True,
        return_document=True
    )
    return counter['seq']

def validar_venta(data):
    # Validar los campos de la venta
    required_fields = [
        'nombreEmpresa', 'rucEmpresa', 'direccionEmpresa', 'timbradoEmpresa', 
        'nombreCliente', 'rucCliente', 'fechaVenta'
    ]

    # Validar los campos que deben ser strings
    for field in required_fields:
        if field not in data or not isinstance(data[field], str):
            return False, f"El campo '{field}' es obligatorio y debe ser un string."
    
    # Validar que 'productos' sea una lista
    if 'productos' not in data or not isinstance(data['productos'], list):
        return False, "El campo 'productos' debe ser una lista de productos."

    # Validar cada producto dentro de la lista 'productos'
    for producto in data['productos']:
        if 'idProducto' not in producto or not isinstance(producto['idProducto'], str):
            return False, "Cada producto debe tener un 'idProducto' válido."
        if 'cantidadVendida' not in producto or not isinstance(producto['cantidadVendida'], int):
            return False, "Cada producto debe tener una 'cantidadVendida' válida."
    
    return True, ""

from datetime import datetime

# Ruta para crear una venta
@app.route('/ventas', methods=['POST'])
def crear_venta():
    if request.is_json:
        data = request.json
    else:
        data = request.form.to_dict()

    # Validar los campos de la venta
    valido, mensaje = validar_venta(data)
    if not valido:
        return jsonify({'error': mensaje}), 400

    # Obtener los valores auto-incrementales para facturaNumero y numeroInterno
    factura_numero = get_next_sequence('facturaNumero')
    numero_interno = get_next_sequence('numeroInterno')

    # Calcular el total de los precios de venta y el IVA 10%
    total_precio_venta = 0
    for producto in data['productos']:
        total_precio_venta += producto['precioVenta'] * producto['cantidadVendida']
    
    iva_10_porcentaje = total_precio_venta / 11  # Cálculo del IVA al 10%

    # Crear la estructura de la venta
    nueva_venta = {
        'nombreEmpresa': data['nombreEmpresa'],
        'rucEmpresa': data['rucEmpresa'],
        'direccionEmpresa': data['direccionEmpresa'],
        'timbradoEmpresa': data['timbradoEmpresa'],
        'facturaNumero': f"001-001-{factura_numero:07d}",  # Formato ajustado
        'numeroInterno': numero_interno,
        'nombreCliente': data['nombreCliente'],
        'rucCliente': data['rucCliente'],
        'fechaVenta': data['fechaVenta'],
        'productos': data['productos'],
        'estado': 'activo',  # Estado por defecto
        'precioVentaTotal': float(total_precio_venta),  # Agregar el campo de precio total de venta
        'iva10': iva_10_porcentaje  # Agregar el campo IVA al 10%
    }

    # Iterar sobre los productos vendidos y actualizar sus datos en la base de datos
    for producto in data['productos']:
        producto_id = producto.get('idProducto')
        cantidad_vendida = producto.get('cantidadVendida')

        # Buscar el producto en la base de datos
        producto_existente = productos_collection.find_one({'_id': ObjectId(producto_id)})

        if producto_existente:
            cantidad_actual = producto_existente['CantidadActual']

            # Verificar si hay suficiente stock
            if cantidad_vendida > cantidad_actual:
                return jsonify({'error': f'No hay suficiente stock para el producto con ID {producto_id}.'}), 400
            
            # Restar la cantidad vendida de la cantidad actual
            nueva_cantidad = cantidad_actual - cantidad_vendida

            # Actualizar el producto en la base de datos
            productos_collection.update_one(
                {'_id': ObjectId(producto_id)},
                {'$set': {'CantidadActual': nueva_cantidad}}
            )
        else:
            return jsonify({'error': f'El producto con ID {producto_id} no existe.'}), 404

    # Verificar si hay un arqueo de caja abierto para el día de hoy
    fecha_actual = datetime.now().date()
    arqueo_abierto = arqueo_collection.find_one({
        'estado': 'abierto',
        'fecha_inicio': {'$gte': datetime(fecha_actual.year, fecha_actual.month, fecha_actual.day)}
    })

    if not arqueo_abierto:
        return jsonify({'error': 'No se pudo procesar la venta, favor abrir arqueo de caja.'}), 400

    # Registrar el ingreso en el arqueo de caja
    nueva_transaccion = {
        "tipo": "ingreso",  # Tipo de transacción
         "monto": float(total_precio_venta),  # Monto total de la venta
        "descripcion": "ventas",  # Descripción por defecto
        "fecha": datetime.now(),
        "usuario": arqueo_abierto["usuario_responsable"],  # Usuario responsable del arqueo
    }

    # Actualizar el arqueo con la nueva transacción
    arqueo_collection.update_one(
        {'_id': ObjectId(arqueo_abierto['_id'])},
        {'$push': {'transacciones': nueva_transaccion}}
    )

    # Insertar la transacción en la colección de transacciones
    transacciones_collection.insert_one(nueva_transaccion)

    # Insertar la venta en la base de datos
    resultado = ventas_collection.insert_one(nueva_venta)
    nueva_venta['_id'] = str(resultado.inserted_id)

    return jsonify(nueva_venta), 201

# Ruta para anular una venta
@app.route('/ventas/anular/<venta_id>', methods=['PUT'])
def anular_venta(venta_id):
    # Buscar la venta por su ID
    venta = ventas_collection.find_one({'_id': ObjectId(venta_id)})

    if not venta:
        return jsonify({'error': 'La venta no existe.'}), 404

    # Verificar si la venta ya está anulada
    if venta['estado'] == 'anulado':
        return jsonify({'error': 'La venta ya está anulada.'}), 400

    # Cambiar el estado de la venta a 'anulado'
    ventas_collection.update_one(
        {'_id': ObjectId(venta_id)},
        {'$set': {'estado': 'anulado'}}
    )

    # Revertir la cantidad de los productos vendidos
    for producto in venta['productos']:
        producto_id = producto['idProducto']
        cantidad_vendida = producto['cantidadVendida']

        # Buscar el producto en la base de datos
        producto_existente = productos_collection.find_one({'_id': ObjectId(producto_id)})

        if producto_existente:
            # Sumar la cantidad vendida de vuelta a la cantidad actual
            nueva_cantidad = producto_existente['CantidadActual'] + cantidad_vendida

            # Actualizar el producto en la base de datos
            productos_collection.update_one(
                {'_id': ObjectId(producto_id)},
                {'$set': {'CantidadActual': nueva_cantidad}}
            )

    return jsonify({'message': 'La venta ha sido anulada y las cantidades revertidas.'}), 200

# Ruta para obtener todas las ventas
@app.route('/ventas', methods=['GET'])
def obtener_ventas():
    try:
        # Obtener todas las ventas de la colección
        ventas = ventas_collection.find()

        # Convertir las ventas en una lista y serializar ObjectId a string
        lista_ventas = []
        for venta in ventas:
            venta['_id'] = str(venta['_id'])  # Convertir ObjectId a string
            lista_ventas.append(venta)
        
        # Retornar la lista de ventas en formato JSON
        return jsonify(lista_ventas), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ruta para obtener la cantidad total de ventas
@app.route('/ventas/cantidad', methods=['GET'])
def obtener_cantidad_ventas():
    try:
        # Obtener la cantidad total de ventas de la colección
        cantidad_total_ventas = ventas_collection.count_documents({})

        # Retornar la cantidad total de ventas en formato JSON
        return jsonify({'cantidad_total_ventas': cantidad_total_ventas}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Ruta para obtener la suma de precioVentaTotal de todas las ventas
@app.route('/ventas/suma', methods=['GET'])
def obtener_suma_precio_venta_total():
    try:
        # Usar agregación para sumar el campo precioVentaTotal
        resultado = ventas_collection.aggregate([
            {
                '$group': {
                    '_id': None,  # No agrupar por ningún campo específico
                    'total_precio_venta': {'$sum': '$precioVentaTotal'}
                }
            }
        ])

        # Extraer el resultado de la agregación
        suma_precio_venta_total = list(resultado)[0]['total_precio_venta'] if resultado else 0

        # Retornar la suma en formato JSON
        return jsonify({'suma_precio_venta_total': suma_precio_venta_total}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Ruta para obtener los detalles de una venta específica por su ID
@app.route('/ventas/<venta_id>', methods=['GET'])
def obtener_venta_por_id(venta_id):
    try:
        # Buscar la venta por su ID
        venta = ventas_collection.find_one({'_id': ObjectId(venta_id)})

        if not venta:
            return jsonify({'error': 'La venta no existe.'}), 404

        # Convertir el ObjectId a string para el campo _id
        venta['_id'] = str(venta['_id'])

        # Retornar los detalles de la venta en formato JSON
        return jsonify(venta), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ruta para generar un reporte de ventas en PDF agrupado por fecha
@app.route('/ventas/reporte', methods=['GET'])
def generar_reporte_ventas_pdf():
    try:
        # Obtener todas las ventas de la colección
        ventas = ventas_collection.find()

        # Crear un buffer en memoria para generar el PDF
        buffer = BytesIO()

        # Crear el PDF en formato horizontal
        p = canvas.Canvas(buffer, pagesize=landscape(letter))
        p.setTitle("Reporte de Ventas")

        # Agregar título al PDF
        p.setFont("Helvetica-Bold", 16)
        p.drawString(250, 550, "Reporte de Ventas")  # Centrado para formato horizontal

        # Definir la posición inicial para los datos
        x = 50
        y = 500

        # Agrupar las ventas por fecha
        ventas_por_fecha = {}
        for venta in ventas:
            fecha = venta.get('fechaVenta', 'N/A')
            if fecha not in ventas_por_fecha:
                ventas_por_fecha[fecha] = []
            ventas_por_fecha[fecha].append(venta)

        p.setFont("Helvetica", 12)

        # Agregar las ventas al PDF agrupadas por fecha
        for fecha, ventas_del_dia in ventas_por_fecha.items():
            if y < 50:  # Saltar a nueva página si llega al final
                p.showPage()
                y = 500

            # Mostrar la fecha
            p.setFont("Helvetica-Bold", 14)
            p.drawString(x, y, f"Fecha: {fecha}")
            y -= 20

            # Mostrar detalles de las ventas en esa fecha
            p.setFont("Helvetica", 12)
            for venta in ventas_del_dia:
                if y < 50:  # Saltar a nueva página si llega al final
                    p.showPage()
                    y = 500

                venta_id = str(venta.get('_id', 'N/A'))[-4:]  # Mostrar solo los últimos 4 dígitos del ID
                cliente = venta.get('nombreCliente', 'N/A')
                total = venta.get('precioVentaTotal', 'N/A')

                p.drawString(x, y, f"ID: ...{venta_id} | Cliente: {cliente} | Total: {total}")
                y -= 20

        # Finalizar el PDF
        p.showPage()
        p.save()

        # Enviar el archivo PDF como respuesta
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="reporte_ventas.pdf", mimetype='application/pdf')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =================================== INICIO ARQUEO DE CAJA ===============================
arqueo_collection = db['arqueos_caja']
transacciones_collection = db['transacciones_caja']

# # Esquema de una transacción de caja
def transaccion_schema(data):
    return {
        "tipo": data["tipo"],  # 'ingreso' o 'egreso'
        "monto": data["monto"],
        "descripcion": data["descripcion"],
        "fecha": datetime.now(),
        "usuario": data.get("usuario", "desconocido"),
    }

# Esquema para el arqueo de caja (sin detalles de efectivo al abrir)
def arqueo_caja_schema(data):
    return {
        "fecha_inicio": data["fecha_inicio"],
        "fecha_fin": data.get("fecha_fin"),
        "saldo_inicial": data["saldo_inicial"],
        "saldo_final": data.get("saldo_final"),
        "usuario_responsable": data["usuario_responsable"],
        "estado": data.get("estado", "abierto"),  # 'abierto' o 'cerrado'
        "transacciones": [],
    }

def detalles_efectivo_schema(data):
    return {
        "monedas": data.get("monedas", 0),  # Cantidad total de monedas
        "billetes": data.get("billetes", 0),  # Cantidad total de billetes
        "tarjetas": data.get("tarjetas", 0),  # Total procesado con tarjetas
        "otros": data.get("otros", 0),  # Otros medios de pago, si aplica
    }

# Ruta para crear un nuevo arqueo de caja
@app.route('/arqueos_caja', methods=['POST'])
def crear_arqueo():
    data = request.json

    nuevo_arqueo = arqueo_caja_schema({
        "fecha_inicio": datetime.now(),
        "saldo_inicial": data["saldo_inicial"],
        "usuario_responsable": data["usuario_responsable"],
    })

    resultado = arqueo_collection.insert_one(nuevo_arqueo)
    nuevo_arqueo['_id'] = str(resultado.inserted_id)

    return jsonify(nuevo_arqueo), 201

# Ruta para registrar una transacción en el arqueo de caja
@app.route('/arqueos_caja/<arqueo_id>/transacciones', methods=['POST'])
def registrar_transaccion(arqueo_id):
    data = request.json

    arqueo = arqueo_collection.find_one({'_id': ObjectId(arqueo_id)})

    if not arqueo or arqueo['estado'] == 'cerrado':
        return jsonify({'error': 'Arqueo no válido o cerrado.'}), 400

    nueva_transaccion = transaccion_schema(data)

    # Convertir ObjectId a string antes de devolverlo en la respuesta
    arqueo_collection.update_one(
        {'_id': ObjectId(arqueo_id)},
        {'$push': {'transacciones': nueva_transaccion}}
    )

    # Guardar la transacción en la colección de transacciones y convertir ObjectId a string
    transaccion_id = transacciones_collection.insert_one(nueva_transaccion).inserted_id
    nueva_transaccion['_id'] = str(transaccion_id)  # Convertir ObjectId a string

    return jsonify(nueva_transaccion), 201

# Ruta para cerrar el arqueo de caja
@app.route('/arqueos_caja/<arqueo_id>/cerrar', methods=['PUT'])
def cerrar_arqueo(arqueo_id):
    data = request.json  # Datos enviados al cerrar el arqueo

    arqueo = arqueo_collection.find_one({'_id': ObjectId(arqueo_id)})

    if not arqueo or arqueo['estado'] == 'cerrado':
        return jsonify({'error': 'El arqueo ya está cerrado o no existe.'}), 400

    # Convertir valores a números para evitar el error de concatenación
    saldo_inicial = float(arqueo['saldo_inicial'])  # Asegurarte de que sea numérico
    total_ingresos = sum(float(t['monto']) for t in arqueo['transacciones'] if t['tipo'] == 'ingreso')
    total_egresos = sum(float(t['monto']) for t in arqueo['transacciones'] if t['tipo'] == 'egreso')

    # Calcular el saldo final
    saldo_final = saldo_inicial + total_ingresos - total_egresos

    # Verificar que los detalles de efectivo se hayan proporcionado al cerrar
    detalles_efectivo = detalles_efectivo_schema(data["detalles_efectivo"])

    # Calcular el total de efectivo ingresado
    total_efectivo = (
        float(detalles_efectivo.get("monedas", 0)) +
        float(detalles_efectivo.get("billetes", 0)) +
        float(detalles_efectivo.get("tarjetas", 0)) +
        float(detalles_efectivo.get("otros", 0))
    )

    # Verificar si el total de efectivo coincide con el saldo final
    if total_efectivo != saldo_final:
        return jsonify({
            'error': 'El total de efectivo no coincide con el saldo final.',
            'saldo_sistema': saldo_final,  # Mostrar el saldo calculado por el sistema
            'total_efectivo_ingresado': total_efectivo  # Mostrar el efectivo ingresado
        }), 400

    # Actualizar el arqueo con el saldo final, los detalles de efectivo y cerrar el arqueo
    arqueo_collection.update_one(
        {'_id': ObjectId(arqueo_id)},
        {
            '$set': {
                'saldo_final': saldo_final,
                'estado': 'cerrado',
                'fecha_fin': datetime.now(),
                'detalles_efectivo': detalles_efectivo,
            }
        }
    )

    return jsonify({'message': 'El arqueo ha sido cerrado.', 'saldo_final': saldo_final}), 200
# Ruta para obtener el arqueo de caja por ID
@app.route('/arqueos_caja/<arqueo_id>', methods=['GET'])
def obtener_arqueo(arqueo_id):
    arqueo = arqueo_collection.find_one({'_id': ObjectId(arqueo_id)})

    if not arqueo:
        return jsonify({'error': 'El arqueo no existe.'}), 404

    arqueo['_id'] = str(arqueo['_id'])
    return jsonify(arqueo), 200

# Ruta para listar todos los arqueos de caja
@app.route('/arqueos_caja', methods=['GET'])
def listar_arqueos():
    arqueos = list(arqueo_collection.find())
    for arqueo in arqueos:
        arqueo['_id'] = str(arqueo['_id'])
    return jsonify(arqueos), 200

# Ruta para generar un reporte de todos los arqueos de caja en PDF
@app.route('/arqueos_caja/reporte', methods=['GET'])
def generar_reporte_todos_arqueos_pdf():
    try:
        # Obtener todos los arqueos de la colección
        arqueos = arqueo_collection.find()

        # Crear un buffer en memoria para generar el PDF
        buffer = BytesIO()

        # Crear el PDF en formato horizontal
        p = canvas.Canvas(buffer, pagesize=landscape(letter))
        p.setTitle("Reporte de Arqueos de Caja")

        # Agregar título al PDF
        p.setFont("Helvetica-Bold", 16)
        p.drawString(250, 550, "Reporte de Arqueos de Caja")  # Centrado para formato horizontal

        # Definir la posición inicial para los datos
        x = 50
        y = 500

        # Recorrer todos los arqueos y agregar sus detalles al PDF
        for arqueo in arqueos:
            if y < 100:  # Saltar a nueva página si llega al final
                p.showPage()
                y = 500

            p.setFont("Helvetica-Bold", 14)
            p.drawString(x, y, f"ID del Arqueo: {str(arqueo['_id'])[-4:]}")
            y -= 20
            p.drawString(x, y, f"Usuario Responsable: {arqueo.get('usuario_responsable', 'N/A')}")
            y -= 20
            p.drawString(x, y, f"Estado: {arqueo.get('estado', 'N/A')}")
            y -= 20
            p.drawString(x, y, f"Fecha de Inicio: {arqueo.get('fecha_inicio', 'N/A')}")
            y -= 20
            p.drawString(x, y, f"Fecha de Fin: {arqueo.get('fecha_fin', 'N/A')}")
            y -= 20
            p.drawString(x, y, f"Saldo Inicial: {arqueo.get('saldo_inicial', 'N/A')}")
            y -= 20
            p.drawString(x, y, f"Saldo Final: {arqueo.get('saldo_final', 'N/A')}")
            y -= 30

            # Detalles de efectivo
            detalles_efectivo = arqueo.get('detalles_efectivo', {})
            p.setFont("Helvetica", 12)
            p.drawString(x, y, "--- Detalles de Efectivo ---")
            y -= 20
            p.drawString(x, y, f"Billetes: {detalles_efectivo.get('billetes', 0)}")
            y -= 20
            p.drawString(x, y, f"Monedas: {detalles_efectivo.get('monedas', 0)}")
            y -= 20
            p.drawString(x, y, f"Otros: {detalles_efectivo.get('otros', 0)}")
            y -= 20
            p.drawString(x, y, f"Tarjetas: {detalles_efectivo.get('tarjetas', 0)}")
            y -= 30

            # Detalles de las transacciones
            transacciones = arqueo.get('transacciones', [])
            if transacciones:
                p.drawString(x, y, "--- Transacciones ---")
                y -= 20
                for transaccion in transacciones:
                    if y < 100:  # Saltar a nueva página si llega al final
                        p.showPage()
                        y = 500

                    descripcion = transaccion.get('descripcion', 'N/A')
                    fecha = transaccion.get('fecha', 'N/A')
                    monto = transaccion.get('monto', 'N/A')
                    tipo = transaccion.get('tipo', 'N/A')
                    usuario = transaccion.get('usuario', 'N/A')

                    p.drawString(x, y, f"Descripción: {descripcion} | Fecha: {fecha} | Monto: {monto} | Tipo: {tipo} | Usuario: {usuario}")
                    y -= 20

            y -= 30  # Espacio entre arqueos

        # Finalizar el PDF
        p.showPage()
        p.save()

        # Enviar el archivo PDF como respuesta
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="reporte_arqueos_caja.pdf", mimetype='application/pdf')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===================================FIN DE ARQUEO CAJA ===================================        
# ====================================INICIO DE CLIENTE ====================================
# Conexión a la colección de clientes
clientes_collection = db['clientes']

# Función para validar los datos del cliente
def validar_cliente(data):
    required_fields = ['nombreCliente', 'rucCliente', 'telefonoCliente']

    for field in required_fields:
        if field not in data or not isinstance(data[field], str):
            return False, f"El campo '{field}' es obligatorio y debe ser un string."
    return True, ""
# Ruta para crear un cliente
@app.route('/clientes', methods=['POST'])
def crear_cliente():
    if request.is_json:
        data = request.json
    else:
        data = request.form.to_dict()

    # Validar los campos del cliente
    valido, mensaje = validar_cliente(data)
    if not valido:
        return jsonify({'error': mensaje}), 400

    # Crear el nuevo cliente
    nuevo_cliente = {
        'nombreCliente': data['nombreCliente'],
        'rucCliente': data['rucCliente'],
        'telefonoCliente': data['telefonoCliente']
    }

    # Insertar el cliente en la base de datos
    resultado = clientes_collection.insert_one(nuevo_cliente)
    nuevo_cliente['_id'] = str(resultado.inserted_id)
    return jsonify(nuevo_cliente), 201
# Ruta para obtener todos los clientes
@app.route('/clientes', methods=['GET'])
def obtener_clientes():
    try:
        # Obtener todos los clientes de la colección
        clientes = list(clientes_collection.find())

        # Convertir los ObjectId a string para enviar en la respuesta JSON
        for cliente in clientes:
            cliente['_id'] = str(cliente['_id'])

        return jsonify(clientes), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# ======================INICIO EMPRESAS ======================
# Conexión a la colección de empresas
empresas_collection = db['empresas']

# Función para validar los datos de la empresa
def validar_empresa(data):
    required_fields = ['nombreEmpresa', 'rucEmpresa', 'direccionEmpresa', 'timbradoEmpresa']

    for field in required_fields:
        if field not in data or not isinstance(data[field], str):
            return False, f"El campo '{field}' es obligatorio y debe ser un string."
    return True, ""
# Ruta para crear una empresa
@app.route('/empresas', methods=['POST'])
def crear_empresa():
    if request.is_json:
        data = request.json
    else:
        data = request.form.to_dict()

    # Validar los campos de la empresa
    valido, mensaje = validar_empresa(data)
    if not valido:
        return jsonify({'error': mensaje}), 400

    # Crear la nueva empresa con estado 'activa'
    nueva_empresa = {
        'nombreEmpresa': data['nombreEmpresa'],
        'rucEmpresa': data['rucEmpresa'],
        'direccionEmpresa': data['direccionEmpresa'],
        'timbradoEmpresa': data['timbradoEmpresa'],
        'estado': 'activa'  # Estado inicial de la empresa
    }

    # Insertar la empresa en la base de datos
    resultado = empresas_collection.insert_one(nueva_empresa)
    nueva_empresa['_id'] = str(resultado.inserted_id)
    return jsonify(nueva_empresa), 201
    
# Ruta para obtener todas las empresas
@app.route('/empresas', methods=['GET'])
def obtener_empresas():
    try:
        # Obtener todas las empresas de la colección
        empresas = list(empresas_collection.find())

        # Convertir los ObjectId a string para enviar en la respuesta JSON
        for empresa in empresas:
            empresa['_id'] = str(empresa['_id'])

        return jsonify(empresas), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ruta para obtener la empresa activa más reciente
@app.route('/empresas/activas', methods=['GET'])
def obtener_empresa_activa_mas_reciente():
    try:
        # Obtener solo la empresa activa más reciente, ordenada por fecha de creación o actualización
        empresa_activa = empresas_collection.find_one({"estado": "activa"}, sort=[('fechaActualizacion', -1)])

        # Convertir el ObjectId a string para enviar en la respuesta JSON
        if empresa_activa:
            empresa_activa['_id'] = str(empresa_activa['_id'])

        return jsonify(empresa_activa), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

 # Ruta para anular una empresa (cambiar estado a 'anulada')
@app.route('/empresas/anular/<id>', methods=['PUT'])
def anular_empresa(id):
    try:
        # Intentar convertir el id en un ObjectId
        empresa_id = ObjectId(id)

        # Actualizar el estado de la empresa a 'anulada'
        result = empresas_collection.update_one(
            {'_id': empresa_id},
            {'$set': {'estado': 'anulada'}}
        )

        if result.matched_count == 0:
            return jsonify({'error': 'Empresa no encontrada'}), 404

        return jsonify({'message': 'Empresa anulada con éxito'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500       
# ================= INICIO PROVEEDOR =======================

proveedores_collection = db['proveedores']

# Función para validar los datos del proveedor
def validar_proveedor(data):
    required_fields = ['nombreProveedor', 'rucProveedor', 'direccionProveedor', 'telefonoProveedor']
    for field in required_fields:
        if field not in data or not isinstance(data[field], str):
            return False, f"El campo '{field}' es obligatorio y debe ser un string."
    return True, ""

# Ruta para crear un proveedor
@app.route('/proveedores', methods=['POST'])
def crear_proveedor():
    if request.is_json:
        data = request.json
    else:
        data = request.form.to_dict()

    # Validar los campos del proveedor
    valido, mensaje = validar_proveedor(data)
    if not valido:
        return jsonify({'error': mensaje}), 400

    # Crear el nuevo proveedor
    nuevo_proveedor = {
        'nombreProveedor': data['nombreProveedor'],
        'rucProveedor': data['rucProveedor'],
        'direccionProveedor': data['direccionProveedor'],
        'telefonoProveedor': data['telefonoProveedor'],
        'estado': 'activo'  # Estado por defecto
    }

    # Insertar el proveedor en la base de datos
    resultado = proveedores_collection.insert_one(nuevo_proveedor)
    nuevo_proveedor['_id'] = str(resultado.inserted_id)
    return jsonify(nuevo_proveedor), 201

# Ruta para obtener todos los proveedores
@app.route('/proveedores', methods=['GET'])
def obtener_proveedores():
    try:
        # Obtener todos los proveedores de la colección
        proveedores = list(proveedores_collection.find())

        # Convertir los ObjectId a string para enviar en la respuesta JSON
        for proveedor in proveedores:
            proveedor['_id'] = str(proveedor['_id'])

        return jsonify(proveedores), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Ruta para obtener solo los proveedores con estado activo
@app.route('/proveedores/activos', methods=['GET'])
def obtener_proveedores_activos():
    try:
        # Obtener solo los proveedores que tienen estado 'activo'
        proveedores_activos = list(proveedores_collection.find({'estado': 'activo'}))

        # Convertir los ObjectId a string para enviar en la respuesta JSON
        for proveedor in proveedores_activos:
            proveedor['_id'] = str(proveedor['_id'])

        # Devolver un array vacío si no hay proveedores activos
        return jsonify(proveedores_activos if proveedores_activos else []), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# Ruta para obtener solo los proveedores con estado anulado
@app.route('/proveedores/anulados', methods=['GET'])
def obtener_proveedores_anulados():
    try:
        # Obtener solo los proveedores que tienen estado 'anulado'
        proveedores_anulados = list(proveedores_collection.find({'estado': 'anulado'}))

        # Convertir los ObjectId a string para enviar en la respuesta JSON
        for proveedor in proveedores_anulados:
            proveedor['_id'] = str(proveedor['_id'])

        # Devolver un array vacío si no hay proveedores anulados
        return jsonify(proveedores_anulados if proveedores_anulados else []), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# Ruta para anular un proveedor
@app.route('/proveedores/anular/<proveedor_id>', methods=['PUT'])
def anular_proveedor(proveedor_id):
    # Buscar el proveedor por su ID
    proveedor = proveedores_collection.find_one({'_id': ObjectId(proveedor_id)})

    if not proveedor:
        return jsonify({'error': 'El proveedor no existe.'}), 404

    # Verificar si el proveedor ya está anulado
    if proveedor['estado'] == 'anulado':
        return jsonify({'error': 'El proveedor ya está anulado.'}), 400

    # Cambiar el estado del proveedor a 'anulado'
    proveedores_collection.update_one(
        {'_id': ObjectId(proveedor_id)},
        {'$set': {'estado': 'anulado'}}
    )

    return jsonify({'message': 'El proveedor ha sido anulado exitosamente.'}), 200

# =================== INICIO CATEGORIAS =====================
# Conexión a la colección de categorías
# Conexión a la colección de categorías
categorias_collection = db['categorias']

# Modificación en la validación y creación de categoría
def validar_categoria(data):
    if 'nombreCategoria' not in data or not isinstance(data['nombreCategoria'], str):
        return False, "El campo 'nombreCategoria' es obligatorio y debe ser un string."
    return True, ""

# Ruta para crear una categoría con estado 'activo'
@app.route('/categorias', methods=['POST'])
def crear_categoria():
    if request.is_json:
        data = request.json
    else:
        data = request.form.to_dict()

    # Validar los campos de la categoría
    valido, mensaje = validar_categoria(data)
    if not valido:
        return jsonify({'error': mensaje}), 400

    # Verificar si la categoría ya existe
    categoria_existente = categorias_collection.find_one({'nombreCategoria': data['nombreCategoria']})
    if categoria_existente:
        return jsonify({'error': 'La categoría con este nombre ya existe.'}), 400

    # Crear la nueva categoría con estado 'activo'
    nueva_categoria = {
        'nombreCategoria': data['nombreCategoria'],
        'estado': 'activo'  # Estado por defecto al crear
    }

    # Insertar la categoría en la base de datos
    resultado = categorias_collection.insert_one(nueva_categoria)
    nueva_categoria['_id'] = str(resultado.inserted_id)
    return jsonify(nueva_categoria), 201

# Ruta para obtener todas las categorías
@app.route('/categorias', methods=['GET'])
def obtener_categorias():
    try:
        # Obtener todas las categorías de la colección
        categorias = list(categorias_collection.find())
        
        # Convertir los ObjectId a string para poder enviar en la respuesta JSON
        for categoria in categorias:
            categoria['_id'] = str(categoria['_id'])
        
        return jsonify(categorias), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ruta para anular una categoría
@app.route('/categorias/anular/<categoria_id>', methods=['PUT'])
def anular_categoria(categoria_id):
    # Buscar la categoría por su ID
    categoria = categorias_collection.find_one({'_id': ObjectId(categoria_id)})

    if not categoria:
        return jsonify({'error': 'La categoría no existe.'}), 404

    # Verificar si la categoría ya está anulada
    if categoria['estado'] == 'anulado':
        return jsonify({'error': 'La categoría ya está anulada.'}), 400

    # Cambiar el estado de la categoría a 'anulado'
    categorias_collection.update_one(
        {'_id': ObjectId(categoria_id)},
        {'$set': {'estado': 'anulado'}}
    )

    return jsonify({'message': 'La categoría ha sido anulada exitosamente.'}), 200
# Ruta para obtener todas las categorías activas
@app.route('/categorias/activas', methods=['GET'])
def obtener_categorias_activas():
    try:
        # Obtener solo las categorías con estado 'activo'
        categorias_activas = list(categorias_collection.find({'estado': 'activo'}))
        
        # Convertir los ObjectId a string para poder enviar en la respuesta JSON
        for categoria in categorias_activas:
            categoria['_id'] = str(categoria['_id'])
        
        return jsonify(categorias_activas), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ruta para obtener todas las categorías anuladas
@app.route('/categorias/anuladas', methods=['GET'])
def obtener_categorias_anuladas():
    try:
        # Obtener solo las categorías con estado 'anulado'
        categorias_anuladas = list(categorias_collection.find({'estado': 'anulado'}))
        
        # Convertir los ObjectId a string para poder enviar en la respuesta JSON
        for categoria in categorias_anuladas:
            categoria['_id'] = str(categoria['_id'])
        
        return jsonify(categorias_anuladas), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
