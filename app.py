from flask import Flask, redirect, render_template, request, jsonify
from flask import send_from_directory, current_app as app
from werkzeug.utils import secure_filename
from analisis import *

UPLOAD_FOLDER = '/home/eduardo/Downloads/AnalisisDeDatos/archivos/'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'json'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def obtenerListaArchivos():
    lista = os.listdir(os.path.join(app.config['UPLOAD_FOLDER']))
    return lista
def obtenerEncabezados(file):
    try:
        if '.csv' in file:  # El archivo es un csv
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER']) + file)
            encabezados = df.columns.values.tolist()
            # return encabezados.to_json(orient="records")
            return {
                "encabezados": encabezados,
                "codigo": 100,
                "mensaje": "OK"
            }

        if '.xlsx' in file:  # El archivo es un excel
            df = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER']) + file)
            encabezados = df.columns.values.tolist()
            # return encabezados.to_json(orient="records")
            return {
                "encabezados": encabezados,
                "codigo": 100,
                "mensaje": "OK"
            }

        if '.xlsx' in file:  # El archivo es un excel
            df = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER']) + file)
            encabezados = df.columns.values.tolist()
            # return encabezados.to_json(orient="records")
            return {
                "encabezados": encabezados,
                "codigo": 100,
                "mensaje": "OK"
            }

        if '.json' in file:  # El archivo es un excel
            df = pd.read_json(os.path.join(app.config['UPLOAD_FOLDER']) + file)
            encabezados = df.columns.values.tolist()
            # return encabezados.to_json(orient="records")
            return {
                "encabezados": encabezados,
                "codigo": 100,
                "mensaje": "OK"
            }
    except Exception as e:
        return {
            "mensaje": str(e),
            "codigo": 666
        }
def obtenerParametros(option):
    if option == '1':
        parametros = [
            {'id': 'titulo', 'nombre': 'Título reporte', 'valorActual': "Regresion Lineal"},
            {'id': 'featureX', 'nombre': 'Feature (X)', 'valorActual': "---"},
            {'id': 'featureY', 'nombre': 'Feature (Y)', 'valorActual': "---"}

        ]
        return parametros
    if option == '2':
        parametros = [
            {'id': 'titulo', 'nombre': 'Título reporte', 'valorActual': "Regresion Polinomial"},
            {'id': 'grados', 'nombre': 'Grados', 'valorActual': "3"},
            {'id': 'featureX', 'nombre': 'Feature (X)', 'valorActual': "---"},
            {'id': 'featureY', 'nombre': 'Feature (Y)', 'valorActual': "---"}

        ]
        return parametros
    if option == '3':
        parametros = [
            {'id': 'titulo', 'nombre': 'Título reporte', 'valorActual': "Clasificador Gaussiano"},
            {'id': 'featureX', 'nombre': 'Feature (X)', 'valorActual': "---"},
            {'id': 'featureY', 'nombre': 'Feature (Y)', 'valorActual': "---"}

        ]
        return parametros
    if option == '4':
        parametros = [
            {'id': 'titulo', 'nombre': 'Título reporte', 'valorActual': "Clasificador de arboles de decision"},
            {'id': 'featureX', 'nombre': 'Feature (X)', 'valorActual': "---"},
            {'id': 'featureY', 'nombre': 'Feature (Y)', 'valorActual': "---"}

        ]
        return parametros
    if option == '5':
        parametros = [
            {'id': 'titulo', 'nombre': 'Título reporte', 'valorActual': "Redes neuronales"},
            {'id': 'featureX', 'nombre': 'Feature (X)', 'valorActual': "---"},
            {'id': 'featureY', 'nombre': 'Feature (Y)', 'valorActual': "---"}

        ]
        return parametros

@app.route("/")
def home():
    return render_template('index.html', listaArchivos=obtenerListaArchivos())
@app.route("/menu")
def menu():
    return render_template("menu.html", listaArchivos=obtenerListaArchivos())
@app.route("/getParametros", methods=["POST", "GET"])
def getParametros():
    option = '1'
    if request.method == "POST":
        option = request.form["option"]
    return jsonify(obtenerParametros(option))
@app.route("/cargarArchivo", methods=["POST"])
def cargarArchivoEntrada():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"codigo": 100, "mensaje": "No se ha encontadro el archivo."})
        file = request.files['file']
        if file.filename == '':
            return jsonify({"codigo": 100, "mensaje": "No se ha seleccionado el archivo."})
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({"codigo": 200, "mensaje": "Archivo " + file.filename + " almacenado y cargado correctamente.",
                        "archivos": obtenerListaArchivos(), "archivo": file.filename})
@app.route("/getCampos", methods=["POST"])
def cargarCampos():
    if request.method == "POST":
        nombreArchivo = request.form["archivo"]
        return jsonify(obtenerEncabezados(nombreArchivo))
@app.route("/analisis", methods=["POST"])
def analisis():
    if request.method == "POST":
        archivoAnalisis = request.form["archivoAnalisis"]
        codigoAnalisis = request.form["tipoAnalisis"]

        x_name = request.form["featureX"]
        y_name = request.form["featureY"]
        titulo = request.form["titulo"]

        if codigoAnalisis == '1' or codigoAnalisis == '2':
            grados = 0
            if codigoAnalisis == '2':
                grados = int(request.form["grados"])

            return jsonify(polinomial(x_name, y_name, archivoAnalisis, grados, titulo))

        if codigoAnalisis == '3': return jsonify(gaussiano(x_name, y_name, archivoAnalisis, titulo))

        if codigoAnalisis == '4': return jsonify(arbol(x_name, y_name, archivoAnalisis, titulo))

        if codigoAnalisis == '5': return jsonify(redesBien(x_name, y_name, archivoAnalisis, titulo))

    return jsonify({"codigo": 400})




@app.route("/obtenerPDF/<archivo>", methods=["GET"])
def reports(archivo):
    if request.method == "GET":
        workingdir = os.path.abspath(os.getcwd())
        filepath = workingdir + '/pdfs/'
        return send_from_directory(filepath, archivo)
@app.route("/descargar", methods=["POST"])
def descargar():
    return redirect("/static/ast.gv.pdf")


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug = True, host = "0.0.0.0", port = 5000)