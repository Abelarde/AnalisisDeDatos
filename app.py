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
            {'id': 'grados', 'nombre': 'Grados', 'valorActual': "6"},
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




@app.route("/obtenerPDF/<archivo>", methods=["GET"])
def reports(archivo):
    if request.method == "GET":
        workingdir = os.path.abspath(os.getcwd())
        filepath = workingdir + '/pdfs/'
        return send_from_directory(filepath, archivo)


@app.route("/getCampos", methods=["POST"])
def cargarCampos():
    if request.method == "POST":
        nombreArchivo = request.form["archivo"]
        return jsonify(obtenerEncabezados(nombreArchivo))


@app.route("/analisis", methods=["POST"])
def analisis():
    if request.method == "POST":
        codigoAnalisis = request.form["tipoAnalisis"]
        archivoAnalisis = request.form["archivoAnalisis"]
        tipoAnalisis = request.form["tipoAnalisis"]
        tipoRegresion = request.form["tipoRegresion"]
        if (
                codigoAnalisis == '1' or codigoAnalisis == '2' or codigoAnalisis == '6' or codigoAnalisis == '9' or codigoAnalisis == '13'):
            pais = request.form["nombrePais"]
            titulo = request.form["titulo"]
            feature = request.form["feature"]
            infecciones = request.form["etiquetaInfecciones"]
            etiquetaPais = request.form["etiquetaPais"]
            # predicciones = str(request.form.getlist("valoresPredecidos")).split(",")
            predicciones = request.form.getlist("valoresPredecidos")
            predicciones = predicciones[0]
            # (archivo, pais, infecciones, etiquetaPais, predicciones)
            if (tipoRegresion == '1'):
                resultados = TendenciaInfeccionLineal(archivoAnalisis, pais, infecciones, etiquetaPais, feature,
                                                      predicciones, titulo)
                return jsonify(resultados)
            if (tipoRegresion == '2' or tipoRegresion == '0'):
                grados = int(request.form["grados"])
                resultados = TendenciaInfeccionRegresionPolinomial(archivoAnalisis, pais, infecciones, etiquetaPais,
                                                                   feature, predicciones, grados, titulo)
                return jsonify(resultados)
        if (codigoAnalisis == '3'):
            pais = request.form["nombrePais"]
            titulo = request.form["titulo"]
            feature = request.form["feature"]
            infecciones = request.form["etiquetaInfecciones"]
            etiquetaPais = request.form["etiquetaPais"]
            # predicciones = str(request.form.getlist("valoresPredecidos")).split(",")
            predicciones = request.form.getlist("valoresPredecidos")
            predicciones = predicciones[0]
            # (archivo, pais, infecciones, etiquetaPais, predicciones)
            if (tipoRegresion == '1'):
                resultados = IndiceProgresion(archivoAnalisis, pais, infecciones, etiquetaPais, feature, predicciones,
                                              titulo)
                return jsonify(resultados)
            if (tipoRegresion == '2' or tipoRegresion == '0'):
                grados = int(request.form["grados"])
                resultados = IndiceProgresion(archivoAnalisis, pais, infecciones, etiquetaPais, feature, predicciones,
                                              grados, titulo)
                return jsonify(resultados)

        if (codigoAnalisis == '4'):
            titulo = request.form["titulo"]
            feature = request.form["feature"]
            infecciones = request.form["etiquetaMortalidad"]
            etiquetaDepartamento = request.form["etiquetaDepartamento"]
            departamento = request.form["departamento"]
            # predicciones = str(request.form.getlist("valoresPredecidos")).split(",")
            predicciones = request.form.getlist("valoresPredecidos")
            predicciones = predicciones[0]
            etiquetaMunicipio = request.form["etiquetaMunicipio"]
            municipio = request.form["municipio"]
            # (archivo, pais, infecciones, etiquetaPais, predicciones)
            if (tipoRegresion == '1'):
                resultados = prediccionMortandadDepartamento(archivoAnalisis, departamento, etiquetaMunicipio,
                                                             municipio, infecciones, etiquetaDepartamento, feature,
                                                             predicciones, titulo)
                return jsonify(resultados)
            if (tipoRegresion == '2' or tipoRegresion == '0'):
                grados = int(request.form["grados"])
                resultados = prediccionMortandadDepartamentoPoli(archivoAnalisis, departamento, etiquetaMunicipio,
                                                                 municipio, infecciones, etiquetaDepartamento, feature,
                                                                 predicciones, grados, titulo)
                return jsonify(resultados)
                # archivo, pais, infecciones, etiquetaPais, predicciones =[]

        if (codigoAnalisis == '5' or codigoAnalisis == '7' or codigoAnalisis == '22' or codigoAnalisis == '24'):
            pais = request.form["nombrePais"]
            titulo = request.form["titulo"]
            feature = request.form["feature"]
            infecciones = request.form["etiquetaInfecciones"]
            etiquetaPais = request.form["etiquetaPais"]
            etiquetaMuertes = request.form["etiquetaMuertes"]
            # predicciones = str(request.form.getlist("valoresPredecidos")).split(",")
            predicciones = request.form.getlist("valoresPredecidos")
            predicciones = predicciones[0]
            # (archivo, pais, infecciones, etiquetaPais, predicciones)
            if (tipoRegresion == '1'):
                resultados = ReporteLinea05(archivoAnalisis, pais, infecciones, etiquetaMuertes, etiquetaPais, feature,
                                            predicciones, titulo)
                return jsonify(resultados)
            if (tipoRegresion == '2' or tipoRegresion == '0'):
                grados = int(request.form["grados"])
                resultados = ReporteLinea05(archivoAnalisis, pais, infecciones, etiquetaMuertes, etiquetaPais, feature,
                                            predicciones, grados, titulo)
                return jsonify(resultados)
        if (codigoAnalisis == '8'):
            pais = request.form["nombrePais"]
            titulo = request.form["titulo"]
            feature = request.form["feature"]
            infecciones = request.form["etiquetaInfecciones"]
            etiquetaPais = request.form["etiquetaPais"]
            # predicciones = str(request.form.getlist("valoresPredecidos")).split(",")
            predicciones = request.form.getlist("valoresPredecidos")
            predicciones = predicciones[0]
            # (archivo, pais, infecciones, etiquetaPais, predicciones)
            if (tipoRegresion == '1'):
                resultados = PrediccionCasosAnio(archivoAnalisis, pais, infecciones, etiquetaPais, feature,
                                                 predicciones, titulo)
                return jsonify(resultados)
            if (tipoRegresion == '2' or tipoRegresion == '0'):
                grados = int(request.form["grados"])
                resultados = PrediccionCasosAnioPolinomial(archivoAnalisis, pais, infecciones, etiquetaPais, feature,
                                                           predicciones, grados, titulo)
                return jsonify(resultados)

        if (codigoAnalisis == '17'):
            pais = request.form["nombrePais"]
            titulo = request.form["titulo"]
            feature = request.form["feature"]
            infecciones = request.form["etiquetaInfecciones"]
            etiquetaPais = request.form["etiquetaPais"]
            etiquetaMuertes = request.form["etiquetaMuertes"]
            # predicciones = str(request.form.getlist("valoresPredecidos")).split(",")
            predicciones = request.form.getlist("valoresPredecidos")
            predicciones = predicciones[0]
            # (archivo, pais, infecciones, etiquetaPais, predicciones)
            if (tipoRegresion == '1'):
                resultados = ReporteLinea17(archivoAnalisis, pais, infecciones, etiquetaMuertes, etiquetaPais, feature,
                                            predicciones, titulo)
                return jsonify(resultados)
            if (tipoRegresion == '2' or tipoRegresion == '0'):
                grados = int(request.form["grados"])
                resultados = ReporteLinea17(archivoAnalisis, pais, infecciones, etiquetaMuertes, etiquetaPais, feature,
                                            predicciones, grados, titulo)
                return jsonify(resultados)
        if (codigoAnalisis == '8'):
            pais = request.form["nombrePais"]
            titulo = request.form["titulo"]
            feature = request.form["feature"]
            infecciones = request.form["etiquetaInfecciones"]
            etiquetaPais = request.form["etiquetaPais"]
            # predicciones = str(request.form.getlist("valoresPredecidos")).split(",")
            predicciones = request.form.getlist("valoresPredecidos")
            predicciones = predicciones[0]
            # (archivo, pais, infecciones, etiquetaPais, predicciones)
            if (tipoRegresion == '1'):
                resultados = PrediccionCasosAnio(archivoAnalisis, pais, infecciones, etiquetaPais, feature,
                                                 predicciones, titulo)
                return jsonify(resultados)
            if (tipoRegresion == '2' or tipoRegresion == '0'):
                grados = int(request.form["grados"])
                resultados = PrediccionCasosAnioPolinomial(archivoAnalisis, pais, infecciones, etiquetaPais, feature,
                                                           predicciones, grados, titulo)
                return jsonify(resultados)

        if (codigoAnalisis == '11'):
            pais = request.form["nombrePais"]
            titulo = request.form["titulo"]
            feature = request.form["feature"]
            infecciones = request.form["etiquetaInfecciones"]
            etiquetaPais = request.form["etiquetaPais"]
            # predicciones = str(request.form.getlist("valoresPredecidos")).split(",")
            predicciones = request.form.getlist("valoresPredecidos")
            predicciones = predicciones[0]
            # (archivo, pais, infecciones, etiquetaPais, predicciones)
            if (tipoRegresion == '1'):
                grados = int(request.form["grados"])
                resultados = PorcentajeInfectadosPolinomial(archivoAnalisis, pais, infecciones, etiquetaPais, feature,
                                                            predicciones, grados, titulo)
                # resultados = PorcentajeInfectadosPolinomial(archivoAnalisis, pais, infecciones, etiquetaPais, feature, predicciones, titulo)
                return jsonify(resultados)
            if (tipoRegresion == '2' or tipoRegresion == '0'):
                grados = int(request.form["grados"])
                resultados = PorcentajeInfectadosPolinomial(archivoAnalisis, pais, infecciones, etiquetaPais, feature,
                                                            predicciones, grados, titulo)
                # resultados = TendenciaInfeccionRegresionPolinomial(archivoAnalisis, pais, infecciones, etiquetaPais, feature, predicciones, grados ,titulo)
                return jsonify(resultados)
    return jsonify({"codigo": 400})

@app.route("/descargar", methods=["POST"])
def descargar():
    return redirect("/static/ast.gv.pdf")


if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug = True, host = "0.0.0.0", port = 5000)