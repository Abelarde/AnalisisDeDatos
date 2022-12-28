from flask import Flask, redirect, url_for, render_template, request
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/analisis", methods=["POST","GET"])
def analisis():
    #if request.method == "POST":
    #    inpt = request.form["inpt"];
    #    global tmp_val
    #    tmp_val=inpt
    #    return redirect(url_for("analisis"))
    #else:
    #   if tmp_val == '':
    #        return render_template('analisis.html', initial='', input='')
    #    genAux = Codigo3D()
    #    genAux.cleanAll()
    #    gen = genAux.getInstance()
    #    result = parse(tmp_val)
    #    global parsed_tree
    #    parsed_tree = result
    #    return render_template('analisis.html', initial=tmp_val, input=gen.getCode())
    return render_template('index.html')


@app.route('/reporte')
def reporte():
    return render_template('reporte.html')

@app.route('/reporte/simbolo')
def simbolo():
    #dig = graphviz.Source(parsed_tree.getDotTable(parsed_tree.AST))
    #chart_output = dig.pipe(format='svg')
    #chart_output = base64.b64encode(chart_output).decode('utf-8')
    #return render_template('reporte.html', chart=chart_output)
    return render_template('index.html')

@app.route('/reporte/error')
def error():
    #dig = graphviz.Source(parsed_tree.getDotErr(parsed_tree.AST))
    #chart_output = dig.pipe(format='svg')
    #chart_output = base64.b64encode(chart_output).decode('utf-8')
    #return render_template('reporte.html', chart=chart_output)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
