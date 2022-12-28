// referencia a codemirror: https://codemirror.net/doc/manual.html
CodeMirror.fromTextArea(document.getElementById("default"),{
    lineNumbers : true,
    mode: "julia",
    theme : "seti",
    matchBrackets: true
});

CodeMirror.fromTextArea(document.getElementById("output"),{
    lineNumbers : true,
    mode: "go",
    theme : "seti",
    readOnly: true,
    matchBrackets: true
});