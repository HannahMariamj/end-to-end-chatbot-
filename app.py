from flask import Flask, render_template,request
from load_model_for_inference import model, tokenizer
from rag_module import final_result

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    result=final_result(input, model, tokenizer)
    print("Response: ",result)
    return str(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)