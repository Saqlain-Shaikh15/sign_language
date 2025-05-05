from flask import Flask, render_template, redirect, url_for
import subprocess
import sys

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/isl')
def run_isl():
    subprocess.Popen([sys.executable, "app_isl.py"])
    return redirect(url_for('home'))

@app.route('/asl')
def run_asl():
    subprocess.Popen([sys.executable, "app_asl.py"])
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
