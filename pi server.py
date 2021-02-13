from flask import Flask
#, render_template,request,redirect,url_for
import RPi.GPIO as GPIO
from time import sleep


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)




rp, gp, bp = 17, 27 , 22
GPIO.setup(rp, GPIO.OUT)
GPIO.setup(gp, GPIO.OUT)
GPIO.setup(bp, GPIO.OUT)


r=GPIO.PWM(rp,1000)

g=GPIO.PWM(gp,1000)

b=GPIO.PWM(bp,1000)

r.start(0)
g.start(0)
b.start(0)

select =[r,g,b]

def reset():
    r.ChangeDutyCycle(0)
    g.ChangeDutyCycle(0)
    b.ChangeDutyCycle(0)
app = Flask(__name__)


import logging
log = logging.getLogger('werkzeug')
log.disabled = True
app.logger.disabled = True

@app.route("/f/<x>")
def do_forward(x):
    reset()
    m = int(x[0])
    z = int(((int(x[1:])/255)*100))
    #print(m,z)
    if m ==0:
        r.ChangeDutyCycle(z)
    elif m == 1:
        g.ChangeDutyCycle(z)
    elif m == 2:
        b.ChangeDutyCycle(z)
    

    return ("", 204)

@app.route("/")
def hello():
    r.ChangeDutyCycle(100)
    g.ChangeDutyCycle(100)
    b.ChangeDutyCycle(100)
    return ("", 204)

    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)