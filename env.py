from selenium import webdriver
import cv2
import psutil
import subprocess
import os
import math
import subprocess as sp
from collections import deque
import time
import numpy as np
"""
def funcdelay(cnframes,nfps):
    def F(func):
        global funcdelay_functions

        try: 
            df = funcdelay_functions
        except:
            df = funcdelay_functions = {}

        def wrapper(*args, **kwargs):
            try:
                d = df[func]
            except: d = deque(maxlen = cnframes)

            while len(d) < d.maxlen:
                d.append(func(*args, **kwargs))
                time.sleep(1 / nfps)

            return d.popleft()

        return wrapper
    return F
    document.querySelector('body > div:nth-child(18) > span:nth-child(1) > span:nth-child(2)').innerText
"""

def getXid(c):
    c.service.process # is a Popen instance for the chromedriver process
    p = psutil.Process(c.service.process.pid)

    pids = tuple( [ str(x.pid) for x in p.children(recursive=True)] )

    plist =  "|".join(pids)
    ot = os.popen("wmctrl -lp | grep -E \"" + plist + '" | sed "s/\s.*//"')
    return min( ot.read().split() )


class environment:
    def __init__(self):
        self.fshape=(64,64,1)

        self.fpower = np.prod(self.fshape)
        self.driver = webdriver.Chrome()
        self.points = 0;
        self.xid = getXid(self.driver)
        print(self.xid)
        if not self.xid :
            raise ("cannot get a xid")

        cv2.startWindowThread()
        cv2.namedWindow("slither-prev")
        self.driver.set_window_size(600, 600)
        self.initpage()
        self.__start_capture()

    def initpage(self):

        self.driver.get("http://www.slither.io")
        self.xpath = xpath = self.driver.find_element_by_xpath
        self.xp_scoreframe = "/html/body/div[13]/span[1]/span[2]"
        self.startbutton = xpath('//*[@id="playh"]/div/div/div[3]')
        self.loginframe = xpath('//*[@id="login"]')
        self.scoreframe = None

        self.driver.execute_script("""
            window.sl_acctime = 0;
            setInterval(()=>{
                if(sl_acctime > 0) sl_acctime -= 0.33;
                else if(sl_acctime < 0) {
                    setAcceleration(0);
                    sl_acctime = 0;
                }
            },333);

            window.sl_accel = () => {
                setAcceleration(1);
                sl_acctime = 0.2;
            } ;
            window.a = null;
            window.getscore = () => {
                try {
                    if (document.body && document.body.children && document.body.children[1].style["display"] == "inline")
                        return "0";
                    if ( !a || !('children' in a) )
                        a = document.querySelector('body > div:nth-child(18)');
                    if ( a && 'children' in a )
                        return a.children[0].children[1].innerText;
                    return "0";
                } catch(error) { console.error(error); return '0'; }
            }
        var old_accel = window.setAcceleration
        window.issnakeaccelerated = 0
        window.setAcceleration = (b) => { old_accel(b) ; window.issnakeaccelerated=b } 
        """)

    def __start_capture(self):
            self.ffmpeg = cv2.VideoCapture(
                """ximagesrc xid={0} ! videoconvert ! videocrop top=130  ! video/x-raw,framerate=15/1 ! videoscale ! video/x-raw,width=64,height=64,format=GRAY8 ! appsink""".format(self.xid), cv2.CAP_GSTREAMER)
        #ximagesrc major opcode of failed request 73 (x_getimage)
        #env = os.environ
        #env["GST_GL_XINITTHREADS"] = "1"
        #print("Selenium window is found. {0}".format(self.xid))
        #gststr = """gst-launch-1.0 ximagesrc xid={0} ! videoconvert ! videocrop top=130  ! video/x-raw,framerate=15/1 ! videoscale ! video/x-raw,width=48,height=48,format=GRAY8 ! filesink location=/dev/stdout""".format(self.xid).split(" ");
        #print( " ".join(gststr))
        #self.ffmpeg = sp.Popen(gststr,stdout=sp.PIPE, bufsize=10**8, env=env )

    def __mousemove(self,*c):
        print("sendingmove ",end="")
        self.points = self.driver.execute_script("xm={0}; ym={1}; return window.getscore();".format(*c))
        print("done.")

    """def _frame(self):
        while True:
            r,f =  self.ffmpeg.read()
            if r:
                cv2.imshow("slither-prev",f)
                return f.reshape((1,)+self.fshape)"""

    def _frame(self):
        while True:
            r,f =  self.ffmpeg.read()
            if r:
                cv2.imshow("slither-prev",f)
            return f.reshape((1,)+self.fshape)
        #while True:
        #    print ("getting frame ", end='\r')
        #    f = self.ffmpeg.stdout.read(self.fpower)
        #    f = np.frombuffer(f,dtype='uint8')[-self.fpower:]
        #    if len(f) < self.fpower: continue
        #    f.shape = ((1,)+self.fshape)
        #    cv2.imshow("slither-prev",f[0])
        #    return f
    
    def is_alive(self):
        return "display: none;" in self.loginframe.get_attribute("style")

    def get_score_and_last_player_move(self):
        score, x, y, accel = map(float,self.driver.execute_script("return [window.getscore(), xm, ym, issnakeaccelerated].join();").split(",") )
        ang = ( 1 - ( np.arctan2(x,y) / math.pi ) ) * 0.5

        #if ang < 0: ang += 2
        action = int(np.round(8 * ang)) 
        if action == 8.0: action = 0
        if int(accel) and np.random.rand(1)[0] < 0.333:
            return 8, score
        return action, score

    def action(self,a,s = 8):
        #angletopix = lambda x: (math.sin( x / 12 * math.pi )*200, math.cos(((x/12) - 1)*math.pi) * 200 )
        if a == 8:
            self.driver.execute_script("sl_accel();")
            return
        v = a/s * math.pi * 2
        self.__mousemove( math.sin(v) * 100, math.cos(v - math.pi) * 100 )
        

    def score(self):
        if int(self.points) > 0: return self.points
        try:
            return self.driver.execute_script("return window.getscore()") or "10"
        except: return "10"
        #for x in range(5):
        #    time.sleep(0.05)
        #    h = self.score()
        #    if h != "": return int(h)
        #return 0;

    def start(self):
        self.startbutton.click()

"""    def score(self):
        while True:
            try:
                return self.scoreframe.text or 0
                break
            except: pass

            try:
                self.scoreframe = self.xpath(self.xp_scoreframe)
                return scoreframe.text or 0
                break
            except:pass
"""

if __name__ == "__main__":
    e = environment()
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    i = 0
    e.start()
    while True:
        cv2.imshow('preview',e._frame())
        e.action(i)
        if i > 8: i=0
        else: i += 0.1
    input()
    #e.start()
    #while True:
    #    print("\"",e.score(),"\"")
