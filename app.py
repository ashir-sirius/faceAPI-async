from flask import Flask, render_template, request

from faces import detect_face

#import threading

import asyncio

#from fastapi import FastAPI
#import uvicorn


#import nest_asyncio
#nest_asyncio.apply()

#loop = asyncio.get_event_loop()
app = Flask(__name__)

#app = FastAPI()

@app.route('/', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'GET':
        return render_template('index.html')
    
    if request.method == 'POST':
        imagefile1 = request.files['imagefile1']
        imagefile2 = request.files['imagefile2']

        image_path1 = "./images/" + imagefile1.filename
        imagefile1.save(image_path1)

        image_path2 = "./images/" + imagefile2.filename
        imagefile2.save(image_path2)

        files = request.files.getlist("images")
        
        #loop = asyncio.get_running_loop()
        #results = loop.run_until_complete(detect_face(image_path1, image_path2))
        #loop.close()
        
        results = asyncio.run(detect_face(image_path1, image_path2))
        
        #results = await asyncio.gather(detect_face(image_path1, image_path2))
        
        #results = asyncio.create_task(detect_face(image_path1, image_path2))
        
        
        print(results)
    
        return render_template('index.html', prediction=results)


if __name__ == '__main__':
    #print(f"In flask global level: {threading.current_thread().name}")
    app.run(port=3000, debug=False)


    #uvicorn.run(app, debug=True)
    
    #asyncio.run(predict()) 