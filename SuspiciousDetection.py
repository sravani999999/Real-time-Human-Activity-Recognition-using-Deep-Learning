from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2 
import shutil
import os
from imageai.Prediction.Custom import CustomImagePrediction
from PIL import ImageTk, Image
from imutils import paths

main = Tk()
main.title("Human Activity Detection")
main.geometry("1200x1200")

# Load background image
bg_image = Image.open("background_image.jpg")  # Change "background_image.jpg" to your image path
bg_image = bg_image.resize((1200, 1200), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(bg_image)
background_label = Label(main, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

global filename

execution_path = os.getcwd()
prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("model.h5")
prediction.setJsonPath("model_class.json")
prediction.loadModel(num_objects=2)

def upload():
    global filename
    filename = askopenfilename(initialdir="videos")
    pathlabel.config(text=filename)

def generateFrame():
    global filename
    text.delete('1.0', END)
    if not os.path.exists('frames'):
        os.mkdir('frames')
    else:
        shutil.rmtree('frames')
        os.mkdir('frames')
    vidObj = cv2.VideoCapture(filename) 
    count = 0
    success = 1
    while success:
        success, image = vidObj.read() 
        if count < 500:
            cv2.imwrite("frames/frame%d.jpg" % count, image)
            text.insert(END, "frames/frame."+str(count)+" saved\n")
            print("frames/frame."+str(count)+" saved")
        else:
            break
        count += 1
    pathlabel.config(text="Frame generation process completed. All frames saved inside frame folder")

def detectActivity():
    imagePaths = sorted(list(paths.list_images("frames")))
    count = 0
    option = 0
    text1.delete('1.0', END)
    for imagePath in imagePaths:
        predictions, probabilities = prediction.predictImage(imagePath, result_count=1)
        for eachPrediction, eachProbability in zip(predictions, probabilities):
            if float(eachProbability) > 80:
                count = count + 1
            if float(eachProbability) < 80:
                count = 0
            if count > 10:
                option = 1
                print(imagePath+" is predicted as "+eachPrediction+" with probability : " +str(eachProbability))
                text1.insert(END, imagePath+" "+eachPrediction+" with probability : " +str(eachProbability)+"\n\n")
                count = 0
        print(imagePath+" processed")
    if option == 0:
        text1.insert(END, "No human activity found in given footage")   

font = ('times', 20, 'bold')
title = Label(main, text='Human Activity Detection From CCTV Footage')
title.config(fg='black')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=5,y=5)

font1 = ('times', 14, 'bold')
upload_button = Button(main, text="Upload CCTV Footage", command=upload)
upload_button.place(x=450,y=120)
upload_button.config(font=font1)  

pathlabel = Label(main, bg='white', fg='black', font=font1)
pathlabel.place(x=300,y=100)

depth_button = Button(main, text="Decompile Video to Frames", command=generateFrame)
depth_button.place(x=300,y=170)
depth_button.config(font=font1) 

user_interest_button = Button(main, text="Detect Human Activity Frame", command=detectActivity)
user_interest_button.place(x=650,y=170)
user_interest_button.config(font=font1) 

font1 = ('times', 12, 'bold')
text = Text(main, height=25, width=50)
text.place(x=60,y=230)
text.config(font=font1)

text1 = Text(main, height=25, width=50)
text1.place(x=600,y=230)
text1.config(font=font1)

main.mainloop()
