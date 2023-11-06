import cv2
from ultralytics import YOLO
import time
import os
import glob
import RPi.GPIO as GPIO
from PIL import Image
from ultralytics.utils import ASSETS
from ultralytics.models.yolo.classify import ClassificationPredictor

model = YOLO('/home/admin/Documents/train18_best.pt')
#Variable that holds the path where the screenshot will be saved.
imgPath = '/home/admin/Documents/img/'
#Grabbing the USB camera source.
camera = cv2.VideoCapture(0)
confidenceMossMatInProgress = 0.5
gpioPinNumber = 23
#Some USB cameras require a few initial frames to work properly.
#For instance, in the initial frames of the camera that was used,
#had a green tint before the camera produced normal pictures.
skipFrames = 10
#If startProcess gets set to True than the motor starts.
startProcess = False
#Threshold for positive results before startProcess is set to True.
threshold = 3
#countImage for a sequence of positive results.
countImage = 0

def capture_screenshot():
    #Skipping the first few frames.
    camera.set(cv2.CAP_PROP_POS_FRAMES, skipFrames)
    #Captures a frame form the USB camera.
    #'frame' holdes the image data and 'frameReturn'
    #indicates whether the frame read was successfull.
    frameReturn, frame = camera.read()
    #If the frame couldn't be captured, an error is printed.
    if not frameReturn:
        print("Can't read frames for some reason.")
    else:
        #Get the current Time and with 'time.strftime()'.
        currentTime = time.strftime("%H-%M-%S", time.localtime())
        #Construct the filename for the screenshot.
        screenshotFilename = "screenshot_" + currentTime + ".png"
        #Try to save the captured frame as an image. 
        try:
            cv2.imwrite(imgPath + screenshotFilename, frame)
            #If the save was successfull, it prints a message
            #with the filename.
            print("Screenshot saved: " + screenshotFilename)
        #If there is an error during the saving process
        #it catches the exception and prints the error.
        except Exception as e:
            print("Error saving screenshot: " + str(e))

#Defines the function newest_image() and returns a string.
def newest_image()->str:
    #Finding all files with the .png extension in the
    #directory by the specified path "imgPath".
    #'os.path.join(imgPath, "*.png")' constructs the path.
    pngImage = glob.glob(os.path.join(imgPath, "*.png"))
    #Check whether pngImage is empty, if so print error message.
    if not pngImage:
        print("Error: No image in the directory.")
    #Finding the most recently modified PNG file.
    #The 'newestPngImage' variable now holds the path to the
    #most recently modified file.
    else:
        newestPngImage = max(pngImage, key=os.path.getmtime)
        #The code attempts to open the most recent PNG.
        #The 'with' statement is used to ensure that the
        #image is closed properly.
        try:
            with Image.open(newestPngImage) as img:
                #If the image is successfully opened, the
                #function returns the path of the newest png
                #as a string.
                return (str(newestPngImage))
        #If there happens to be an error while attempting to
        #open the image, it catches and prints the error message.
        except Exception as e:
            print("Error loading the PNG image: ", e)

def image_classification():
    #Call the function newest_image() and pass it to the model.
    results = model(newest_image())
    #If a result exists...
    if results[0] is not None or results[0].boxes.conf is not None:
        #...save confidence level in conf.
        conf = results[0].probs.top5conf
        print(conf)
        #If confidence level of the "moosmatte_in_arbeit" class is
        #greater than the previous defined threshold the function
        #returns True.
        if conf[1] <=confidenceMossMatInProgress:
            print("Moosmatte in Arbeit")
            return True
        #The opposite of the if branch and returns False.
        else:
            print("Kein Abstandsgewirke mit Moos erkannt")
            return False

def clear_images():
    #Returns a list of all files in the given directory.
    listFiles = os.listdir(imgPath)
    #Each file with the extension .png gets deleted.
    for file in listFiles:
        if file.endswith('.png'):
            os.remove(os.path.join(imgPath, file))
            print("Image deleted.")
            
def evaluate_moss(mossDetected):
    #Accessing variable outside of local scope.
    global countImage
    global startProcess
    print(str(countImage))
    #If a moss mat has been detected. 
    if mossDetected:
        #If the threshold is less than countImage
        #then start the motor.
        if countImage >= threshold:
            startProcess = True
        else:
            #If the countImage is less than threshold
            #then keep the motor off and increment the counter.
            startProcess = False
            countImage = countImage + 1
    #If no moss mat has been detected.
    else:
        #Reset startProcess to false and countImage value.
        startProcess = False
        countImage = 0
        
def start_motor():
    #Accessing variable outside of local scope.
    global startProcess
    #If startProcess is True the GPIO pin is set to low.
    if startProcess:
        GPIO.output(gpioPinNumber, GPIO.LOW)
    #If startProcess is Flase the GPIO pin is set to high.
    else:
        GPIO.output(gpioPinNumber, GPIO.HIGH)

#Set GPIO mode to Broadcom -> refers to the
#physical pin numbers on the Raspberry Pi.
GPIO.setmode(GPIO.BCM)
#Configures the GPIO pin 'gpioPinNumber' as an output pin.
GPIO.setup(gpioPinNumber, GPIO.OUT)

while True:
    print("----------------------")
    capture_screenshot()
    #Image_classification() retruns a True if a moos mat has
    #been detected.
    mossDetected = image_classification()
    #evaluate_moss() evaluates the sequence of images.
    evaluate_moss(mossDetected)
    start_motor()
    clear_images()
    print("Sleeping for 1 second.\n\n")
    time.sleep(1)