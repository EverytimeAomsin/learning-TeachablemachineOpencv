import tensorflow.keras
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('flower_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
# cam = cv2.VideoCapture(1)
img = np.zeros((500,500,3),np.uint8)
text = ""

while True:
    img = cv2.imread('./img/3.jpg')
    # _,img = cam.read()
    img = cv2.resize(img,(224, 224)) #224 เป็นขนาด standard img model


    b,g,r,a = 255,255,255,0

    #โหลดไฟล์ฟอนต์
    fontpath = "./RSU_Text.ttf" # <== 这里是宋体路径 
    font = ImageFont.truetype(fontpath, 30)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    

 
    #turn the image into a numpy array
    image_array = np.asarray(img)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    # print(prediction)
    for i in prediction:
        if i[0] > 0.7:
            text ="เดซี่"
        if i[1] > 0.7:
            text ="แดนดิไลออน"
        if i[2] > 0.7:
            text ="กุหลาบ"
        if i[3] > 0.7:
            text ="ทานตะวัน"
        if i[4] > 0.7:
            text ="ทิวลิป"
        # print(text)
        # cv2.putText(img,text,(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)
        draw.rectangle((0, 400, 600, 190), fill=(0, 0, 0) )
        draw.line((0, 188, 450, 188), fill=(255, 255, 255), width=2)
        draw.text((80, 190), "ดอก" + text, font = font, fill = (b, g, r, a))
        img = np.array(img_pil)
        img = cv2.resize(img,(500, 500))
   # Display 
    cv2.imshow("Flower detection", img);cv2.waitKey();cv2.destroyAllWindows()
 