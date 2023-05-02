from flask import Flask,request,render_template,redirect

from distutils.log import debug
from flask import Flask, request, jsonify
import os

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2 
import numpy as np
from PIL import Image
from matplotlib import pyplot as pl

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

labels = [{'name':'Black Widow', 'id':1, 'type':'Venomous', 'color':'Red' ,
	            'details':'Black widow spider bites are very toxic and occur symptoms such as, Anxiety, Difficulty breathing, Headache, High blood pressure, increased sweating and Muscle weakness. Seek medical help right away.',
			    'health':['• Clean the area with soap and water.'
		        ,'•	Wrap ice in a clean cloth and place it on the bite area. Leave it on for 10 minutes and then off for 10 minutes. Repeat this process. If the person has blood flow problems, decrease the time that the ice is on the area to prevent possible skin damage.'
		        ,'•	Keep the affected area still, if possible, to prevent the venom from spreading. A homemade splint may be helpful if the bite was on the arms, legs, hands, or feet.'
		        ,'•	Loosen clothing and remove rings and other tight jewelry.']}, 
	      {'name':'Blue Tarantula', 'id':2, 'type':'Venomous', 'color':'Red' ,
	            'details':'However, this species is less dangerous than most of its relatives. It is a spider, and it is venomous. Although, this animal is not known for being excessively aggressive, and it will not attack a human just because it wants to do so.',
			    'health':['• Immediately clean the bite area with soap and water.'
		        ,'•	Apply a damp cloth with cold water or ice to the bite area to reduce swelling.'
		        ,'•	Elevate the bite area, if possible.'
		        ,'•	Seek medical attention for severe symptoms.'
		        ,'•	A healthcare provider may prescribe antibiotics to prevent infection.']}, 
	      {'name':'Bold Jumper', 'id':3, 'type':'Non Venomous', 'color':'Green' ,
	            'details':'Jumping spiders are not poisonous. Despite their less-than-appealing appearance, jumping spiders are not dangerous to humans.','health':[]}, 
	      {'name':'Brown Grass Spider', 'id':4, 'type':'Non Venomous', 'color':'Green' ,
	            'details':'Grass spiders are not poisonous, nor do they present a threat to humans.','health':[]}, 
	      {'name':'Brown Recluse Spider', 'id':5, 'type':'Venomous', 'color':'Red' ,
	            'details':'A bite from a brown recluse spider will not be instantly noticed because its bite is painless. Bite reactions vary from mild irritation to a potentially dangerous reaction. The bite will occur symptoms such as, Rash, Fever, Dizziness, Vomiting, Restlessness or difficulty sleeping, Swelling, Bruising, Pain surrounding muscles near the bite, Pain in your abdomen, back, chest and legs.',
			    'health':['• Immediately clean the bite area with soap and water.'
		        ,'•	Apply a damp cloth with cold water or ice to the bite area to reduce swelling.'
		        ,'•	Elevate the bite area, if possible.'
		        ,'•	Seek medical attention for severe symptoms.'
		        ,'•	A healthcare provider may prescribe antibiotics to prevent infection.']},
	      {'name':'Deinopis Spider', 'id':6, 'type':'Non Venomous', 'color':'Green' ,
	            'details':'These fascinating spiders have very mild venom and are harmless to people.','health':[]},
	      {'name':'Golden Orb Weaver', 'id':7, 'type':'Non Venomous', 'color':'Green' ,
	            'details':'Often mistaken for a dangerous creature, the Australian golden orb-weaving spider is in fact harmless to humans.','health':[]},
	      {'name':'Hobo Spider', 'id':8, 'type':'Venomous', 'color':'Red' ,
	            'details':'The hobo spider is not dangerous. Their venom is no more toxic than that of other spiders and they are no more aggressive or likely to bite people than other spiders. But if you are allergic to  them it may cause a major issue.',
			    'health':['• Clean the bite area with mild soap and water'
		        ,'•	Apply a cool compress to the bite site to reduce pain and swelling'
		        ,'•	Elevate your arm or leg if that’s where the bite occurred']},
	      {'name':'Huntsman Spider', 'id':9, 'type':'Non Venomous', 'color':'Green' ,
	            'details':"These spiders look so large and hairy, despite their often large and hairy appearance, huntsman spiders are not considered to be dangerous spiders. As with most spiders, they do possess venom, and a bite may cause some ill effects but won't cause deadly impacts.",'health':[]},
	      {'name':'Ladybird Mimic Spider', 'id':10, 'type':'Non Venomous', 'color':'Green' ,
	            'details':'Red ladybugs tend to be more predatory and able to defend themselves. Red is a deterrent to many larger predators, including birds. However, they are not as poisonous as orange ladybugs. Orange-tinted ladybugs (which are mostly Asian lady beetles) tend to have the most toxins in their bodies. Therefore, they may be the most allergenic to humans. These spiders are harmless to human.','health':[]},
	      {'name':'Peacock Spider', 'id':11, 'type':'Non Venomous', 'color':'Green' ,
	            'details':"Like many arachnids, peacock spiders are venomous, but they are completely harmless to humans. Their prey consists of small invertebrates such as flies and moths, but unlike other spiders they don't use webs for hunting. As Mr Schubert said, They slowly approach their prey, trying not be detected.",'health':[]},
	      {'name':'Red Knee Trantula', 'id':12, 'type':'Non Venomous', 'color':'Green' ,
	            'details':"Mexican red knee tarantulas produce a venom that is toxic to its prey—insects, small frogs, lizards, and mice. But it is not a threat to humans. The spider's venom paralyzes prey. Bites are relatively harmless to humans, comparable to the sting of a bee or wasp.",'health':[]},
	      {'name':'Spiny Backed Orb Weaver', 'id':13, 'type':'Non Venomous', 'color':'Green' ,
	            'details':'Spiny-backed orb weaver spiders are mostly harmless. Their large webs often startle and annoy people, but the pests pose no serious health risks.','health':[]},
	      {'name':'White Kneed Tarantula', 'id':14, 'type':'Venomous', 'color':'Red' ,
	            'details':'If a tarantula bites you, you may have pain at the site of the bite similar to a bee sting. The area of the bite may become warm and red. And symptoms like Breathing difficulty, Loss of blood flow to major organs (an extreme reaction), Itchiness, shock, Rapid heart rate, Skin rash may occur.',
			    'health':['Wash the area with soap and water. Place ice (wrapped in a clean cloth or other covering) on the site of the sting for 10 minutes and then off for 10 minutes. Repeat this process. If the person has blood flow problems, reduce the time the ice is used to prevent possible skin damage.']},
	      {'name':'Yellow Garden Spider', 'id':15, 'type':'Non Venomous', 'color':'Green' ,
	            'details':'These spiders produce venom that is harmless to humans, but helps to immobilize prey like flies, bees, and other flying insects that are caught in the web. They spin webs in sunny areas with plants on which they can anchor the webs. They may also be seen in backyard gardens.','health':[]}]

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-5')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def bbox_fn(image_name):
    # name = image_name + '.jpg'
    name = str(image_name)
    print(name)
    img = cv2.imread(os.path.join('static','Images',name))
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.25,
                agnostic_mode=False)
    
    detection_threshold = 0.25

    image = image_np_with_detections
    scores = list(filter(lambda x:x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    scores = detections['detection_scores'][:len(scores)]
   
    try : 
        id = labels[classes[0]]['id']
    
    except:
        id = 0
    
  
    return id

app = Flask(__name__)


app.config["IMAGE_UPLOADS"] = "./static/Images"
#app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]

from werkzeug.utils import secure_filename


@app.route('/predict',methods = ["GET","POST"])
def upload_image():
	if request.method == "POST":
		image = request.files['file']
		if image.filename == '':
			print("Image must have a file name")
			return redirect(request.url)

		filename = secure_filename(image.filename)
		
		basedir = os.path.abspath(os.path.dirname(__file__))
		image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))

		id = bbox_fn(filename)
		if id == 0:
			print("Only Upload Spider Images")
			return render_template('main.html',imgtype='other')
		SpiderName = [{'name': labels[id-1]['name'],
		                'type': labels[id-1]['type'],
			            'color': labels[id-1]['color']}]
		Details = labels[id-1]['details']


		return render_template("main.html",filename=filename, spidername=SpiderName, details=Details, health=labels[id-1]['health'])

	return render_template('main.html')

@app.route('/home',methods = ["GET","POST"])
def home():
	return render_template('home.html')

@app.route('/',methods = ["GET"])
def login():
    uname = request.args.get("username")
    password = request.args.get("password")
    if (uname == 'admin') and (password == 'admin'):
        return redirect('/home')
	    
    return render_template('login.html')

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static',filename = "/Images" + filename), code=301)


app.run(debug=True,port=2000)