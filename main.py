from flask import Flask, request, render_template, jsonify
from datetime import datetime
import db
app = Flask(__name__)
places_dict = {
    "gunja": {"ids": [431, 432],"url":"https://m.hak.hr/kamera.asp?g=2&k=206"},
    "zupanja": {"ids": [79, 80],"url":"https://m.hak.hr/kamera.asp?g=2&k=44"},
    "slavonski_brod": {"ids": [195, 196],"url":"https://m.hak.hr/kamera.asp?g=2&k=140"},
    "svilaj": {"ids": [461, 462],"url":"https://m.hak.hr/kamera.asp?g=2&k=211"},
    "stara_gradiska": {"ids": [59, 60],"url":"https://m.hak.hr/kamera.asp?g=2&k=32"},
    "maljevac": {"ids": [429, 430],"url":"https://m.hak.hr/kamera.asp?g=2&k=177"},
    "kamensko": {"ids": [317, 318],"url":"https://m.hak.hr/kamera.asp?g=2&k=192"},
    "arzano": {"ids": [315, 316],"url":"https://m.hak.hr/kamera.asp?g=2&k=193"},
    "vinjani_gornji": {"ids": [994, 995],"url":"https://m.hak.hr/kamera.asp?g=2&k=282"},
    "vinjani_donji": {"ids": [302, 303],"url":"https://m.hak.hr/kamera.asp?g=2&k=39"},
    "novasela/bijaca": {"ids": [201, 202],"url":"https://m.hak.hr/kamera.asp?g=2&k=137"},
    "metkovic": {"ids": [320, 319],"url":"https://m.hak.hr/kamera.asp?g=2&k=136"},
    "klek/neum1": {"ids": [203, 204],"url":"https://m.hak.hr/kamera.asp?g=2&k=138"},
    "zatondoli/neum2": {"ids": [206],"url":"https://m.hak.hr/kamera.asp?g=2&k=139"},
    "brgat": {"ids": [453, 454],"url":"https://m.hak.hr/kamera.asp?g=2&k=208"}
}
@app.route('/')
def index():
    granica = request.args.get('granica', None)
    if granica not in places_dict:
        granica = None  # Ensure it's handled in the template
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    vehicle_count = 0  # Default count if no granica selected
    if granica:
        vehicle_count = db.count_cars_ids(granica)  # Replace with the actual function logic
    return render_template(
        'index.html', 
        granica=granica, 
        places_dict=places_dict, 
        timestamp=timestamp, 
        vehicle_count=vehicle_count
    )

@app.route('/count_cars')
def count_cars_endpoint():
    granica = request.args.get('granica', None)
    if granica in places_dict:
        vehicle_count = db.count_cars_ids(granica)  # Function to analyze traffic image
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return jsonify({'granica': granica,'places_dict':places_dict, 'vehicle_count': vehicle_count, 'timestamp': timestamp})
    return jsonify({'error': 'Invalid granica'}), 400
if __name__ == "__main__":
    app.run(debug=True)
