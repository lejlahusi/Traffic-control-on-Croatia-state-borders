# 🚦 Traffic Monitoring Web App

A web-based application to monitor traffic density at various border checkpoints using live camera feeds, object detection (YOLO), and real-time updates.

## 📌 Features

- Display traffic images from selected border locations.
- Count stationary vehicles using YOLO object detection.
- Show real-time vehicle counts and update timestamps.
- Periodically fetch updated data and refresh images.
- Interactive web interface using Flask and Jinja2 templating.

---

## 🛠️ Technologies Used

- **Flask** – Web server and routing.
- **HTML/Jinja2** – Dynamic templating.
- **JavaScript (Fetch API)** – Real-time data updates.
- **YOLO / OpenCV (via `db.count_cars_ids`)** – Vehicle detection.
- **Python** – Backend logic and integration.

---

## 📂 Project Structure

```
project/
│
├── static/
│   └── output_image_<id>.jpg      # Updated traffic images
│
├── templates/
│   ├── base.html                  # Base layout
│   └── index.html                 # Main UI logic
│
├── db.py                          # Contains logic to count vehicles using YOLO
├── main.py                        # Flask application
└── README.md                      # This file
```

---

## 🌍 Available Locations

The app supports several checkpoints. Example values for the `granica` parameter:

- `zupanja`
- `gunja`
- `slavonski_brod`
- `svilaj`
- `stara_gradiska`
- `maljevac`
- `kamensko`
- `arzano`
- `vinjani_gornji`
- `vinjani_donji`
- `novasela/bijaca`
- `metkovic`
- `klek/neum1`
- `zatondoli/neum2`
- `brgat`

---

## 📈 API Endpoint

- `GET /count_cars?granica=<location>`  
  Returns current vehicle count and timestamp in JSON format.

Example:
```json
{
  "granica": "zupanja",
  "vehicle_count": 5,
  "timestamp": "2025-06-17 14:55:23"
}
```

## 🧠 Future Improvements

- Add real-time heatmaps or charts.
- Multiprocessing for all state borders
- Integrate historical traffic trend analysis.
- Add user authentication and admin control panel.

---
