# 🚨 AI-Based Multi-Disaster Prediction & Rescue System

An AI-powered disaster management and emergency rescue system designed to support **disaster prediction, real-time victim detection, location tracking, and shortest rescue path planning**.

The system combines **Machine Learning, Computer Vision, Drone Surveillance, GPS, Geospatial Mapping, and Smart Routing** to assist rescue teams during disasters such as **Floods, Wildfires, Cyclones, and Earthquakes**.

---

## 📌 Project Overview

During natural disasters, rescue teams face major challenges such as:

- Identifying victims in affected areas
- Locating victims accurately
- Finding the nearest rescue team
- Selecting safe and shortest routes
- Avoiding flooded, damaged, or blocked roads
- Predicting disaster risk and generating early warnings
- Monitoring affected areas in real time

This project provides an integrated solution where **drone cameras capture live aerial footage**, the footage is transmitted to the system/laptop, and AI-based computer vision detects victims.

Once a victim is detected, the system determines the victim's location and calculates an **optimal rescue route from the nearest rescue team**, while avoiding blocked or dangerous zones.

The system also includes ML-based disaster prediction modules for **flood, wildfire, cyclone, and earthquake risk assessment**.

---

# 🎯 Objectives

- Detect victims in disaster-affected areas using **drone camera footage**.
- Track victim locations using **GPS and geographical coordinates**.
- Identify the **nearest rescue team**.
- Calculate the **shortest/optimal rescue path**.
- Avoid blocked areas such as floods, fire, rubble, and building collapse zones.
- Predict disaster risks using Machine Learning.
- Generate risk levels and disaster alerts.
- Provide a centralized dashboard for rescue operations.

---

# 🏗️ System Architecture

```text
                 ┌─────────────────────────┐
                 │   Disaster Data Sources │
                 │ Flood / Fire / Cyclone  │
                 │      / Earthquake       │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Data Preprocessing    │
                 │ Feature Engineering     │
                 │ Data Cleaning            │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    ML Prediction        │
                 │ XGBoost / LightGBM      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Risk Score & Alerts     │
                 │ Low / Moderate / High   │
                 │ / Critical               │
                 └─────────────────────────┘


 Drone Camera
      │
      │ Live Video
      ▼
┌───────────────────┐
│ Laptop / Server   │
│ Video Processing  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ YOLOv8 + OpenCV   │
│ Victim Detection  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Victim Location   │
│ GPS Coordinates   │
└─────────┬─────────┘
          │
          ▼
┌────────────────────────────┐
│ Nearest Rescue Team        │
│ + Distance Calculation     │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Smart Route Planning       │
│ OSRM + OpenStreetMap        │
│ + Haversine Distance       │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ Safe & Shortest Rescue     │
│ Route to Victim             │
└────────────────────────────┘
## 🖥️ System Screenshots

### 1. Landing Page

![Landing Page](Images/Landing_page.png)

The landing page provides an overview of the AI-based disaster prediction, victim detection, and rescue path planning system.

---

### 2. Person Detection

![Person Detection](Images/Persons_Detection.jpeg)

The computer vision module uses YOLOv8 to detect people/victims from disaster-affected area footage captured by the drone.

---

### 3. Shortest Rescue Path on Map

![Shortest Path on Map](Images/Shortest_Path_on_map.png)

The system identifies the victim's location and calculates an optimal rescue route using geographical mapping and routing.

---

### 4. Victim Details

![Victims Details](Images/Victims_Details.png)

The system displays important information about detected victims to support rescue team decision-making.

---

### 5. Live Disaster Alert

![Live Disaster Alert](Images/Live_Disaster_Alert.png)

The system provides live disaster alerts and displays the current disaster status.

---

### 6. Live Disaster Alerts

![Live Disaster Alerts](Images/Live_Disaster_Alerts_2.png)

Additional live alert information helps rescue teams monitor ongoing disaster situations.

---

### 7. Live Disaster Alerts – Detailed View

![Live Disaster Alerts Detailed View](Images/Live_Disaster_Alerts_3.png)

The detailed alert view provides additional information about detected disaster events and their severity.

---

### 8. Live Alert Dashboard

![Live Alert Dashboard](Images/Live_Alert_Dashboard.png)

The dashboard provides a centralized view of active alerts and disaster-related information.

---

### 9. Live Alert Dashboard – Detailed View

![Live Alert Dashboard Detailed View](Images/Live_Alert_Dashboard_2.png)

The detailed dashboard provides additional monitoring information for rescue operations.

---

### 10. Disaster Statistics

![Disaster Statistics](Images/Disaster_Statistics.png)

The system presents disaster-related statistics to support data-driven decision-making.
