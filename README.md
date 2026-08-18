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

The landing page provides an overview of the AI-based disaster prediction, victim detection, and rescue path planning system.

![Landing Page](https://github.com/RutujaWarkhade/AI-Based-Early-Disaster-Warning-Victim-Detection-and-Rescue-Path-System/blob/main/Images/Landing_page.png)

---

### 2. Person Detection

The computer vision module uses YOLOv8 to detect people/victims from disaster-affected area footage captured by the drone.

![Person Detection](https://github.com/RutujaWarkhade/AI-Based-Early-Disaster-Warning-Victim-Detection-and-Rescue-Path-System/blob/main/Images/Persons_Detection.jpeg)

---

### 3. Shortest Rescue Path on Map

After detecting a victim, the system identifies the victim's location and calculates an optimal rescue route using geographical mapping and routing.

![Shortest Path on Map](https://github.com/RutujaWarkhade/AI-Based-Early-Disaster-Warning-Victim-Detection-and-Rescue-Path-System/blob/main/Images/Shortest_Path_on_map.png)

---

### 4. Victim Details

The system displays important information about detected victims to support rescue team decision-making.

![Victims Details](https://github.com/RutujaWarkhade/AI-Based-Early-Disaster-Warning-Victim-Detection-and-Rescue-Path-System/blob/main/Images/Victims_Details.png)

---

### 5. Live Disaster Alert

The system provides live disaster alerts and displays the current disaster status to support early response.

![Live Disaster Alert](https://github.com/RutujaWarkhade/AI-Based-Early-Disaster-Warning-Victim-Detection-and-Rescue-Path-System/blob/main/Images/Live_Disaster_Alert.png)

---

### 6. Live Disaster Alerts

Additional live alert information is presented to help rescue teams monitor ongoing disaster situations.

![Live Disaster Alerts](https://github.com/RutujaWarkhade/AI-Based-Early-Disaster-Warning-Victim-Detection-and-Rescue-Path-System/blob/main/Images/Live_Disaster_Alerts_2.png)

---

### 7. Live Disaster Alerts – Detailed View

A detailed alert view provides additional information about detected disaster events and their severity.

![Live Disaster Alerts Detailed View](https://github.com/RutujaWarkhade/AI-Based-Early-Disaster-Warning-Victim-Detection-and-Rescue-Path-System/blob/main/Images/Live_Disaster_Alerts_3.png)

---

### 8. Live Alert Dashboard

The dashboard provides a centralized view of active alerts and disaster-related information.

![Live Alert Dashboard](https://github.com/RutujaWarkhade/AI-Based-Early-Disaster-Warning-Victim-Detection-and-Rescue-Path-System/blob/main/Images/Live_Alert_Dashboard.png)

---

### 9. Live Alert Dashboard – Detailed View

The detailed dashboard provides additional monitoring information for ongoing rescue and disaster-management operations.

![Live Alert Dashboard Detailed View](https://github.com/RutujaWarkhade/AI-Based-Early-Disaster-Warning-Victim-Detection-and-Rescue-Path-System/blob/main/Images/Live_Alert_Dashboard_2.png)

---

### 10. Disaster Statistics

The system presents disaster-related statistics to help understand disaster patterns and support data-driven decision-making.

![Disaster Statistics](https://github.com/RutujaWarkhade/AI-Based-Early-Disaster-Warning-Victim-Detection-and-Rescue-Path-System/blob/main/Images/Disaster_Statistics.png)

