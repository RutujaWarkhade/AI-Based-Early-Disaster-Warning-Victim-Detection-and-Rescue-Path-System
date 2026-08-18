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
