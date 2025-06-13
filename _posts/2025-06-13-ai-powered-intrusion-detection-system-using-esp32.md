---
title: "AI Powered Intrusion Detection System Using ESP32 and Edge Impulse."
date: 2025-03-29
categories: [Projects,Edge AI,Iot,Security]
tags: [Projects,Edge AI,Iot,ESP32,Security]
---

# Introduction

In today’s hyper-connected world, the Internet of Things (IoT) has transformed how we interact with technology, from smart home devices to industrial sensors. However, this connectivity comes with a significant challenge: **security**. \
IoT devices are increasingly targeted by cyberattacks, such as UDP flood attacks, which can overwhelm and disrupt their functionality. We developed an AI-powered intrusion detection system (IDS) to address this issue. Our project leverages Edge AI to detect UDP flood attacks in real-time using two ESP32 microcontrollers, ensuring low-latency, efficient, and lightweight security for IoT environments.

The goal of this project was to create a system capable of identifying malicious network traffic directly on resource-constrained devices, eliminating the need for cloud-based processing. By combining packet sniffing, machine learning, and real-time alerting, our solution achieves an impressive 98.99% accuracy in detecting UDP flood attacks.

In this article, I’ll walk you through every step of our journey, from collecting network data to training a neural network, deploying it on ESP32 devices, and testing the system in real-world scenarios. 

All the code, datasets, and resources are available in our GitHub repository [AI-IDS](https://github.com/ammarlouah/AI-IDS). 

Whether you’re an IoT enthusiast, an AI practitioner, or a cybersecurity buff, this article will give you a detailed look at how we brought Edge AI to IoT security.

# System Architecture

The architecture of our system is designed to operate efficiently on resource-constrained IoT devices while ensuring real-time detection and alerting for UDP flood attacks. The system leverages two ESP32 microcontrollers, each with distinct roles, working in tandem to capture, analyze, and respond to network traffic. Below, we break down the components and their interactions, highlighting how they collaborate to achieve robust security.

# System Architecture

The architecture of our system is designed to operate efficiently on resource-constrained IoT devices while ensuring real-time detection and alerting for UDP flood attacks. The system leverages two ESP32 microcontrollers, each with distinct roles, working in tandem to capture, analyze, and respond to network traffic. Below, we break down the components and their interactions, highlighting how they collaborate to achieve robust security.

## ESP32 A: Sniffer and Classifier

The first ESP32, referred to as **ESP32 A**, serves as the primary data collection and analysis unit. Its role is to capture Wi-Fi packets, extract relevant features, and classify traffic as either "normal" or "UDP flood" using a machine learning model. Here’s how it works:

- **Packet Capture**: ESP32 A is configured in **promiscuous mode**, allowing it to capture all Wi-Fi packets within its range, regardless of their destination. This mode is essential for monitoring network traffic without interfering with the network’s normal operation.
- **Packet Filtering**: To focus on the target IoT device, ESP32 A filters packets based on the device’s MAC address. This ensures that only relevant traffic is analyzed, reducing noise from unrelated devices in the network.
- **Feature Extraction**: For every 1-second window, ESP32 A computes a set of features to characterize the traffic:
  - **target_pkts**: The number of packets directed to the target device’s MAC address.
  - **pkt_ratio**: The ratio of target packets to the total number of packets captured in the window.
  - **avg_pkt_size**: The average size (in bytes) of the target packets. These features were selected for their ability to distinguish normal traffic from UDP flood attacks, which typically involve a high volume of small, rapid packets.
- **Classification**: The extracted features are fed into a neural network model, trained using Edge Impulse, to classify the traffic. The model outputs a prediction label ("normal" or "UDP flood") along with a confidence score (e.g., 0.95 for UDP flood). The model is optimized for the ESP32’s limited computational resources, ensuring fast inference.
- **Communication**: After classification, ESP32 A sends the prediction results to ESP32 B via a UART interface (TX pin on GPIO17 of ESP32 A to RX pin on GPIO16 of ESP32 B). The data is transmitted in a simple format, such as `PRED:udp_flood,0.95`, ensuring reliable and low-latency communication.

## ESP32 B: Alert System

The second ESP32, **ESP32 B**, acts as the alert and notification hub. It processes the predictions from ESP32 A and triggers alerts when an attack is detected. Its key responsibilities include:

- **Wi-Fi Connectivity**: Unlike ESP32 A, which operates in promiscuous mode, ESP32 B connects to a Wi-Fi network to enable communication with external services, specifically a Telegram bot for sending alerts.
- **Prediction Processing**: ESP32 B receives predictions from ESP32 A via UART (RX pin on GPIO16). It maintains a **sliding window** of the last 10 predictions to ensure robust decision-making. This window helps mitigate false positives by requiring a majority consensus before triggering an alert.
- **Alert Mechanism**: If at least 6 out of the 10 predictions in the sliding window indicate a "UDP flood" attack, ESP32 B sends an alert message via Telegram. The message, sent through a pre-configured Telegram bot to a specified chat, reads: **"UDP flood detected by ESP2!"**. This threshold (≥ 6/10) balances sensitivity and reliability, ensuring alerts are triggered only when there’s strong evidence of an attack.
- **Error Handling**: ESP32 B is programmed to handle potential communication errors over UART, such as missing or corrupted predictions, ensuring the system remains operational even in noisy environments.

## System Integration

The two ESP32s communicate seamlessly via UART, creating a modular and scalable architecture. ESP32 A focuses on computationally intensive tasks (packet capture and classification), while ESP32 B handles external communication and alerting. This division of labor optimizes resource usage, as each microcontroller is tailored to its specific role. The use of UART for inter-device communication ensures low-latency, reliable data transfer, critical for real-time operation.

The system is designed to be lightweight, running entirely on edge devices without reliance on cloud infrastructure. This makes it suitable for IoT deployments where internet connectivity may be intermittent or where low latency is critical. The architecture also allows for future scalability, such as adding more ESP32 units to monitor multiple devices or integrating additional attack detection models.

# Data Collection

The success of our system hinges on a high-quality dataset that captures the nuances of both normal and malicious network traffic. To train our neural network model to detect UDP flood attacks, we collected data using ESP32 A, one of our two ESP32 microcontrollers, and simulated realistic attack scenarios. This section outlines the setup, features extracted, scenarios (including the attack simulation using Kali Linux), data logging process, and the resulting dataset.

## Setup for Data Collection

We configured **ESP32 A** to capture Wi-Fi packets in **promiscuous mode**, enabling it to intercept all Wi-Fi packets within its range, regardless of their destination. This mode is ideal for network monitoring, as it allows comprehensive traffic analysis without requiring the ESP32 to join the network.

- **Hardware**: ESP32 A was programmed via the Arduino IDE to operate in promiscuous mode and process incoming packets efficiently.
- **Network Environment**: Data collection occurred in a controlled Wi-Fi network, including a target device and other devices generating background traffic.
- **Filtering**: To focus on the target device, we filtered packets based on its MAC address, reducing noise from unrelated devices in the network.

## Features Extracted

For each **1-second window**, ESP32 A computed a set of features to characterize the captured traffic. These features were selected to highlight differences between normal traffic and UDP flood attacks, which typically involve a high volume of small, rapid packets. The extracted features were:

- **timestamp**: The start time of the 1-second window, used for logging purposes.
- **duration_ms**: The window duration (fixed at 1000 ms).
- **total_pkts**: The total number of Wi-Fi packets captured in the window.
- **target_pkts**: The number of packets directed to the target device’s MAC address.
- **pkt_ratio**: The ratio of target packets to total packets (`target_pkts / total_pkts`), indicating the proportion of traffic targeting the device.
- **bytes_total**: The total size (in bytes) of packets directed to the target device.
- **avg_pkt_size**: The average size of target packets (`bytes_total / target_pkts`), reflecting typical packet size.
- **unique_MACs**: The number of unique source MAC addresses, indicating the diversity of sending devices.
- **mean_IAT_us**: The mean inter-arrival time (in microseconds) between consecutive target packets, capturing packet frequency.
- **IAT_variance**: The variance of inter-arrival times, indicating the consistency of packet arrivals.

These features were calculated in real-time by ESP32 A, leveraging its processing capabilities to handle high packet volumes.

## Simulated Scenarios

We collected data under two scenarios to create a comprehensive dataset:

1. **Normal Traffic**:

   - **Description**: Represented typical network activity without malicious interference.
   - **Activities**: Included routine IoT device operations (e.g., periodic data transmissions) and background traffic from other devices (e.g., browsing, streaming).
   - **Data Volume**: We collected **1000 rows** of data, each representing a 1-second window of normal traffic.
   - **Purpose**: To establish a baseline for legitimate traffic patterns, enabling the model to recognize normal behavior.

2. **UDP Flood Attack**:

   - **Description**: Simulated a UDP flood attack targeting the device at IP address 192.168.137.193.
   - **Simulation Method**: We used **Kali Linux** with the `hping3` tool to generate the attack. The command was:

     ```
     hping3 --flood --rand-source --udp -p 53 192.168.137.193
     ```
     - `--flood`: Sends packets as fast as possible.
     - `--rand-source`: Randomizes source IP addresses to mimic a distributed attack.
     - `--udp`: Uses UDP packets.
     - `-p 53`: Targets port 53 (commonly associated with DNS, but used here to simulate a flood).
     - `192.168.137.193`: The target device’s IP address.
   - **Data Volume**: We collected **500 rows** of data, each representing a 1-second window during the attack.
   - **Purpose**: To capture the characteristics of malicious traffic, such as high packet volume, small packet sizes, and reduced inter-arrival times.

## Data Logging

To store the data for model training, we implemented a robust logging mechanism:

- **Process**: ESP32 A computed features for each 1-second window and transmitted them to a computer via a serial connection (USB). A Python script on the computer listened to the serial port, parsed the feature data, and saved it to CSV files.
- **File Structure**: Two CSV files were created:
  - **Normal traffic**: 1000 rows, labeled as “normal.”
  - **UDP flood traffic**: 500 rows, labeled as “udp_flood.” Each row included the features listed above and a label indicating the scenario.
- **Balanced Dataset**: To ensure a balanced dataset for training, we created a final dataset of **1000 rows** by selecting all 500 rows of UDP flood data and randomly sampling 500 rows from the 1000 rows of normal traffic. This resulted in **500 normal** and **500 udp_flood** rows, ensuring equal representation of both classes.
- **Validation**: We inspected the CSV files to verify data integrity, checking for missing values, outliers, or inconsistencies in feature calculations.

## Challenges and Solutions

Data collection presented several challenges, which we addressed systematically:

- **Noisy Wi-Fi Environment**: Unrelated traffic from nearby devices could skew the data. We mitigated this by filtering packets based on the target device’s MAC address and conducting experiments in a controlled network with minimal external interference.
- **ESP32 Resource Constraints**: The ESP32’s limited memory and processing power required optimized code to process packets in real-time, especially during high-traffic UDP flood scenarios. We streamlined feature calculations to prevent data loss.
- **Realistic Attack Simulation**: The `hping3` command was carefully configured to replicate real-world UDP flood attacks. Randomizing source IPs (`--rand-source`) ensured the simulation mimicked distributed attacks, while targeting port 53 added realism, as it’s a common attack vector.

## Outcome

The data collection phase produced a robust dataset:

- **Raw Data**: 1000 rows of normal traffic and 500 rows of UDP flood traffic.
- **Balanced Dataset**: 1000 rows total (500 normal, 500 udp_flood), saved as CSV files for preprocessing and model training. This dataset captured the essential characteristics of both traffic types, providing a solid foundation for developing our model.

# Model Training

Using the Edge Impulse platform, we transformed our collected dataset into a neural network model optimized for deployment on the resource-constrained ESP32 microcontroller. This section details the preprocessing, model design, training process, and evaluation, leveraging the insights from our Edge Impulse project (available at https://studio.edgeimpulse.com/public/698375/live).

## Data Preprocessing

Before training, we preprocessed the dataset collected from ESP32 A to ensure it was suitable. The raw dataset consisted of **1000 rows** of normal traffic and **500 rows** of UDP flood traffic, each row representing features extracted over a 1-second window. The features included `timestamp`, `duration_ms`, `total_pkts`, `target_pkts`, `pkt_ratio`, `bytes_total`, `avg_pkt_size`, `unique_MACs`, `mean_IAT_us`, and `IAT_variance`.

- **Data Cleaning**:

  - We removed irrelevant columns that didn’t contribute to distinguishing normal from malicious traffic, such as `timestamp` and `duration_ms` (fixed at 1000 ms). Other features like `unique_MACs` were excluded after analysis, as they showed low correlation with attack patterns.
  - The final features selected for training were:
    - **target_pkts**: Number of packets targeting the device.
    - **pkt_ratio**: Ratio of target packets to total packets.
    - **avg_pkt_size**: Average size of target packets. 
  - These features were chosen because they effectively captured the high packet volume, disproportionate target traffic, and smaller packet sizes characteristic of UDP flood attacks.

- **Labeling**:

  - Each row was labeled as either “normal” or “udp_flood” based on the scenario (normal traffic or simulated attack using `hping3` on Kali Linux).

- **Splitting the dataset** :

  - We split the dataset into **80% training** (800 samples) and **20% testing** (200 samples) to evaluate model performance on unseen data.

## Model Design

In Edge Impulse, we designed a neural network tailored for the ESP32’s limited computational resources. The model was part of an **Impulse**, which is Edge Impulse’s pipeline for processing and learning (https://docs.edgeimpulse.com/docs/impulses). 

The Impulse consisted of:

- **Input Block**: The input layer accepted the three selected features (`target_pkts`, `pkt_ratio`, `avg_pkt_size`).
- **Learning Block**: A fully connected neural network with the following architecture:
  - **Input Layer**: 3 neurons, corresponding to the three input features.
  - **Dense Layer 1**: 20 neurons with ReLU activation to capture complex patterns in the data.
  - **Dense Layer 2**: 10 neurons with ReLU activation for further feature processing.
  - **Output Layer**: 2 neurons with softmax activation, representing the probabilities for “normal” and “udp_flood” classes.

This architecture was chosen for its balance between expressiveness (to model the traffic patterns) and efficiency (to run on the ESP32). The model’s compact size ensured it could fit within the ESP32’s memory constraints.

## Training Process

The model was trained using Edge Impulse’s cloud-based training environment, which optimized hyperparameters for edge devices:

- **Epochs**: We trained the model for **50 epochs**, allowing sufficient iterations to learn patterns without overfitting.
- **Learning Rate**: Edge Impulse automatically tuned the learning rate, starting with a small value (e.g., 0.001) and adjusting it to minimize the loss function.

During training, Edge Impulse provided real-time metrics, including accuracy.

## Model Evaluation

After training, the model achieved an impressive **98.99% accuracy** on the test set (200 samples), as reported by Edge Impulse. Key evaluation metrics included:

- **Accuracy**: 98.99%, indicating the model correctly classified 198 out of 200 test samples.
- **Confusion Matrix**: The matrix showed minimal false positives and false negatives, with strong performance in distinguishing normal traffic from UDP flood attacks.
- **Precision and Recall**: High precision (correctly identifying UDP flood attacks) and recall (capturing most attack instances) confirmed the model’s reliability.
- **On-Device Performance**: Edge Impulse’s profiling tool estimated the model’s inference time at approximately 1 ms on the ESP32.

The high accuracy was attributed to the clear separation between normal and attack traffic in the feature space, as visualized in Edge Impulse’s feature explorer (available in the project dashboard at https://studio.edgeimpulse.com/public/698375/live).

## Model Export

Once trained, the model was exported as an **Arduino library** compatible with the ESP32. Edge Impulse’s deployment tool optimized the model for the ESP32’s microcontroller environment, converting it to a format that could be integrated into the Arduino IDE. The exported library included:

- The neural network weights and architecture.
- Functions to preprocess input features and run inference.
- A lightweight inference engine for efficient execution on the ESP32.

This **Arduino library** was later integrated, enabling real-time classification of network traffic, as described in the deployment section.

## Challenges and Solutions

Training the model presented a few challenges:

- **Feature Selection**: Choosing the right features was critical. We initially considered all 10 features but found that `target_pkts`, `pkt_ratio`, and `avg_pkt_size` provided the best discriminative power, reducing model complexity without sacrificing accuracy.
- **Edge Constraints**: We kept the model small (two dense layers, 30 total neurons) to ensure it could run efficiently on the ESP32, which has limited memory and processing power.

## Outcome

The training phase resulted in a highly accurate (98.99%) neural network model, optimized for deployment on ESP32 A. The model effectively distinguished normal traffic from UDP flood attacks, leveraging the three key features to achieve robust performance. The Edge Impulse platform streamlined the process, from data upload to model export, making it accessible for edge AI development. The trained model was ready for integration into our system, as detailed in the deployment section.

# Deployment on ESP32

After training the neural network model in Edge Impulse with high accuracy, we deployed it in the ESP32. 

**ESP32 A** handles packet sniffing, feature extraction, and inference, while **ESP32 B** processes predictions and sends Telegram alerts. This section details the deployment process, based on the provided firmware for both devices, including model integration, UART communication, and alert mechanisms.

## Exporting the Model

The trained model was exported from Edge Impulse as an **Arduino library**, optimized for the ESP32:

- **Model Details**: The neural network takes three input features (`target_pkts`, `pkt_ratio`, `avg_pkt_size`) and classifies traffic as "normal" or "udp_flood". The Arduino library includes the model weights and inference functions.
- **Export Process**: Using Edge Impulse’s “Arduino Library” deployment option, we downloaded a `.zip` file, which was imported into the Arduino IDE for use in ESP32 A’s firmware.
- **Efficiency**: The model is lightweight, fitting within the ESP32’s memory constraints and executing inference quickly for real-time operation.

## Deploying on ESP32 A: Sniffer and Classifier

**ESP32 A** captures Wi-Fi packets, computes features, runs the Edge Impulse model, and sends predictions to ESP32 B via UART. Here’s how it was implemented:

- **Firmware Setup**:

  - **Libraries**: Uses `Arduino.h`, `WiFi.h`, `esp_wifi.h`, and the Edge Impulse library (`ammarlouah-project-1_inferencing.h`).
  - **Wi-Fi Sniffing**: Configured in promiscuous mode (`esp_wifi_set_promiscuous(true)`) on channel 1, with a callback (`sniffer_cb`) to process packets.
  - **Packet Filtering**: Filters packets by a target MAC address (e.g., `TARGET_MAC`), tracking only relevant traffic.
  - **Feature Extraction**: Every 1-second window (`WINDOW_MS = 1000`), it calculates:
    - `target_pkts`: Number of packets matching the target MAC (`cntTarget`).
    - `pkt_ratio`: Ratio of target packets to total packets (`cntTarget / cntTotal`).
    - `avg_pkt_size`: Average size of all packets (`bytesTotal / cntTotal`).
  - **Inference**: Features are fed into the Edge Impulse model via `run_classifier`, producing a prediction (e.g., "udp_flood") and probability.
  - **UART Output**: Predictions are sent to ESP32 B via `Serial1` (TX on GPIO17) at 115200 baud in the format `PRED:<label>,<prob>` (e.g., `PRED:udp_flood,0.95`). Errors are sent as `ERR:<code>`.

- **Implementation Notes**:

  - Uses critical sections (`portENTER_CRITICAL`) to ensure atomic updates of packet statistics.
  - Resets counters after each window to prepare for the next cycle.
  - Logs activity to `Serial` for debugging.

## Deploying on ESP32 B: Alert System

**ESP32 B** receives predictions, analyzes them with a sliding window, and sends Telegram alerts when a UDP flood is detected. Here’s the deployment breakdown:

- **Firmware Setup**:

  - **Libraries**: Uses `Arduino.h`, `WiFi.h`, and `WiFiClientSecure.h` for connectivity and HTTPS requests.
  - **Wi-Fi**: Connects to a network using `WIFI_SSID` and `WIFI_PASSWORD`, retrying every 10 seconds if disconnected.
  - **UART Input**: Receives data from ESP32 A via `HardwareSerial MySerial` (RX on GPIO16) at 115200 baud, parsing lines like `PRED:udp_flood,0.95`.
  - **Sliding Window**: Maintains a buffer (`bufferFlood`) of the last 10 predictions (`WINDOW_SIZE = 10`), counting "udp_flood" occurrences (`floodCount`).
  - **Alert Logic**:
    - Threshold: If `floodCount >= 6` (`THRESHOLD = 6`) and not in flood state (`inFloodState == false`), it enters the flood state and sends a Telegram alert: "UDP flood detected by ESP2!".
    - Recovery: If `floodCount < 6` and in flood state, it exits the flood state and sends: "UDP flood cleared on ESP2".
    - Consecutive confirmation (`USE_CONSECUTIVE_CONFIRM`) is disabled (`false`), so alerts trigger immediately without requiring multiple confirmations.
  - **Telegram Alerts**: Uses `WiFiClientSecure` with `setInsecure()` to send HTTPS GET requests to `api.telegram.org`. Messages are URL-encoded and sent with a 5-second timeout.

- **Configuration**:

  - Telegram credentials (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`).
  - Initial startup sends a test message: "ESP2 started and connected to Wi-Fi".

- **Error Handling**:

  - Reconnects to Wi-Fi if the connection drops.
  - Limits `incomingBuffer` to 200 characters to prevent overflow.
  - Logs errors (e.g., `ERR:` messages) to `Serial`.

## Integration and Testing

- **UART Communication**: ESP32 A (TX on GPIO17) sends predictions to ESP32 B (RX on GPIO16) at 115200 baud, ensuring reliable one-way data transfer.
- **Testing**: Validated with simulated UDP floods:
  - ESP32 A detected floods, sent predictions (e.g., `PRED:udp_flood,0.95`).
  - ESP32 B tracked predictions, triggered alerts when `floodCount >= 6`, and cleared them when conditions normalized.
- **Robustness**: Handled high packet rates and maintained real-time performance.

# Testing and Results
<style>
.video-container {
    position: relative;
    padding-bottom: 56.25%; /* 16:9 aspect ratio */
    height: 0;
    overflow: hidden;
}

.video-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}
</style>
<div class="video-container">
    <iframe src="https://www.youtube.com/embed/EU13Hn265dQ" frameborder="0" allowfullscreen></iframe>
</div>

# Conclusion

In summary, this project successfully demonstrated the development and deployment of the system. By integrating NN model directly onto resource-constrained edge devices, we achieved remarkable accuracy in detecting UDP flood attacks in real-time. The implementation of an efficient alerting mechanism, utilizing Telegram for immediate notifications, further enhances the system's practicality and responsiveness. This work underscores the transformative potential of edge AI in bolstering IoT security, paving the way for safer and more reliable connected environments. Looking ahead, future research could focus on expanding the system’s detection capabilities to address a broader range of cyber threats or optimizing it for even lower power consumption, ensuring its viability across diverse IoT applications.
