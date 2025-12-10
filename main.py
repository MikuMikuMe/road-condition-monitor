Creating a comprehensive "road-condition-monitor" system involves a combination of IoT devices to collect data, a backend to process the data, and a machine learning model to predict road conditions. Below is a simplified version of such a system implemented in Python. This example will demonstrate data collection, storing, and processing, but note that it's highly conceptual without hardware integration:

```python
import random
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SensorDataSimulator:
    """
    Simulate sensor data from IoT devices that could measure temperature, humidity, road vibration, etc.
    """
    def __init__(self):
        # Define possible road condition states
        self.states = ["Dry", "Wet", "Icy"]

    def generate_data(self):
        # Simulate sensor data
        return {
            "temperature": random.uniform(-10, 35),  # in Celsius
            "humidity": random.uniform(0, 100),      # in percentage
            "vibration": random.uniform(0, 5),       # arbitrary units
            "condition": random.choice(self.states)  # real condition for training/testing
        }

class RoadConditionMonitor:
    """
    Monitor road conditions in real-time using sensor data and provide predictions using a trained ML model.
    """
    def __init__(self):
        self.model = RandomForestClassifier()
        self.sensor_data = SensorDataSimulator()
        self.training_data = []
        self.target_data = []

    def collect_data(self, samples=100):
        try:
            for _ in range(samples):
                data = self.sensor_data.generate_data()
                # Log data collection process
                logging.info(f"Collected data: {data}")
                self.training_data.append([
                    data["temperature"],
                    data["humidity"],
                    data["vibration"]
                ])
                self.target_data.append(data["condition"])
        except Exception as e:
            logging.error(f"Error collecting data: {e}")

    def train_model(self):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.training_data, self.target_data, test_size=0.2, random_state=42
            )
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logging.info(f"Model trained with accuracy: {accuracy:.2f}")
        except Exception as e:
            logging.error(f"Error training model: {e}")

    def predict_condition(self, temperature, humidity, vibration):
        try:
            data_point = np.array([[temperature, humidity, vibration]])
            prediction = self.model.predict(data_point)
            logging.info(f"Predicted condition: {prediction[0]}")
            return prediction[0]
        except Exception as e:
            logging.error(f"Error predicting condition: {e}")
            return None

if __name__ == "__main__":
    monitor = RoadConditionMonitor()
    
    # Collect sample data
    monitor.collect_data(samples=500)
    
    # Train the model
    monitor.train_model()

    # Simulate prediction
    temperature = random.uniform(-10, 35)
    humidity = random.uniform(0, 100)
    vibration = random.uniform(0, 5)
    
    # Log prediction request
    logging.info(f"Requesting prediction for data - Temperature: {temperature}, Humidity: {humidity}, Vibration: {vibration}")
    
    # Get prediction
    predicted_condition = monitor.predict_condition(temperature, humidity, vibration)
```

### Notes:
1. **SensorDataSimulator**: This is a mock class that simulates data input. In a real IoT application, this would be replaced with actual sensor data.
   
2. **RandomForestClassifier**: Used here for its simplicity in handling classification tasks. Depending on the complexity of the data, you might choose a more sophisticated model.

3. **Error Handling**: Each operation such as data collection, model training, and prediction is enclosed in try-except blocks to handle potential exceptions gracefully.

4. **Logging**: Logging is configured to provide detailed information about the system's operations and any issues encountered.

5. **Scalability**: This example is highly simplified. Real-world implementations would require handling more sophisticated data pipelines, multiple sensors, data aggregation across larger geographic regions, handling asynchronous data input, and potentially cloud-based storage and processing.