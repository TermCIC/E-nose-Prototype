import serial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from collections import deque
import threading
import time
import sys  # Needed for safe program termination
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from settings import find_working_port

cmap = plt.get_cmap('tab10')

# Serial port configuration
SERIAL_PORT = find_working_port()  # Change to match your device
BAUD_RATE = 9600
READING_CSV_FILE = 'enose_readings.csv'
CONTROL_CSV_FILE = 'enose_control_summary.csv'
CORRECTION_CSV_FILE = 'enose_correction_coefficients.csv'
MAX_ENTRIES = 10  # Stop after collecting 10 entries

# Storage for real-time PCA
buffer_size = 50  # Keep last 50 readings
data_buffer = deque(maxlen=buffer_size)

# Global flag to stop threads
stop_flag = threading.Event()

# Start timestamp
timestamp = time.time()

# Check if the CSV file exists, create it if not
if not os.path.exists(CONTROL_CSV_FILE):
    with open(CONTROL_CSV_FILE, 'w') as f:
        f.write("""GM102B_mean,GM302B_mean,GM502B_mean,GM702B_mean,temperature_mean,humidity_mean,GM102B_max,GM302B_max,GM502B_max,GM702B_max,temperature_max,humidity_max,GM102B_min,GM302B_min,GM502B_min,GM702B_min,temperature_min,humidity_min,GM102B_sd,GM302B_sd,GM502B_sd,GM702B_sd,temperature_sd,humidity_sd,GM102B_final,GM302B_final,GM502B_final,GM702B_final,temperature_final,humidity_final,treatment,timestamp\n""")

if not os.path.exists(CORRECTION_CSV_FILE):
    with open(CORRECTION_CSV_FILE, 'w') as f:
        f.write("sensor,a,b,intercept,treatment,timestamp\n")

if not os.path.exists(READING_CSV_FILE):
    with open(READING_CSV_FILE, 'w') as f:
        f.write(
            "GM102B_rel,GM302B_rel,GM502B_rel,GM702B_rel,temperature,humidity,treatment,timestamp,time\n")


def start_pca_plot_thread():
    pca_thread = threading.Thread(target=plot_pca, daemon=True)
    pca_thread.start()


def send_command(ser, command):
    """Sends a command to the serial device."""
    try:
        if ser and ser.is_open:
            ser.write((command + '\n').encode())  # Ensure newline for command
            print(f">>> Sent command: {command}")
        else:
            print("Serial port not open.")
    except Exception as e:
        print(f"Error sending command: {e}")


def read_serial(treatment=None, record_control_air=False):
    # Ask user for treatment name
    if not treatment:
        print("Enter the treatment name: ")
        treatment = input().strip()
    """Controls pumps, collects control and sample e-nose data, and saves normalized values to CSV."""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("Connected to Serial Port:", SERIAL_PORT)

        # Initial state: All pumps OFF
        send_command(ser, "PUMP1 OFF")
        send_command(ser, "PUMP2 OFF")

        # === STEP 1: Collect control air readings ===
        print(
            "Step 1: Running control air (PUMP2 ON) and collecting 300 control readings...")
        send_command(ser, "PUMP2 ON")

        control_readings = []
        control_count = 0

        while control_count < 300 and not stop_flag.is_set():
            line = ser.readline().decode(errors='replace').strip()
            try:
                values = list(map(float, line.split(',')))
                if len(values) == 6:
                    control_readings.append(values)
                    control_count += 1
                    print(f"Control {control_count}/300:", values)
            except ValueError:
                continue  # Skip malformed lines

            time.sleep(1)

        send_command(ser, "PUMP2 OFF")

        if not control_readings:
            print("No control readings collected.")
            return

        # Calculate control mean and sd per sensor
        import numpy as np
        control_array = np.array(control_readings)
        control_mean = np.mean(control_array, axis=0)
        control_max = np.max(control_array, axis=0)
        control_min = np.min(control_array, axis=0)
        control_sd = np.std(control_array, axis=0, ddof=1)
        final_values = control_array[-1]

        # Get environmental correction coefficients from historical control data
        correction_coeffs = get_env_correction_coefficients()

        print(f"Control Mean: {control_mean}")
        print(f"Control SD: {control_sd}")
        print(f"Control Max: {control_max}")
        print(f"Control Min: {control_min}")
        print("Final sensor readings:", final_values)

        if record_control_air:
            # Save control summary
            with open(CONTROL_CSV_FILE, 'a') as f:
                row = list(control_mean) + list(control_max) + \
                    list(control_min) + list(control_sd) + list(final_values)
                row = [f"{v:.4f}" for v in row]  # optional: format for readability
                row.append(treatment)
                row.append(str(timestamp))
                f.write(",".join(row) + "\n")
        else:
            print("Control air collected but not saved to CSV.")


        # === STEP 2: Collect sample air readings ===
        print("Step 2: Running sample air (PUMP1 ON) and collecting 100 sample readings...")
        send_command(ser, "PUMP1 ON")

        sample_readings = []
        sample_count = 0
        first_adjusted_gas = None

        while sample_count < 100 and not stop_flag.is_set():
            line = ser.readline().decode(errors='replace').strip()
            values = line.split(',')

            # Expecting 6 values: 4 gas + 2 (temp, humidity)
            if len(values) == 6 and all(v.replace('.', '', 1).isdigit() for v in values):
                values = list(map(float, values))

                gas_values = values[:4]
                temp_hum_values = values[4:]

                # Adjust only gas sensor readings
                temperature = temp_hum_values[0]
                humidity = temp_hum_values[1]

                sensor_labels = ['GM102B_mean', 'GM302B_mean',
                                 'GM502B_mean', 'GM702B_mean']
                adjusted_gas = []

                for i, v in enumerate(gas_values):
                    f = final_values[i]  # baseline value from control
                    sensor = sensor_labels[i]
                    # fallback to 0 if missing
                    a, b, intercept = correction_coeffs.get(sensor, (0, 0, 0))
                    correction = a * temperature + b * humidity
                    baseline = f + correction
                    adjusted = v - baseline
                    adjusted_gas.append(adjusted)

                if first_adjusted_gas is None:
                    first_adjusted_gas = adjusted_gas.copy()

                normalized_gas = [val - base for val,
                                  base in zip(adjusted_gas, first_adjusted_gas)]

                # Combine adjusted gas readings with unadjusted temperature and humidity
                full_reading = normalized_gas + temp_hum_values

                sample_readings.append(adjusted_gas)
                sample_count += 1
                print(f"Sample {sample_count}/100:", full_reading)

                with open(READING_CSV_FILE, 'a') as f:
                    f.write(",".join(map(str, full_reading)) +
                            f",{treatment}" + f",{timestamp}" + f",{sample_count}\n")

            time.sleep(1)

        send_command(ser, "PUMP1 OFF")
        send_command(ser, "PUMP2 OFF")

        # === Normalize and save data ===
        if not sample_readings:
            print("No sample readings received.")

    except Exception as e:
        print(f"Serial Read Error: {e}")
    finally:
        print("Cleaning up and closing serial port.")
        if ser and ser.is_open:
            ser.close()
        # Do not exit the whole program


def get_env_correction_coefficients(control_csv=CONTROL_CSV_FILE):
    try:
        df = pd.read_csv(control_csv)
        df = df.dropna()

        voc_sensors = ['GM102B_mean', 'GM302B_mean',
                       'GM502B_mean', 'GM702B_mean']
        temp_col = 'temperature_mean'
        hum_col = 'humidity_mean'

        coefficients = {}

        latest_treatment = df['treatment'].iloc[-1] if 'treatment' in df.columns else 'unknown'
        latest_timestamp = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else time.time()

        for sensor in voc_sensors:
            X = df[[temp_col, hum_col]]
            y = df[sensor]
            model = LinearRegression().fit(X, y)
            a = model.coef_[0]
            b = model.coef_[1]
            intercept = model.intercept_
            coefficients[sensor] = (a, b, intercept)

            print(f"{sensor} -> a: {a:.4f}, b: {b:.4f}, intercept: {intercept:.4f}")

            # Save to CSV
            with open(CORRECTION_CSV_FILE, 'a') as f:
                f.write(
                    f"{sensor},{a:.6f},{b:.6f},{intercept:.6f},{latest_treatment},{latest_timestamp}\n")

        return coefficients

    except Exception as e:
        print(f"Error during regression analysis: {e}")
        return {}


def plot_pca():
    """Runs in the main thread to update PCA plot."""
    plt.ion()
    fig, ax = plt.subplots()
    pca = PCA(n_components=2)

    while not stop_flag.is_set():
        try:
            # Read CSV and check for missing values
            df = pd.read_csv(READING_CSV_FILE)
            df.dropna(inplace=True)

            if len(df) > 2:
                # Extract treatments and assign colors
                treatments = df['treatment'].astype(str)
                unique_treatments = sorted(treatments.unique())

                cmap = matplotlib.colormaps.get_cmap(
                    'tab10')  # updated to avoid deprecation
                color_map = {label: cmap(i % 10)
                             for i, label in enumerate(unique_treatments)}
                colors = treatments.map(color_map)

                # PCA on sensor columns (exclude treatment and timestamp)
                transformed_data = pca.fit_transform(df.iloc[:, :-5])

                # Plot
                ax.clear()
                ax.scatter(
                    transformed_data[:, 0],
                    transformed_data[:, 1],
                    c=colors,
                    alpha=0.6
                )

                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title('PCA of Gas Sensor Readings')

                # Add legend
                handles = [
                    Line2D([0], [0], marker='o', color='w', label=label,
                           markerfacecolor=color_map[label], markersize=8)
                    for label in unique_treatments
                ]
                ax.legend(handles=handles, title="Treatment", loc="best")

                plt.draw()
                plt.pause(1)
        except Exception as e:
            print(f"PCA Plot Error: {e}")

    print("PCA thread stopped.")


def run_enose_capture(treatment_input=None, record_control_air=False):
    global treatment, timestamp
    stop_flag.clear()
    if treatment_input is None:
        print("Enter the treatment name: ")
        treatment = input().strip()
    else:
        treatment = treatment_input
    timestamp = time.time()

    serial_thread = threading.Thread(
        target=read_serial, args=(treatment,record_control_air), daemon=True)
    serial_thread.start()


def run_enose_capture_hourly(treatment_base_name="auto", record_control_air=False):
    """Runs e-nose capture every hour with a timestamped treatment name."""
    try:
        while True:
            # Generate unique treatment name using timestamp
            time_str = time.strftime("%Y%m%d_%H%M%S")
            treatment_name = f"{treatment_base_name}_{time_str}"
            print(f"\n>>> Starting new e-nose capture: {treatment_name}")

            # Run one capture
            run_enose_capture(treatment_input=treatment_name, record_control_air=False)

            # Sleep for 1 hour
            print("Waiting for 1 hour until next capture...")
            time.sleep(3600)  # 3600 seconds = 1 hour

    except KeyboardInterrupt:
        print("\nHourly capture interrupted by user.")


def run_enose_capture_10min(treatment_base_name="auto", record_control_air=False, num_cycles=1):
    """Runs e-nose capture every 10 minutes with a timestamped treatment name."""
    try:
        for cycle in range(num_cycles):
            # Generate unique treatment name using timestamp
            time_str = time.strftime("%Y%m%d_%H%M%S")
            treatment_name = f"{treatment_base_name}_{time_str}"
            print(f"\n>>> Starting capture {cycle + 1}/{num_cycles}: {treatment_name}")

            # Run one capture
            run_enose_capture(treatment_input=treatment_name, record_control_air=record_control_air)

            # Wait for capture thread to finish before next round
            while threading.active_count() > 2:  # 1 main + 1 PCA + capture thread
                time.sleep(1)

            # Countdown for next round (if not the last round)
            if cycle < num_cycles - 1:
                print("Waiting 10 minutes until next capture...")
                for i in range(600, 0, -1):
                    print(f"Next capture in {i} seconds", end='\r')
                    time.sleep(1)

        print(f"\n✅ Completed all {num_cycles} measurement(s) for: {treatment_base_name}")

    except KeyboardInterrupt:
        print("\n⛔ Measurement interrupted by user.")

    finally:
        # Stop PCA plot thread and cleanup
        stop_flag.set()
        print("Stopping PCA plot and exiting...")
        time.sleep(2)  # Allow PCA thread to exit gracefully
        plt.close("all")
        sys.exit(0)  # Graceful shutdown


if __name__ == "__main__":
    # Prompt user for treatment base name and number of cycles
    treatment_name = input("Enter the treatment name: ").strip()
    try:
        num_cycles = int(input("How many times to measure (every 10 min)? ").strip())
    except ValueError:
        print("Invalid number of cycles. Defaulting to 1.")
        num_cycles = 1

    # Start the data collection thread
    capture_thread = threading.Thread(
        target=run_enose_capture_10min,
        args=(treatment_name, False, num_cycles),
        daemon=True
    )
    capture_thread.start()

    # Run PCA plot in main thread
    plot_pca()
