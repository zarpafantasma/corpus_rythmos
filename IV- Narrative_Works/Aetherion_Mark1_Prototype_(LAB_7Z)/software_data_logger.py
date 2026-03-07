#!/usr/bin/env python3
"""
============================================================================
AETHERION MARK 1 - DATA LOGGER
============================================================================

Real-time data acquisition and logging system for Aetherion Mark 1 
vacuum gradient thruster testing.

Features:
  - Serial communication with STM32 MCU
  - Multi-channel data logging (position, temperature, acceleration)
  - Real-time plotting
  - CSV export with metadata
  - Safety monitoring with alerts

Author: RTM Engineering Team
Date: February 2026
Revision: 1.0

Requirements:
  pip install pyserial numpy matplotlib pandas

Usage:
  python data_logger.py --port COM3 --output test_001.csv
  python data_logger.py --port /dev/ttyUSB0 --plot --duration 120

============================================================================
"""

import argparse
import csv
import datetime
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Deque

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("WARNING: numpy not installed. Some features disabled.")
    np = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    PLOTTING_AVAILABLE = True
except ImportError:
    print("WARNING: matplotlib not installed. Plotting disabled.")
    PLOTTING_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration parameters"""
    # Serial
    port: str = ""
    baudrate: int = 115200
    timeout: float = 0.1
    
    # Logging
    output_file: str = ""
    log_interval_ms: int = 10  # 100 Hz
    buffer_size: int = 10000
    
    # Safety thresholds (from Red Team Advisory)
    temp_warning: float = 70.0
    temp_critical: float = 90.0
    
    # Calibration
    balance_sensitivity: float = 10.0  # nN/µm (update after CAL-1)
    
    # Display
    plot_enabled: bool = False
    plot_window: int = 1000  # samples
    verbose: bool = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SensorData:
    """Single data point from sensors"""
    timestamp_ms: int = 0
    position_um: float = 0.0
    temp1_c: float = 0.0
    temp2_c: float = 0.0
    temp3_c: float = 0.0
    temp4_c: float = 0.0
    accel_x_g: float = 0.0
    accel_y_g: float = 0.0
    accel_z_g: float = 0.0
    voltage_cmd: float = 0.0
    frequency_hz: float = 0.0
    mode: str = "IDLE"
    
    def temp_max(self) -> float:
        """Return maximum piezo temperature"""
        return max(self.temp1_c, self.temp2_c, self.temp3_c, self.temp4_c)
    
    def to_list(self) -> list:
        """Convert to list for CSV writing"""
        return [
            self.timestamp_ms,
            self.position_um,
            self.temp1_c,
            self.temp2_c,
            self.temp3_c,
            self.temp4_c,
            self.accel_x_g,
            self.accel_y_g,
            self.accel_z_g,
            self.voltage_cmd,
            self.frequency_hz,
            self.mode
        ]
    
    @staticmethod
    def csv_header() -> list:
        """Return CSV header row"""
        return [
            "timestamp_ms",
            "position_um",
            "temp1_c",
            "temp2_c",
            "temp3_c",
            "temp4_c",
            "accel_x_g",
            "accel_y_g",
            "accel_z_g",
            "voltage_cmd",
            "frequency_hz",
            "mode"
        ]


@dataclass
class TestMetadata:
    """Test session metadata"""
    test_id: str = ""
    operator: str = ""
    date: str = ""
    start_time: str = ""
    end_time: str = ""
    mode: str = ""
    frequency_hz: float = 0.0
    voltage_v: float = 0.0
    duration_ms: int = 0
    pressure_torr: float = 0.0
    ambient_temp_c: float = 0.0
    calibration_factor: float = 0.0
    notes: str = ""


# =============================================================================
# SERIAL COMMUNICATION
# =============================================================================

class AetherionSerial:
    """Serial communication with Aetherion MCU"""
    
    def __init__(self, config: Config):
        self.config = config
        self.serial: Optional[serial.Serial] = None
        self.connected = False
        self.rx_buffer = ""
        
    def connect(self) -> bool:
        """Connect to serial port"""
        try:
            self.serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout
            )
            self.connected = True
            print(f"Connected to {self.config.port} at {self.config.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"ERROR: Could not connect to {self.config.port}: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial port"""
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.connected = False
        print("Disconnected from serial port")
    
    def send_command(self, cmd: str) -> bool:
        """Send command to MCU"""
        if not self.connected:
            return False
        try:
            self.serial.write(f"{cmd}\r\n".encode())
            if self.config.verbose:
                print(f"TX: {cmd}")
            return True
        except serial.SerialException as e:
            print(f"ERROR: Send failed: {e}")
            return False
    
    def read_line(self) -> Optional[str]:
        """Read line from MCU"""
        if not self.connected:
            return None
        try:
            if self.serial.in_waiting > 0:
                data = self.serial.readline().decode('utf-8', errors='ignore').strip()
                if data and self.config.verbose:
                    print(f"RX: {data}")
                return data
        except serial.SerialException:
            pass
        return None
    
    def request_status(self) -> Optional[SensorData]:
        """Request and parse status from MCU"""
        self.send_command("status_raw")
        
        # Wait for response
        time.sleep(0.05)
        
        # Read response lines
        lines = []
        while True:
            line = self.read_line()
            if line:
                lines.append(line)
            else:
                break
        
        # Parse response
        return self._parse_status(lines)
    
    def _parse_status(self, lines: List[str]) -> Optional[SensorData]:
        """Parse status response into SensorData"""
        data = SensorData()
        data.timestamp_ms = int(time.time() * 1000)
        
        for line in lines:
            try:
                if line.startswith("POS:"):
                    data.position_um = float(line.split(":")[1])
                elif line.startswith("T1:"):
                    data.temp1_c = float(line.split(":")[1])
                elif line.startswith("T2:"):
                    data.temp2_c = float(line.split(":")[1])
                elif line.startswith("T3:"):
                    data.temp3_c = float(line.split(":")[1])
                elif line.startswith("T4:"):
                    data.temp4_c = float(line.split(":")[1])
                elif line.startswith("AX:"):
                    data.accel_x_g = float(line.split(":")[1])
                elif line.startswith("AY:"):
                    data.accel_y_g = float(line.split(":")[1])
                elif line.startswith("AZ:"):
                    data.accel_z_g = float(line.split(":")[1])
                elif line.startswith("V:"):
                    data.voltage_cmd = float(line.split(":")[1])
                elif line.startswith("F:"):
                    data.frequency_hz = float(line.split(":")[1])
                elif line.startswith("MODE:"):
                    data.mode = line.split(":")[1].strip()
            except (ValueError, IndexError):
                continue
        
        return data


# =============================================================================
# DATA LOGGER
# =============================================================================

class DataLogger:
    """Main data logging class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.serial = AetherionSerial(config)
        self.data_buffer: Deque[SensorData] = deque(maxlen=config.buffer_size)
        self.running = False
        self.recording = False
        self.csv_file = None
        self.csv_writer = None
        self.metadata = TestMetadata()
        self.start_time = 0
        
        # Plotting data
        self.plot_time: Deque[float] = deque(maxlen=config.plot_window)
        self.plot_position: Deque[float] = deque(maxlen=config.plot_window)
        self.plot_temp: Deque[float] = deque(maxlen=config.plot_window)
        
        # Statistics
        self.sample_count = 0
        self.warning_count = 0
        
    def start(self) -> bool:
        """Start the data logger"""
        if not self.serial.connect():
            return False
        
        self.running = True
        self.start_time = time.time()
        print("Data logger started")
        return True
    
    def stop(self):
        """Stop the data logger"""
        self.running = False
        self.stop_recording()
        self.serial.disconnect()
        print(f"Data logger stopped. Total samples: {self.sample_count}")
    
    def start_recording(self, filename: str, metadata: TestMetadata):
        """Start recording to CSV file"""
        try:
            self.metadata = metadata
            self.metadata.date = datetime.datetime.now().strftime("%Y-%m-%d")
            self.metadata.start_time = datetime.datetime.now().strftime("%H:%M:%S")
            
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write metadata header
            self._write_metadata_header()
            
            # Write column headers
            self.csv_writer.writerow(SensorData.csv_header())
            
            self.recording = True
            print(f"Recording started: {filename}")
            
        except IOError as e:
            print(f"ERROR: Could not open file: {e}")
            self.recording = False
    
    def stop_recording(self):
        """Stop recording"""
        if self.recording:
            self.metadata.end_time = datetime.datetime.now().strftime("%H:%M:%S")
            self.recording = False
            
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None
            
            print("Recording stopped")
    
    def _write_metadata_header(self):
        """Write metadata to CSV header"""
        self.csv_writer.writerow(["# Aetherion Mark 1 Test Data"])
        self.csv_writer.writerow([f"# Test ID: {self.metadata.test_id}"])
        self.csv_writer.writerow([f"# Operator: {self.metadata.operator}"])
        self.csv_writer.writerow([f"# Date: {self.metadata.date}"])
        self.csv_writer.writerow([f"# Start Time: {self.metadata.start_time}"])
        self.csv_writer.writerow([f"# Mode: {self.metadata.mode}"])
        self.csv_writer.writerow([f"# Frequency: {self.metadata.frequency_hz} Hz"])
        self.csv_writer.writerow([f"# Voltage: {self.metadata.voltage_v} V"])
        self.csv_writer.writerow([f"# Pressure: {self.metadata.pressure_torr} Torr"])
        self.csv_writer.writerow([f"# Calibration Factor: {self.metadata.calibration_factor} nN/um"])
        self.csv_writer.writerow([f"# Notes: {self.metadata.notes}"])
        self.csv_writer.writerow(["#"])
    
    def acquire_sample(self) -> Optional[SensorData]:
        """Acquire single data sample"""
        data = self.serial.request_status()
        
        if data:
            self.sample_count += 1
            self.data_buffer.append(data)
            
            # Update plot data
            elapsed = time.time() - self.start_time
            self.plot_time.append(elapsed)
            self.plot_position.append(data.position_um)
            self.plot_temp.append(data.temp_max())
            
            # Write to CSV if recording
            if self.recording and self.csv_writer:
                self.csv_writer.writerow(data.to_list())
            
            # Safety checks
            self._check_safety(data)
            
        return data
    
    def _check_safety(self, data: SensorData):
        """Check safety thresholds"""
        temp_max = data.temp_max()
        
        if temp_max >= self.config.temp_critical:
            print(f"\n!!! CRITICAL: Temperature {temp_max:.1f}°C >= {self.config.temp_critical}°C !!!")
            print("!!! SENDING E-STOP COMMAND !!!\n")
            self.serial.send_command("estop")
            self.warning_count += 1
            
        elif temp_max >= self.config.temp_warning:
            if self.warning_count % 10 == 0:  # Don't spam
                print(f"WARNING: Temperature {temp_max:.1f}°C >= {self.config.temp_warning}°C")
            self.warning_count += 1
    
    def run_acquisition_loop(self, duration_s: float = None):
        """Run continuous acquisition loop"""
        interval = self.config.log_interval_ms / 1000.0
        end_time = time.time() + duration_s if duration_s else None
        
        print(f"Acquiring data at {1000/self.config.log_interval_ms:.0f} Hz")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Acquire sample
                data = self.acquire_sample()
                
                # Display status
                if data and self.sample_count % 100 == 0:
                    self._print_status(data)
                
                # Check duration limit
                if end_time and time.time() >= end_time:
                    print("\nDuration limit reached")
                    break
                
                # Maintain sample rate
                elapsed = time.time() - loop_start
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
    
    def _print_status(self, data: SensorData):
        """Print current status line"""
        elapsed = time.time() - self.start_time
        thrust = data.position_um * self.config.balance_sensitivity
        
        print(f"[{elapsed:6.1f}s] "
              f"Pos: {data.position_um:7.2f} µm | "
              f"Thrust: {thrust:7.1f} nN | "
              f"Temp: {data.temp_max():5.1f}°C | "
              f"Mode: {data.mode}")
    
    def calculate_statistics(self) -> dict:
        """Calculate statistics from buffered data"""
        if not self.data_buffer or not np:
            return {}
        
        positions = [d.position_um for d in self.data_buffer]
        temps = [d.temp_max() for d in self.data_buffer]
        
        stats = {
            'samples': len(self.data_buffer),
            'duration_s': (self.data_buffer[-1].timestamp_ms - 
                          self.data_buffer[0].timestamp_ms) / 1000.0,
            'position_mean': np.mean(positions),
            'position_std': np.std(positions),
            'position_min': np.min(positions),
            'position_max': np.max(positions),
            'temp_mean': np.mean(temps),
            'temp_max': np.max(temps),
            'thrust_mean': np.mean(positions) * self.config.balance_sensitivity,
            'thrust_std': np.std(positions) * self.config.balance_sensitivity,
        }
        
        return stats
    
    def print_statistics(self):
        """Print session statistics"""
        stats = self.calculate_statistics()
        
        if not stats:
            print("No data collected")
            return
        
        print("\n" + "="*60)
        print("SESSION STATISTICS")
        print("="*60)
        print(f"Samples:          {stats['samples']}")
        print(f"Duration:         {stats['duration_s']:.1f} s")
        print(f"Position Mean:    {stats['position_mean']:.3f} µm")
        print(f"Position Std:     {stats['position_std']:.3f} µm")
        print(f"Position Range:   {stats['position_min']:.3f} to {stats['position_max']:.3f} µm")
        print(f"Thrust Mean:      {stats['thrust_mean']:.1f} nN")
        print(f"Thrust Std:       {stats['thrust_std']:.1f} nN")
        print(f"Max Temperature:  {stats['temp_max']:.1f} °C")
        print("="*60 + "\n")


# =============================================================================
# REAL-TIME PLOTTING
# =============================================================================

class RealTimePlotter:
    """Real-time data visualization"""
    
    def __init__(self, logger: DataLogger, config: Config):
        self.logger = logger
        self.config = config
        self.fig = None
        self.axes = None
        self.lines = {}
        
    def setup_plot(self):
        """Setup matplotlib figure"""
        if not PLOTTING_AVAILABLE:
            return
        
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('AETHERION MARK 1 - Real-Time Monitor', fontsize=14)
        
        # Position/Thrust plot
        self.axes[0].set_ylabel('Position (µm)')
        self.axes[0].set_title('Torsion Balance Deflection')
        self.axes[0].grid(True, alpha=0.3)
        self.lines['position'], = self.axes[0].plot([], [], 'g-', linewidth=1)
        
        # Temperature plot
        self.axes[1].set_ylabel('Temperature (°C)')
        self.axes[1].set_title('Piezo Array Temperature')
        self.axes[1].grid(True, alpha=0.3)
        self.axes[1].axhline(y=self.config.temp_warning, color='yellow', 
                             linestyle='--', label='Warning')
        self.axes[1].axhline(y=self.config.temp_critical, color='red', 
                             linestyle='--', label='Critical')
        self.lines['temp'], = self.axes[1].plot([], [], 'orange', linewidth=1)
        self.axes[1].legend(loc='upper right')
        
        # Thrust calculation
        self.axes[2].set_xlabel('Time (s)')
        self.axes[2].set_ylabel('Thrust (nN)')
        self.axes[2].set_title('Calculated Thrust')
        self.axes[2].grid(True, alpha=0.3)
        self.lines['thrust'], = self.axes[2].plot([], [], 'cyan', linewidth=1)
        
        plt.tight_layout()
        
    def update_plot(self, frame):
        """Update plot with new data"""
        if not self.logger.plot_time:
            return self.lines.values()
        
        times = list(self.logger.plot_time)
        positions = list(self.logger.plot_position)
        temps = list(self.logger.plot_temp)
        thrusts = [p * self.config.balance_sensitivity for p in positions]
        
        # Update data
        self.lines['position'].set_data(times, positions)
        self.lines['temp'].set_data(times, temps)
        self.lines['thrust'].set_data(times, thrusts)
        
        # Adjust axes
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        
        return self.lines.values()
    
    def run(self):
        """Run animated plot"""
        if not PLOTTING_AVAILABLE:
            print("Plotting not available (matplotlib not installed)")
            return
        
        self.setup_plot()
        
        ani = FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)
        plt.show()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def list_ports():
    """List available serial ports"""
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("No serial ports found")
        return
    
    print("\nAvailable serial ports:")
    print("-" * 50)
    for port in ports:
        print(f"  {port.device}: {port.description}")
    print()


def interactive_setup() -> tuple:
    """Interactive setup wizard"""
    print("\n" + "="*60)
    print("AETHERION MARK 1 - DATA LOGGER SETUP")
    print("="*60 + "\n")
    
    # List ports
    list_ports()
    
    # Get port
    port = input("Enter serial port (e.g., COM3 or /dev/ttyUSB0): ").strip()
    
    # Get test info
    test_id = input("Enter test ID (e.g., AT-001): ").strip() or "TEST"
    operator = input("Enter operator name: ").strip() or "Unknown"
    
    # Get test parameters
    mode = input("Enter mode (OMV/TPH/HYBRID) [OMV]: ").strip().upper() or "OMV"
    
    try:
        freq = float(input("Enter frequency in Hz [2000]: ").strip() or "2000")
    except ValueError:
        freq = 2000.0
    
    try:
        voltage = float(input("Enter voltage in V [150]: ").strip() or "150")
    except ValueError:
        voltage = 150.0
    
    try:
        cal_factor = float(input("Enter calibration factor in nN/µm [10.0]: ").strip() or "10.0")
    except ValueError:
        cal_factor = 10.0
    
    notes = input("Enter notes (optional): ").strip()
    
    # Create config
    config = Config(
        port=port,
        balance_sensitivity=cal_factor,
        verbose=False
    )
    
    # Create metadata
    metadata = TestMetadata(
        test_id=test_id,
        operator=operator,
        mode=mode,
        frequency_hz=freq,
        voltage_v=voltage,
        calibration_factor=cal_factor,
        notes=notes
    )
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"AETHERION_{test_id}_{timestamp}.csv"
    
    return config, metadata, filename


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Aetherion Mark 1 Data Logger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_logger.py --list-ports
  python data_logger.py --port COM3 --output test.csv
  python data_logger.py --port /dev/ttyUSB0 --plot --duration 60
  python data_logger.py --interactive
        """
    )
    
    parser.add_argument('--port', '-p', help='Serial port')
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--duration', '-d', type=float, help='Recording duration (seconds)')
    parser.add_argument('--plot', action='store_true', help='Enable real-time plotting')
    parser.add_argument('--list-ports', action='store_true', help='List available ports')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive setup')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--cal-factor', type=float, default=10.0, 
                        help='Calibration factor nN/µm (default: 10.0)')
    
    args = parser.parse_args()
    
    # List ports and exit
    if args.list_ports:
        list_ports()
        return
    
    # Interactive mode
    if args.interactive:
        config, metadata, filename = interactive_setup()
    else:
        # Command line mode
        if not args.port:
            parser.print_help()
            print("\nERROR: --port is required (or use --interactive)")
            return
        
        config = Config(
            port=args.port,
            balance_sensitivity=args.cal_factor,
            verbose=args.verbose,
            plot_enabled=args.plot
        )
        
        metadata = TestMetadata(
            test_id="CLI_TEST",
            calibration_factor=args.cal_factor
        )
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = args.output or f"AETHERION_LOG_{timestamp}.csv"
    
    # Create logger
    logger = DataLogger(config)
    
    # Start
    if not logger.start():
        return
    
    # Start recording
    logger.start_recording(filename, metadata)
    
    # Run with or without plotting
    if config.plot_enabled and PLOTTING_AVAILABLE:
        # Start acquisition in background thread
        acq_thread = threading.Thread(
            target=logger.run_acquisition_loop,
            args=(args.duration,)
        )
        acq_thread.daemon = True
        acq_thread.start()
        
        # Run plotter in main thread
        plotter = RealTimePlotter(logger, config)
        plotter.run()
    else:
        # Run acquisition in main thread
        logger.run_acquisition_loop(args.duration)
    
    # Stop and print stats
    logger.stop()
    logger.print_statistics()
    
    print(f"\nData saved to: {filename}")


if __name__ == "__main__":
    main()
