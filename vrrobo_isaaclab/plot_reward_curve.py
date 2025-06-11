import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

#!/usr/bin/env python3
"""
This script extracts the scalar data for 'Train/mean_reward' from two TensorBoard event files
and plots both reward curves for comparison.
"""

# Paths to the TensorBoard event files, update if needed.
EVENT_FILE1 = (
    "/home/zhust/codes/go2_isaaclab/logs/rsl_rl/unitree_go2_gsfix/"
    "2025-04-01_17-09-02/events.out.tfevents.1743498560.zhust-System-Product-Name.1430095.0"
)
EVENT_FILE2 = (
    "/home/zhust/codes/go2_isaaclab/logs/rsl_rl/unitree_go2_gsfix/"
    "2025-04-04_00-39-50/events.out.tfevents.1743698398.zhust-System-Product-Name.2319220.0"
)

def extract_scalar(tag: str, event_file: str):
    # Initialize EventAccumulator with scalars size guidance set to 0 to load all events.
    event_acc = EventAccumulator(event_file, size_guidance={'scalars': 0})
    event_acc.Reload()
    
    if tag not in event_acc.Tags().get("scalars", []):
        print(f"Tag '{tag}' not found in the event file: {event_file}")
        sys.exit(1)
    
    return event_acc.Scalars(tag)

def main():
    tag = "Train/mean_reward"
    
    # Verify event files exist
    for file in [EVENT_FILE1, EVENT_FILE2]:
        if not os.path.exists(file):
            print(f"Event file not found: {file}")
            sys.exit(1)
    
    # Extract rewards from both event files.
    scalar_events1 = extract_scalar(tag, EVENT_FILE1)
    scalar_events2 = extract_scalar(tag, EVENT_FILE2)
    
    # Unpack steps and values for both runs.
    steps1 = [event.step for event in scalar_events1]
    values1 = [event.value for event in scalar_events1]
    
    steps2 = [event.step for event in scalar_events2]
    values2 = [event.value for event in scalar_events2]
    
    def smooth(values, factor=0.9):
        smoothed = []
        s = values[0]
        smoothed.append(s)
        for v in values[1:]:
            s = factor * s + (1 - factor) * v
            smoothed.append(s)
        return smoothed

    smoothed_values1 = smooth(values1, 0.99)
    smoothed_values2 = smooth(values2, 0.99)

    # Set Times New Roman as the font for the plot.
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    # Plot both original and smoothed reward curves for comparison.
    plt.figure(figsize=(4, 2.5))
    # Run 1: plot original (faded) and smoothed curves.
    plt.plot(steps1, values1, linestyle='-', color='b', alpha=0.4)
    plt.plot(steps1, smoothed_values1, linestyle='-', color='b', label="ViT encoder")
    # Run 2: plot original (faded) and smoothed curves.
    plt.plot(steps2, values2, linestyle='-', color='r', alpha=0.4)
    plt.plot(steps2, smoothed_values2, linestyle='-', color='r', label="CNN encoder")
    plt.xlabel("Training Step")
    plt.ylabel("Train/mean_reward")
    # plt.title("Comparison of Training Mean Reward Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot.
    plt.show()
    # Save the plot to a file.
    output_path = "reward_curve_comparison.png"
    plt.savefig(output_path)

if __name__ == "__main__":
    main()