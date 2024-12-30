import io
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
from PIL import Image
from numpy import ndarray
from tqdm import tqdm

from data.neuro_mnist_loader import load_n_mnist_events
from utils.globals import PATH_TO_N_MNIST

"""
Visualise the N-MNIST dataset as a gif
"""

# CONSTANTS
DURATION = 50


def create_frames(fig, ax, events_dict: Dict[int, Dict[str, ndarray]], max_frames: int) -> list:
    frames = []
    fps = 60

    # Create a color map for visualization
    c1 = '#21c65b'  # Green for ON events
    c2 = '#aa46f2'  # Purple for OFF events

    prev_events = {i: {'x': None, 'y': None, 'p': None} for i in range(9)}

    progress_bar = tqdm(range(max_frames), desc='Creating frames', leave=False)

    for frame_idx in progress_bar:
        # Clear previous frame client_runs
        for subplot_ax in ax.flat:
            subplot_ax.clear()
            subplot_ax.set_facecolor('black')

        # Process each subplot
        for idx, subplot_ax in enumerate(ax.flat):
            event_num = idx + 1
            if event_num not in events_dict:
                continue

            event = events_dict[event_num]
            td_x, td_y = event['x'], event['y']
            td_p, td_ts = event['polarity'], event['timestamp']

            # Calculate time window for current frame
            min_timestamp = td_ts.min()
            max_timestamp = td_ts.max()
            total_duration = max_timestamp - min_timestamp
            events_per_frame = max(1, total_duration // fps)

            start_time = frame_idx * events_per_frame
            end_time = start_time + events_per_frame

            # Get events in the current time window
            mask = (td_ts >= start_time) & (td_ts < end_time)
            x_segment = td_x[mask]
            y_segment = td_y[mask]
            p_segment = td_p[mask]

            # Plot current events
            if len(x_segment) > 0:
                subplot_ax.scatter(x_segment, y_segment, c=np.where(p_segment == 1, c1, c2),  marker='.', s=10)

            # Plot previous events with fade effect
            prev = prev_events[idx]
            if prev['x'] is not None:
                subplot_ax.scatter(prev['x'], prev['y'], c=np.where(prev['p'] == 1, c1, c2), marker='.', alpha=0.5, s=10)

            # Update previous events
            prev_events[idx] = {
                'x': x_segment,
                'y': y_segment,
                'p': p_segment
            }

            # Configure subplot
            subplot_ax.invert_yaxis()
            subplot_ax.set_xticks([])
            subplot_ax.set_yticks([])
            subplot_ax.set_title(f'Digit {event_num}', color='white')

        # Adjust layout and save frame
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='black', dpi=100)
        buf.seek(0)
        frame = Image.open(buf)
        frames.append(frame.copy())  # Create a copy to prevent memory issues
        buf.close()

    return frames


def create_gifs(events_dict: Dict[int, Dict[str, ndarray]], out_path: str):
    # Create figure with subplots
    fig, ax = plt.subplots(3, 3, figsize=(12, 12), facecolor='black')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Calculate maximum frames
    num_frames_list = [len(frame_dict["timestamp"]) for frame_dict in events_dict.values()]
    max_frames = min(max(num_frames_list) // 60, 100)  # Limit frames for reasonable file size

    # Create and save frames
    frames = create_frames(fig, ax, events_dict, max_frames)

    # Save as GIF
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION,
        loop=0
    )

    plt.close(fig)
    print(f"GIF saved as {out_path}")


def load_multiple_nmnist():
    """
    Loads multiple N-MNIST events into memory.
    Returns: A dictionary of N-MNIST events
    """
    events_dict = {}
    spiking_numbers = list(range(1, 10))  # Numbers 1-9

    for num in spiking_numbers:
        num_path = os.path.join(PATH_TO_N_MNIST, 'Train', str(num))
        if not os.path.exists(num_path):
            print(f"Warning: Path {num_path} does not exist")
            continue

        bins = os.listdir(num_path)
        if not bins:
            print(f"Warning: No bin files found in {num_path}")
            continue

        # Get a random bin file
        bin_file = rng.choice(bins)
        file_path = os.path.join(num_path, bin_file)

        td_x, td_y, td_ts, td_p = load_n_mnist_events(file_path)

        # Normalize timestamps to start from 1
        replacement_ts = np.arange(1, len(td_ts) + 1)

        events_dict[num] = {
            'x': td_x,
            'y': td_y,
            'polarity': td_p,
            'timestamp': replacement_ts
        }

    return events_dict


# Main execution
if __name__ == "__main__":
    # Create necessary directories
    for directory in ['frames', 'gifs']:
        os.makedirs(directory, exist_ok=True)

    output_path = 'gifs/n_mnist_animation.gif'

    events = load_multiple_nmnist()
    create_gifs(events, output_path)

    print('gif saved to ', output_path)