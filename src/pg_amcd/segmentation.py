import numpy as np
from typing import Tuple, List, Dict, Any

def select_max_energy_segment_indices(
    signal: np.ndarray, 
    segment_points: int = 10000
) -> Tuple[int, int]:
    """Finds the indices of the start and end of the segment_points window 
    containing the highest energy.
    """
    N = len(signal)
    if N <= segment_points:
        return 0, N
        
    squared_signal = np.square(signal)
    window = np.ones(segment_points)
    rolling_energy = np.convolve(squared_signal, window, mode='valid')
    start_idx = int(np.argmax(rolling_energy))
    end_idx = start_idx + segment_points
    return start_idx, end_idx

def generate_sliding_windows(
    time: np.ndarray,
    signal: np.ndarray,
    fs: float,
    window_seconds: float = 1.0,
    overlap_ratio: float = 0.75
) -> List[Dict[str, Any]]:
    """Generates overlapping sliding windows over the input signal.
    
    Returns:
        A list of dictionaries containing:
            'start_idx': Start index in time/signal arrays
            'end_idx': End index (exclusive) in time/signal arrays
            'start_time': Start time of the window (seconds)
            'end_time': End time of the window (seconds)
            'time_segment': Sliced time array
            'signal_segment': Sliced signal array
    """
    segment_points = int(window_seconds * fs)
    step_points = int(segment_points * (1.0 - overlap_ratio))
    if step_points <= 0:
        step_points = 1
        
    N = len(signal)
    windows = []
    
    start_idx = 0
    while start_idx + segment_points <= N:
        end_idx = start_idx + segment_points
        windows.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': float(time[start_idx]),
            'end_time': float(time[end_idx - 1]),
            'time_segment': time[start_idx:end_idx],
            'signal_segment': signal[start_idx:end_idx]
        })
        start_idx += step_points
        
    # If no window fits, return at least one window covering the whole signal
    if not windows and N > 0:
        windows.append({
            'start_idx': 0,
            'end_idx': N,
            'start_time': float(time[0]),
            'end_time': float(time[-1]),
            'time_segment': time,
            'signal_segment': signal
        })
        
    return windows
