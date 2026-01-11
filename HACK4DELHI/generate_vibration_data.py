"""
Smart Railway Vibration Dataset Generator
==========================================
Strategy:
1. Generate realistic NORMAL train passage patterns
2. Transform normal patterns into ABNORMAL tampering patterns
   by applying mathematical transformations (frequency shifts, 
   amplitude modulation, impulse injection)

This approach ensures abnormal data is based on real track conditions
but with tampering signatures added.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

# Create output directories
os.makedirs('training_data/normal', exist_ok=True)
os.makedirs('training_data/abnormal', exist_ok=True)

print("=" * 70)
print("🚆 SMART RAILWAY VIBRATION DATASET GENERATOR")
print("=" * 70)

# ============================================================================
# STEP 1: GENERATE NORMAL PATTERNS (Train Passing)
# ============================================================================

def generate_normal_train_passage(duration=2.0, sample_rate=100, seed=None):
    """
    Generate realistic normal train passage vibration
    
    Characteristics:
    - Smooth envelope (approach → pass → leave)
    - Low frequency (5-20 Hz) - wheel impacts
    - Moderate amplitude (0.02 - 0.15 g)
    - Periodic oscillations
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Train type variations
    train_types = ['freight', 'passenger', 'metro']
    train_type = np.random.choice(train_types)
    
    if train_type == 'freight':
        base_freq = np.random.uniform(6, 10)    # Slower, heavier
        amplitude = np.random.uniform(0.06, 0.12)
        noise_level = 0.015
    elif train_type == 'passenger':
        base_freq = np.random.uniform(10, 15)   # Faster, smoother
        amplitude = np.random.uniform(0.04, 0.08)
        noise_level = 0.01
    else:  # metro
        base_freq = np.random.uniform(13, 18)
        amplitude = np.random.uniform(0.03, 0.06)
        noise_level = 0.008
    
    # Gaussian envelope (train approaching and passing)
    envelope = np.exp(-((t - duration/2) ** 2) / (duration/3))
    
    # Primary oscillation (wheel impacts on rail)
    primary = amplitude * np.sin(2 * np.pi * base_freq * t)
    
    # Harmonics (track natural frequencies)
    harmonic1 = 0.3 * amplitude * np.sin(2 * np.pi * base_freq * 2 * t + np.random.uniform(0, 2*np.pi))
    harmonic2 = 0.15 * amplitude * np.sin(2 * np.pi * base_freq * 3 * t + np.random.uniform(0, 2*np.pi))
    
    # Rail joints (periodic impacts every ~12m)
    joint_freq = base_freq / np.random.uniform(5, 7)
    joints = 0.15 * amplitude * np.sin(2 * np.pi * joint_freq * t)
    
    # Random micro-variations (track irregularities)
    micro_var = 0.05 * amplitude * np.random.randn(len(t))
    
    # Background ambient noise
    noise = noise_level * np.random.randn(len(t))
    
    # Combine all components
    acceleration = envelope * (primary + harmonic1 + harmonic2 + joints + micro_var) + noise
    
    return t, acceleration

def generate_idle_track(duration=2.0, sample_rate=100, seed=None):
    """
    Generate idle track vibration (no train)
    Very low amplitude environmental noise
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Very low amplitude ambient vibrations
    ambient = 0.008 * np.random.randn(len(t))
    
    # Slow drift (temperature, wind)
    drift = 0.003 * np.sin(2 * np.pi * 0.3 * t)
    
    # Occasional micro-events (birds, distant traffic)
    num_events = np.random.randint(2, 5)
    events = np.zeros_like(t)
    for _ in range(num_events):
        idx = np.random.randint(0, len(t) - 20)
        event_len = np.random.randint(10, 20)
        events[idx:idx+event_len] += 0.01 * np.exp(-np.arange(event_len)/5)
    
    acceleration = ambient + drift + events
    
    return t, acceleration

# ============================================================================
# STEP 2: TRANSFORM NORMAL → ABNORMAL (Tampering Signatures)
# ============================================================================

def apply_cutting_transformation(t, normal_acc):
    """
    Transform normal signal → cutting/sawing pattern
    
    Method: Add high-frequency continuous vibration
    """
    # High frequency cutting vibration (40-80 Hz)
    cutting_freq = np.random.uniform(45, 75)
    cutting_amplitude = np.random.uniform(0.4, 0.9)
    
    # Continuous cutting with on/off periods
    on_periods = np.random.rand(len(t)) > 0.25  # 75% cutting time
    on_periods = np.convolve(on_periods, np.ones(15)/15, mode='same') > 0.5
    
    # Main cutting signal
    cutting = cutting_amplitude * np.sin(2 * np.pi * cutting_freq * t)
    
    # Tool vibration modulation (chatter)
    modulation = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)
    
    # High frequency harmonics
    harmonic = 0.25 * cutting_amplitude * np.sin(2 * np.pi * cutting_freq * 1.5 * t)
    
    # Combine: amplify normal + add cutting signature
    abnormal = 2.0 * normal_acc + on_periods * (cutting * modulation + harmonic)
    
    # Random impulses (tool catching)
    num_impulses = np.random.randint(3, 7)
    for _ in range(num_impulses):
        idx = np.random.randint(0, len(t) - 30)
        impulse_len = 25
        impulse = np.random.uniform(0.6, 1.2) * np.exp(-np.arange(impulse_len)/6)
        abnormal[idx:idx+impulse_len] += impulse
    
    return abnormal

def apply_hammering_transformation(t, normal_acc):
    """
    Transform normal signal → hammering pattern
    
    Method: Add periodic sharp impulses
    """
    # Number of hammer strikes
    num_strikes = np.random.randint(10, 18)
    
    # Somewhat regular timing
    base_interval = len(t) / num_strikes
    strike_indices = []
    for i in range(num_strikes):
        jitter = np.random.uniform(-base_interval*0.2, base_interval*0.2)
        idx = int(i * base_interval + jitter)
        if 0 <= idx < len(t) - 50:
            strike_indices.append(idx)
    
    # Start with amplified normal signal
    abnormal = 1.5 * normal_acc
    
    # Add hammer strikes
    for idx in strike_indices:
        # Impact magnitude
        impact_mag = np.random.uniform(1.0, 2.2)
        
        # Sharp impact with exponential decay
        decay_len = np.random.randint(35, 55)
        decay = impact_mag * np.exp(-np.arange(decay_len) / 10)
        
        # Metal ringing (high frequency resonance)
        ring_freq = np.random.uniform(120, 180)
        ringing = 0.4 * np.sin(2 * np.pi * ring_freq * np.arange(decay_len) / 100)
        
        # Combine impact and ringing
        strike = decay * (1 + ringing)
        
        # Add to signal
        end_idx = min(idx + decay_len, len(abnormal))
        actual_len = end_idx - idx
        abnormal[idx:end_idx] += strike[:actual_len]
    
    return abnormal

def apply_grinding_transformation(t, normal_acc):
    """
    Transform normal signal → grinding pattern
    
    Method: Add very high frequency continuous vibration
    """
    # Very high frequency grinding (60-120 Hz)
    grind_freq = np.random.uniform(65, 110)
    grind_amplitude = np.random.uniform(0.5, 0.9)
    
    # Varying intensity (pressure changes)
    intensity = 1 + 0.4 * np.sin(2 * np.pi * 2.5 * t)
    
    # Main grinding signal
    grinding = grind_amplitude * intensity * np.sin(2 * np.pi * grind_freq * t)
    
    # Multiple harmonics
    harm1 = 0.3 * grind_amplitude * np.sin(2 * np.pi * grind_freq * 1.4 * t)
    harm2 = 0.2 * grind_amplitude * np.sin(2 * np.pi * grind_freq * 2.1 * t)
    
    # Random amplitude spikes (tool catching)
    spikes = np.zeros_like(t)
    num_spikes = np.random.randint(15, 25)
    for _ in range(num_spikes):
        idx = np.random.randint(0, len(t) - 20)
        spike_len = 18
        spike_mag = np.random.uniform(0.4, 0.9)
        spike = spike_mag * np.exp(-np.arange(spike_len)/5)
        spikes[idx:idx+spike_len] += spike
    
    # Combine: amplify normal + add grinding
    abnormal = 2.5 * normal_acc + grinding + harm1 + harm2 + spikes
    
    return abnormal

def apply_drilling_transformation(t, normal_acc):
    """
    Transform normal signal → drilling pattern
    
    Method: Add high frequency with increasing amplitude (penetration)
    """
    # Drill rotation frequency
    drill_freq = np.random.uniform(85, 130)
    
    # Progressive penetration (increasing amplitude)
    penetration = 0.5 + 0.8 * (t / t[-1])  # Ramps from 0.5 to 1.3
    
    # Main drilling vibration
    drilling = penetration * 0.7 * np.sin(2 * np.pi * drill_freq * t)
    
    # Periodic thrust (feed rate)
    thrust_freq = np.random.uniform(6, 10)
    thrust = 0.4 * penetration * np.sin(2 * np.pi * thrust_freq * t)
    
    # High frequency harmonics
    harmonic = 0.25 * penetration * np.sin(2 * np.pi * drill_freq * 1.6 * t)
    
    # Drill chatter
    chatter = 0.2 * np.random.randn(len(t))
    
    # Combine: amplify normal + add drilling
    abnormal = 1.8 * normal_acc + drilling + thrust + harmonic + chatter
    
    return abnormal

def apply_random_tampering(t, normal_acc):
    """
    Randomly choose a tampering type and apply transformation
    """
    tampering_types = [
        apply_cutting_transformation,
        apply_hammering_transformation,
        apply_grinding_transformation,
        apply_drilling_transformation
    ]
    
    # Randomly select tampering method
    transform_func = np.random.choice(tampering_types)
    
    return transform_func(t, normal_acc)

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def save_csv(time, acceleration, filename):
    """Save vibration data to CSV"""
    df = pd.DataFrame({
        'time': np.round(time, 2),
        'acceleration': np.round(acceleration, 3)
    })
    df.to_csv(filename, index=False)

# ============================================================================
# MAIN GENERATION LOOP
# ============================================================================

print("\n📊 STEP 1: Generating NORMAL vibration patterns...")
print("-" * 70)

normal_patterns = []
normal_count = 0

# Generate train passages (60 samples)
for i in range(60):
    t, acc = generate_normal_train_passage(
        duration=np.random.uniform(1.5, 2.5),
        seed=i
    )
    filename = f'training_data/normal/train_passage_{i+1:03d}.csv'
    save_csv(t, acc, filename)
    normal_patterns.append((t, acc, filename))
    normal_count += 1
    print(f"  ✓ {filename}")

# Generate idle tracks (10 samples)
for i in range(10):
    t, acc = generate_idle_track(
        duration=np.random.uniform(1.5, 2.5),
        seed=60+i
    )
    filename = f'training_data/normal/idle_track_{i+1:03d}.csv'
    save_csv(t, acc, filename)
    normal_patterns.append((t, acc, filename))
    normal_count += 1
    print(f"  ✓ {filename}")

print(f"\n✅ Generated {normal_count} NORMAL samples")

# ============================================================================
# STEP 2: Transform Normal → Abnormal
# ============================================================================

print("\n🚨 STEP 2: Transforming to ABNORMAL patterns (tampering)...")
print("-" * 70)

abnormal_count = 0

# Transform each normal pattern into abnormal
for i, (t, normal_acc, source_file) in enumerate(normal_patterns):
    
    # Apply random tampering transformation
    abnormal_acc = apply_random_tampering(t, normal_acc)
    
    filename = f'training_data/abnormal/tampering_{i+1:03d}.csv'
    save_csv(t, abnormal_acc, filename)
    abnormal_count += 1
    
    source_name = os.path.basename(source_file)
    print(f"  ✓ {filename} (from {source_name})")

print(f"\n✅ Generated {abnormal_count} ABNORMAL samples")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("📊 DATASET GENERATION COMPLETE")
print("=" * 70)
print(f"\n📁 Location: training_data/")
print(f"   └─ normal/     : {normal_count} files (train passages + idle)")
print(f"   └─ abnormal/   : {abnormal_count} files (tampering transformed)")
print(f"\n📈 Total samples: {normal_count + abnormal_count}")
print("\n🎯 Strategy:")
print("   - Normal: Realistic train passage physics")
print("   - Abnormal: Normal patterns + tampering signatures")
print("\n✅ Ready for ML model training!")
print("   Next step: python ml_models/train_vibration_model.py")
print("=" * 70)