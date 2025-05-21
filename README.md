# EXP.NO.8-Simulation-of-QPSK

# AIM
To analyse the modulation of QPSK Signal

# SOFTWARE REQUIRED

Python

# ALGORITHMS
```
Make Random Bits
→ Create a list of 0s and 1s (2 bits for each symbol).

Group the Bits
→ Take every 2 bits and turn them into 1 symbol.

Assign Angles
→ Each symbol gets a different angle:

00 → 0°

01 → 90°

10 → 180°

11 → 270°

Make the Signal
→ For each symbol, draw a wave with the right angle.

Join the Waves
→ Put all the waves together to make the full QPSK signal.

Draw the Signal
→ Show the signal using plots (real part, imaginary part, full signal).

```

# PROGRAM
import numpy as np
import matplotlib.pyplot as plt

num_symbols = 10  # Number of QPSK symbols (each with 2 bits)
T = 1.0  # Symbol period
fs = 100.0  # Sampling frequency
t = np.arange(0, T, 1/fs)

# Generate 2 bits per symbol
bits = np.random.randint(0, 2, num_symbols * 2)

# Separate into I (cosine) and Q (sine) bits
i_bits = bits[0::2]  # Even-indexed bits
q_bits = bits[1::2]  # Odd-indexed bits

# Map bits: 0 → -1, 1 → +1
i_values = 2 * i_bits - 1
q_values = 2 * q_bits - 1

# Initialize signal arrays
i_signal = np.array([])
q_signal = np.array([])
combined_signal = np.array([])
symbol_times = []

for i in range(num_symbols):
    i_carrier = i_values[i] * np.cos(2 * np.pi * t / T)
    q_carrier = q_values[i] * np.sin(2 * np.pi * t / T)
    symbol_times.append(i * T)
    i_signal = np.concatenate((i_signal, i_carrier))
    q_signal = np.concatenate((q_signal, q_carrier))
    combined_signal = np.concatenate((combined_signal, i_carrier + q_carrier))

t_total = np.arange(0, num_symbols * T, 1/fs)

# Plotting
plt.figure(figsize=(14, 9))

# In-phase (cosine) component
plt.subplot(3, 1, 1)
plt.plot(t_total, i_signal, label='In-phase (cos)', color='blue')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0.8, f'{i_bits[i]}', fontsize=12, color='black')
plt.title('In-phase Component (Cosine) - One Bit per Symbol')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Quadrature (sine) component
plt.subplot(3, 1, 2)
plt.plot(t_total, q_signal, label='Quadrature (sin)', color='orange')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0.8, f'{q_bits[i]}', fontsize=12, color='black')
plt.title('Quadrature Component (Sine) - One Bit per Symbol')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Combined QPSK waveform
plt.subplot(3, 1, 3)
plt.plot(t_total, combined_signal, label='QPSK Signal = I + Q', color='green')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, 0.8, f'{i_bits[i]}{q_bits[i]}', fontsize=12, color='black')
plt.title('Combined QPSK Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# OUTPUT
![QPSK OUTPUT](https://github.com/user-attachments/assets/f579cdd8-1853-4667-8645-7874657e2096)


 
# RESULT / CONCLUSIONS

Thus QPSK modulation is implemented using Scilab code.
