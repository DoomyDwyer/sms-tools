from scipy.signal import get_window
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models'))
import sineModelMultiRes as sm
import utilFunctions as UF

def createOutputFile(y, fs, output_file):
    target_directory = os.path.dirname(output_file)
    # Create the directory if it doesn't already exist
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)
    # Delete the output file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)
    UF.wavwrite(y, fs, output_file)

# Sound 1, analysis & synthesis, create output file and plot
inputFile = 'sounds/141185_2244250-lq.wav'
(fs, x) = UF.wavread(inputFile)

windows = ['blackman', 'blackman', 'blackman']
M = [3031, 779, 295]
w = []
for i in range(len(windows)):
    w.append(get_window(windows[i], M[i], M[i] % 2 == 0))

N = [4096, 1024, 512]
B = [340, 900, 22050]
t = -90

y = sm.sineModelMultiRes(x, fs, w, N, t, B)

# write the synthesized sound obtained from the sinusoidal synthesis
outputFile = 'output_sounds/A10-b-1.wav'
createOutputFile(y, fs, outputFile)

plt.figure(1, figsize=(9.5, 7))

plt.subplot(2, 1, 1)
plt.plot(x)
plt.title('x')

plt.subplot(2, 1, 2)
plt.plot(y)
plt.title('y')

plt.tight_layout()
plt.show()

# Sound 2, analysis & synthesis, create output file and plot
inputFile = 'sounds/353773_2305278-lq.wav'
(fs, x) = UF.wavread(inputFile)

windows = ['blackman', 'blackman', 'blackman']
M = [5089, 2647, 1059]
w = []
for i in range(len(windows)):
    w.append(get_window(windows[i], M[i], M[i] % 2 == 0))

N = [8192, 4096, 2048]
B = [100, 250, 22050]
t = -90

y = sm.sineModelMultiRes(x, fs, w, N, t, B)

# write the synthesized sound obtained from the sinusoidal synthesis
outputFile = 'output_sounds/A10-b-2.wav'
createOutputFile(y, fs, outputFile)

plt.figure(1, figsize=(9.5, 7))

plt.subplot(2, 1, 1)
plt.plot(x)
plt.title('x')

plt.subplot(2, 1, 2)
plt.plot(y)
plt.title('y')

plt.tight_layout()
plt.show()
