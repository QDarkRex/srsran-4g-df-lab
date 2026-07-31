import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = "/tmp/csi_capture.bin"
TARGET_PRB = 9 
NUM_RE = TARGET_PRB * 12

def load_csi():
    records = []
    try:
        raw_bytes = np.fromfile(FILE_PATH, dtype=np.uint8)
        pointer = 0
        while pointer < len(raw_bytes):
            # 1. Read Header
            header = raw_bytes[pointer : pointer + 8].view(np.uint32)
            if len(header) < 2: break
            
            n_re = header[1]
            pointer += 8
            
            # 2. Extract Data
            data_bytes = raw_bytes[pointer : pointer + (n_re * 4)]
            if len(data_bytes) == (n_re * 4):
                data = data_bytes.view(np.float32)
                
                # STRICT CHECK: Only keep if it matches our target RE count
                if n_re == NUM_RE and len(data) == NUM_RE:
                    records.append(data.copy()) # Use copy() to keep memory clean
            
            pointer += (n_re * 4)
            
    except Exception as e:
        print(f"Error reading file: {e}")

    # Final check to ensure we found anything
    if not records:
        return np.array([])
        
    return np.stack(records) # stack is safer for building the 2D array

print(f"Extracting PRB {TARGET_PRB} from {FILE_PATH}...")
csi_matrix = load_csi()

if csi_matrix.size > 0:
    print(f"Successfully loaded {csi_matrix.shape[0]} frames.")
    
    # Normalizing so we can see movement patterns regardless of signal strength
    csi_norm = (csi_matrix - np.min(csi_matrix)) / (np.max(csi_matrix) - np.min(csi_matrix) + 1e-9)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(csi_norm.T, aspect='auto', cmap='magma', interpolation='none')
    plt.colorbar(label="Normalized Magnitude")
    plt.title(f"LTE CSI Radar - PRB: {TARGET_PRB}")
    plt.ylabel("Subcarrier Index")
    plt.xlabel("Captured Bursts")
    
    # Save the file (crucial for headless terminals)
    plt.savefig("csi_heatmap.png")
    print("Heatmap saved to csi_heatmap.png")
    
    try:
        plt.show()
    except:
        print("No display detected (Headless mode).")
else:
    print(f"No valid PRB {TARGET_PRB} records found. Try increasing your ping packet size.")