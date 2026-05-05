import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import argparse
import os

def plot_velocity_relationship(gt_midi_path, pred_midi_path, output_png="velocity_audit.png"):
    """
    Creates a visual audit of how well the model reproduces drum dynamics,
    now with per-class breakdown to identify which instruments are 'bimodal'.
    """
    def get_velocity_map(path):
        midi = pretty_midi.PrettyMIDI(path)
        notes = []
        for inst in midi.instruments:
            for note in inst.notes:
                notes.append((note.start, note.pitch, note.velocity))
        # Sort by time then pitch for alignment
        return sorted(notes, key=lambda x: (x[0], x[1]))

    if not os.path.exists(gt_midi_path) or not os.path.exists(pred_midi_path):
        print(f"Error: One of the files does not exist: {gt_midi_path} or {pred_midi_path}")
        return

    gt_notes = get_velocity_map(gt_midi_path)
    pred_notes = get_velocity_map(pred_midi_path)

    # MIDI Pitch to Name Mapping
    pitch_map = {
        36: "Kick", 38: "Snare", 42: "HHC", 46: "HHO",
        47: "TomMid", 43: "TomLow", 50: "TomHigh",
        49: "Crash1", 57: "Crash2", 51: "Ride"
    }

    # Alignment and grouping by class
    class_data = {name: {"gt": [], "pred": []} for name in pitch_map.values()}
    matched_gt = []
    matched_pred = []
    
    for gt_t, gt_p, gt_v in gt_notes:
        best_match = None
        min_diff = 0.05 
        for pr_t, pr_p, pr_v in pred_notes:
            if gt_p == pr_p:
                diff = abs(gt_t - pr_t)
                if diff < min_diff:
                    min_diff = diff
                    best_match = pr_v
        
        if best_match is not None:
            matched_gt.append(gt_v)
            matched_pred.append(best_match)
            class_name = pitch_map.get(gt_p, f"Note({gt_p})")
            if class_name not in class_data:
                class_data[class_name] = {"gt": [], "pred": []}
            class_data[class_name]["gt"].append(gt_v)
            class_data[class_name]["pred"].append(best_match)

    if not matched_gt:
        print("No matching notes found between the two MIDI files within 50ms.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. SCATTER PLOT (Color coded by class)
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_data)))
    for (name, data), color in zip(class_data.items(), colors):
        if len(data["gt"]) > 0:
            ax1.scatter(data["gt"], data["pred"], alpha=0.6, c=[color], label=name, edgecolors='none', s=20)
            
    ax1.plot([0, 127], [0, 127], 'r--', alpha=0.8, label='Perfect 1:1')
    ax1.set_title("Velocity Correlation: Colored by Class")
    ax1.set_xlabel("Ground Truth Velocity")
    ax1.set_ylabel("Predicted Velocity")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # 2. HISTOGRAM
    ax2.hist(matched_gt, bins=30, alpha=0.3, label='Ground Truth', color='black', density=True)
    ax2.hist(matched_pred, bins=30, alpha=0.5, label='Predicted', color='#e67e22', density=True)
    ax2.set_title("Density Distribution (Normalized)")
    ax2.set_xlabel("Velocity Value")
    ax2.set_ylabel("Density")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_png)
    
    # Text Analysis
    print(f"\n--- Velocity Audit Report ---")
    print(f"{'Class':<12} | {'Count':<6} | {'Correlation':<12}")
    print("-" * 35)
    
    overall_corr = np.corrcoef(matched_gt, matched_pred)[0, 1]
    
    for name, data in class_data.items():
        if len(data["gt"]) > 1:
            # Check for zero variance to avoid nan
            if np.std(data["gt"]) > 0 and np.std(data["pred"]) > 0:
                corr = np.corrcoef(data["gt"], data["pred"])[0, 1]
                corr_str = f"{corr:.4f}"
            else:
                corr_str = "N/A (No Var)"
            print(f"{name:<12} | {len(data['gt']):<6} | {corr_str}")
            
    print("-" * 35)
    print(f"{'OVERALL':<12} | {len(matched_gt):<6} | {overall_corr:.4f}")
    print(f"\nAnalysis saved to {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze velocity relationship between two MIDI files.')
    parser.add_argument('gt_midi', help='Path to the Ground Truth MIDI file')
    parser.add_argument('pred_midi', help='Path to the Predicted MIDI file')
    parser.add_argument('--output', '-o', default='velocity_audit.png', help='Output PNG filename')

    args = parser.parse_args()
    
    plot_velocity_relationship(args.gt_midi, args.pred_midi, args.output)