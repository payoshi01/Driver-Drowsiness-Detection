
import os
import sys
import cv2
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET, LABELS_CSV

OUTPUT_CSV = os.path.join(DATASET, "labels_refined.csv")

LABELS = {
    "1": "Focused",
    "2": "Drowsy",
    "3": "Yawning",
    "4": "LookingAway"
}

def main():
    if not os.path.exists(LABELS_CSV):
        sys.exit(f"Labels file not found: {LABELS_CSV}")
    df = pd.read_csv(LABELS_CSV)
    # Only review own recordings currently labelled Drowsy or LookingAway
    mask = (
        (df["source"] == "own") &
        (df["label"].isin(["Drowsy", "LookingAway"]))
    )
    indices = df.index[mask].tolist()
    if not indices:
        sys.exit("No Drowsy or LookingAway frames found.")
    print(f"Frames to review: {len(indices)}")
    print("\nKeyboard shortcuts:")
    print("1 = Focused")
    print("2 = Drowsy")
    print("3 = Yawning")
    print("4 = LookingAway")
    print("S = Skip (keep original label)")
    print("Q = Save and quit")

    for position, idx in enumerate(indices, start=1):
        row = df.loc[idx]
        image_path = os.path.join(DATASET, row["filepath"])
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read: {image_path}")
            continue
        display = image.copy()
        info = [
            f"Frame {position}/{len(indices)}",
            f"Original label: {row['label']}",
            f"Subject: {row['subject_id']}",
            "Press 1/2/3/4 to relabel | S to skip | Q to save"
        ]
        for i, line in enumerate(info):
            cv2.putText(
                display, line, (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0), 2
            )
        cv2.imshow("Frame Label Review", display)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in [ord("1"), ord("2"), ord("3"), ord("4")]:
                new_label = LABELS[chr(key)]
                df.at[idx, "label"] = new_label
                df.at[idx, "label_id"] = {
                    "Focused": 0,
                    "Drowsy": 1,
                    "Yawning": 2,
                    "LookingAway": 3
                }[new_label]
                print(f"{position}: {row['label']} -> {new_label}")
                break
            elif key == ord("s"):
                print(f"{position}: Skipped")
                break

            elif key == ord("q"):
                df.to_csv(OUTPUT_CSV, index=False)
                cv2.destroyAllWindows()
                print(f"\nSaved progress to {OUTPUT_CSV}")
                return

    df.to_csv(OUTPUT_CSV, index=False)
    cv2.destroyAllWindows()

    print(f"\nCompleted! Refined labels saved to:")
    print(OUTPUT_CSV)

if __name__ == "__main__":
    main()
