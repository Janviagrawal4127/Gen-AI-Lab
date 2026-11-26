
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

video_path = input("Enter video file path: ")

if not os.path.exists(video_path):
    print("Video file not found!")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Unable to open video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nDeepFake Video Analysis")
print("Frames:", total_frames)
print("FPS:", fps)

previous_frame = None
frame_count = 0
blur_scores = []
frame_differences = []
sample_frames = []
artifact_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_scores.append(blur_score)

    if previous_frame is not None:
        difference = np.mean(
            cv2.absdiff(gray, previous_frame)
        )

        frame_differences.append(difference)

        if difference > 40:
            artifact_count += 1

    previous_frame = gray

    if frame_count % max(1, int(fps)) == 0:
        sample_frames.append(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

cap.release()

average_blur = np.mean(blur_scores)
average_difference = np.mean(frame_differences) if frame_differences else 0

print("\nVideo Analysis Results")
print("Total Frames Analyzed:", frame_count)
print("Average Blur Score:", round(average_blur, 2))
print("Average Frame Difference:", round(average_difference, 2))
print("Frames with High Changes:", artifact_count)

if average_blur < 100:
    print("Observation: Video contains potentially blurred frames.")
else:
    print("Observation: Video frames appear relatively sharp.")

if artifact_count > 0:
    print("Observation: Sudden frame changes detected.")
else:
    print("Observation: No major sudden frame changes detected.")

plt.figure(figsize=(12, 5))
plt.plot(blur_scores)
plt.title("Frame Blur Analysis")
plt.xlabel("Frame Number")
plt.ylabel("Laplacian Variance")
plt.grid()
plt.show()

if frame_differences:
    plt.figure(figsize=(12, 5))
    plt.plot(frame_differences)
    plt.title("Frame Difference Analysis")
    plt.xlabel("Frame Number")
    plt.ylabel("Mean Pixel Difference")
    plt.grid()
    plt.show()

if sample_frames:
    count = min(5, len(sample_frames))

    plt.figure(figsize=(15, 5))

    for i in range(count):
        plt.subplot(1, count, i + 1)
        plt.imshow(sample_frames[i])
        plt.title("Frame " + str(i + 1))
        plt.axis("off")

    plt.tight_layout()
    plt.show()

print("\nEthical Implications of DeepFake Technology:")
print("1. DeepFakes can spread misinformation and fake news.")
print("2. They can damage the reputation of individuals.")
print("3. They may be used for identity theft and fraud.")
print("4. Non-consensual fake videos violate privacy.")
print("5. DeepFakes can manipulate public opinion.")

print("\nVideo analysis completed successfully!")
