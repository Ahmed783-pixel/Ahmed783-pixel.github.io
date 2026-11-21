# Website agent handoff: Kinematics Labelling of Surgical Video Datasets

## Your task

Update my personal website's project entry using the context below. Match the site's existing visual style and keep the public-facing description concise. Present this as an ongoing computer vision project with completed tip-localization experiments and a planned learned velocity-estimation stage. Do not imply that physical velocity prediction on unlabelled videos has already been trained or validated.

This document summarizes our discussion beginning with my clarification of the project goal. It is a handoff, not a claim that every proposed component is implemented. Repository observations below reflect the code inspected during that discussion; verify the current branch before making new implementation claims. This request supplies content for a website update, not instructions to implement or train the research pipeline.

## Project goal

**Kinematics labelling of previously unlabelled endoscopic video datasets.**

The intended system localizes surgical instrument tips and predicts their associated velocities from video sequences. We first use labelled data to train tip localization and prepare physically scaled velocity supervision. A velocity model will then learn from sequences of RGB frames, tip detections and associated velocity labels. Ultimately, it should annotate previously unlabelled endoscopic videos with tip locations and estimated velocities without requiring robot kinematics at inference time.

The desired velocity units are millimetres per second, not only pixels per second. Camera-frame horizontal and vertical components are of interest; predicting all three camera-frame components was discussed as an option, not finalized as a requirement.

## Completed work and repository evidence

Repository: https://github.com/yxLuo06/Computer-Vison-Course-project

### 1. EndoVis tip localization and tracking

- Fine-tuned YOLO26n-Pose on EndoVis to predict instrument bounding boxes and five landmarks: left clasper, right clasper, head, shaft and endpoint.
- Defined the tracked tip as the midpoint between the two clasper landmarks.
- Associated detections across frames using tip and head locations.
- Calculated image-plane velocity after detection from coordinate differences and frame timing. This calculation is in pixels/second; it is not a learned velocity output.
- The tracker leaves velocity missing for new detections, uncertain matches and the first frame after a detection gap.

Files inspected:

- https://github.com/yxLuo06/Computer-Vison-Course-project/blob/main/experiments/endovis-pose/README.md
- https://github.com/yxLuo06/Computer-Vison-Course-project/blob/main/experiments/endovis-pose/scripts/train_pose_full.py
- https://github.com/yxLuo06/Computer-Vison-Course-project/blob/main/experiments/endovis-pose/scripts/check_two_tool_tracking.py

### 2. SurgPose adaptation

The SurgPose experiment was found on the **Will branch**, under `experiments/surgpose-tip/`; it was not in the main-branch experiments listing inspected at that time.

- Adapted the EndoVis checkpoint to SurgPose instrument jaw-tip localization.
- The documented training schedule used sequence 000000 for a 10-epoch pilot and sequences 000001–000010 for a second 10-epoch run.
- Retained five keypoint output slots, but supervised the two jaw-tip slots using IDs 4 and 5 for PSM1 and IDs 11 and 12 for PSM3. Unverified remaining slots were masked rather than assigned invented anatomical correspondences.
- Used left and right camera images as separate detection-training examples; paired views at the same time step shared the train/validation split.
- Rendered predicted and annotated jaw tips in synchronized left/right views for held-out sequence 000011 as a qualitative demonstration.
- This rendering is a stereo detection overlay. The inspected script does not triangulate physical positions or calculate mm/s velocity labels.

Files inspected:

- https://github.com/yxLuo06/Computer-Vison-Course-project/blob/Will/experiments/surgpose-tip/README.md
- https://github.com/yxLuo06/Computer-Vison-Course-project/blob/Will/experiments/surgpose-tip/scripts/train_surgpose_000000.py
- https://github.com/yxLuo06/Computer-Vison-Course-project/blob/Will/experiments/surgpose-tip/scripts/train_surgpose_range.py
- https://github.com/yxLuo06/Computer-Vison-Course-project/blob/Will/experiments/surgpose-tip/scripts/render_stereo.py

### Status boundary

**Completed:** tip-detection training, tracking experiments with derived pixel velocity, and SurgPose stereo tip-detection visualization.

**Not established in the inspected code:** a physical velocity training table, a trained velocity regression model, or validated mm/s prediction on previously unlabelled endoscopic videos. A separate local notebook could contain additional work, but that was not verified in this discussion.

Do not call SurgPose an existing velocity-labelled dataset without qualification. Our proposed velocity labels would be derived from its annotations/calibration and valid timing information.

## Proposed next stage: prepare velocity supervision

1. Choose a consistent point definition: one jaw endpoint or the midpoint between both endpoints. The current EndoVis and SurgPose experiments use different evaluation definitions, so reconcile them before creating velocity labels.
2. Match the same annotated physical keypoint in the left and right images.
3. Use stereo calibration, with appropriate distortion handling, to triangulate camera-frame positions. Confirm calibration translation units before expressing positions in mm. If using a midpoint, reconstruct both endpoints and compute their 3D midpoint.
4. Verify actual capture timestamps or acquisition intervals. A video encoded/exported at 30 FPS does not by itself establish the physical sampling interval of the captured trajectory.
5. Estimate velocity from position changes over time. For equally spaced samples, a central difference is `v[t] = (p[t+1] - p[t-1]) / (2 * dt)`.
6. Exclude missing or unreliable stereo correspondences, examine reconstruction errors and derivative noise, and save labels with validity information.
7. Split by entire sequences before forming overlapping video clips to avoid train/test leakage.

Use dataset annotations to derive supervision and keep YOLO predictions separate for evaluation. Stereo-derived labels are reference estimates with annotation/calibration uncertainty, not independently measured perfect ground truth.

An example training table links sequence ID, frame/timestamp, instrument ID, tip pixel coordinates, reconstructed position, velocity components and validity flags. The video frames remain separate files accessed through those identifiers.

The immediate proposed milestone is one sequence with checked 3D positions, verified timing and plausible velocity trajectories. It has not been claimed as complete.

## Proposed velocity model

Keep the trained YOLO model as the tip localizer and initially train a separate PyTorch velocity predictor. Modifying YOLO internally is optional; adding an output head to a single image does not provide an explicit observation of motion.

Suggested baseline, not a finalized or implemented architecture:

**RGB clip + tracked tip coordinates/confidences + timing → image encoder → temporal model (for example a GRU) → velocity regression head.**

The data loader uses the table to load each clip and its target velocity. Velocity labels are targets for the loss, never input features. A possible training objective compares predicted and reference velocity vectors with a regression loss.

At inference, an unlabelled video provides frames and timing; YOLO supplies tip detections and association provides instrument tracks. The temporal model predicts velocity. The intended output is per-instrument annotations and video overlays, with an exportable table of predictions.

If using moving or resized crops, retain original image coordinates and crop/scale metadata so preprocessing does not erase or distort the motion and scale cues needed for velocity prediction.

## Optional optical-flow experiment

Optical flow was discussed as an additional motion input, not implemented work. A model such as RAFT can estimate apparent pixel displacement between frames. Flow around the tracked instrument could complement RGB features and tip trajectories in the velocity predictor.

Optical flow alone does not supply physical scale or directly output mm/s. Physical velocity labels remain necessary for the proposed supervised regression approach. Compare the RGB/trajectory baseline against an otherwise comparable model with optical flow on held-out sequences.

RAFT reference: https://arxiv.org/abs/2003.12039

## Monocular deployment and evaluation

Monocular video contains one camera viewpoint; stereo video contains synchronized left/right viewpoints. Using only SurgPose's left video is monocular input even though stereo data can be used offline to construct training labels.

For intended monocular deployment, train the predictor with the information available at deployment. Stereo can generate supervision without becoming a required inference input.

Metric velocity from arbitrary monocular video is not uniquely determined by image motion alone. A trained model can learn scale/depth cues and dataset priors, but changes in camera geometry, working distance, instrument appearance and camera motion can impair transfer. Do not promise universally accurate mm/s annotations from arbitrary video.

Evaluate localization and velocity separately, using held-out sequences with reference labels. A qualitative overlay on genuinely unlabelled video demonstrates application, not measured velocity accuracy. Report coverage/missing predictions as well as errors. Camera-relative velocity must not be described as robot-base or world-relative velocity without the corresponding transformations and motion handling.

## Future motivation

Longer-term, explore surgical gesture recognition using tip trajectories/velocities and broader image-motion patterns. This is a downstream research direction, not an implemented project result. The primary project remains kinematics labelling of video datasets.

## Suggested website copy

### Title

Kinematics Labelling of Surgical Video Datasets

### Short project card

Developing a vision-based pipeline to annotate endoscopic videos with surgical instrument tip locations and velocity estimates. Current work includes YOLO Pose fine-tuning on EndoVis and SurgPose, instrument tracking, and stereo tip-localization demonstrations. The next stage explores supervised temporal models for physical velocity prediction on previously unlabelled videos.

### Longer description

This project aims to turn unlabelled endoscopic videos into datasets annotated with instrument-tip locations and estimated velocities. We have fine-tuned YOLO Pose for surgical instrument landmarks, implemented tracking with image-plane motion estimates, and adapted the detector to SurgPose stereo imagery. The planned next stage uses calibrated stereo annotations and verified timing to construct physical velocity supervision, then trains a temporal visual model to estimate tip velocity from video sequences. Optical flow will be explored as an additional motion representation. The resulting annotations could support surgical motion analysis and future gesture-recognition research.

### Suggested status label

In progress — tip localization implemented; learned physical velocity estimation planned.

### Skills/technology labels

Implemented: Python, PyTorch, Ultralytics YOLO Pose, OpenCV, keypoint detection, instrument tracking, EndoVis, SurgPose.

Planned/exploratory: stereo triangulation for velocity supervision, temporal regression, optical flow, monocular kinematics estimation.

## Instructions for the website-editing agent

- Use the goal and status distinction above as the central narrative.
- Preserve the site's existing styling and unrelated content.
- Do not claim that YOLO was trained on velocity, that stereo rendering performs 3D reconstruction, or that mm/s prediction on unlabelled video is already validated.
- Do not convert within-sequence pose mAP into a claim about physical velocity accuracy or cross-dataset generalization.
- Do not invent benchmark results, real-time performance, clinical deployment, publication status or individual ownership of team work.
- Use team-oriented language for shared implementation; confirm my individual contributions before writing specific first-person achievement claims.
- Link to the repository and verify branch-specific files if adding implementation details or demonstrations.
- Use actual project media if available and appropriate for public display. Keep concept illustrations distinct from measured outputs; check dataset/media reuse conditions before publishing original surgical footage.
- Keep technical caveats concise in the public entry; retain the complete limitations in supporting project documentation.
- The suggested copy is editable, but preserve its distinction between completed work and proposed research.
