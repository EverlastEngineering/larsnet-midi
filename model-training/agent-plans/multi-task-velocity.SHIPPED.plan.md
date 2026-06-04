# **Engineering Plan: Transitioning to Multi-Task Velocity Regression**

> **STATUS: SHIPPED** — This dual-head plan (10 onset classification + 10
> velocity regression channels, masked-MSE loss) was implemented and shipped
> in commit `4a60cc5 successful mtl with velocity`. The model architecture
> (see `model.py`) and loss function (see `MultiTaskDrumLoss` in
> `train_utils.py`) match this plan.
>
> **Outcome of subsequent full training runs:** the architecture worked but
> the training run that followed (visible in commits `0a6593b pre-training`,
> `e7aae84 Ugh a mess`, `6a24b7d opencode improvements`) did not produce a
> usable model in practice. Root cause was not isolated before the rescue
> branch was created. A future iteration should re-examine convergence,
> velocity distribution in outputs, and whether the pos_weight rebalance
> (5.10..156.13 → 2.0..10.0, see rescue commit) helped or hurt.
>
> Preserved as the design-of-record for what's currently in `model.py`.

---

## **Context**

Current model uses classification probability as a proxy for velocity, causing a "binary" dynamic range (0 or 127). We are moving to a dual-head output where classification and regression are handled separately.

## **Phase 1: Model Architecture (model.py)**

**Objective:** Expand the output space from 10 to 20 dimensions.

* **Task 1.1:** Locate the final Fully Connected layer (self.fc).  
* **Task 1.2:** Change nn.Linear(input\_dim, 10\) to nn.Linear(input\_dim, 20\).  
* **Task 1.3:** Ensure the forward pass returns raw logits. **Do not** apply Sigmoid inside the forward function; let the loss function and inference script handle activations for better numerical stability.

## **Phase 2: Target Generation (train\_utils.py)**

**Objective:** Update build\_targets to encode both "IsHit" and "VelocityValue".

* **Task 2.1:** Modify build\_targets to create a tensor of shape \[TotalFrames, 20\].  
* **Task 2.2:** Channels 0-9 should contain the binary onset heatmap (1.0 for hit, 0.0 for silence).  
* **Task 2.3:** Channels 10-19 should contain the normalized velocity (midi\_velocity / 127.0) at the exact frame of the hit, and 0.0 everywhere else.

## **Phase 3: Loss Function (train\_utils.py)**

**Objective:** Implement MultiTaskDrumLoss to balance classification and regression.

* **Task 3.1:** Create a custom nn.Module for loss.  
* **Task 3.2:** Split the prediction/target tensors into onset\_part (first 10\) and velocity\_part (last 10).  
* **Task 3.3:** Use BCEWithLogitsLoss (with pos\_weight) for the onset part.  
* **Task 3.4:** Use MSELoss for the velocity part, but **apply a mask**. Only calculate MSE on frames where the Ground Truth onset is 1.0. This prevents the model from trying to predict "0 velocity" for silent frames, which would suppress the signal.  
* **Task 3.5:** Return Loss\_Onset \+ (Weight \* Loss\_Velocity). Start with Weight \= 2.0.

## **Phase 4: Inference Logic (inference.py / heatmap\_utils)**

**Objective:** Decouple probability from velocity during MIDI generation.

* **Task 4.1:** Update the note extraction loop. Use channels 0-9 (passed through a Sigmoid) to detect peaks/onsets based on your threshold.  
* **Task 4.2:** Once an onset is found at frame F, look at channel F \+ 10 (the corresponding velocity channel).  
* **Task 4.3:** Set MIDI velocity \= prediction\[F, class\_idx \+ 10\] \* 127.

## **Verification**

* Run velocity\_analysis.py after the first epoch.  
* **Success Criteria:** The scatter plot should begin to show a diagonal trend rather than two horizontal clusters at the extremes.