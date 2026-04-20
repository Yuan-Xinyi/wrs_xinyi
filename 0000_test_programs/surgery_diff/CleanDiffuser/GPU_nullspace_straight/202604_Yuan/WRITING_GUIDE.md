# Writing Guide for This Paper

This document records the current project understanding and writing preferences, so future revisions stay aligned with the intended T-RO-style framing.

Hard rule:

- Any content enclosed in `\xinyi{...}` must remain completely unchanged in future revisions. Do not modify, rewrite, shorten, expand, or paraphrase even a single character inside `\xinyi{}` blocks unless the user explicitly requests it.

## Core Problem Framing

The paper is about farsighted inverse kinematics for path-following manipulation.

The key idea is not merely to find a feasible IK solution for the current Cartesian target, but to select a joint configuration that supports better future continuation along a prescribed path.

Typical motivating scenarios:

- A single manipulator pushes a box along a given path or direction, and success is measured by how far it can keep pushing.
- A dual-arm robot cooperatively pushes or transports an object along a given direction.

The core claim is:

- Different feasible IK solutions for the same current task condition can lead to very different downstream path-following performance.
- Therefore, IK should be treated as a farsighted configuration selection problem, not only an instantaneous feasibility problem.

## Method Positioning

The method should be described as a unified framework with three parts:

1. A diffusion-based IK generator.
2. A contrastive performance network, LNet.
3. A damped Jacobian-based final correction step.

Important implementation-aligned details:

- The task condition is based on `position + direction + target_normal`.
- The condition vector is 9D: `[p; d; n]`.
- The diffusion model performs conditional kinematic inpainting over the feasible joint manifold.
- LNet is trained with contrastive or ranking-style supervision, not simple absolute regression as the main narrative.
- Guidance is injected during reverse diffusion using the gradient of the learned contrastive score.
- Final Cartesian consistency is recovered with damped Jacobian position correction.

Preferred description:

- The method amortizes expensive simulate-then-select search into generative sampling plus learned guidance.
- The method collapses future motion utility into present-time configuration generation.

Avoid overstating:

- Do not describe the method as full trajectory planning between two endpoints.
- Do not position it as generic point-to-point planning.
- Do not overemphasize pose-wide constraints if the current implementation mainly enforces position and task-direction geometry.

## Problem Definition Language

Preferred terms:

- `farsighted inverse kinematics`
- `farsighted configuration selection`
- `path-following manipulation`
- `future continuation length`
- `rollout length`
- `future motion capability`
- `task-conditioned feasible solution manifold`

Use carefully:

- `straight-line manipulation` is acceptable when discussing the current implementation.
- `path-following manipulation` is preferred as the broader framing.

Avoid using language that sounds too narrow unless needed for a technical subsection:

- `instantaneous IK only`
- `pointwise feasibility only`

## Introduction Guidance

The introduction should follow this logic:

1. Start from manipulation scenarios where the robot must continue moving an object along a prescribed path.
2. Explain why current feasible IK is insufficient for sustained pushing or cooperative path following.
3. Highlight redundancy and the difficulty of continuous IK along a path.
4. Motivate a generative solution.
5. Introduce diffusion + contrastive guidance + Jacobian correction.
6. Present contributions in a concise T-RO/T-ASE style.

Important emphasis:

- The difficulty is not just finding one feasible configuration.
- The real challenge is selecting a configuration that preserves future maneuverability.

## Related Work Structure

The related work should stay in the following order unless there is a strong reason to change it:

### 1. Redundant IK and Continuous IK Along a Path

Focus:

- IK redundancy
- null-space methods
- continuous IK difficulties along trajectories
- drift, discontinuity, singularity, and loss of maneuverability

Main positioning:

- Existing work mostly solves pointwise IK or local redundancy resolution.
- Our work targets continuous path-following quality through current configuration selection.

### 2. Generative Models for Planning and Optimization Initialization

Focus:

- diffusion and other generative models in planning
- generative priors for optimization initialization

Main positioning:

- Existing work often addresses point-to-point planning or start-goal trajectory generation.
- Our work is not about endpoint-to-endpoint planning.
- Our work uses generation to select a current IK configuration for future path following.

### 3. Path Following in Robotic Manipulation

Focus:

- path following versus general trajectory tracking
- pushing, insertion, polishing, cooperative transport

Main positioning:

- Path following depends strongly on local kinematic quality.
- Our contribution is on the IK selection stage that enables better continuation.

### 4. Reachability and Future Motion Capability

Focus:

- reachability
- dexterity
- manipulability
- continuation feasibility

Main positioning:

- Existing metrics are often global descriptors or local surrogates.
- Our method uses rollout length as the operational measure of future capability and injects it into generation.

### 5. Guidance in Generative Models

Focus:

- classifier guidance
- classifier-free guidance
- energy or value guidance
- task-guided generation

Main positioning:

- Our method uses a contrastive ranking network over IK configurations as the guidance source.

## Title Preferences

The current preferred title style is:

- problem first
- method second
- application context last

Current preferred title:

- `Farsighted Inverse Kinematics via Contrastively Guided Diffusion for Path-Following Manipulation`

Title style preferences:

- Make it sound like a T-RO paper.
- Keep it concise and technical.
- Avoid overly long marketing-style phrasing.
- Prefer `via ...` or `for ...` constructions common in robotics papers.

## Abstract Guidance

The abstract should clearly cover:

1. Conventional IK is usually instantaneous feasibility.
2. In path-following manipulation, feasible IK solutions differ in future continuation ability.
3. The proposed method uses conditional diffusion to sample feasible configurations.
4. LNet provides contrastive guidance based on rollout-length ranking.
5. Damped Jacobian correction recovers final Cartesian consistency.
6. The method improves farsighted path-following capability over conventional baselines.

Preferred tone:

- concise
- technical
- no hype
- T-RO style

## Methods Section Guidance

The Methods section should stay close to the actual implementation.

Essential points to preserve:

- Condition vector: `position`, `direction`, `target_normal`
- Diffusion prior takes normalized condition information
- Guided sampling uses contrastive score gradients
- Multiple guidance strengths may be evaluated, such as `0.1, 1, 5, 10`
- Final correction uses damped Jacobian position correction
- Evaluation uses rollout continuation length

Do not casually add unsupported claims such as:

- full pose control if not actually enforced
- dynamics-aware optimization if not in current implementation
- full motion planning if only configuration selection is implemented

## Experiments Guidance

The experiments should be framed around the question:

- Can the proposed method select IK configurations that allow longer downstream path following?

Current implementation-aligned experimental ingredients:

- Franka Research 3 platform
- GPU null-space straight-line tracker
- sphere-based self-collision checking
- anchor condition from dataset
- rollout-based measurement of continuation length

Important metrics:

- realized rollout length
- gain over dataset/reference configuration
- raw position error before correction
- corrected position consistency
- LNet score
- diffusion predicted length proxy
- runtime

Useful ablations:

- without LNet guidance
- ranking versus regression scorer
- without Jacobian correction
- different guidance strengths
- different sampling steps
- different damping or correction tolerances

## Writing Style Preferences

Use IEEE T-RO style language:

- formal
- precise
- restrained
- technically grounded

Preferred habits:

- make claims with clear scope
- distinguish between general framing and current implementation
- use strong transitions between problem, limitation, and contribution

Avoid:

- exaggerated novelty claims
- vague AI buzzwords
- overly promotional wording
- long decorative titles

## How Future Revisions Should Proceed

When revising the paper in the future:

1. Check that the framing still matches path-following manipulation, not endpoint planning.
2. Keep the problem statement aligned with farsighted IK selection.
3. Keep related work in the five-part structure above.
4. Keep method descriptions consistent with the actual code and evaluation pipeline.
5. Prefer T-RO-style concise wording over broad conceptual language.

If there is a conflict between elegant phrasing and implementation fidelity:

- prioritize fidelity to the actual method.
