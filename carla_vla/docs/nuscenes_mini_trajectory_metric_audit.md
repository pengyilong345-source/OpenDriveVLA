# nuScenes-mini trajectory metric audit

This audit uses only local OpenDriveVLA/UniAD code. Full-validation trajectory
pickles are not present locally and no full-val GT values are used.

## Representation

- The planning head emits six `(x, y)` increments and applies `torch.cumsum`
  (`planning_head.py:190-192`), so decoded trajectories are **absolute future
  offsets from the current reference frame**, not stepwise displacements.
- `NuScenesTraj.get_sdc_planning_label` follows six native `sample.next` links,
  maps each future LIDAR origin through future LIDAR -> ego -> global -> current
  ego -> current LIDAR, and stores `(x, y, yaw)` (`trajectory_api.py:225-269`).
  Evaluation uses only x/y.
- GT and prediction are both in current LIDAR coordinates and metres. Local
  comments are not fully consistent about semantic axis names: the BEV utility
  calls x forward/y lateral, while the official command heuristic treats x as
  the left/right discriminator. The L2 path does not resolve or alter this
  naming: it compares emitted prediction and official GT coordinates directly,
  with no sign flip (`metric.py:209-213` leaves candidate flips commented).
  Collision code performs its own sign/axis handling, which is irrelevant here
  because mini occupancy GT is unavailable.
- nuScenes keyframes are nominally 0.5 s apart. The first future point is
  t=0.5 s, not t=0. Thus indices 1, 3, and 5 correspond to 1, 2, and 3 s.

## Shapes and masks

The official evaluator reshapes predictions to the GT mask shape and supports a
leading current point only when the time dimension is odd. For this mini path,
GT and masks use `float32` arrays of shape `(1, 6, 2)`. Each available future
point has mask `[1,1]`; unavailable future points retain numeric zero only as
storage and have mask `[0,0]`, so they are never treated as stationary GT.

The local full-val GT files `gt/gt_traj.pkl` and `gt/gt_traj_mask.pkl` are not
installed, so their dtype cannot be inspected directly. The selected mini
schema matches the evaluator's expected batch/time/xy layout.

## Metrics

`PlanningMetric.compute_L2` computes
`sqrt(sum((prediction_xy-GT_xy)^2 * mask, xy))`. The official UniAD table reports
pointwise L2 at indices 1, 3, and 5. Its STP-3 table reports cumulative means of
the first 2, 4, and 6 points. This mini evaluator reports both:

- L2@1s, @2s, @3s: masked pointwise values at indices 1, 3, 5;
- ADE: mean masked waypoint L2 over all valid points;
- FDE: L2 at each sample's last valid waypoint;
- cumulative 1/2/3 s L2 for comparison with the local STP-3 calculation.

Invalid GT points are excluded by the mask. A parsed all-zero prediction is a
real prediction and is evaluated normally. A parse failure remains missing and
is excluded from numeric trajectory metrics; it is never replaced with zeros.
The stock `retrieve_traj` pads short parses with their final parsed point and
would fail on an empty parse, but this experiment preserves the already-stored
parsed trajectory rather than invoking that repair logic.

Collision metrics require real mini planning occupancy/segmentation GT. That
asset is unavailable, so collision is reported as `not_computed`; full-val
occupancy is never substituted.
