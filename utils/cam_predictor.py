

from sam2.build_sam import build_sam2_camera_predictor

def process_anns(points, labels, boxes):
    def _process(ann): 
        return np.array([ann]) if ann != [] else None
    points = _process(points)
    labels = _process(labels)
    boxes = _process(boxes)
    return points, labels, boxes

sam2_cam_model = build_sam2_camera_predictor(self.config_file, self.checkpoint_file, device=self.device)

self.predictor.load_first_frame(self.live_frame)
for ann_obj_id, (points, labels, boxes) in enumerate(zip(self.input_points, self.input_labels, self.input_boxes)):
    points, labels, boxes = process_anns(points, labels, boxes)
    self.log(("id:", ann_obj_id, "pt:", points, "lb:", labels, "bx:", boxes), debug_only=True)
    
    if all(x is None for x in [points, labels, boxes]):
        continue
    
    _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
        frame_idx=0,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        bbox=boxes,
    )
    
# Assign current masks
for i, out_obj_id in enumerate(out_obj_ids):
    self.log((out_mask_logits[i] > 0.0).cpu().numpy().shape, debug_only=True)
    mask_list.append((out_mask_logits[i] > 0.0).cpu().numpy())
self.masks = np.concatenate(mask_list, axis=0)