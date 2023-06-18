import os
import random
import colorsys
import json
import numpy as np
import cv2
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon



class Visualize():
    def __init__(self, image, ax = None, title="", preffix = "", save_dir = "./plot", figsize=(16, 16)) -> None:
        if not ax:
            self.fig, self.ax = plt.subplots(1, figsize=figsize) # type: ignore
        self.gt_color = 'y'
        self.pred_color = 'r'
        self.active_class:list = []

        self.save_dir = save_dir

        height, width = image.shape[:2]
        self.image = image.astype(np.uint32).copy()
        self.title = title
        self.ax.set_ylim(height + 10, -10)
        self.ax.set_xlim(-10, width + 10)
        self.ax.axis('off')
        self.ax.set_title(title)

        self.preffix = preffix

    def set_color(self, gt_colot, pred_color):
        self.gt_color   = gt_colot
        self.pred_color = pred_color

    def set_active_class(self, active_class):
        self.active_class = active_class

    def clear(self):
        self.ax.clear()

    def show(self, if_save = True):
        self.ax.imshow(self.image)
        if if_save:
            plt.savefig(os.path.join(self.save_dir, "{}{}.jpg".format(self.preffix, self.title)), dpi=600)
        else:
            plt.show()
        
    def plot_instance(self, boxes, class_ids, 
                      class_names:list[str] = [], scores:list[float]=[], 
                      gt_boxes:list[np.ndarray]=[], masks:list[np.ndarray]=[]):
        assert len(class_ids) == boxes.shape[0], "class_ids 的长度必须与 boxes 长度一致"
        assert len(class_names) == 0 or len(class_names) == boxes.shape[0], "class_names 的长度必须与 boxes 长度一致或为空"
        assert len(scores) == 0 or len(scores) == boxes.shape[0], "scores 的长度必须与 boxes 长度一致或为空"
        assert len(gt_boxes) == 0 or len(gt_boxes) == boxes.shape[0], "gt_boxes 的长度必须与 boxes 长度一致或为空"
        assert len(masks) == 0 or len(masks) == boxes.shape[0], "masks 的长度必须与 boxes 长度一致或为空"
        
        colors = self.random_colors(boxes.shape[0])
        #根据包围框大小自动适应标注大小
        for i in range(boxes.shape[0]):
            color = colors[i]
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            self.ax.add_patch(p)

            class_id = class_ids[i]
            score = "{:.3f}".format(scores[i])  if len(scores) else ""
            label = class_names[i]              if len(class_names) else ""
            caption = label + ' ' + score
            self.ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")
        if len(gt_boxes):
            for i in range(len(gt_boxes)):
                color = self.gt_color
                y1, x1, y2, x2 = gt_boxes[i]
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                    alpha=0.7, linestyle="dashed",
                                    edgecolor=color, facecolor='none')
                self.ax.add_patch(p)

    def plot_kp(self,   pred_kp:np.ndarray = np.array([]),  pred_kp_visib = np.array([]),
                        gt_kp = np.array([]),               gt_kp_visib = np.array([])):
        ax = self.ax
        for kp, kp_visib, color in zip((pred_kp, gt_kp), (pred_kp_visib, gt_kp_visib), (self.pred_color, self.gt_color)):
            if kp_visib.size == 0:
                kp_visib = np.ones(kp.shape[0:2], np.bool_)
            for i in range(np.shape(kp)[0]):
                kp_size = np.max(kp[i], axis=0) - np.min(kp[i], axis=0)
                fontsize = int(np.sqrt(kp_size[0] * kp_size[1]/80))
                self._plot_kp(ax, kp[i], kp_visib[i], color, fontsize)

    def plot_bbox_3d(self, pred_bbox_3d = np.array([]), gt_bbox_3d = np.array([])):
        ax = self.ax
        for bbox_3d, color in zip((pred_bbox_3d, gt_bbox_3d), (self.pred_color, self.gt_color)):
            if bbox_3d.size > 0:
                self._plot_bbox_3d(ax, bbox_3d, color)

    @staticmethod
    def _plot_bbox_3d(ax:plt.Axes, bbox_3d:np.ndarray, linecolor)->None:
        # 画线
        plt.plot([bbox_3d[0,1], bbox_3d[1,1]], [bbox_3d[0,0], bbox_3d[1,0]], color=linecolor)
        plt.plot([bbox_3d[0,1], bbox_3d[6,1]], [bbox_3d[0,0], bbox_3d[6,0]], color=linecolor)
        plt.plot([bbox_3d[6,1], bbox_3d[7,1]], [bbox_3d[6,0], bbox_3d[7,0]], color=linecolor)
        plt.plot([bbox_3d[1,1], bbox_3d[7,1]], [bbox_3d[1,0], bbox_3d[7,0]], color=linecolor)

        plt.plot([bbox_3d[2,1], bbox_3d[3,1]], [bbox_3d[2,0], bbox_3d[3,0]], color=linecolor)
        plt.plot([bbox_3d[2,1], bbox_3d[4,1]], [bbox_3d[2,0], bbox_3d[4,0]], color=linecolor)
        plt.plot([bbox_3d[4,1], bbox_3d[5,1]], [bbox_3d[4,0], bbox_3d[5,0]], color=linecolor)
        plt.plot([bbox_3d[3,1], bbox_3d[5,1]], [bbox_3d[3,0], bbox_3d[5,0]], color=linecolor)

        plt.plot([bbox_3d[0,1], bbox_3d[2,1]], [bbox_3d[0,0], bbox_3d[2,0]], color=linecolor)
        plt.plot([bbox_3d[1,1], bbox_3d[3,1]], [bbox_3d[1,0], bbox_3d[3,0]], color=linecolor)
        plt.plot([bbox_3d[7,1], bbox_3d[5,1]], [bbox_3d[7,0], bbox_3d[5,0]], color=linecolor)
        plt.plot([bbox_3d[6,1], bbox_3d[4,1]], [bbox_3d[6,0], bbox_3d[4,0]], color=linecolor)
        # 画点['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w'], color=' coral '
        for i in range(8):
            ax.scatter(bbox_3d[i, 1], bbox_3d[i, 0], s=20, c=linecolor)
            ax.text(bbox_3d[i, 1], bbox_3d[i, 0], "p{}".format(i), c='k')



    @staticmethod
    def _plot_kp(ax:plt.Axes, kp:np.ndarray, kp_visib, linecolor, size)->None:
        kpnum = kp.shape[0]
        for pos, visib, idx in zip(kp, kp_visib, range(kpnum)):
            if visib:
                ax.scatter( pos[1], pos[0], s=size, c=linecolor)
                ax.text(    pos[1], pos[0], "p{}".format(idx), c='w', fontdict={'size':size})   
            else:
                pass 
        # ax.show()

    @staticmethod
    def random_colors(N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

def verify_dataset(dataset, verify_num):
    dataset_size = len(dataset.image_info)
    sample_idx = np.linspace(0, dataset_size, verify_num, endpoint= False, dtype=np.int32)
    for idx in sample_idx:
        plt.clf()
        rgb = dataset.load_image(idx)
        masks, ids = dataset.load_mask(idx)
        masks = np.transpose(masks, (2,0,1))
        keypoints = dataset.load_keypoint(idx)
        compose = np.zeros(rgb.shape)
        for mask in masks:
            mask = mask.astype(np.uint8) * 255
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            compose += mask
        compose = np.clip(compose, 0, 255)
        compose = (0.3 * compose + 0.7 * rgb).astype(np.uint8)
        plt.imshow(compose)
        for kp in keypoints:
            kp = np.reshape(kp, (24, 2))
            plt.scatter(kp[:,1], kp[:,0])
        plt.show()

if __name__ == "__main__":
    from utils import ShapesDataset
    class_info_json   = r"./class_info_json/class_info_grasp.json"
    dataset_json = r"./dataset_json/grasp_kp_morrison.json"
    dataset = ShapesDataset(class_info_json)
    with open(dataset_json) as f:
        info = json.load(f)
    dataset.load_from_json(info)
    dataset.prepare()
    verify_dataset(dataset, 50)

############################################################
#  Visualization
############################################################


# def apply_mask(image, mask, color, alpha=0.5):
#     """Apply the given mask to the image.
#     """
#     for c in range(3):
#         image[:, :, c] = np.where(mask == 1,
#                                   image[:, :, c] *
#                                   (1 - alpha) + alpha * color[c] * 255,
#                                   image[:, :, c])
#     return image
# def display_instances(image, boxes, class_ids, class_names,
#                       scores=None, title="",
#                       figsize=(16, 16), pred_kp:np.ndarray = None, gt_kp = None, pred_kp_visib = None, gt_kp_visib = None, 
#                       pred_bbox_3d = None, gt_bbox_3d = None, 
#                       ax=None,
#                       show_mask=True, show_bbox=True, show_bbox_3d=True,
#                       colors=None, captions=None, active_class = None):
#     """
#     boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
#     masks: [height, width, num_instances]
#     class_ids: [num_instances]
#     class_names: list of class names of the dataset
#     scores: (optional) confidence scores for each box
#     title: (optional) Figure title
#     show_mask, show_bbox: To show masks and bounding boxes or not
#     figsize: (optional) the size of the image
#     colors: (optional) An array or colors to use with each object
#     captions: (optional) A list of strings to use as captions for each object

#     pred_kp: [num_instance, kpnum, 2]
#     gt_kp: [num_instance, kpnum, 2]
#     pred_kp_visib: [num_instance, kpnum]
#     gt_kp_visib:[num_instance, kpnum]
#     """
#     # Number of instances
#     N = boxes.shape[0]
#     # if not N:
#     #     print("\n*** No instances to display *** \n")
#     # else:
#     #     assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

#     # If no axis is passed, create one and automatically call show()
#     auto_show = False
#     if not ax:
#         _, ax = plt.subplots(1, figsize=figsize)
#         auto_show = True

#     # Generate random colors
#     colors = colors or random_colors(N)

#     # Show area outside image boundaries.
#     height, width = image.shape[:2]
#     ax.set_ylim(height + 10, -10)
#     ax.set_xlim(-10, width + 10)
#     ax.axis('off')
#     ax.set_title(title)

#     masked_image = image.astype(np.uint32).copy()
#     size = (np.sqrt((boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1]))/10).astype(np.int32) #根据包围框大小自动适应标注大小

#     for i in range(np.shape(gt_kp)[0]):
#         plot_kp(ax, gt_kp[i], gt_kp_visib[i],  'y', size[i])

#     if gt_bbox_3d is not None:
#         for i in range(np.shape(gt_kp)[0]):
#             plot_bbox_3d(ax, gt_bbox_3d[i], 'y')

#     for i in range(N):
#         color = colors[i]
#         if active_class is not None:
#             if class_ids[i] not in np.where(active_class)[0]:
#                 continue
#         # Bounding box
#         if not np.any(boxes[i]):
#             # Skip this instance. Has no bbox. Likely lost in image cropping.
#             continue
#         y1, x1, y2, x2 = boxes[i]
#         if show_bbox:
#             p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
#                                 alpha=0.7, linestyle="dashed",
#                                 edgecolor=color, facecolor='none')
#             ax.add_patch(p)

#         # 3维包围盒（如果有输入）
#         if pred_kp is not None:
#             plot_kp(ax, pred_kp[i], pred_kp_visib[i],  'r', size[i])

#         if pred_bbox_3d is not None:
#             plot_bbox_3d(ax, pred_bbox_3d[i], 'r')

#         # Label
#         if not captions:
#             class_id = class_ids[i]
#             score = scores[i] if scores is not None else None
#             label = class_names[class_id]
#             caption = "{} {:.3f}".format(label, score) if score else label
#         else:
#             caption = captions[i]
#         ax.text(x1, y1 + 8, caption,
#                 color='w', size=11, backgroundcolor="none")

#         # # Mask
#         # mask = masks[:, :, i]
#         # if show_mask:
#         #     masked_image = apply_mask(masked_image, mask, color)

#         # # Mask Polygon
#         # # Pad to ensure proper polygons for masks that touch image edges.
#         # padded_mask = np.zeros(
#         #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
#         # padded_mask[1:-1, 1:-1] = mask
#         # contours = find_contours(padded_mask, 0.5)
#         # for verts in contours:
#         #     # Subtract the padding and flip (y, x) to (x, y)
#         #     verts = np.fliplr(verts) - 1
#         #     p = Polygon(verts, facecolor="none", edgecolor=color)
#         #     ax.add_patch(p)
#     ax.imshow(masked_image.astype(np.uint8))
#     plt.savefig(r"C:/Users/NingXiao/Desktop/val_result/{}.jpg".format(title), dpi=600)
#     # if auto_show:
#     #     plt.show()
# def display_differences(image,
#                         gt_box, gt_class_id, gt_mask,
#                         pred_box, pred_class_id, pred_score, pred_mask,
#                         class_names, title="", ax=None,
#                         show_mask=True, show_box=True,
#                         iou_threshold=0.5, score_threshold=0.5):
#     """Display ground truth and prediction instances on the same image."""
#     # Match predictions to ground truth
#     gt_match, pred_match, overlaps = utils.compute_matches(
#         gt_box, gt_class_id, gt_mask,
#         pred_box, pred_class_id, pred_score, pred_mask,
#         iou_threshold=iou_threshold, score_threshold=score_threshold)
#     # Ground truth = green. Predictions = red
#     colors = [(0, 1, 0, .8)] * len(gt_match)\
#            + [(1, 0, 0, 1)] * len(pred_match)
#     # Concatenate GT and predictions
#     class_ids = np.concatenate([gt_class_id, pred_class_id])
#     scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
#     boxes = np.concatenate([gt_box, pred_box])
#     masks = np.concatenate([gt_mask, pred_mask], axis=-1)
#     # Captions per instance show score/IoU
#     captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
#         pred_score[i],
#         (overlaps[i, int(pred_match[i])]
#             if pred_match[i] > -1 else overlaps[i].max()))
#             for i in range(len(pred_match))]
#     # Set title if not provided
#     title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
#     # Display
#     display_instances(
#         image,
#         boxes, masks, class_ids,
#         class_names, scores, ax=ax,
#         show_bbox=show_box, show_mask=show_mask,
#         colors=colors, captions=captions,
#         title=title)


# def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
#     """
#     anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
#     proposals: [n, 4] the same anchors but refined to fit objects better.
#     """
#     masked_image = image.copy()

#     # Pick random anchors in case there are too many.
#     ids = np.arange(rois.shape[0], dtype=np.int32)
#     ids = np.random.choice(
#         ids, limit, replace=False) if ids.shape[0] > limit else ids

#     fig, ax = plt.subplots(1, figsize=(12, 12))
#     if rois.shape[0] > limit:
#         plt.title("Showing {} random ROIs out of {}".format(
#             len(ids), rois.shape[0]))
#     else:
#         plt.title("{} ROIs".format(len(ids)))

#     # Show area outside image boundaries.
#     ax.set_ylim(image.shape[0] + 20, -20)
#     ax.set_xlim(-50, image.shape[1] + 20)
#     ax.axis('off')

#     for i, id in enumerate(ids):
#         color = np.random.rand(3)
#         class_id = class_ids[id]
#         # ROI
#         y1, x1, y2, x2 = rois[id]
#         p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
#                               edgecolor=color if class_id else "gray",
#                               facecolor='none', linestyle="dashed")
#         ax.add_patch(p)
#         # Refined ROI
#         if class_id:
#             ry1, rx1, ry2, rx2 = refined_rois[id]
#             p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
#                                   edgecolor=color, facecolor='none')
#             ax.add_patch(p)
#             # Connect the top-left corners of the anchor and proposal for easy visualization
#             ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

#             # Label
#             label = class_names[class_id]
#             ax.text(rx1, ry1 + 8, "{}".format(label),
#                     color='w', size=11, backgroundcolor="none")

#             # Mask
#             m = utils.unmold_mask(mask[id], rois[id]
#                                   [:4].astype(np.int32), image.shape)
#             masked_image = apply_mask(masked_image, m, color)

#     ax.imshow(masked_image)

#     # Print stats
#     print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
#     print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
#     print("Positive Ratio: {:.2f}".format(
#         class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# # TODO: Replace with matplotlib equivalent?
# def draw_box(image, box, color):
#     """Draw 3-pixel width bounding boxes on the given image array.
#     color: list of 3 int values for RGB.
#     """
#     y1, x1, y2, x2 = box
#     image[y1:y1 + 2, x1:x2] = color
#     image[y2:y2 + 2, x1:x2] = color
#     image[y1:y2, x1:x1 + 2] = color
#     image[y1:y2, x2:x2 + 2] = color
#     return image


# def display_top_masks(image, mask, class_ids, class_names, limit=4):
#     """Display the given image and the top few class masks."""
#     to_display = []
#     titles = []
#     to_display.append(image)
#     titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
#     # Pick top prominent classes in this image
#     unique_class_ids = np.unique(class_ids)
#     mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
#                  for i in unique_class_ids]
#     top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
#                                     key=lambda r: r[1], reverse=True) if v[1] > 0]
#     # Generate images and titles
#     for i in range(limit):
#         class_id = top_ids[i] if i < len(top_ids) else -1
#         # Pull masks of instances belonging to the same class.
#         m = mask[:, :, np.where(class_ids == class_id)[0]]
#         m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
#         to_display.append(m)
#         titles.append(class_names[class_id] if class_id != -1 else "-")
#     display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


# def plot_precision_recall(AP, precisions, recalls):
#     """Draw the precision-recall curve.

#     AP: Average precision at IoU >= 0.5
#     precisions: list of precision values
#     recalls: list of recall values
#     """
#     # Plot the Precision-Recall curve
#     _, ax = plt.subplots(1)
#     ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
#     ax.set_ylim(0, 1.1)
#     ax.set_xlim(0, 1.1)
#     _ = ax.plot(recalls, precisions)


# def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
#                   overlaps, class_names, threshold=0.5):
#     """Draw a grid showing how ground truth objects are classified.
#     gt_class_ids: [N] int. Ground truth class IDs
#     pred_class_id: [N] int. Predicted class IDs
#     pred_scores: [N] float. The probability scores of predicted classes
#     overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictions and GT boxes.
#     class_names: list of all class names in the dataset
#     threshold: Float. The prediction probability required to predict a class
#     """
#     gt_class_ids = gt_class_ids[gt_class_ids != 0]
#     pred_class_ids = pred_class_ids[pred_class_ids != 0]

#     plt.figure(figsize=(12, 10))
#     plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.yticks(np.arange(len(pred_class_ids)),
#                ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
#                 for i, id in enumerate(pred_class_ids)])
#     plt.xticks(np.arange(len(gt_class_ids)),
#                [class_names[int(id)] for id in gt_class_ids], rotation=90)

#     thresh = overlaps.max() / 2.
#     for i, j in itertools.product(range(overlaps.shape[0]),
#                                   range(overlaps.shape[1])):
#         text = ""
#         if overlaps[i, j] > threshold:
#             text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
#         color = ("white" if overlaps[i, j] > thresh
#                  else "black" if overlaps[i, j] > 0
#                  else "grey")
#         plt.text(j, i, "{:.3f}/n{}".format(overlaps[i, j], text),
#                  horizontalalignment="center", verticalalignment="center",
#                  fontsize=9, color=color)

#     plt.tight_layout()
#     plt.xlabel("Ground Truth")
#     plt.ylabel("Predictions")


# def draw_boxes(image, boxes=None, refined_boxes=None,
#                masks=None, captions=None, visibilities=None,
#                title="", ax=None):
#     """Draw bounding boxes and segmentation masks with different
#     customizations.

#     boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
#     refined_boxes: Like boxes, but draw with solid lines to show
#         that they're the result of refining 'boxes'.
#     masks: [N, height, width]
#     captions: List of N titles to display on each box
#     visibilities: (optional) List of values of 0, 1, or 2. Determine how
#         prominent each bounding box should be.
#     title: An optional title to show over the image
#     ax: (optional) Matplotlib axis to draw on.
#     """
#     # Number of boxes
#     assert boxes is not None or refined_boxes is not None
#     N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

#     # Matplotlib Axis
#     if not ax:
#         _, ax = plt.subplots(1, figsize=(12, 12))

#     # Generate random colors
#     colors = random_colors(N)

#     # Show area outside image boundaries.
#     margin = image.shape[0] // 10
#     ax.set_ylim(image.shape[0] + margin, -margin)
#     ax.set_xlim(-margin, image.shape[1] + margin)
#     ax.axis('off')

#     ax.set_title(title)

#     masked_image = image.astype(np.uint32).copy()
#     for i in range(N):
#         # Box visibility
#         visibility = visibilities[i] if visibilities is not None else 1
#         if visibility == 0:
#             color = "gray"
#             style = "dotted"
#             alpha = 0.5
#         elif visibility == 1:
#             color = colors[i]
#             style = "dotted"
#             alpha = 1
#         elif visibility == 2:
#             color = colors[i]
#             style = "solid"
#             alpha = 1

#         # Boxes
#         if boxes is not None:
#             if not np.any(boxes[i]):
#                 # Skip this instance. Has no bbox. Likely lost in cropping.
#                 continue
#             y1, x1, y2, x2 = boxes[i]
#             p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
#                                   alpha=alpha, linestyle=style,
#                                   edgecolor=color, facecolor='none')
#             ax.add_patch(p)

#         # Refined boxes
#         if refined_boxes is not None and visibility > 0:
#             ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
#             p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
#                                   edgecolor=color, facecolor='none')
#             ax.add_patch(p)
#             # Connect the top-left corners of the anchor and proposal
#             if boxes is not None:
#                 ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

#         # Captions
#         if captions is not None:
#             caption = captions[i]
#             # If there are refined boxes, display captions on them
#             if refined_boxes is not None:
#                 y1, x1, y2, x2 = ry1, rx1, ry2, rx2
#             ax.text(x1, y1, caption, size=11, verticalalignment='top',
#                     color='w', backgroundcolor="none",
#                     bbox={'facecolor': color, 'alpha': 0.5,
#                           'pad': 2, 'edgecolor': 'none'})

#         # Masks
#         if masks is not None:
#             mask = masks[:, :, i]
#             masked_image = apply_mask(masked_image, mask, color)
#             # Mask Polygon
#             # Pad to ensure proper polygons for masks that touch image edges.
#             padded_mask = np.zeros(
#                 (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
#             padded_mask[1:-1, 1:-1] = mask
#             contours = find_contours(padded_mask, 0.5)
#             for verts in contours:
#                 # Subtract the padding and flip (y, x) to (x, y)
#                 verts = np.fliplr(verts) - 1
#                 p = Polygon(verts, facecolor="none", edgecolor=color)
#                 ax.add_patch(p)
#     ax.imshow(masked_image.astype(np.uint8))


# def display_table(table):
#     """Display values in a table format.
#     table: an iterable of rows, and each row is an iterable of values.
#     """
#     html = ""
#     for row in table:
#         row_html = ""
#         for col in row:
#             row_html += "<td>{:40}</td>".format(str(col))
#         html += "<tr>" + row_html + "</tr>"
#     html = "<table>" + html + "</table>"
#     IPython.display.display(IPython.display.HTML(html))


# def display_weight_stats(model):
#     """Scans all the weights in the model and returns a list of tuples
#     that contain stats about each weight.
#     """
#     layers = model.get_trainable_layers()
#     table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
#     for l in layers:
#         weight_values = l.get_weights()  # list of Numpy arrays
#         weight_tensors = l.weights  # list of TF tensors
#         for i, w in enumerate(weight_values):
#             weight_name = weight_tensors[i].name
#             # Detect problematic layers. Exclude biases of conv layers.
#             alert = ""
#             if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
#                 alert += "<span style='color:red'>*** dead?</span>"
#             if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
#                 alert += "<span style='color:red'>*** Overflow?</span>"
#             # Add row
#             table.append([
#                 weight_name + alert,
#                 str(w.shape),
#                 "{:+9.4f}".format(w.min()),
#                 "{:+10.4f}".format(w.max()),
#                 "{:+9.4f}".format(w.std()),
#             ])
#     display_table(table)


# def plot_heatmap(image, heatmap, objnum = 0, save_path=None):
#     '''
#     显示关键点的热图
#     image:   [h,w, 3]
#     heatmap: [h,w, kp, obj]
#     '''
#     mask = np.zeros((heatmap.shape[0], heatmap.shape[1]))
#     for i in range(heatmap.shape[2]):
#         mask += heatmap[:,:, i, objnum]
#     mask = mask/mask.max()
#     image = image/255
#     mask = np.tile(np.expand_dims(mask, -1), [1,1,3])

#     mix = image*0.7 + mask*0.3

#     plt.imshow(mix)
#     plt.show()
#     if save_path is not None:
#         try:
#             plt.savefig(save_path)
#         except:
#             print("要保存的路径不存在")