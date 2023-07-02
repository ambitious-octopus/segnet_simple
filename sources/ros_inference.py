#!/usr/bin/env python3

import torch
import torchvision.transforms as transforms

import cv2
from PIL import Image

import numpy as np
import threading

import rospy
from cv_bridge import CvBridge
from sensor_msgs import msg

import custom_model


class PrednetSegmentation:
    def __init__(self, name, gpu_factor, device, class_names, with_ros=True):
        """Initialize ROS comm and Prednet PyTorch module """

        # Setup Pytorch device
        self.name = name
        self.device = device

        # Setup ROS and Thread processing
        self.frame_queue_lock = threading.Lock()
        self.cv_bridge = CvBridge()
        self.frames = []

        # Setup classes
        self.active_class_indices = [ 0, 1, 2 ]
        active_class_names = [ class_names[i] for i in self.active_class_indices ]

        # Setup model
        torch.cuda.set_per_process_memory_fraction(gpu_factor, 0)
        self.model, self.input_size, self.transform_image = self.LoadModel(len(self.active_class_indices), self.device)

        # Setup ROS publishers and subscribers
        self.with_ros = with_ros
        if self.with_ros:
            self.ros_seg_pubs = [ rospy.Publisher("/prednet_segmentation/{}/image".format(name), msg.Image, queue_size=10) \
                                  for name in active_class_names ]

            self.ros_frame_sub = rospy.Subscriber(cam_topic, msg.Image, self._ros_frame_cb)

    def LoadModel(self, num_classes, device):
        """Load pytorch model and image transforms"""
        model, input_size = custom_model.initialize_model(num_classes+1, keep_feature_extract=True, use_pretrained=False)

        state_dict = torch.load("train.pth", map_location=device)

        model = model.to(device)
        model.load_state_dict(state_dict)
        model.eval()

        #transforms_image =  transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #])
        transform_image = transforms.Compose(
                [
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )

        return model, input_size, transform_image

    def ExecuteStep(self):
        """Execute one segmentation step. """
        # Wait for first frame
        if not len(self.frames):
            print("No frames received")
            return

        # Segment images
        with self.frame_queue_lock:
            image = self.frames[-1]

            segment_images = self.SegmentImage(image)

            self.frames.clear()

        # Publish segmentation masks
        for i in range(0, len(self.active_class_indices)):
            ros_img = self.cv_bridge.cv2_to_imgmsg(segment_images[i], encoding="passthrough")
            self.ros_seg_pubs[i].publish(ros_img)

    def SegmentImage(self, image):
        with torch.no_grad():
            out = self.model(image)["out"]
            _, preds = torch.max(out, 1)
            preds = preds.to("cpu")
            preds_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)

            print(preds_np.shape)
            self.frames.clear()

        segment_images = []
        for i in range(0, len(self.active_class_indices)):
            class_img = np.where(preds_np == self.active_class_indices[i], 1.0, 0.0)
            segment_images.append(class_img)

        return segment_images

    def TestFile(self, filein, fileout_base="seg_img"):
        pil_image = Image.open(filein)
        torch_img = self._pil_to_torch(pil_image)

        seg_imgs = self.SegmentImage(torch_img)
        for i in range(0, len(seg_imgs)):
            seg_img = Image.fromarray(seg_imgs[i])
            seg_img = seg_img.convert("L")
            seg_img.save(f"{fileout_base}_{i}.png")

    def _pil_to_torch(self, pil_image):
        torch_img = self.transform_image(pil_image)
        torch_img = torch_img.unsqueeze(0)

        torch_img = torch_img.to(self.device)

        return torch_img

    def _convert_cv2_to_torch(self, cv_frame):
        """Convert from cv2 image to pytorch tensor"""
        image = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image)
        return self._pil_to_torch(pil_image)

    def _push_ros_frame(self, ros_frame):
        """Push ros frame onto self.frames"""
        # Convert to cv image
        cv_image = self.cv_bridge.imgmsg_to_cv2(ros_frame, desired_encoding="passthrough")
        tensor_frame = self._convert_cv2_to_torch(cv_image)

        # Insert into frame array, store self.n_frames previous frames
        with self.frame_queue_lock:
            self.frames.append(tensor_frame)

    def _ros_frame_cb(self, data):
        # Push new frame onto stack
        self._push_ros_frame(data)
        self.ExecuteStep()


if __name__ == "__main__":
    try:
        rospy.init_node("prednet_segmentation")

        gpu_fact = rospy.get_param("segmentation_gpu_factor", 1.0)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cam_topic="/camera/camera/image"

        seg_class_names = [
                'power_drill',
                'screwdriver',
                'plate',
        ]

        module = PrednetSegmentation("prednet_segmentation",
                                     gpu_factor=gpu_fact,
                                     device=device,
                                     class_names=seg_class_names)

    except rospy.ROSInterruptException:
        pass

    rospy.spin()
