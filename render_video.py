#
# Modified from render.py to support video output
#
import matplotlib.pyplot as plt
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import cv2
import numpy as np
from utils.general_utils import safe_state, depth_with_colormap
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
from typing import List, Tuple
from scipy.spatial.transform import Rotation, Slerp

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


class RenderCamera:
    def __init__(self, R, T, FoVx, FoVy, image_height, image_width, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        self.R = torch.tensor(R, dtype=torch.float32).cuda()
        self.T = torch.tensor(T, dtype=torch.float32).cuda()
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height = image_height
        self.image_width = image_width

        self.zfar = 1e10
        self.znear = 0.01

        self.trans = torch.tensor(trans, dtype=torch.float32).cuda()
        self.scale = scale

        self.world_view_transform = torch.tensor(
            getWorld2View2(self.R.cpu().numpy(),
                           self.T.cpu().numpy(),
                           trans, scale)).transpose(0, 1).cuda()

        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar,
            fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def get_render_poses(cameras, n_interp: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract camera poses from existing cameras and interpolate between them"""
    base_poses = []

    # Collect poses from all cameras
    for cam in cameras:
        base_poses.append((cam.R, cam.T))

    # If interpolation is needed, insert transition frames between adjacent cameras
    if n_interp > 0:
        interpolated_poses = []
        n_poses = len(base_poses)

        for i in range(n_poses):
            # Add current keyframe
            interpolated_poses.append(base_poses[i])

            # Interpolate between current frame and next frame (except for the last frame)
            if i < n_poses - 1:
                R1, T1 = base_poses[i]
                R2, T2 = base_poses[i + 1]

                # Interpolate between keyframes
                rot1 = Rotation.from_matrix(R1)
                rot2 = Rotation.from_matrix(R2)
                slerp = Slerp([0, 1], Rotation.from_matrix([R1, R2]))

                for t in np.linspace(0, 1, n_interp + 2)[1:-1]:
                    R = slerp(t).as_matrix()
                    T = (1 - t) * T1 + t * T2
                    interpolated_poses.append((R, T))

        return interpolated_poses
    else:
        return base_poses


def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR for OpenCV
    return img


def render_video(model_path, iteration, views, gaussians, pipeline, background,
                 train_test_exp, separate_sh, fps=30, n_interp=3):
    """Render video with interpolated camera poses"""
    video_dir = os.path.join(model_path, f"videos_{iteration}")
    makedirs(video_dir, exist_ok=True)

    # Get video dimensions and camera parameters
    template_camera = views[0]
    h, w = template_camera.image_height, template_camera.image_width

    # Generate camera pose sequence
    print(f"Generating camera poses from {len(views)} views...")
    render_poses = get_render_poses(views, n_interp)
    print(f"Generated {len(render_poses)} poses after interpolation")

    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writers = {}
    video_paths = {
        "render": os.path.join(video_dir, "rendered.mp4"),
        "clear": os.path.join(video_dir, "clear.mp4"),
        "depth": os.path.join(video_dir, "depth.mp4"),
        "attenuation": os.path.join(video_dir, "attenuation.mp4"),
        "backscatter": os.path.join(video_dir, "backscatter.mp4")
    }

    # Record rendering performance
    render_times = []  # Only for recording 3DGS rendering time

    try:
        # Initialize video writers
        for key, path in video_paths.items():
            writer = cv2.VideoWriter()
            success = writer.open(path, fourcc, fps, (w, h), True)
            if not success:
                raise RuntimeError(f"Failed to open video writer for {path}")
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
            video_writers[key] = writer

        num = 0
        # Render each frame
        for idx, (R, T) in enumerate(tqdm(render_poses, desc=f"Rendering frames")):

            # Create rendering camera
            view = RenderCamera(
                R=R,
                T=T,
                FoVx=template_camera.FoVx,
                FoVy=template_camera.FoVy,
                image_height=h,
                image_width=w
            )

            # Record rendering time (use CUDA synchronization to ensure accurate timing)
            torch.cuda.synchronize()
            render_start = time.time()

            # Render
            render_pkg = render(view, gaussians, pipeline, background,
                                use_trained_exp=train_test_exp, separate_sh=separate_sh)

            torch.cuda.synchronize()
            render_end = time.time()
            render_time = render_end - render_start
            render_times.append(render_time)

            # Prepare output
            outputs = {
                "render": render_pkg["render"],
                "clear": render_pkg["restore_scene"],
                "depth": depth_with_colormap(render_pkg["depth"], colormap='turbo'),
                "attenuation": render_pkg["attenuation"],
                "backscatter": render_pkg["backscatter"]
            }

            if train_test_exp:
                for key in outputs:
                    outputs[key] = outputs[key][..., outputs[key].shape[-1] // 2:]

            # Write frames
            for key, tensor in outputs.items():
                frame = tensor_to_numpy(tensor)
                video_writers[key].write(frame)

            # Display rendering progress and performance
            if idx % 10 == 0:
                recent_fps = 1.0 / np.mean(render_times[-10:]) if len(render_times) >= 10 else 0
                print(f"Frame {idx}/{len(render_poses)}, Recent rendering speed: {recent_fps:.2f} FPS")

    except Exception as e:
        print(f"Error during rendering: {str(e)}")
        raise

    finally:
        # Close video writers
        for writer in video_writers.values():
            if writer is not None:
                writer.release()

        # Save rendering performance statistics
        stats_path = os.path.join(video_dir, 'render_performance.txt')
        with open(stats_path, 'w') as f:
            avg_time = np.mean(render_times) * 1000  # Convert to milliseconds
            avg_fps = 1.0 / np.mean(render_times)
            min_fps = 1.0 / np.max(render_times)
            max_fps = 1.0 / np.min(render_times)

            f.write("3DGS Rendering Performance Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total frames: {len(render_times)}\n")
            f.write(f"Average render time: {avg_time:.2f} ms\n")
            f.write(f"Average render speed: {avg_fps:.2f} FPS\n")
            f.write(f"Min render speed: {min_fps:.2f} FPS\n")
            f.write(f"Max render speed: {max_fps:.2f} FPS\n\n")

            f.write("Frame-by-frame render times:\n")
            f.write("Frame\tTime(ms)\tFPS\n")
            for i, t in enumerate(render_times):
                f.write(f"{i}\t{t * 1000:.2f}\t{1.0 / t:.2f}\n")

        # Display rendering results
        print("\nRendering completed:")
        print(f"Average rendering speed: {avg_fps:.2f} FPS")
        for path in video_paths.values():
            if os.path.exists(path):
                file_size = os.path.getsize(path)
                print(f"Generated video at {path} (size: {file_size / 1024 / 1024:.2f} MB)")


def render_set(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
               fps: int, n_interp: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        dataset.eval = True
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Use Testing camera sequence
        # all_cameras = scene.getTrainCameras()
        all_cameras = scene.getTestCameras()

        render_video(dataset.model_path, scene.loaded_iter,
                     all_cameras, gaussians, pipeline, background,
                     dataset.train_test_exp, SPARSE_ADAM_AVAILABLE,
                     fps, n_interp)


if __name__ == "__main__":
    parser = ArgumentParser(description="Video rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fps", default=10, type=int,
                        help="Output video FPS")
    parser.add_argument("--n_interp", default=23, type=int,
                        help="Number of interpolated frames between keyframes")
    args = get_combined_args(parser)
    print("Rendering video for " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_set(model.extract(args), args.iteration,
               pipeline.extract(args), args.fps, args.n_interp)