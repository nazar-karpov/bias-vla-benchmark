"""Проверка: рендерит ли SAPIEN на этой машине через NVIDIA Vulkan (на RunPod с драйвером 580
take_picture дедлочился). Запускать с timeout."""
import time
import numpy as np
import sapien

t0 = time.time()
print("sapien", sapien.__version__, flush=True)
scene = sapien.Scene()
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [1, 1, 1])
b = scene.create_actor_builder()
b.add_box_visual(half_size=[0.1, 0.1, 0.1], material=[0.8, 0.2, 0.2])
b.build_static(name="box")
cam = scene.add_camera(name="cam", width=320, height=240, fovy=1.0, near=0.05, far=10)
cam.set_entity_pose(sapien.Pose([-1, 0, 0.3]))
scene.update_render()
print("before take_picture", round(time.time() - t0, 1), flush=True)
cam.take_picture()
rgba = cam.get_picture("Color")
print("render ok", rgba.shape, float(np.asarray(rgba)[..., :3].mean()), "t=%.1fs" % (time.time() - t0), flush=True)
