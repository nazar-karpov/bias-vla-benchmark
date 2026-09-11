"""Рендер SAPIEN на явно заданном устройстве: python test_render_dev.py cuda:1"""
import sys, time
import sapien
spec = sys.argv[1] if len(sys.argv) > 1 else "cuda"
t0 = time.time()
d = sapien.Device(spec)
print(f"resolved: spec={spec} name={d.name} cuda_id={d.cuda_id} pci={d.pci_string} is_cuda={d.is_cuda()}", flush=True)
scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(d)])
scene.add_ground(0)
cam = scene.add_camera("c", 64, 64, 1.0, 0.1, 10)
cam.set_local_pose(sapien.Pose([-2, 0, 1]))
scene.update_render(); cam.take_picture(); img = cam.get_picture("Color")
print(f"RENDER_OK {spec} {img.shape} {time.time()-t0:.1f}s", flush=True)
