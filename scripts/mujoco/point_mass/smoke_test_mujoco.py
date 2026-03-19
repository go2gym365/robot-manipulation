import mujoco
import numpy as np


XML = """
<mujoco model="point_mass">
    <option timestep="0.01" gravity="0 0 0"/>
    <worldbody>
        <body name="point" pos="0 0 0">
            <joint name="slide_x" type="slide" axis="1 0 0"/>
            <joint name="slide_y" type="slide" axis="0 1 0"/>
            <geom type="sphere" size="0.05" rgba="0.2 0.6 0.9 1"/>
        </body>
    </worldbody>
    <actuator>
        <motor joint="slide_x" ctrlrange="-1 1" gear="1"/>
        <motor joint="slide_y" ctrlrange="-1 1" gear="1"/>
    </actuator>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    print("nq:", model.nq) # position state 갯수
    print("nv:", model.nv) # velocity state 갯수
    print("nu:", model.nu) # control input 갯수

    for t in range(100):
        data.ctrl[:] = np.array([0.5, -0.2], dtype=np.float64)
        mujoco.mj_step(model, data)

        if t % 10 == 0:
            print(
                f"step={t:3d} "
                f"qpos={data.qpos.copy()} "
                f"qvel={data.qvel.copy()}"
            )


if __name__ == "__main__":
    main()