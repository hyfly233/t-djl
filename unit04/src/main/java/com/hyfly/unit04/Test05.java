package com.hyfly.unit04;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;

@Slf4j
public class Test05 {

    public static void main(String[] args) {
        System.out.println(Device.cpu());
        System.out.println(Device.gpu());
        System.out.println(Device.gpu(1));

        System.out.println("GPU count: " + Engine.getInstance().getGpuCount());
        Device d = Device.gpu(1);

        System.out.println(d.getDevices());

        System.out.println(tryGpu(0));
        System.out.println(tryGpu(3));

        Arrays.toString(tryAllGpus());


        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.create(new int[]{1, 2, 3});
            Device device = x.getDevice();
            log.info("Device: " + device);

            NDArray x1 = manager.ones(new Shape(2, 3), DataType.FLOAT32, tryGpu(0));

            log.info("x1: " + x1.toDebugString(true));

            NDArray y = manager.randomUniform(-1, 1, new Shape(2, 3), DataType.FLOAT32, tryGpu(1));
            log.info("y: " + y.toDebugString(true));

            NDArray z = x1.toDevice(tryGpu(1), true);
            System.out.println(x1);
            System.out.println(z);

            NDArray add = y.add(z);

            log.info("add: " + add.toDebugString(true));
        }

    }

    public static Device tryGpu(int i) {
        return Engine.getInstance().getGpuCount() > i ? Device.gpu(i) : Device.cpu();
    }

    /* Return all available GPUs or the [CPU] if no GPU exists */
    public static Device[] tryAllGpus() {
        int gpuCount = Engine.getInstance().getGpuCount();
        if (gpuCount > 0) {
            Device[] devices = new Device[gpuCount];
            for (int i = 0; i < gpuCount; i++) {
                devices[i] = Device.gpu(i);
            }
            return devices;
        }
        return new Device[]{Device.cpu()};
    }
}
