package com.hyfly.unit04;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.NormalInitializer;
import com.hyfly.unit04.entity.MyInit;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test02 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray x = manager.randomUniform(0, 1, new Shape(2, 4));

            SequentialBlock net = new SequentialBlock();

            net.add(Linear.builder().setUnits(8).build());
            net.add(Activation.reluBlock());
            net.add(Linear.builder().setUnits(1).build());
            net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
            net.initialize(manager, DataType.FLOAT32, x.getShape());

            ParameterStore ps = new ParameterStore(manager, false);
            net.forward(ps, new NDList(x), false).head();

            //
            ParameterList params = net.getParameters();
            // Print out all the keys (unique!)
            for (var pair : params) {
                System.out.println(pair.getKey());
            }

            // Use the unique key to access the Parameter
            NDArray dense0Weight = params.get("01Linear_weight").getArray();
            NDArray dense0Bias = params.get("01Linear_bias").getArray();

            // Use indexing to access the Parameter
            NDArray dense1Weight = params.valueAt(2).getArray();
            NDArray dense1Bias = params.valueAt(3).getArray();

            System.out.println(dense0Weight);
            System.out.println(dense0Bias);

            System.out.println(dense1Weight);
            System.out.println(dense1Bias);

            //
            dense0Weight.getGradient();

            //
            SequentialBlock rgnet = new SequentialBlock();
            rgnet.add(block2());
            rgnet.add(Linear.builder().setUnits(10).build());
            rgnet.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
            rgnet.initialize(manager, DataType.FLOAT32, x.getShape());

            rgnet.forward(ps, new NDList(x), false).singletonOrThrow();

            log.info(rgnet.toString());

            //
            for (var param : rgnet.getParameters()) {
                System.out.println(param.getValue().getArray());
            }

            //
            Block majorBlock1 = rgnet.getChildren().get(0).getValue();
            Block subBlock2 = majorBlock1.getChildren().valueAt(1);
            Block linearLayer1 = subBlock2.getChildren().valueAt(0);
            NDArray bias = linearLayer1.getParameters().valueAt(1).getArray();

            log.info(bias.toDebugString(true));

            //
            net.setInitializer(new ConstantInitializer(1), Parameter.Type.WEIGHT);
            net.initialize(manager, DataType.FLOAT32, x.getShape());
            Block linearLayer = net.getChildren().get(0).getValue();
            NDArray weight = linearLayer.getParameters().get(0).getValue().getArray();

            log.info(weight.toDebugString(true));

            //
            net = getNet();
            net.setInitializer(new ConstantInitializer(42), Parameter.Type.WEIGHT);
            net.initialize(manager, DataType.FLOAT32, new Shape(2, 4));
            Block linearLayer2 = net.getChildren().get(0).getValue();
            NDArray weight2 = linearLayer2.getParameters().get(0).getValue().getArray();
            log.info(weight2.toDebugString(true));

            //
            net = getNet();
            net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
            net.initialize(manager, DataType.FLOAT32, new Shape(2, 4));
            Block linearLayer3 = net.getChildren().valueAt(0);
            NDArray weight3 = linearLayer3.getParameters().valueAt(0).getArray();
            log.info(weight3.toDebugString(true));

            //
            net = getNet();
            ParameterList params2 = net.getParameters();

            params2.get("01Linear_weight").setInitializer(new NormalInitializer());
            params2.get("03Linear_weight").setInitializer(Initializer.ONES);

            net.initialize(manager, DataType.FLOAT32, new Shape(2, 4));

            System.out.println(params2.valueAt(0).getArray());
            System.out.println(params2.valueAt(2).getArray());

            //
            net = getNet();
            net.setInitializer(new MyInit(), Parameter.Type.WEIGHT);
            net.initialize(manager, DataType.FLOAT32, x.getShape());
            Block linearLayer4 = net.getChildren().valueAt(0);
            NDArray weight4 = linearLayer4.getParameters().valueAt(0).getArray();
            log.info(weight4.toDebugString(true));

            //
            NDArray weightLayer = net.getChildren().valueAt(0)
                    .getParameters().valueAt(0).getArray();
            weightLayer.addi(7);
            weightLayer.divi(9);
            weightLayer.set(new NDIndex(0, 0), 2020); // set the (0, 0) index to 2020
            log.info(weightLayer.toDebugString(true));

            //
            SequentialBlock net1 = new SequentialBlock();

            // 我们需要给共享层一个名称，以便可以引用它的参数。
            Block shared = Linear.builder().setUnits(8).build();
            SequentialBlock sharedRelu = new SequentialBlock();
            sharedRelu.add(shared);
            sharedRelu.add(Activation.reluBlock());

            net1.add(Linear.builder().setUnits(8).build());
            net1.add(Activation.reluBlock());
            net1.add(sharedRelu);
            net1.add(sharedRelu);
            net1.add(Linear.builder().setUnits(10).build());

            NDArray x1 = manager.randomUniform(-10f, 10f, new Shape(2, 20), DataType.FLOAT32);

            net1.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
            net1.initialize(manager, DataType.FLOAT32, x1.getShape());

            net1.forward(ps, new NDList(x1), false).singletonOrThrow();

            // Check that the parameters are the same
            NDArray shared1 = net1.getChildren().valueAt(2)
                    .getParameters().valueAt(0).getArray();
            NDArray shared2 = net1.getChildren().valueAt(3)
                    .getParameters().valueAt(0).getArray();
            shared1.eq(shared2);
        }


    }

    public static SequentialBlock getNet() {
        SequentialBlock net = new SequentialBlock();
        net.add(Linear.builder().setUnits(8).build());
        net.add(Activation.reluBlock());
        net.add(Linear.builder().setUnits(1).build());
        return net;
    }

    public static SequentialBlock block1() {
        SequentialBlock net = new SequentialBlock();
        net.add(Linear.builder().setUnits(32).build());
        net.add(Activation.reluBlock());
        net.add(Linear.builder().setUnits(16).build());
        net.add(Activation.reluBlock());
        return net;
    }

    public static SequentialBlock block2() {
        SequentialBlock net = new SequentialBlock();
        for (int i = 0; i < 4; i++) {
            net.add(block1());
        }
        return net;
    }

}
