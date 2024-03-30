package com.hyfly.unit04;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.Translator;
import com.hyfly.unit04.entity.FixedHiddenMLP;
import com.hyfly.unit04.entity.MLP;
import com.hyfly.unit04.entity.MySequential;
import com.hyfly.unit04.entity.NestMLP;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test01 {

    public static void main(String[] args) throws Exception {

        try (NDManager manager = NDManager.newBaseManager()) {
            int inputSize = 20;
            NDArray x = manager.randomUniform(0, 1, new Shape(2, inputSize)); // (2, 20) shape

            try (Model model = Model.newInstance("lin-reg")) {
                SequentialBlock net = new SequentialBlock();

                net.add(Linear.builder().setUnits(256).build());
                net.add(Activation.reluBlock());
                net.add(Linear.builder().setUnits(10).build());
                net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
                net.initialize(manager, DataType.FLOAT32, x.getShape());

                model.setBlock(net);

                //
                Translator translator = new NoopTranslator();

                NDList xList = new NDList(x);

                Predictor predictor = model.newPredictor(translator);

                NDArray ndArray = ((NDList) predictor.predict(xList)).singletonOrThrow();

                log.info(ndArray.toDebugString(true));

                //
                MLP net1 = new MLP(inputSize);
                net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
                net.initialize(manager, DataType.FLOAT32, x.getShape());

                model.setBlock(net1);

                Predictor predictor2 = model.newPredictor(translator);

                NDArray ndArray1 = ((NDList) predictor2.predict(xList)).singletonOrThrow();

                log.info(ndArray1.toDebugString(true));

                //
                MySequential net2 = new MySequential();
                net2.add(Linear.builder().setUnits(256).build());
                net2.add(Activation.reluBlock());
                net2.add(Linear.builder().setUnits(10).build());

                net2.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
                net2.initialize(manager, DataType.FLOAT32, x.getShape());

                Model model2 = Model.newInstance("my-sequential");
                model2.setBlock(net2);

                Predictor predictor3 = model2.newPredictor(translator);
                NDArray ndArray2 = ((NDList) predictor3.predict(xList)).singletonOrThrow();

                log.info(ndArray2.toDebugString(true));

                //
                FixedHiddenMLP net3 = new FixedHiddenMLP();

                net3.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
                net3.initialize(manager, DataType.FLOAT32, x.getShape());

                Model model3 = Model.newInstance("fixed-mlp");
                model3.setBlock(net3);

                Predictor predictor4 = model3.newPredictor(translator);
                NDArray ndArray3 = ((NDList) predictor4.predict(xList)).singletonOrThrow();

                log.info(ndArray3.toDebugString(true));

                //
                SequentialBlock chimera = new SequentialBlock();

                chimera.add(new NestMLP());
                chimera.add(Linear.builder().setUnits(20).build());
                chimera.add(new FixedHiddenMLP());

                chimera.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
                chimera.initialize(manager, DataType.FLOAT32, x.getShape());
                Model model4 = Model.newInstance("chimera");
                model4.setBlock(chimera);

                Predictor predictor5 = model4.newPredictor(translator);
                NDArray ndArray4 = ((NDList) predictor5.predict(xList)).singletonOrThrow();

                log.info(ndArray4.toDebugString(true));
            }
        }
    }
}
