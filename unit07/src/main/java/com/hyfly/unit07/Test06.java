package com.hyfly.unit07;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.recurrent.RNN;
import ai.djl.training.ParameterStore;
import com.hyfly.utils.timemachine.RNNModel;
import com.hyfly.utils.timemachine.TimeMachine;
import com.hyfly.utils.timemachine.TimeMachineDataset;
import com.hyfly.utils.timemachine.Vocab;

public class Test06 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            int batchSize = 32;
            int numSteps = 35;

            TimeMachineDataset dataset = new TimeMachineDataset.Builder()
                    .setManager(manager).setMaxTokens(10000).setSampling(batchSize, false)
                    .setSteps(numSteps).build();
            dataset.prepare();
            Vocab vocab = dataset.getVocab();

            int numHiddens = 256;
            RNN rnnLayer = RNN.builder().setNumLayers(1)
                    .setStateSize(numHiddens).optReturnState(true).optBatchFirst(false).build();

            NDList state = beginState(manager, batchSize, 1, numHiddens);
            System.out.println(state.size());
            System.out.println(state.get(0).getShape());

            //
            NDArray X = manager.randomUniform(0, 1, new Shape(numSteps, batchSize, vocab.length()));

            NDList input = new NDList(X, state.get(0));
            rnnLayer.initialize(manager, DataType.FLOAT32, input.getShapes());
            NDList forwardOutput = rnnLayer.forward(new ParameterStore(manager, false), input, false);
            NDArray Y = forwardOutput.get(0);
            NDArray stateNew = forwardOutput.get(1);

            System.out.println(Y.getShape());
            System.out.println(stateNew.getShape());

            //
            Device device = manager.getDevice();
            RNNModel net = new RNNModel(rnnLayer, vocab.length());
            net.initialize(manager, DataType.FLOAT32, X.getShape());
            TimeMachine.predictCh8("time traveller", 10, net, vocab, device, manager);

            //
            int numEpochs = Integer.getInteger("MAX_EPOCH", 500);

            int lr = 1;
            TimeMachine.trainCh8((Object) net, dataset, vocab, lr, numEpochs, device, false, manager);
        }
    }

    public static NDList beginState(NDManager manager, int batchSize, int numLayers, int numHiddens) {
        return new NDList(manager.zeros(new Shape(numLayers, batchSize, numHiddens)));
    }
}
