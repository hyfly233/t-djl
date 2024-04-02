package com.hyfly.unit08;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.recurrent.LSTM;
import com.hyfly.utils.timemachine.RNNModel;
import com.hyfly.utils.timemachine.TimeMachine;
import com.hyfly.utils.timemachine.TimeMachineDataset;
import com.hyfly.utils.timemachine.Vocab;

public class Test03 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            int batchSize = 32;
            int numSteps = 35;

            TimeMachineDataset dataset = new TimeMachineDataset.Builder()
                    .setManager(manager)
                    .setMaxTokens(10000)
                    .setSampling(batchSize, false)
                    .setSteps(numSteps)
                    .build();
            dataset.prepare();
            Vocab vocab = dataset.getVocab();

            //
            int vocabSize = vocab.length();
            int numHiddens = 256;
            int numLayers = 2;
            Device device = manager.getDevice();
            LSTM lstmLayer =
                    LSTM.builder()
                            .setNumLayers(numLayers)
                            .setStateSize(numHiddens)
                            .optReturnState(true)
                            .optBatchFirst(false)
                            .build();

            RNNModel model = new RNNModel(lstmLayer, vocabSize);

            //
            int numEpochs = Integer.getInteger("MAX_EPOCH", 500);

            int lr = 2;
            TimeMachine.trainCh8(model, dataset, vocab, lr, numEpochs, device, false, manager);
        }
    }
}
