package com.hyfly.unit08;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.recurrent.LSTM;
import com.hyfly.utils.timemachine.RNNModel;
import com.hyfly.utils.timemachine.TimeMachine;
import com.hyfly.utils.timemachine.TimeMachineDataset;
import com.hyfly.utils.timemachine.Vocab;

public class Test04 {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            // 加载数据
            int batchSize = 32;
            int numSteps = 35;
            Device device = manager.getDevice();
            TimeMachineDataset dataset = new TimeMachineDataset.Builder()
                    .setManager(manager)
                    .setMaxTokens(10000)
                    .setSampling(batchSize, false)
                    .setSteps(numSteps)
                    .build();
            dataset.prepare();
            Vocab vocab = dataset.getVocab();

            // 通过设置 .optBidirectional(true) 来定义双向LSTM模型
            int vocabSize = vocab.length();
            int numHiddens = 256;
            int numLayers = 2;
            LSTM lstmLayer =
                    LSTM.builder()
                            .setNumLayers(numLayers)
                            .setStateSize(numHiddens)
                            .optReturnState(true)
                            .optBatchFirst(false)
                            .optBidirectional(true)
                            .build();

            // Train the model
            RNNModel model = new RNNModel(lstmLayer, vocabSize);
            int numEpochs = Integer.getInteger("MAX_EPOCH", 500);

            int lr = 1;
            TimeMachine.trainCh8(model, dataset, vocab, lr, numEpochs, device, false, manager);
        }
    }
}
