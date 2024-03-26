package com.hyfly.unit01;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Test06 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            float[] fairProbsArr = new float[6];

            for (int i = 0; i < fairProbsArr.length; i++) {
                fairProbsArr[i] = 1f / 6;
            }
            NDArray fairProbs = manager.create(fairProbsArr);
            NDArray ndArray = manager.randomMultinomial(1, fairProbs);

            log.info(ndArray.toDebugString(true));
        }
    }
}
