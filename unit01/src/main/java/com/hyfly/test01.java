package com.hyfly;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class test01 {

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {

            log.info("1 ====================================");

            NDArray one = manager.arange(12);
            log.info(one.toDebugString(true));
            log.info(String.valueOf(one.getShape()));

            log.info("2 ====================================");

            NDArray two = manager.ones(new Shape(3, 4));
            log.info(two.toDebugString(true));
            log.info(String.valueOf(two.getShape()));

            log.info("3 ====================================");

            NDArray three = manager.randomNormal(new Shape(3, 3, 4));
            log.info(three.toDebugString(true));
            log.info(String.valueOf(three.getShape()));
        }
    }
}
